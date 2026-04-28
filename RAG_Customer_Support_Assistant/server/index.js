import cors from "cors";
import express from "express";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..");
const pythonFile = path.join(projectRoot, "rag_agent.py");
const distDir = path.join(projectRoot, "dist");
const pythonBin = process.env.PYTHON_BIN || "python";
const PORT = Number(process.env.PORT || 7860);
const HOST = process.env.HOST || "0.0.0.0";
const allowedOrigins = new Set([
  "http://127.0.0.1:5173",
  "http://localhost:5173",
  "http://127.0.0.1:4173",
  "http://localhost:4173",
]);

const app = express();

app.disable("x-powered-by");
app.use(
  helmet({
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false,
  }),
);
if (process.env.NODE_ENV !== "production") {
  app.use(
    cors({
      origin(origin, callback) {
        if (!origin || allowedOrigins.has(origin)) {
          return callback(null, true);
        }
        return callback(new Error("Origin not allowed by CORS policy."));
      },
    }),
  );
}
app.use(express.json({ limit: "8kb" }));

const chatLimiter = rateLimit({
  windowMs: 60 * 1000,
  limit: 12,
  standardHeaders: "draft-7",
  legacyHeaders: false,
  message: {
    error: "Too many requests. Please wait a minute before asking again.",
  },
});

function validateQuestion(question) {
  if (typeof question !== "string") {
    return "Question must be a string.";
  }

  const cleaned = question.trim();
  if (!cleaned) {
    return "Question cannot be empty.";
  }

  if (cleaned.length > 500) {
    return "Question is too long. Keep it under 500 characters.";
  }

  return null;
}

function runRagQuestion(question) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonBin,
      ["-W", "ignore", pythonFile, "--question", question, "--json"],
      {
        cwd: projectRoot,
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONIOENCODING: "utf-8",
        },
      },
    );

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });

    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error("RAG engine timed out while processing the question."));
    }, 120000);

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (code) => {
      clearTimeout(timeout);

      if (code !== 0) {
        return reject(
          new Error(stderr.trim() || `RAG engine exited with code ${code}.`),
        );
      }

      try {
        const parsed = JSON.parse(stdout.trim());
        return resolve(parsed);
      } catch (error) {
        return reject(
          new Error(
            `Failed to parse RAG response. ${error.message}. Raw output: ${stdout.trim()}`,
          ),
        );
      }
    });
  });
}

app.get("/api/health", (_req, res) => {
  res.json({
    ok: true,
    api: "online",
    ragEngine: path.basename(pythonFile),
  });
});

app.post("/api/chat", chatLimiter, async (req, res) => {
  const error = validateQuestion(req.body?.question);
  if (error) {
    return res.status(400).json({ error });
  }

  try {
    const result = await runRagQuestion(req.body.question.trim());
    return res.json({
      answer: result.answer,
      escalated: Boolean(result.requires_human),
      sources: Array.isArray(result.sources) ? result.sources : [],
      relevanceScores: Array.isArray(result.relevance_scores)
        ? result.relevance_scores
        : [],
      question: result.question,
    });
  } catch (runtimeError) {
    return res.status(500).json({
      error: "The assistant could not process your request right now.",
      detail:
        process.env.NODE_ENV === "development" ? runtimeError.message : undefined,
    });
  }
});

app.use((error, _req, res, _next) => {
  const message =
    error?.message === "Origin not allowed by CORS policy."
      ? error.message
      : "Unexpected server error.";

  res.status(message === error?.message ? 403 : 500).json({ error: message });
});

if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));

  app.get("*", (req, res, next) => {
    if (req.path.startsWith("/api/")) {
      return next();
    }
    return res.sendFile(path.join(distDir, "index.html"));
  });
}

app.listen(PORT, HOST, () => {
  console.log(`RAG API listening on http://${HOST}:${PORT}`);
});
