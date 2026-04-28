import subprocess, sys, re
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

def clean(text):
    """Make text safe for latin-1 PDF encoding."""
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text

def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # A4 is 210mm wide. With l/r margins of 15 each, usable width = 180mm
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(15)

    in_code_block = False

    for line in lines:
        raw = line.rstrip()

        # Handle code/mermaid fences
        if "```" in raw.strip() and raw.strip().startswith("```"):
            in_code_block = not in_code_block
            if not in_code_block:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(130, 130, 130)
                pdf.cell(0, 5, clean("[See diagram/code on GitHub]"), ln=True)
                pdf.ln(1)
            continue
        if in_code_block:
            continue

        # Strip markdown formatting
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"`(.+?)`", r"[\1]", text)
        text = clean(text)

        # Blank line
        if not text.strip():
            pdf.ln(2)
            continue

        if raw.startswith("# ") and not raw.startswith("## ") and not raw.startswith("### "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(10, 50, 120)
            pdf.ln(4)
            pdf.cell(0, 10, clean(raw[2:].strip()), ln=True)
            pdf.ln(3)

        elif raw.startswith("## ") and not raw.startswith("### "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(20, 80, 160)
            pdf.ln(4)
            pdf.cell(0, 8, clean(raw[3:].strip()), ln=True)
            # Draw underline
            y = pdf.get_y()
            pdf.set_draw_color(20, 80, 160)
            pdf.set_line_width(0.4)
            pdf.line(15, y, 195, y)
            pdf.ln(3)

        elif raw.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(40, 40, 40)
            pdf.ln(2)
            pdf.cell(0, 7, clean(raw[4:].strip()), ln=True)
            pdf.ln(1)

        elif raw.startswith("- ") or raw.startswith("* "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(20)
            pdf.multi_cell(170, 6, clean("- " + text[2:].strip()))

        elif re.match(r"^\d+\.\s", raw):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(20)
            pdf.multi_cell(170, 6, clean(text.strip()))

        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(15)
            pdf.multi_cell(180, 6, clean(text.strip()))

    try:
        pdf.output(pdf_path)
        print(f"  Created: {pdf_path}")
    except PermissionError:
        fallback = f"{Path(pdf_path).stem}_updated.pdf"
        pdf.output(fallback)
        print(f"  Created: {fallback} (original file was locked)")


files = [
    ("HLD.md",                     "HLD.pdf"),
    ("LLD.md",                     "LLD.pdf"),
    ("Technical_Documentation.md", "Technical_Documentation.pdf"),
]

print("Generating PDFs...")
for md, out in files:
    md_to_pdf(md, out)
print("\nAll 3 PDFs are ready in your rag folder!")
