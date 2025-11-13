import os
import json
import pandas as pd
import pymupdf
import openai
from tqdm import tqdm
import unicodedata

# -------------------------------
# Configuration
# -------------------------------

openai.api_key = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = "gpt-4.1-mini"
TEMPERATURE = 0.3

ANALYTICAL_CSV = "analytical_papers_summary.csv"
CROSS_CSV = "cross_comparison_matrix.csv"

PAPERS_FOLDER = "papers"
PARSED_TEXT_FOLDER = "parsed_text"

os.makedirs(PARSED_TEXT_FOLDER, exist_ok=True)

# -------------------------------
# Helper functions
# -------------------------------

def clean_text(text):
    if not text:
        return text
    # Normalize unicode to standard form
    text = unicodedata.normalize("NFKC", text)
    # Optionally replace any remaining weird quotes with straight quotes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return text

# -------------------------------
# PDF Parsing
# -------------------------------

def parse_pdf(pdf_file_path):
    """Extracts text from a PDF with page markers and saves to parsed_text folder."""
    try:
        doc = pymupdf.open(pdf_file_path)
        all_text = ""
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text")
            all_text += f"\n--- Page {i + 1} ---\n{text}"
        doc.close()

        # Save to text file in parsed_text folder
        txt_filename = os.path.join(PARSED_TEXT_FOLDER, os.path.basename(pdf_file_path).replace(".pdf", ".txt"))
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(all_text)
        print(f"Saved extracted text to {txt_filename}")

        return all_text

    except Exception as e:
        print(f"Error opening PDF {pdf_file_path}: {e}")
        return ""

# -------------------------------
# GPT Analysis
# -------------------------------

def analyze_paper(pdf_text, paper_name):
    """
    Extracts a concise analytical summary tailored to blockchain research.
    Main fields use brief sentences; 'other_details' captures nuance as newline-separated bullets.
    """
    prompt = f"""
You are an expert academic analyst specializing in blockchain research.
Summarize the following paper in **structured JSON** form.

Rules:
- Output must be **only raw JSON** (no markdown, no code fences).
- Main fields (pilot, period_of_study, methods, themes, blockchain_failures, blockchain_successes, sentiment) should be **brief sentences**, enough to convey the main idea quickly, e.g., "Survey of 150 participants. Case study in Ghana.", "Used Ethereum; offline NFC cards." or "N/A".
- Use full sentences **only** in 'methods', 'blockchain_failures', 'blockchain_successes', and 'other_details'.
- In **'other_details'**, each point must begin with the Unicode bullet character **'• '** (U+2022), followed by one sentence per line (newline-separated, not as a list).
- Do **not** use dashes, asterisks, or numbers for bullets — only '• '.
- If information is missing, write "N/A".
- Include a 'title' field (use the PDF filename or inferred title).

Return JSON in exactly this format:
{{
    "title": "...",
    "pilot": "...",
    "period_of_study": "...",
    "methods": "...",
    "themes": "...",
    "blockchain_failures": "...",
    "blockchain_successes": "...",
    "sentiment": "...",
    "other_details": "... (newline-separated bullets) ..."
}}

Paper text:
\"\"\"
{pdf_text}
\"\"\"
"""
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        result_text = clean_text(response.choices[0].message.content.strip())
        extracted = json.loads(result_text)

        # Ensure CSV-friendly newline-separated other_details
        if "other_details" in extracted and isinstance(extracted["other_details"], list):
            extracted["other_details"] = "\n".join([d.strip("- ").strip() for d in extracted["other_details"]])

        return extracted

    except json.JSONDecodeError:
        print(f"Error parsing GPT JSON for paper {paper_name}:\n{result_text}")
        return None
    except Exception as e:
        print(f"Error analyzing paper {paper_name}: {e}")
        return None

# -------------------------------
# CSV Export
# -------------------------------

def save_analytical_csv(analyses):
    df = pd.DataFrame(analyses)
    df.index.name = "Paper"
    df.reset_index(inplace=True)
    df.rename(columns={"index": "paper_name"}, inplace=True)
    df.to_csv(ANALYTICAL_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved analytical summary to {ANALYTICAL_CSV}")

def generate_cross_comparison_matrix(analyses):
    """
    Generate a pairwise cross-comparison matrix with GPT, returning concise full-sentence explanations:
    1-2 sentences on the most salient similarities, and 1-2 on differences.
    """
    paper_names = [p["paper_name"] for p in analyses]
    matrix = pd.DataFrame(index=paper_names, columns=paper_names)

    for i, paper_a in enumerate(analyses):
        for j, paper_b in enumerate(analyses):
            if i == j:
                matrix.iloc[i, j] = "Same paper."
            else:
                # Generate comparison explanation using GPT
                prompt = f"""
You are an expert academic research analyst. Compare the following two papers. 
Write a concise explanation **with 1-2 sentences on the most important similarities** 
and **1-2 sentences on the most important differences** in topics, methods, novelty, 
main findings, and sector. Do not write paragraphs; keep it to 2–4 sentences total.

Paper A:
{json.dumps(paper_a, indent=2)}

Paper B:
{json.dumps(paper_b, indent=2)}
"""
                try:
                    response = openai.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    explanation = clean_text(response.choices[0].message.content.strip())
                    matrix.iloc[i, j] = explanation
                except Exception as e:
                    print(f"Error generating comparison for {paper_a['paper_name']} vs {paper_b['paper_name']}: {e}")
                    matrix.iloc[i, j] = "Comparison failed."

    matrix.to_csv(CROSS_CSV)
    print(f"Saved cross-paper comparison matrix to {CROSS_CSV}")

# -------------------------------
# Main Processing
# -------------------------------

def process_papers(folder_path):
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    analyses = []

    for pdf_path in tqdm(pdf_files, desc="Analyzing papers"):
        paper_name = os.path.basename(pdf_path)
        text = parse_pdf(pdf_path)
        if not text:
            print(f"Skipping {paper_name} because no text could be extracted.")
            continue

        analysis = analyze_paper(text, paper_name)
        if analysis:
            analysis["paper_name"] = paper_name
            analyses.append(analysis)

    if analyses:
        save_analytical_csv(analyses)
        generate_cross_comparison_matrix(analyses)
    else:
        print("No papers were successfully analyzed.")

# -------------------------------
# Entry Point
# -------------------------------

if __name__ == "__main__":
    process_papers(PAPERS_FOLDER)
