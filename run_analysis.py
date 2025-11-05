import os
import json
import pandas as pd
import pymupdf
import openai
from tqdm import tqdm

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
    Extract rich analytical summary in JSON with full-sentence fields.
    """
    prompt = f"""
You are an expert academic research analyst. Summarize the following paper in JSON.
Use full sentences and provide clear, complete explanations. Return JSON strictly as:

{{
    "empirical": "...",
    "authors": "...",
    "year": "...",
    "topics": "...",
    "keywords": "...",
    "sector": "...",
    "methods": "...",
    "sample_size": "...",
    "data_type": "...",
    "novelty": "...",
    "main_findings": "...",
    "strengths": "...",
    "limitations": "...",
    "practical_relevance": "..."
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
        result_text = response.choices[0].message.content.strip()
        extracted = json.loads(result_text)
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
    df.to_csv(ANALYTICAL_CSV, index=False)
    print(f"Saved analytical summary to {ANALYTICAL_CSV}")

def generate_cross_comparison_matrix(analyses):
    """
    Generate a pairwise cross-comparison matrix with GPT, returning full-sentence explanations.
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
You are an expert academic research analyst. Compare the following two papers and write one
full sentence explaining how they are similar or different in topics, methods, novelty,
main findings, and sector. Focus on analytical nuance.

Paper A:
{json.dumps(paper_a, indent=2)}

Paper B:
{json.dumps(paper_b, indent=2)}

Write exactly one concise, full sentence explaining the relationship between Paper A and Paper B.
"""
                try:
                    response = openai.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3
                    )
                    explanation = response.choices[0].message.content.strip()
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
