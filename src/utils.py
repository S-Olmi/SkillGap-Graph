
import PyPDF2
import io
import sys
import os
import re
from typing import List, Dict, Any, Optional

def extract_text_from_pdf(pdf_path: str) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""


def extract_text_from_plain_file(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"[ERROR] Impossibile leggere il file {path}: {e}")
        return ""


def get_text_from_source(path: str) -> str:

    if os.path.isdir(path):
        print(f"[INFO] Analisi cartella rilevata: {path}")
        combined_text = []
        for root, dirs, files in os.walk(path):
            for file in files:
                # Saltiamo cartelle nascoste (.git) e file non testuali pesanti
                if not file.startswith('.') and file.endswith(('.md', '.txt', '.py', '.js', '.cpp')):
                    file_path = os.path.join(root, file)
                    combined_text.append(f"--- FILE: {file} ---\n")
                    combined_text.append(extract_text_from_plain_file(file_path))
        return "\n".join(combined_text)

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in [".md", ".txt", ".markdown"]:
        return extract_text_from_plain_file(path)
    else:
        print(f"[WARN] Estensione {ext} non supportata esplicitamente, provo lettura testuale.")
        return extract_text_from_plain_file(path)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'[^\w\s#+]', '', text.lower())
    return " ".join(text.split())

def contains_token(text: str, token: str) -> bool:
    t_norm = normalize_text(text)
    tk_norm = normalize_text(token)

    if not tk_norm:
        return False

    if len(tk_norm) <= 2:
        pattern = rf"\b{re.escape(tk_norm)}\b"
        return bool(re.search(pattern, t_norm))

    return tk_norm in t_norm

def read_multiline_input() -> str:
    return sys.stdin.read()

def read_stdin_if_available() -> Optional[str]:
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return None

def make_initial_state(cv_text: str, job_text: str, available_sources: Dict) -> Dict[str, Any]:
    return {
        "cv_raw_text": cv_text,
        "job_raw_text": job_text,
        "available_sources": available_sources,
        "candidate_skills": [],
        "role_requirements": [],
        "soft_requirements": [],
        "skill_gap": {},
        "match_score": 0.0,
        "canonical_skills": [],
        "potential_rag_targets": {},
        "triage_decision": "ACCEPT", # Default prudente
        "analysis_path": "LIGHT",     # Default
        "rag_ready": False,
        "explanatory_context": {},
        "report": "",
        "decisions": []
    }

def render_explanation(state: Dict[str, Any]):
    print("\n" + "= " *30)
    print("LOG DECISIONALE DELL'AGENTE")
    print("= " *30)
    for i, step in enumerate(state.get("decisions", []), 1):
        print(f"{i}. [{step['step']}] -> {step['rule']}")
    print("= " *30 + "\n")