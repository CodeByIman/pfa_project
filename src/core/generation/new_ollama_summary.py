import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
import unicodedata

logger = logging.getLogger(__name__)


def _check_ollama() -> bool:
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def _light_clean(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize('NFKC', text)
    # Remove obvious OCR artifacts and LaTeX noise lightly
    t = re.sub(r"\(cid:[^\)]+\)", " ", t)
    t = re.sub(r"\$\$[^$]{5,}\$\$", " ", t)
    t = re.sub(r"\\\[[\s\S]{1,400}?\\\]", " ", t)  # \[ ... ]
    t = re.sub(r"\\\([\s\S]{1,300}?\\\)", " ", t)  # \( ... )
    t = re.sub(r"\[(?:\d+[ ,;-]?)+\]", " ", t)        # [12, 3]
    # Normalize excessive newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def summarize_with_new_prompt(
    text: str,
    title: Optional[str] = None,
    authors: Optional[List[str]] = None,
    year: Optional[str] = None,
    model: str = 'mistral'
) -> str:
    """
    Call Ollama/Mistral with the user's specified prompt to produce a long, well-structured summary.
    """
    if not _check_ollama():
        return "Ollama is not available on localhost:11434"

    cleaned = _light_clean(text)

    authors_str = ", ".join((authors or [])[:3]) + ("..." if authors and len(authors) > 3 else "")
    title = title or "Untitled"
    year = year or ""

    prompt = f"""
Based on this text extracted from a research paper, reorganize the information.
Using ONLY the following structured extractive content, write a fluent, human-like summary (8–12 sentences, 2–3 paragraphs, no bullets) that starts with "This paper" and explains the problem/context, methodology, key results, and implications.

PAPER DETAILS (if available):
Title: {title}
Authors: {authors_str}
Year: {year}

CONTENT TO SUMMARIZE (between <<< and >>>):
<<<
{cleaned}
>>>
""".strip()

    # Save prompt for audit
    try:
        data_root = Path(__file__).resolve().parents[3] / 'data'
        prompt_dir = data_root / 'evaluation' / 'prompts'
        prompt_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        safe_title = re.sub(r'[^a-zA-Z0-9_-]+', '_', title)[:60]
        prompt_file = prompt_dir / f"prompt_new_{ts}_{safe_title}.txt"
        with prompt_file.open('w', encoding='utf-8') as pf:
            pf.write(prompt)
    except Exception:
        pass

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_predict': 1200,
                    'num_ctx': 8192,
                    'repeat_penalty': 1.05,
                    'presence_penalty': 0.1
                }
            },
            timeout=45
        )
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            # Ensure it starts with "This paper" if model prefixed something
            result = re.sub(r'^.*?(This paper)', r'\1', result, flags=re.DOTALL) or result
            return result
        else:
            return f"Ollama HTTP error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "Ollama request timed out"
    except Exception as e:
        logger.exception("New Ollama summary error")
        return f"Error: {e}"
