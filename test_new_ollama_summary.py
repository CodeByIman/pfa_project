import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import requests
import unicodedata

API_URL = "http://127.0.0.1:11434/api/generate"


def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def light_clean(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize('NFKC', text)
    t = re.sub(r"\(cid:[^\)]+\)", " ", t)
    t = re.sub(r"\$\$[^$]{5,}\$\$", " ", t)
    t = re.sub(r"\\\[[\s\S]{1,400}?\\\]", " ", t)
    t = re.sub(r"\\\([\s\S]{1,300}?\\\)", " ", t)
    t = re.sub(r"\[(?:\d+[ ,;-]?)+\]", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_prompt(text: str, title: Optional[str], authors: Optional[List[str]], year: Optional[str]) -> str:
    authors_str = ", ".join((authors or [])[:3]) + ("..." if authors and len(authors) > 3 else "")
    title = title or "Untitled"
    year = year or ""
    cleaned = light_clean(text)
    return f"""
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


def save_prompt(prompt: str, title: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "data" / "evaluation" / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    safe_title = re.sub(r'[^a-zA-Z0-9_-]+', '_', (title or 'untitled'))[:60]
    fpath = out_dir / f"prompt_cli_{ts}_{safe_title}.txt"
    fpath.write_text(prompt, encoding='utf-8')
    return fpath


def call_ollama(prompt: str, model: str, temperature: float, top_p: float, num_predict: int, num_ctx: int, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
        }
    }
    r = requests.post(API_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def warmup_model(model: str, timeout: int = 180) -> None:
    """Do a tiny generate to force-load the model into memory before the main call."""
    try:
        payload = {
            "model": model,
            "prompt": "Hello",
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 10,
                "num_ctx": 2048,
            }
        }
        r = requests.post(API_URL, json=payload, timeout=timeout)
        # Don't raise on warmup; it's best-effort
        _ = r.text
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Test Ollama Mistral summary on arbitrary text (no frontend)")
    parser.add_argument("--text_file", type=str, default=None, help="Path to a txt file containing the extractive content")
    parser.add_argument("--text", type=str, default=None, help="Inline text content (overrides --text_file if provided)")
    parser.add_argument("--title", type=str, default="Test Paper", help="Paper title")
    parser.add_argument("--authors", type=str, nargs='*', default=["John Doe", "Jane Smith"], help="List of authors")
    parser.add_argument("--year", type=str, default="2024", help="Publication year")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model name (e.g., mistral or mistral:7b)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_predict", type=int, default=300)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--no_warmup", action="store_true", help="Skip model warmup call")
    args = parser.parse_args()

    if not check_ollama():
        print("ERROR: Ollama not available at http://localhost:11434. Start Ollama and pull the model.")
        sys.exit(1)

    # Load text
    if args.text is not None:
        raw_text = args.text
    elif args.text_file is not None:
        p = Path(args.text_file)
        if not p.exists():
            print(f"ERROR: text_file not found: {p}")
            sys.exit(1)
        raw_text = p.read_text(encoding='utf-8', errors='ignore')
    else:
        # Default demo text
        raw_text = (
            "This paper introduces Acc Align, a novel multilingual word aligner based on "
            "multilingual sentence transformers (LaBSE). The method first induces word alignments "
            "directly from LaBSE and then applies a simple adapter-based finetuning to further improve performance. "
            "Experiments on seven language pairs show state-of-the-art AER, including strong zero-shot results. "
            "The approach is parameter-efficient and supports multiple language pairs in a single model."
        )

    prompt = build_prompt(raw_text, args.title, args.authors, args.year)
    prompt_path = save_prompt(prompt, args.title)
    print(f"Saved prompt to: {prompt_path}")

    try:
        if not args.no_warmup:
            print("Warming up model (tiny generate)...")
            warmup_model(args.model, timeout=min(args.timeout, 180))

        summary = call_ollama(
            prompt=prompt,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            timeout=args.timeout,
        )
        # Ensure starts from "This paper"
        summary = re.sub(r'^.*?(This paper)', r'\1', summary, flags=re.DOTALL) or summary
        print("\n===== New Ollama Result =====\n")
        print(summary)
        print("\n==============================\n")
    except requests.HTTPError as e:
        print(f"HTTP error: {e} | Response: {getattr(e, 'response', None)}")
        sys.exit(2)
    except requests.Timeout:
        print("Request timed out")
        sys.exit(3)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(4)


if __name__ == "__main__":
    main()
