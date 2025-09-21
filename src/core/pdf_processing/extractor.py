from pathlib import Path
from typing import Optional
import re

try:
	import pdfplumber
	has_plumber = True
except Exception:
	has_plumber = False

try:
	from PyPDF2 import PdfReader
	has_pypdf2 = True
except Exception:
	has_pypdf2 = False


def _normalize_extracted_text(text: str) -> str:
    """Normalize extracted PDF text to restore missing spaces and fix artifacts."""
    if not text:
        return ""
    t = text
    # Replace non-breaking spaces and ligatures
    t = t.replace("\u00A0", " ")
    t = t.replace("\ufb01", "fi").replace("\ufb02", "fl")
    # Remove hyphenation at line breaks: e.g., "convolu-\ntion" -> "convolution"
    t = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", t)
    # Join lines that are not sentence boundaries into spaces
    t = re.sub(r"(?<![\.!?])\n(?=\S)", " ", t)
    # Insert spaces between a lowercase and uppercase boundary (MostGPU -> Most GPU)
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    # Insert spaces between acronyms and following words (GPUprograms -> GPU programs)
    # Use capture groups (no variable-length lookbehind)
    t = re.sub(r"([A-Z]{2,})([A-Z]?[a-z])", r"\1 \2", t)
    # Insert spaces between letters and digits
    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", t)
    t = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", t)
    # Collapse multiple spaces
    t = re.sub(r"[ \t]+", " ", t)
    # Normalize multiple newlines
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 15) -> str:
    """Extract text from PDF with caching to avoid re-extraction and fix spacing issues."""
    # Create cache file path
    cache_path = pdf_path.with_suffix('.txt')
    
    # Return cached text if available (normalize and validate)
    if cache_path.exists():
        try:
            with cache_path.open('r', encoding='utf-8') as f:
                cached = f.read()
            normalized = _normalize_extracted_text(cached)
            # If cached content seems broken (very low space density), force re-extract
            space_ratio = (normalized.count(' ') / max(len(normalized), 1)) if normalized else 0.0
            if space_ratio < 0.03 and has_plumber:
                # Attempt re-extraction instead of returning broken cache
                pass
            else:
                return normalized
        except Exception:
            pass  # If cache is corrupted, re-extract
    
    # Extract text from PDF
    text_parts = []
    if has_plumber:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for i, page in enumerate(pdf.pages[:max_pages]):
                    # Try standard extraction with mild tolerances
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ''
                    # If space density is too low, reconstruct from words
                    space_ratio = (text.count(' ') / max(len(text), 1)) if text else 0.0
                    if space_ratio < 0.05:
                        try:
                            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
                            if words:
                                text = ' '.join(w.get('text', '') for w in words)
                        except Exception:
                            pass
                    text_parts.append(text)
            extracted_text = '\n'.join(text_parts)
        except Exception:
            extracted_text = ''
    else:
        extracted_text = ''
    
    if has_pypdf2 and not extracted_text:
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages[:max_pages]):
                text_parts.append(page.extract_text() or '')
            extracted_text = '\n'.join(text_parts)
        except Exception:
            extracted_text = ''
    
    # Normalize extracted text before caching
    extracted_text = _normalize_extracted_text(extracted_text)

    # Cache the extracted text
    if extracted_text:
        try:
            with cache_path.open('w', encoding='utf-8') as f:
                f.write(extracted_text)
        except Exception:
            pass  # If caching fails, continue with extracted text

    return extracted_text