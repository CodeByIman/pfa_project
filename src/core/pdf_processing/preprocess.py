import re


def clean_scientific_text(text: str) -> str:
    """
    Clean scientific text by removing citations, equations, tables, figures,
    author/affiliation headers, and other non-content elements while preserving
    the main textual content and fixing spacing issues from PDF extraction.
    """
    if not text:
        return ''

    t = text.strip()

    # --- Pre-normalization (prevent fused words) ---
    t = t.replace("\u00A0", " ")  # NBSP -> space
    t = t.replace("\ufb01", "fi").replace("\ufb02", "fl")  # ligatures
    # Fix hyphenation across line breaks and inline hyphenations
    t = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", t)
    t = re.sub(r"(?<=\w)-(?=\w)", "", t)
    # Insert spaces between camelCase/acronym-word and letter-digit boundaries
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    t = re.sub(r"([A-Z]{2,})([A-Z]?[a-z])", r"\1 \2", t)
    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", t)
    t = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", t)
    # Standardize tokens split by extraction
    t = re.sub(r"\bI\s*o\s*T\b", "IoT", t, flags=re.IGNORECASE)
    t = re.sub(r"\bar\s*Xiv\b", "arXiv", t, flags=re.IGNORECASE)

    # Remove trailing sections (references, bibliography...)
    sections_to_remove = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*acknowledgments?\s*\n",
        r"\n\s*appendix\s+[a-z]\s*\n",
        r"\n\s*supplementary\s+material\s*\n",
    ]
    for pattern in sections_to_remove:
        parts = re.split(pattern, t, flags=re.IGNORECASE)
        t = parts[0]

    # Remove inline citations
    citation_patterns = [
        r"\[(?:\s*\d+\s*(?:[,;]\s*\d+)*\s*)\]",
        r"\[(?:\s*\d+\s*-\s*\d+\s*)\]",
        r"\((?:[A-Z][a-zA-Z\-']+\s+et\s+al\.?|[A-Z][a-zA-Z\-']+(?:\s+&\s+[A-Z][a-zA-Z\-']+)?),?\s*\d{4}[a-z]?\)",
        r"\((?:\d{4}[a-z]?)\)",
        r"(?:ref\.|reference)\s+\[\d+\]",
        r"(?:see|cf\.)\s*\[[\d,\s-]+\]",
    ]
    for pattern in citation_patterns:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)

    # Remove equations and LaTeX math
    equation_patterns = [
        r"^\s*\([A-Z]?\d+\)\s*$",
        r"^\s*Eq\.\s*\(\d+\)",
        r"\$[^$]+\$",
        r"\\\[[^\]]+\\\]",
        r"\\begin\{equation\}.*?\\end\{equation\}",
        r"\\begin\{align\}.*?\\end\{align\}",
    ]
    for pattern in equation_patterns:
        t = re.sub(pattern, "", t, flags=re.DOTALL)

    # Remove leading author/affiliation block before Abstract/Introduction
    try:
        lines_all = t.splitlines()
        cut_idx = None
        for i, ln in enumerate(lines_all[:120]):
            if re.search(r"^\s*abstract\b", ln, flags=re.IGNORECASE) or re.search(r"^\s*introduction\b", ln, flags=re.IGNORECASE):
                cut_idx = i
                break
        if cut_idx is not None:
            head = lines_all[:cut_idx]
            keep_head = []
            aff_kw = r"department|university|school|laborator(y|ies)|institute|college|faculty|center|centre|laboratoire|group|lab|dept\.|city|country"
            for ln in head:
                if re.search(aff_kw, ln, flags=re.IGNORECASE):
                    continue
                if re.search(r"\b(ORCID|ORCiD)\b", ln, flags=re.IGNORECASE):
                    continue
                if re.search(r"\b[A-Za-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Za-z]+(?:\s+[A-Z][a-z]+)+)*\b", ln) and len(ln) < 200:
                    continue
                if re.search(r"\b\d{4}\b", ln) and re.search(r"\b[A-Z][a-z]+\b", ln):
                    continue
                if '@' in ln:
                    continue
                keep_head.append(ln)
            t = "\n".join(keep_head + lines_all[cut_idx:])
    except Exception:
        pass

    # Line-wise filtering
    lines = []
    for line in t.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        skip_patterns = [
            r"^table\s+\d+",
            r"^figure\s+\d+",
            r"^fig\.\s*\d+",
            r"^appendix\s+[a-z]",
            r"^section\s+\d+",
            r"^abstract\s*$",
            r"^keywords?\s*:",
            r"^introduction\s*$",
            r"^conclusion\s*$",
            r"^methodology\s*$",
            r"^results\s*$",
            r"^discussion\s*$",
            r"^\d+\.\s*$",
            r"^[A-Z\s]+$" if len(line_stripped) < 50 else r"(?!)",
            r"^.*\s+\d+\s*$" if len(line_stripped.split()) <= 3 else r"(?!)",
            r"^\s*[\d\.\s]+$",
            r"^(?:doi|url|arxiv):",
            r"^in\s+proceedings\s+of",
            r"^(?:copyright|manuscript\s+received|submitted\s+to|accepted\s+for\s+publication)",
            r"^(?:this\s+is\s+a\s+preprint|preprint\s+of)",
        ]
        if any(re.match(p, line_stripped, flags=re.IGNORECASE) for p in skip_patterns):
            continue
        if len(line_stripped) > 10:
            special_chars = sum(1 for c in line_stripped if c in "{}[]()=+-*/\\^_<>|&%$#@~`")
            if special_chars / len(line_stripped) > 0.3:
                continue
        alnum_count = sum(1 for c in line_stripped if c.isalnum())
        if len(line_stripped) > 5 and alnum_count / len(line_stripped) < 0.5:
            continue
        if len(line_stripped) < 20 and not re.search(r"[.!?]$", line_stripped):
            continue
        cleaned_line = line_stripped
        cleaned_line = re.sub(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z]\.)?\s*\d+(?:,\s*\d+)*\b", "", cleaned_line)
        cleaned_line = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", cleaned_line)
        cleaned_line = re.sub(r"https?://[^\s]+", "", cleaned_line)
        cleaned_line = re.sub(r"\s+", " ", cleaned_line).strip()
        if len(cleaned_line) > 10 and re.search(r"[a-zA-Z]", cleaned_line):
            lines.append(cleaned_line)

    result = "\n".join(lines)
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)
    result = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", result)
    result = re.sub(r"\\[a-zA-Z]+", "", result)
    result = re.sub(r" +", " ", result)

    # Remove repeated short headers/footers
    try:
        lines_res = result.splitlines()
        counts = {}
        for ln in lines_res:
            if 5 <= len(ln) <= 80:
                counts[ln] = counts.get(ln, 0) + 1
        repeated = {ln for ln, c in counts.items() if c >= 3}
        if repeated:
            lines_res = [ln for ln in lines_res if ln not in repeated]
            result = "\n".join(lines_res)
    except Exception:
        pass

    return result.strip()