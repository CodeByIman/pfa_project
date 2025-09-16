import re


def clean_scientific_text(text: str) -> str:
	if not text:
		return ''
	t = text
	# Remove references section (simple heuristic)
	parts = re.split(r"\n\s*references\s*\n", t, flags=re.IGNORECASE)
	t = parts[0]
	# Remove inline citations like [12], [3,4], (Smith et al., 2020)
	t = re.sub(r"\[(?:\s*\d+\s*(?:,\s*\d+)*\s*)\]", "", t)
	t = re.sub(r"\((?:[A-Z][a-zA-Z\-']+\s+et\s+al\.|[A-Z][a-zA-Z\-']+),?\s*\d{4}[a-z]?\)", "", t)
	# Drop lines that look like tables or equations
	lines = []
	for line in t.splitlines():
		line_stripped = line.strip()
		if not line_stripped:
			continue
		if re.match(r"^table\s+\d+", line_stripped, flags=re.IGNORECASE):
			continue
		if re.match(r"^figure\s+\d+", line_stripped, flags=re.IGNORECASE):
			continue
		# many non-alnum characters (likely formulas)
		non_alnum_ratio = sum(1 for c in line_stripped if not c.isalnum() and not c.isspace()) / max(1, len(line_stripped))
		if non_alnum_ratio > 0.4:
			continue
		lines.append(line)
	return "\n".join(lines) 