from pathlib import Path
from typing import Optional

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


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 15) -> str:
	"""Extract text from PDF with caching to avoid re-extraction."""
	# Create cache file path
	cache_path = pdf_path.with_suffix('.txt')
	
	# Return cached text if available
	if cache_path.exists():
		try:
			with cache_path.open('r', encoding='utf-8') as f:
				return f.read()
		except Exception:
			pass  # If cache is corrupted, re-extract
	
	# Extract text from PDF
	text_parts = []
	if has_plumber:
		try:
			with pdfplumber.open(str(pdf_path)) as pdf:
				for i, page in enumerate(pdf.pages[:max_pages]):
					text = page.extract_text() or ''
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
	
	# Cache the extracted text
	if extracted_text:
		try:
			with cache_path.open('w', encoding='utf-8') as f:
				f.write(extracted_text)
		except Exception:
			pass  # If caching fails, continue with extracted text
	
	return extracted_text 