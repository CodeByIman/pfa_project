from pathlib import Path
from typing import Optional
import requests
import hashlib


def download_pdf(url: str, dest_dir: Path, timeout: int = 20) -> Optional[Path]:
	"""Download PDF with better error handling. Returns None if download fails."""
	if not url or not url.strip():
		return None
		
	dest_dir.mkdir(parents=True, exist_ok=True)
	name = hashlib.md5(url.encode('utf-8')).hexdigest() + '.pdf'
	path = dest_dir / name
	
	# Return cached file if it exists and has content
	if path.exists() and path.stat().st_size > 0:
		return path
		
	try:
		# Add headers to mimic a browser request
		headers = {
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
		}
		r = requests.get(url, timeout=timeout, headers=headers)
		
		# Check if response is actually a PDF
		content_type = r.headers.get('content-type', '').lower()
		if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
			print(f"Warning: URL {url} doesn't appear to be a PDF (content-type: {content_type})")
			return None
			
		r.raise_for_status()
		path.write_bytes(r.content)
		return path
		
	except requests.exceptions.HTTPError as e:
		if e.response.status_code == 403:
			print(f"Access forbidden (403) for {url} - likely requires authentication")
		elif e.response.status_code == 404:
			print(f"PDF not found (404) at {url}")
		else:
			print(f"HTTP error {e.response.status_code} for {url}")
		return None
	except requests.exceptions.RequestException as e:
		print(f"Network error downloading {url}: {e}")
		return None
	except Exception as e:
		print(f"Unexpected error downloading {url}: {e}")
		return None 