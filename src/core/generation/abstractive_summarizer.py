"""
Abstractive Summarization Module

This module provides abstractive summarization using Hugging Face transformers
and Ollama with Mistral model. Supports multiple models: T5, BART, Pegasus, 
and Mistral via Ollama with graceful fallbacks.
"""

from typing import Optional
import functools
import requests
import json

try:
	import torch
	from transformers import pipeline
	has_tf = True
except Exception:
	has_tf = False


@functools.lru_cache(maxsize=1)
def _get_device() -> int:
	"""Get the best available device (GPU or CPU)."""
	if not has_tf:
		return -1
	return 0 if torch.cuda.is_available() else -1


@functools.lru_cache(maxsize=1)
def _load_summarizer(model_name: str = 't5-small'):
	"""
	Load a summarization pipeline with fallback models.
	
	Args:
		model_name: The model to load (t5-small, facebook/bart-base, google/pegasus-xsum)
	
	Returns:
		Pipeline object or None if loading fails
	"""
	if not has_tf:
		return None
	
	device = _get_device()
	
	# Try primary model
	try:
		return pipeline('summarization', model=model_name, device=device)
	except Exception:
		pass
	
	# Fallback models in order of preference
	fallback_models = [
		'facebook/bart-base',
		'google/pegasus-xsum',
		't5-small'
	]
	
	for fallback_model in fallback_models:
		if fallback_model != model_name:
			try:
				return pipeline('summarization', model=fallback_model, device=device)
			except Exception:
				continue
	
	return None


def summarize_abstractive(text: str, max_length: int = 150, min_length: int = 40) -> str:
	"""
	Generate abstractive summary using Hugging Face transformers (optimized for speed).
	
	Args:
		text: Input text to summarize
		max_length: Maximum length of the summary
		min_length: Minimum length of the summary
	
	Returns:
		Abstractive summary or fallback text if summarization fails
	"""
	if not text or not text.strip():
		return ''
	
	# Load summarizer
	pipe = _load_summarizer()
	if pipe is None:
		# Fallback: return truncated text
		return text[:max_length] + "..." if len(text) > max_length else text
	
	# Clean and prepare text
	text = text.strip()
	
	# Truncate if too long (limit to first 1000 chars for speed)
	if len(text) > 1000:
		text = text[:1000]
	
	# Skip summarization if text is too short
	if len(text) < 100:
		return text
	
	try:
		# Use smaller max_length for speed
		calculated_max = min(max_length, 100)  # Reduced for speed
		calculated_min = min(min_length, 20)   # Reduced for speed
		
		# Generate summary with optimized parameters
		result = pipe(
			text, 
			max_length=calculated_max,
			min_length=calculated_min,
			truncation=True,
			do_sample=False,
			clean_up_tokenization_spaces=True,
			# Add speed optimizations
			early_stopping=True,
			no_repeat_ngram_size=2
		)
		
		if isinstance(result, list) and result:
			summary = result[0].get('summary_text', '').strip()
			return summary if summary else text[:max_length]
		
		return text[:max_length]
		
	except Exception as e:
		print(f"Abstractive summarization error: {e}")
		# Fallback: return truncated text
		return text[:max_length] + "..." if len(text) > max_length else text


def _check_ollama_availability() -> bool:
	"""
	Check if Ollama server is running and Mistral model is available.
	
	Returns:
		True if Ollama is available with Mistral model, False otherwise
	"""
	try:
		# Check if Ollama server is running
		response = requests.get('http://localhost:11434/api/tags', timeout=5)
		if response.status_code == 200:
			models = response.json().get('models', [])
			# Check if Mistral model is available
			for model in models:
				if 'mistral' in model.get('name', '').lower():
					return True
		return False
	except Exception:
		return False


def generate_final_response_mistral(title: str, authors: list, year: str, abstract: str, 
                                   tfidf_summary: str, lsa_summary: str, abstractive_summary: str) -> str:
	"""
	Generate a human-like final response using Mistral that synthesizes all summaries and metadata.
	
	Args:
		title: Paper title
		authors: List of authors
		year: Publication year
		abstract: Paper abstract
		tfidf_summary: TF-IDF extractive summary
		lsa_summary: LSA extractive summary
		abstractive_summary: Abstractive summary from transformers
	
	Returns:
		Human-like final response or fallback text if Mistral is unavailable
	"""
	if not _check_ollama_availability():
		# Fallback to a simple combined response
		return f"This paper '{title}' by {', '.join(authors[:3])} ({year}) focuses on the research described in the abstract. The key findings include insights from multiple analysis methods."
	
	try:
		# Prepare comprehensive prompt for meta-processing
		authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
		
		prompt = f"""Based on this paper and summaries, write a clear 2-3 sentence explanation:

Title: {title}
Authors: {authors_str} ({year})
TF-IDF: {tfidf_summary[:100]}...
LSA: {lsa_summary[:100]}...
Abstractive: {abstractive_summary[:100]}...

Write a natural response starting with "This paper" that explains what the research is about and why it matters. Keep it concise and engaging."""

		# Make request to Ollama API
		response = requests.post(
			'http://localhost:11434/api/generate',
			json={
				'model': 'mistral',
				'prompt': prompt,
				'stream': False,
				'options': {
					'temperature': 0.7,  # Higher temperature for more natural language
					'top_p': 0.9,
					'max_tokens': 300
				}
			},
			timeout=15  # Reduced timeout for faster response
		)
		
		if response.status_code == 200:
			result = response.json()
			final_response = result.get('response', '').strip()
			
			if final_response:
				return final_response
		
		# Fallback if API call fails
		return f"This paper '{title}' by {authors_str} ({year}) presents research on {title.lower()}. Based on multiple analysis methods, the work appears to focus on the key concepts and findings described in the abstract and summaries."
		
	except requests.exceptions.Timeout:
		print(f"Mistral timeout for paper: {title[:50]}... - using fallback")
		return f"This paper '{title}' by {authors_str} ({year}) presents research on {title.lower()[:50]}. Based on the analysis summaries, this work contributes valuable insights to the field."
	except requests.exceptions.RequestException as e:
		print(f"Mistral connection error for paper: {title[:50]}... - using fallback")
		return f"This paper '{title}' by {authors_str} ({year}) presents important research findings. The analysis suggests significant contributions to the field."
	except Exception as e:
		print(f"Mistral processing error for paper: {title[:50]}... - using fallback")
		return f"This paper '{title}' by {authors_str} ({year}) contains valuable research insights based on the available analysis."


def summarize_mistral(text: str, max_length: int = 150) -> str:
	"""
	DEPRECATED: Generate abstractive summary using Ollama with Mistral model.
	This function is kept for backward compatibility but should not be used in the new architecture.
	Use generate_final_response_mistral() instead for meta-processing.
	
	Args:
		text: Input text to summarize
		max_length: Maximum length of the summary (used for fallback)
	
	Returns:
		Abstractive summary using Mistral or fallback summary if Ollama is unavailable
	"""
	print("Warning: summarize_mistral() is deprecated. Use generate_final_response_mistral() for meta-processing.")
	
	if not text or not text.strip():
		return ''
	
	# Limit input text to ~1500 chars to avoid slowness
	text = text.strip()
	if len(text) > 1500:
		text = text[:1500]
	
	# Skip summarization if text is too short
	if len(text) < 100:
		return text
	
	try:
		# Prepare the prompt for Mistral
		prompt = f"Summarize this research paper in 5 bullet points:\n\n{text}"
		
		# Make request to Ollama API
		response = requests.post(
			'http://localhost:11434/api/generate',
			json={
				'model': 'mistral',
				'prompt': prompt,
				'stream': False,
				'options': {
					'temperature': 0.3,
					'top_p': 0.9,
					'max_tokens': 200
				}
			},
			timeout=30
		)
		
		if response.status_code == 200:
			result = response.json()
			summary = result.get('response', '').strip()
			
			if summary:
				# Clean up the summary - remove any extra formatting
				summary = summary.replace('•', '-').replace('*', '-')
				return summary
		
		# If Ollama request fails, fallback to transformer-based summarizer
		print("Ollama request failed, falling back to transformer-based summarizer")
		return summarize_abstractive(text, max_length)
		
	except requests.exceptions.RequestException as e:
		print(f"Ollama connection error: {e}, falling back to transformer-based summarizer")
		return summarize_abstractive(text, max_length)
	except Exception as e:
		print(f"Mistral summarization error: {e}, falling back to transformer-based summarizer")
		return summarize_abstractive(text, max_length)


def get_available_models() -> list:
	"""
	Get list of available summarization models.
	
	Returns:
		List of model names that can be used
	"""
	models = []
	
	# Add Mistral if Ollama is available
	if _check_ollama_availability():
		models.append('mistral')
	
	# Add transformer models if available
	if has_tf:
		models.extend([
			't5-small',
			'facebook/bart-base', 
			'google/pegasus-xsum',
			'facebook/bart-large-cnn',
			't5-base'
		])
	
	return models