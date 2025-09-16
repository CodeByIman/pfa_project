from typing import List
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def _split_into_sentences(text: str) -> List[str]:
	sents = re.split(r'(?<=[.!?])\s+', (text or '').strip())
	return [s.strip() for s in sents if len(s.strip()) > 0]


def summarize_tfidf(text: str, max_sentences: int = 3) -> str:
	sentences = _split_into_sentences(text)
	if not sentences:
		return ''
	vec = TfidfVectorizer(stop_words='english')
	try:
		X = vec.fit_transform(sentences)
	except Exception:
		return ' '.join(sentences[:max_sentences])
	scores = np.asarray(X.mean(axis=1)).ravel()
	idx = np.argsort(-scores)[:max_sentences]
	selected = [sentences[i] for i in sorted(idx)]
	return ' '.join(selected) 