from typing import List
import functools
import numpy as np

try:
	from sentence_transformers import SentenceTransformer
	has_st = True
except Exception:
	has_st = False


@functools.lru_cache(maxsize=1)
def _load_model(name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
	if not has_st:
		return None
	return SentenceTransformer(name)


def embed_texts(texts: List[str]) -> np.ndarray:
	if not has_st:
		# Fallback: simple bag-of-words hashing for deterministic vectors
		from hashlib import md5
		vecs = []
		for t in texts:
			h = md5((t or '').encode('utf-8')).digest()
			arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
			arr = np.pad(arr, (0, max(0, 384 - arr.shape[0])), constant_values=0)[:384]
			arr = arr / (np.linalg.norm(arr) + 1e-8)
			vecs.append(arr)
		return np.vstack(vecs)
	model = _load_model()
	emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
	return emb


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
	b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
	return a_norm @ b_norm.T 