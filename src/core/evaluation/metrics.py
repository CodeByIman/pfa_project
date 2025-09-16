from typing import Dict
from pathlib import Path
import json

try:
	from rouge_score import rouge_scorer
	has_rouge = True
except Exception:
	has_rouge = False

try:
	import bert_score
	has_bertscore = True
except Exception:
	has_bertscore = False


def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
	if not has_rouge:
		return {'rouge1': 0.0, 'rougeL': 0.0}
	scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
	scores = scorer.score(reference, candidate)
	return {k: float(v.fmeasure) for k, v in scores.items()}


def compute_bertscore(candidate: str, reference: str, lang: str = 'en') -> Dict[str, float]:
	if not has_bertscore:
		return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
	P, R, F1 = bert_score.score([candidate], [reference], lang=lang)
	return {'precision': float(P.mean()), 'recall': float(R.mean()), 'f1': float(F1.mean())}


def log_human_feedback(data_dir: Path, query: str, paper_id: str, relevant: bool, notes: str = '') -> None:
	data_dir.mkdir(parents=True, exist_ok=True)
	path = data_dir / 'feedback.jsonl'
	record = {
		'query': query,
		'paper_id': paper_id,
		'relevant': bool(relevant),
		'notes': notes,
	}
	with path.open('a', encoding='utf-8') as f:
		f.write(json.dumps(record, ensure_ascii=False) + '\n') 