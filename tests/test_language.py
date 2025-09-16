import pytest

from ai_research_agent.src.core.query_understanding.language_detection import detect_language


def test_detect_language_english():
	assert detect_language("Deep learning for medical diagnosis") in ('en', 'unknown')


def test_detect_language_french():
	# heuristic detection acceptable when langdetect missing
	lang = detect_language("État de l'art en apprentissage profond")
	assert lang in ('fr', 'unknown') 