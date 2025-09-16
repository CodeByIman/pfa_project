from ai_research_agent.src.core.generation.tfidf_summarizer import summarize_tfidf


def test_tfidf_summarizer_returns_text():
	text = "Sentence one. Sentence two is informative. Sentence three has keywords."
	summary = summarize_tfidf(text, max_sentences=2)
	assert isinstance(summary, str)
	assert len(summary) > 0 