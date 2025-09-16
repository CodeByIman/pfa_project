from ai_research_agent.src.core.generation.lsa_summarizer import summarize_lsa


def test_lsa_summarizer_returns_text():
	text = "Sentence one. Sentence two is informative. Sentence three has keywords. Sentence four adds context."
	summary = summarize_lsa(text, max_sentences=2)
	assert isinstance(summary, str)
	assert len(summary) > 0 