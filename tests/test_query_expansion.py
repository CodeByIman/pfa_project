from ai_research_agent.src.core.query_understanding.query_expansion import expand_query


def test_expand_query_basic():
	entities = {'domain': ['medical ai'], 'methods': ['cnn'], 'datasets': [], 'metrics': [], 'keywords': []}
	expanded = expand_query('deep learning diagnosis', entities)
	assert 'deep' in expanded['expanded_terms']
	assert 'cnn' in expanded['expanded_terms'] 