from typing import Dict, List
import re

try:
	from nltk.corpus import wordnet as wn
	has_wordnet = True
except Exception:
	has_wordnet = False

try:
	from googletrans import Translator
	has_translator = True
except Exception:
	has_translator = False


STOPWORDS = set([
	'the', 'and', 'of', 'a', 'an', 'in', 'to', 'for', 'on', 'with', 'by', 'is', 'are', 'be', 'this', 'that', 'it'
])

FRENCH_STOPWORDS = set([
	'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus', 'par', 'grand', 'ca', 'du', 'la', 'des', 'les'
])


def _tokenize(text: str, lang: str = 'en') -> List[str]:
	stopwords = FRENCH_STOPWORDS if lang == 'fr' else STOPWORDS
	return [t for t in re.split(r"[^a-zA-ZÀ-ÿ0-9\-]+", (text or '').lower()) if t and t not in stopwords]


def _translate_to_english(text: str) -> str:
	"""Translate French text to English for better search results"""
	if not has_translator:
		return text
	
	try:
		translator = Translator()
		result = translator.translate(text, src='fr', dest='en')
		return result.text if result and result.text else text
	except Exception:
		# If translation fails, return original text
		return text


def expand_query(query: str, entities: Dict[str, List[str]], lang: str = 'en') -> Dict[str, List[str]]:
	# If language is French, translate query to English for better search results
	original_query = query
	if lang == 'fr':
		query = _translate_to_english(query)
		# Also translate entity values
		translated_entities = {}
		for key, values in entities.items():
			translated_values = []
			for value in values:
				translated_value = _translate_to_english(value)
				translated_values.append(translated_value)
			translated_entities[key] = translated_values
		entities = translated_entities
	
	# Extract terms from translated query and entities
	terms = set(_tokenize(query, 'en'))  # Use English tokenization for translated text
	for lst in entities.values():
		for kw in lst:
			for t in _tokenize(kw, 'en'):  # Use English tokenization
				terms.add(t)

	expanded: List[str] = sorted(list(terms))
	
	# Apply WordNet expansion only on English terms
	if has_wordnet:
		try:
			for t in list(expanded):
				for syn in wn.synsets(t):
					for lemma in syn.lemma_names():
						if lemma and lemma.lower() not in terms and len(lemma) <= 30:
							terms.add(lemma.lower().replace('_', ' '))
		except LookupError:
			# wordnet corpus missing; skip synonym expansion
			pass
		expanded = sorted(list(terms))

	return {
		'expanded_terms': expanded,
		'expanded_query': ' '.join(expanded),
		'original_query': original_query,
		'translated_query': query if lang == 'fr' else None
	}