import re
import logging
import unicodedata
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

# Common abbreviations that shouldn't trigger sentence breaks
ABBREVIATIONS = {
    'dr', 'prof', 'mr', 'mrs', 'ms', 'vs', 'etc', 'fig', 'eq', 'ref', 'sec', 'ch', 'vol',
    'no', 'pp', 'p', 'al', 'cf', 'i.e', 'e.g', 'viz', 'ca', 'approx', 'est', 'max', 'min',
    'avg', 'std', 'var', 'dept', 'univ', 'corp', 'inc', 'ltd', 'co', 'govt', 'admin'
}

# Keywords that indicate important sentences
IMPORTANCE_KEYWORDS = {
    'conclusion', 'result', 'finding', 'significant', 'important', 'novel', 'contribution',
    'propose', 'demonstrate', 'show', 'prove', 'evidence', 'analysis', 'evaluation',
    'performance', 'accuracy', 'improvement', 'method', 'approach', 'algorithm'
}


def _robust_sentence_split(text: str) -> List[str]:
    """
    Robust sentence segmentation that handles abbreviations, decimals, and citations.
    """
    if not text or not text.strip():
        return []
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Protect abbreviations by temporarily replacing periods
    protected_text = text
    for abbr in ABBREVIATIONS:
        # Case insensitive replacement with word boundaries
        pattern = r'\b' + re.escape(abbr) + r'\.'
        protected_text = re.sub(pattern, abbr + '<!PERIOD!>', protected_text, flags=re.IGNORECASE)
    
    # Protect decimal numbers (e.g., 3.14, 95.2%)
    protected_text = re.sub(r'\b\d+\.\d+', lambda m: m.group().replace('.', '<!DECIMAL!>'), protected_text)
    
    # Protect citations like [1], [Smith et al.], (2020)
    protected_text = re.sub(r'\[([^\]]*\.)*[^\]]*\]', lambda m: m.group().replace('.', '<!CITATION!>'), protected_text)
    protected_text = re.sub(r'\(([^)]*\.)*[^)]*\)', lambda m: m.group().replace('.', '<!PAREN!>'), protected_text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', protected_text)
    
    # Restore protected periods and clean up
    restored_sentences = []
    for sent in sentences:
        sent = sent.replace('<!PERIOD!>', '.').replace('<!DECIMAL!>', '.').replace('<!CITATION!>', '.').replace('<!PAREN!>', '.')
        sent = sent.strip()
        if sent and len(sent) > 10:  # Filter very short fragments
            restored_sentences.append(sent)
    
    return restored_sentences


def _preprocess_text(sentences: List[str], min_length: int = 20, max_length: int = 300) -> List[str]:
    """
    Clean and filter sentences for quality.
    """
    processed = []
    
    for sent in sentences:
        # Remove excessive whitespace and special characters
        cleaned = re.sub(r'\s+', ' ', sent).strip()
        
        # Skip sentences that are too short or too long
        if len(cleaned) < min_length or len(cleaned) > max_length:
            continue
            
        # Skip sentences with too many numbers/symbols (likely tables/formulas)
        if len(re.findall(r'[0-9\(\)\[\]{}=<>+\-*/]', cleaned)) > len(cleaned) * 0.4:
            continue
            
        # Skip sentences that are mostly uppercase (likely headers)
        if len([c for c in cleaned if c.isupper()]) > len(cleaned) * 0.7:
            continue
            
        processed.append(cleaned)
    
    return processed


def _calculate_position_score(idx: int, total: int) -> float:
    """
    Calculate position-based importance score (beginning and end are more important).
    """
    if total <= 1:
        return 1.0
    
    # Normalize position to [0, 1]
    pos = idx / (total - 1)
    
    # U-shaped curve: beginning and end are more important
    if pos <= 0.1:  # First 10%
        return 1.0
    elif pos >= 0.9:  # Last 10%
        return 0.9
    elif pos <= 0.3:  # First 30%
        return 0.8
    else:  # Middle
        return 0.6


def _calculate_keyword_score(sentence: str) -> float:
    """
    Calculate importance score based on presence of key terms.
    """
    sentence_lower = sentence.lower()
    score = 0.0
    
    for keyword in IMPORTANCE_KEYWORDS:
        if keyword in sentence_lower:
            score += 1.0
    
    # Normalize by sentence length to avoid bias toward long sentences
    return min(score / max(len(sentence.split()) / 10, 1), 2.0)


def _calculate_adaptive_components(n_sentences: int, n_features: int) -> int:
    """
    Calculate optimal number of SVD components based on data size.
    """
    # Use 10-20% of sentences as components, with reasonable bounds
    ratio_based = max(int(n_sentences * 0.15), 5)
    feature_based = min(n_features - 1, 50)
    
    return min(ratio_based, feature_based, 50)


def _calculate_multi_criteria_score(lsa_scores: np.ndarray, sentences: List[str]) -> np.ndarray:
    """
    Combine LSA scores with position and keyword importance.
    """
    n_sentences = len(sentences)
    final_scores = np.zeros(n_sentences)
    
    # Normalize LSA scores to [0, 1]
    if lsa_scores.max() > 0:
        normalized_lsa = lsa_scores / lsa_scores.max()
    else:
        normalized_lsa = lsa_scores
    
    for i, sentence in enumerate(sentences):
        # Weighted combination of different factors
        lsa_weight = 0.6
        position_weight = 0.2
        keyword_weight = 0.2
        
        final_scores[i] = (
            lsa_weight * normalized_lsa[i] +
            position_weight * _calculate_position_score(i, n_sentences) +
            keyword_weight * _calculate_keyword_score(sentence)
        )
    
    return final_scores


def _split_into_sentences(text: str) -> List[str]:
    """Legacy function for backward compatibility."""
    return _robust_sentence_split(text)


def summarize_lsa(text: str, max_sentences: int = 30, min_sentence_length: int = 20, 
                 max_sentence_length: int = 300) -> str:
    """
    Enhanced LSA summarization with robust preprocessing and multi-criteria scoring.
    
    Args:
        text: Input text to summarize
        max_sentences: Maximum number of sentences in summary
        min_sentence_length: Minimum sentence length to consider
        max_sentence_length: Maximum sentence length to consider
        
    Returns:
        Extractive summary using enhanced LSA algorithm
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to LSA summarizer")
        return ''
    
    try:
        # Step 1: Robust sentence segmentation
        sentences = _robust_sentence_split(text)
        logger.debug(f"Initial sentences extracted: {len(sentences)}")
        
        if not sentences:
            logger.warning("No sentences found after segmentation")
            return ''
        
        # Step 2: Preprocessing and filtering
        sentences = _preprocess_text(sentences, min_sentence_length, max_sentence_length)
        logger.debug(f"Sentences after preprocessing: {len(sentences)}")
        
        if len(sentences) <= max_sentences:
            return ' '.join(sentences)
        
        # Step 3: TF-IDF vectorization with optimized parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=min(5000, len(sentences) * 10),  # Adaptive feature limit
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,  # Keep rare terms in small documents
            max_df=0.95,  # Remove very common terms
            sublinear_tf=True  # Use log scaling
        )
        
        X = vectorizer.fit_transform(sentences)
        logger.debug(f"TF-IDF matrix shape: {X.shape}")
        
        if X.shape[1] == 0:
            logger.warning("No features extracted from TF-IDF")
            return ' '.join(sentences[:max_sentences])
        
        # Step 4: SVD with adaptive components
        n_components = _calculate_adaptive_components(len(sentences), X.shape[1])
        logger.debug(f"Using {n_components} SVD components")
        
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_svd = svd.fit_transform(X)
        
        # Log explained variance for debugging
        explained_variance_ratio = svd.explained_variance_ratio_.sum()
        logger.debug(f"Explained variance ratio: {explained_variance_ratio:.3f}")
        
        # Step 5: Multi-criteria scoring
        lsa_scores = np.linalg.norm(X_svd, axis=1)
        final_scores = _calculate_multi_criteria_score(lsa_scores, sentences)
        
        # Step 6: Select top sentences and maintain order
        top_indices = np.argsort(-final_scores)[:max_sentences]
        selected_sentences = [sentences[i] for i in sorted(top_indices)]
        
        result = ' '.join(selected_sentences)
        logger.debug(f"Generated summary with {len(selected_sentences)} sentences")
        
        return result
        
    except ValueError as e:
        logger.error(f"LSA summarization failed with ValueError: {e}")
        # Fallback: return first sentences
        return ' '.join(sentences[:max_sentences]) if sentences else ''
    
    except Exception as e:
        logger.error(f"LSA summarization failed with unexpected error: {e}")
        # Emergency fallback: simple truncation
        fallback_sentences = _robust_sentence_split(text)
        return ' '.join(fallback_sentences[:max_sentences]) if fallback_sentences else text[:500]