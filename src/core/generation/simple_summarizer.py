from typing import List
import re


def simple_summarize(text: str, max_sentences: int = 3) -> str:
    """Simple extractive summarizer that avoids model loading issues."""
    if not text or len(text.strip()) < 50:
        return text or ""
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    
    # Simple scoring based on length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = len(sentence) * 0.5  # Length bonus
        if i < 3:  # First sentences bonus
            score += 10
        if i > len(sentences) - 3:  # Last sentences bonus
            score += 5
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = [s[1] for s in scored_sentences[:max_sentences]]
    
    # Sort back to original order
    result_sentences = []
    for sentence in sentences:
        if sentence in top_sentences:
            result_sentences.append(sentence)
    
    return " ".join(result_sentences)

