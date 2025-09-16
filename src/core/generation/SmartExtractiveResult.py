"""
Optimized Pipeline: Smart Extractive Summarization + Ollama for Final Response
No PDFs, No Heavy Transformers - Just Fast & Effective
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class SmartExtractiveResult:
    """Result from smart extractive summarization optimized for Ollama"""
    key_sentences: List[str]
    important_terms: List[str]
    paper_focus: str  # Main topic/method
    contribution: str  # Key contribution
    structured_summary: str  # Well-formatted for Ollama


def smart_extractive_for_ollama(text: str, max_sentences: int = 4) -> SmartExtractiveResult:
    """
    Smart extractive summarization optimized to feed Ollama effectively.
    Extracts key information in a structured way that Ollama can process well.
    
    Args:
        text: Input text (abstract)
        max_sentences: Maximum sentences to extract
    
    Returns:
        SmartExtractiveResult with structured information
    """
    if not text or len(text.strip()) < 50:
        return SmartExtractiveResult(
            key_sentences=["No sufficient content available"],
            important_terms=[],
            paper_focus="Unknown",
            contribution="Not specified",
            structured_summary="Insufficient content for analysis"
        )
    
    # Clean and prepare text
    text = text.strip()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Extract important terms (keywords that indicate research focus)
    research_indicators = [
        'method', 'approach', 'algorithm', 'technique', 'framework', 'model',
        'analysis', 'study', 'evaluation', 'experiment', 'results', 'findings',
        'propose', 'present', 'introduce', 'develop', 'design', 'implement',
        'machine learning', 'deep learning', 'neural network', 'AI', 'data science',
        'classification', 'prediction', 'optimization', 'performance', 'accuracy'
    ]
    
    important_terms = []
    text_lower = text.lower()
    for term in research_indicators:
        if term in text_lower:
            important_terms.append(term)
    
    # Score sentences for importance
    def score_sentence(sentence: str) -> float:
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Higher score for sentences with research indicators
        for term in important_terms:
            if term in sentence_lower:
                score += 2.0
        
        # Higher score for sentences with specific patterns
        patterns = [
            r'\b(we|this paper|our|the method|the approach)\b',
            r'\b(results?|findings?|shows?|demonstrates?)\b',
            r'\b(novel|new|improved|efficient|effective)\b',
            r'\b(compared to|outperforms|achieves)\b'
        ]
        
        for pattern in patterns:
            if re.search(pattern, sentence_lower):
                score += 1.0
        
        # Prefer sentences that aren't too short or too long
        word_count = len(sentence.split())
        if 10 <= word_count <= 30:
            score += 1.0
        
        return score
    
    # Select best sentences
    if len(sentences) <= max_sentences:
        key_sentences = sentences
    else:
        scored_sentences = [(score_sentence(s), s) for s in sentences]
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        key_sentences = [s for _, s in scored_sentences[:max_sentences]]
    
    # Identify paper focus and contribution
    paper_focus = "Research study"
    contribution = "Presents findings"
    
    for sentence in key_sentences:
        sentence_lower = sentence.lower()
        
        # Detect focus
        if any(word in sentence_lower for word in ['method', 'approach', 'algorithm']):
            if 'machine learning' in sentence_lower or 'deep learning' in sentence_lower:
                paper_focus = "Machine Learning methodology"
            elif 'data' in sentence_lower:
                paper_focus = "Data analysis approach"
            else:
                paper_focus = "Methodological approach"
        
        # Detect contribution
        if any(word in sentence_lower for word in ['novel', 'new', 'propose', 'introduce']):
            contribution = "Introduces novel approach"
        elif any(word in sentence_lower for word in ['improve', 'better', 'outperform']):
            contribution = "Improves existing methods"
        elif any(word in sentence_lower for word in ['evaluate', 'compare', 'analysis']):
            contribution = "Provides comparative analysis"
    
    # Create structured summary optimized for Ollama
    structured_summary = f"""
FOCUS: {paper_focus}
CONTRIBUTION: {contribution}
KEY POINTS: {' '.join(key_sentences[:3])}
TERMS: {', '.join(important_terms[:5])}
""".strip()
    
    return SmartExtractiveResult(
        key_sentences=key_sentences,
        important_terms=important_terms,
        paper_focus=paper_focus,
        contribution=contribution,
        structured_summary=structured_summary
    )


def generate_ollama_final_summary(
    title: str,
    authors: List[str],
    year: str,
    extractive_result: SmartExtractiveResult,
    query_context: str = ""
) -> str:
    """
    Generate final human-readable summary using Ollama with structured extractive input.
    This is where the AI magic happens - Ollama transforms extractive data into natural language.
    
    Args:
        title: Paper title
        authors: List of authors
        year: Publication year
        extractive_result: Smart extractive summary
        query_context: Original user query for context
    
    Returns:
        Natural language summary from Ollama
    """
    
    def _check_ollama() -> bool:
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=3)
            return response.status_code == 200
        except:
            return False
    
    if not _check_ollama():
        # Fallback: create a decent summary without Ollama
        authors_str = ', '.join(authors[:2]) + ('...' if len(authors) > 2 else '')
        return f"""This paper "{title}" by {authors_str} ({year}) focuses on {extractive_result.paper_focus.lower()}. {extractive_result.contribution}. Key aspects include: {', '.join(extractive_result.key_sentences[:2])}."""
    
    # Prepare optimized prompt for Ollama
    authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
    
    # Context-aware prompt
    context_hint = ""
    if query_context and len(query_context.strip()) > 3:
        context_hint = f"User is interested in: {query_context}. "
    
    prompt = f"""Write a clear, engaging 2-3 sentence summary of this research paper.

{context_hint}

Paper: "{title}" by {authors_str} ({year})

Research Summary:
- Focus: {extractive_result.paper_focus}
- Contribution: {extractive_result.contribution}
- Key findings: {' '.join(extractive_result.key_sentences[:2])}
- Important concepts: {', '.join(extractive_result.important_terms[:5])}

Write in natural language starting with "This paper" or "This research". Make it informative but accessible."""

    try:
        logger.info(f"🦙 Generating Ollama summary for: {title[:50]}...")
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.4,  # Balanced creativity
                    'top_p': 0.85,
                    'num_predict': 200,   # Reasonable length
                    'stop': ['\n\n', 'Paper:', 'Research:']  # Stop at breaks
                }
            },
            timeout=12  # Reasonable timeout
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            if result and len(result) > 50:  # Ensure meaningful response
                logger.info(f"✅ Ollama summary generated ({len(result)} chars)")
                return result
            else:
                logger.warning("⚠️ Ollama response too short, using fallback")
        else:
            logger.warning(f"⚠️ Ollama HTTP error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        logger.warning("⏰ Ollama timeout, using fast fallback")
    except Exception as e:
        logger.error(f"❌ Ollama error: {e}")
    
    # Smart fallback using extractive data
    return f"""This paper "{title}" by {authors_str} ({year}) presents work on {extractive_result.paper_focus.lower()}. {extractive_result.contribution}. The research {extractive_result.key_sentences[0] if extractive_result.key_sentences else 'contributes to the field'}."""


def fast_pipeline_extractive_ollama(
    query: str,
    papers: List[Any],  # Papers from arxiv search
    max_papers: int = 5
) -> List[Dict[str, Any]]:
    """
    Fast pipeline: Extractive summaries → Ollama final summaries
    No PDFs, no heavy transformers, just smart extraction + AI generation.
    
    Args:
        query: User's search query
        papers: List of papers from arxiv
        max_papers: Maximum papers to process
    
    Returns:
        List of processed papers with Ollama-generated summaries
    """
    
    logger.info(f"🚀 Fast pipeline processing {min(len(papers), max_papers)} papers")
    
    results = []
    
    for i, paper in enumerate(papers[:max_papers]):
        logger.info(f"📄 Processing paper {i+1}/{min(len(papers), max_papers)}: {paper.title[:50]}...")
        
        # Step 1: Smart extractive summarization
        abstract = paper.abstract or "No abstract available"
        extractive_result = smart_extractive_for_ollama(abstract, max_sentences=4)
        
        # Step 2: Generate final summary with Ollama
        final_summary = generate_ollama_final_summary(
            title=paper.title,
            authors=paper.authors,
            year=str(paper.year),
            extractive_result=extractive_result,
            query_context=query
        )
        
        # Create result
        result = {
            'paper_id': paper.id,
            'title': paper.title,
            'authors': paper.authors,
            'year': paper.year,
            'link': paper.pdf_url or paper.entry_url,
            'score': getattr(paper, 'relevance_score', 0.8),  # Default score if not available
            
            # Summaries
            'original_abstract': abstract,
            'extractive_summary': extractive_result.structured_summary,
            'final_ollama_summary': final_summary,  # This is the main one!
            
            # Additional structured data
            'paper_focus': extractive_result.paper_focus,
            'contribution': extractive_result.contribution,
            'key_terms': extractive_result.important_terms,
            'key_sentences': extractive_result.key_sentences,
            
            # For compatibility with your frontend
            'summaries': {
                'extractive': extractive_result.structured_summary,
                'abstractive': final_summary,  # Ollama-generated
                'tfidf': extractive_result.structured_summary,  # Same as extractive
                'combined': final_summary
            },
            'abstractive_summary': final_summary,  # Main display summary
            'final_response': final_summary,
            'method': 'extractive_ollama'
        }
        
        results.append(result)
        logger.info(f"✅ Paper {i+1} processed successfully")
    
    logger.info(f"🎉 Fast pipeline completed: {len(results)} papers processed")
    return results


# Integration with your existing orchestrator
def run_fast_ollama_pipeline(
    query: str,
    max_results: int = 20,
    top_k: int = 5,
    api: str = 'arxiv'
) -> Dict[str, Any]:
    """
    Main entry point for the fast Ollama-based pipeline.
    Replaces your existing run_pipeline for speed and simplicity.
    """
    
    # Import your existing modules
    from ..query_understanding.language_detection import detect_language, translate_to_english
    from ..query_understanding.intent_extraction import detect_intent
    from ..query_understanding.entity_extraction import extract_entities
    from ..query_understanding.query_expansion import expand_query
    from ..retrieval.arxiv_client import search_arxiv
    
    # Language processing (keep your existing logic)
    orig_lang = detect_language(query)
    query_en = translate_to_english(query, orig_lang)
    intent = detect_intent(query_en)
    entities = extract_entities(query_en, lang='en')
    expanded = expand_query(query_en, entities, lang='en')
    search_query = expanded['expanded_query']
    
    logger.info(f"🔍 Searching for: {search_query}")
    
    # Search papers (abstracts only - no PDFs!)
    papers = search_arxiv(search_query, max_results=max_results, sort_by='relevance', api=api)
    logger.info(f"📚 Found {len(papers)} papers")
    
    # Process with fast pipeline
    processed_papers = fast_pipeline_extractive_ollama(
        query=query_en,
        papers=papers,
        max_papers=top_k
    )
    
    # Build response in your expected format
    response = {
        'query_language': orig_lang,
        'intent': intent,
        'entities': entities,
        'expanded_query': search_query,
        'api_used': api,
        'processing_mode': 'fast_extractive_ollama',
        'summary_method': 'extractive_ollama',
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'total_papers_found': len(papers),
            'papers_processed': len(processed_papers),
            'processing_time': 'fast',
            'method_used': 'Smart extractive + Ollama generation'
        },
        'results': processed_papers
    }
    
    return response


# Test function
def test_fast_pipeline():
    """Test the fast pipeline with a sample query"""
    
    print("🧪 Testing Fast Extractive + Ollama Pipeline...")
    
    # Test extractive summarization
    sample_abstract = """
    Machine learning algorithms require large amounts of data to train effectively. 
    This paper proposes a novel data augmentation technique that can improve model 
    performance with limited training data. Our method uses generative adversarial 
    networks to create synthetic training examples. Experiments on image classification 
    tasks show that our approach achieves 15% better accuracy compared to baseline methods.
    """
    
    extractive_result = smart_extractive_for_ollama(sample_abstract)
    print(f"📊 Extractive Result:")
    print(f"  Focus: {extractive_result.paper_focus}")
    print(f"  Contribution: {extractive_result.contribution}")
    print(f"  Key terms: {extractive_result.important_terms}")
    
    # Test Ollama integration
    final_summary = generate_ollama_final_summary(
        title="Novel Data Augmentation for Machine Learning",
        authors=["John Doe", "Jane Smith"],
        year="2024",
        extractive_result=extractive_result,
        query_context="machine learning data augmentation"
    )
    
    print(f"\n🦙 Ollama Final Summary:")
    print(final_summary)
    
    print(f"\n✅ Test completed!")


if __name__ == "__main__":
    test_fast_pipeline()