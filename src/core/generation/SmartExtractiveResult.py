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
import unicodedata
from pathlib import Path

# PDF helpers from existing modules
try:
    from ..pdf_processing.downloader import download_pdf
    from ..pdf_processing.extractor import extract_text_from_pdf
except Exception:
    download_pdf = None
    extract_text_from_pdf = None

logger = logging.getLogger(__name__)

@dataclass
class SmartExtractiveResult:
    """Result from smart extractive summarization optimized for Ollama"""
    key_sentences: List[str]
    important_terms: List[str]
    paper_focus: str  # Main topic/method
    contribution: str  # Key contribution
    structured_summary: str  # Well-formatted for Ollama


def _robust_sentence_split(text: str) -> List[str]:
    """
    Split text into sentences while handling common academic patterns:
    - Abbreviations (e.g., e.g., i.e., Dr., Prof., et al.)
    - Citations like [12], (Smith et al., 2020)
    - Decimal numbers and acronyms
    - Preserve punctuation at end of sentence
    """
    if not text:
        return []
    # Normalize whitespace
    t = unicodedata.normalize('NFKC', text)
    # Protect common abbreviations by temporarily replacing the period
    abbrev = [
        'e.g.', 'i.e.', 'et al.', 'Fig.', 'Eq.', 'Dr.', 'Prof.', 'Mr.', 'Ms.', 'Inc.', 'Ltd.', 'vs.', 'al.'
    ]
    for a in abbrev:
        t = t.replace(a, a.replace('.', '∯'))
    # Split on sentence end punctuation followed by whitespace and a capital or digit
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\(])', t)
    sentences = []
    for p in parts:
        s = p.replace('∯', '.').strip()
        if len(s) >= 20:
            sentences.append(s)
    return sentences


def smart_extractive_for_ollama(
    text: str,
    max_sentences: int = 500,
    target_ratio: float = 0.17,
    min_paragraphs: int = 8,
    max_chars: int = 20000,
    min_lines: int = 30,
    max_lines: int = 100,
    min_line_chars: int = 300
) -> SmartExtractiveResult:
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
    # Robust sentence split to avoid mid-sentence breaks
    sentences = _robust_sentence_split(text)
    
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
    
    # Select sentences to meet target size (~target_ratio of original) but capped by max_chars
    # Score then choose top-K while preserving original order for coherence
    original_len = len(text)
    target_chars = min(max(int(original_len * target_ratio), 800), max_chars)
    scored_sentences = [(score_sentence(s), idx, s) for idx, s in enumerate(sentences)]
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    chosen = []
    chosen_char_count = 0
    for sc, idx, s in scored_sentences:
        if s in chosen:
            continue
        chosen.append(s)
        chosen_char_count += len(s) + 1
        if chosen_char_count >= target_chars:
            break
        if len(chosen) >= max(3*max_sentences, 200):
            # hard stop to avoid too many sentences
            break
    # Restore original order
    key_sentences = sorted(chosen, key=lambda s: sentences.index(s)) if chosen else sentences[:max_sentences]
    
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
    
    # Build paragraphs by grouping sentences sequentially to preserve flow
    # Aim for natural paragraph sizes (4-7 sentences), respecting max_chars
    para_min, para_max = 4, 7
    paragraphs: List[str] = []
    total = 0
    i = 0
    while i < len(key_sentences) and total < max_chars:
        group_size = min(para_max, max(para_min, (len(key_sentences) - i) // max(min_paragraphs, 1) or para_min))
        group = key_sentences[i:i+group_size]
        para = ' '.join(group)
        if total + len(para) + 2 > max_chars:
            break
        paragraphs.append(para)
        total += len(para) + 2
        i += group_size

    # Ensure minimum paragraph count when possible
    if len(paragraphs) < min_paragraphs:
        # Split longer paragraphs or add remaining sentences one by one
        remaining = key_sentences[i:]
        for s in remaining:
            if len(paragraphs) >= min_paragraphs:
                break
            if total + len(s) + 2 > max_chars:
                break
            paragraphs.append(s)
            total += len(s) + 2

    body_text = '\n\n'.join(paragraphs)

    # Create structured summary optimized for Ollama (keep metadata header + long body)
    structured_summary = body_text 
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
    Generate final human-readable summary using Ollama with improved prompt structure.
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
    
    # Prepare authors
    authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')

    context_hint = ""
    if query_context and len(query_context.strip()) > 3:
        context_hint = f"\nUSER QUERY CONTEXT: {query_context}\nMake sure to address aspects relevant to this query.\n"

    # Clean the extractive text more gently
    cleaned_extractive = _clean_extractive_text_gentle(extractive_result.structured_summary)

    # Prompt with explicit content markers to avoid prompt bleed
    prompt = f"""You are an expert research paper summarizer. Write a comprehensive, coherent summary of this research paper using ONLY the information provided below.

PAPER DETAILS:
Title: {title}
Authors: {authors_str}
Year: {year}

{context_hint}

REQUIREMENTS:
- Write 4-6 complete paragraphs (not bullet points)
- Start with: "This paper..."
- Cover: problem/context, methodology, results, and implications
- Use clear, academic language
- Stay strictly within the provided information
- Make it flow naturally as a cohesive narrative

EXTRACTED CONTENT TO SUMMARIZE (between <<< and >>>):
<<<
{cleaned_extractive}
>>>

Now write the comprehensive summary following the requirements above:"""

    try:
        logger.info(f"🦙 Generating Ollama summary for: {title[:50]}...")
        
        # Save the exact prompt for debugging/audit
        try:
            data_root = Path(__file__).resolve().parents[3] / 'data'
            prompt_dir = data_root / 'evaluation' / 'prompts'
            prompt_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            safe_title = re.sub(r'[^a-zA-Z0-9_-]+', '_', title)[:60]
            prompt_file = prompt_dir / f"prompt_{ts}_{safe_title}.txt"
            with prompt_file.open('w', encoding='utf-8') as pf:
                pf.write(prompt)
        except Exception:
            pass

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.2,
                    'top_p': 0.9,
                    'num_predict': 1200,
                    'num_ctx': 8192,
                    'repeat_penalty': 1.05,
                    'presence_penalty': 0.1
                }
            },
            timeout=40
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            
            # Clean up the response
            result = _clean_ollama_response(result)
            
            if result and len(result) > 200:  # Ensure substantially long response
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
    return _create_smart_fallback_summary(title, authors_str, year, extractive_result)


def _clean_extractive_text_gentle(text: str) -> str:
    """
    Gentle cleaning that preserves readability and context
    """
    if not text:
        return ""
    
    # Very light cleaning
    text = unicodedata.normalize('NFKC', text)
    
    # Remove only obvious OCR artifacts
    text = re.sub(r"\(cid:[^\)]+\)", " ", text)
    
    # Remove obvious LaTeX but preserve structure
    text = re.sub(r"\$\$[^$]{5,}\$\$", " [mathematical expression] ", text)
    text = re.sub(r"\\[a-z]+\{[^}]*\}", " ", text)  # LaTeX commands
    
    # Clean up citation markers but keep context
    text = re.sub(r"\[[\d,\s]+\]", "", text)
    
    # Preserve paragraph structure while cleaning
    paragraphs = []
    current_para = []
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        else:
            # Only exclude obviously broken lines
            if len(line) < 10 or line.count(' ') < 2:
                continue
            # Skip if mostly symbols
            symbol_ratio = sum(1 for c in line if not c.isalnum() and c != ' ') / max(len(line), 1)
            if symbol_ratio > 0.7:
                continue
            current_para.append(line)
    
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # Join with proper paragraph separation
    result = '\n\n'.join(p for p in paragraphs if len(p) > 30)
    
    return result[:12000]  # Reasonable limit


def _clean_ollama_response(response: str) -> str:
    """
    Clean up the Ollama response to remove artifacts
    """
    if not response:
        return ""
    
    # Remove any prompt echo or instruction artifacts
    response = re.sub(r'^.*?This paper', 'This paper', response, flags=re.DOTALL)
    
    # Remove any trailing instruction echoes
    response = re.sub(r'\n\n(Now write|REQUIREMENTS|PAPER DETAILS).*', '', response, flags=re.DOTALL)
    
    # Clean up excessive newlines but preserve paragraph structure
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Remove any incomplete sentences at the end
    sentences = response.split('.')
    if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
        response = '.'.join(sentences[:-1]) + '.'
    
    return response.strip()


def _create_smart_fallback_summary(title: str, authors_str: str, year: str, extractive_result: SmartExtractiveResult) -> str:
    """
    Create a coherent fallback summary when Ollama fails
    """
    # Use the extractive sentences to build a better fallback
    key_sentences = extractive_result.key_sentences[:4]
    
    if len(key_sentences) >= 2:
        # Create a more natural fallback
        intro = f'This paper "{title}" by {authors_str} ({year}) '
        
        if extractive_result.paper_focus:
            intro += f"focuses on {extractive_result.paper_focus.lower()}. "
        
        content = ' '.join(key_sentences[:3])
        
        conclusion = ""
        if extractive_result.contribution:
            conclusion = f" {extractive_result.contribution}"
        
        return intro + content + conclusion
    
    # Basic fallback if not enough sentences
    return f"""This paper "{title}" by {authors_str} ({year}) presents research on {extractive_result.paper_focus.lower()}. {extractive_result.contribution}. The work contributes to understanding in this field through systematic analysis and methodology."""
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
    # Prepare output directory to save extractive summaries used by Ollama
    try:
        # Default data directory relative to this file if orchestrator helper is not available
        data_root = Path(__file__).resolve().parents[3] / 'data'
        out_dir = data_root / 'evaluation' / 'extractive'
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        out_dir = None
    
    for i, paper in enumerate(papers[:max_papers]):
        logger.info(f"📄 Processing paper {i+1}/{min(len(papers), max_papers)}: {paper.title[:50]}...")
        
        # Step 1: Choose source text: prefer first pages of PDF if available to achieve longer extractive summary
        source_text = paper.abstract or "No abstract available"
        if download_pdf and extract_text_from_pdf and getattr(paper, 'pdf_url', None):
            try:
                data_root = Path(__file__).resolve().parents[3] / 'data'
                pdf_dir = data_root / 'pdfs'
                pdf_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = download_pdf(paper.pdf_url, pdf_dir)
                if pdf_path is not None:
                    extracted_text = extract_text_from_pdf(pdf_path, max_pages=20)
                    if extracted_text and len(extracted_text.strip()) > 1000:
                        source_text = extracted_text
            except Exception as _e:
                # Fall back to abstract silently
                pass

        extractive_result = smart_extractive_for_ollama(
            source_text,
            max_sentences=800,
            target_ratio=0.3,
            min_paragraphs=10,
            max_chars=22000,
            min_lines=40,
            max_lines=120,
            min_line_chars=400
        )
        
        # Step 2: Generate final summary with Ollama
        final_summary = generate_ollama_final_summary(
            title=paper.title,
            authors=paper.authors,
            year=str(paper.year),
            extractive_result=extractive_result,
            query_context=query
        )

        # Optionally save extractive summary to disk for inspection
        if out_dir is not None:
            try:
                ts = datetime.now().strftime('%Y%m%d-%H%M%S')
                safe_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(paper.id or 'unknown'))[:60]
                filename = out_dir / f"{ts}_{safe_id}_extractive.txt"
                with filename.open('w', encoding='utf-8') as f:
                    f.write(f"Title: {paper.title}\n")
                    f.write(f"Authors: {', '.join(paper.authors[:5])}\n")
                    f.write(f"Year: {paper.year}\n")
                    f.write(f"Paper ID: {paper.id}\n")
                    f.write(f"Link: {paper.pdf_url or paper.entry_url}\n")
                    f.write("\n=== Structured Extractive Summary (used by Ollama) ===\n")
                    f.write(extractive_result.structured_summary + "\n\n")
                    f.write("Key sentences:\n")
                    for s in extractive_result.key_sentences:
                        f.write(f"- {s}\n")
                    f.write("\nImportant terms: " + ", ".join(extractive_result.important_terms) + "\n")
                    f.write("Focus: " + extractive_result.paper_focus + "\n")
                    f.write("Contribution: " + extractive_result.contribution + "\n")
            except Exception as e:
                logger.warning(f"Could not save extractive summary: {e}")
        
        # Create result
        result = {
            'paper_id': paper.id,
            'title': paper.title,
            'authors': paper.authors,
            'year': paper.year,
            'link': paper.pdf_url or paper.entry_url,
            'score': getattr(paper, 'relevance_score', 0.8),  # Default score if not available
            
            # Summaries
            'original_abstract': paper.abstract or "",
            'source_text': source_text,
            'source_origin': 'pdf' if (download_pdf and extract_text_from_pdf and getattr(paper, 'pdf_url', None) and source_text != (paper.abstract or "No abstract available")) else 'abstract',
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