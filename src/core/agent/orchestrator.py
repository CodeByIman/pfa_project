# Updated orchestrator.py with Ollama integration

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import os
import json
from datetime import datetime

from ..query_understanding.language_detection import detect_language, translate_to_english
from ..query_understanding.intent_extraction import detect_intent
from ..query_understanding.entity_extraction import extract_entities
from ..query_understanding.query_expansion import expand_query
from ..retrieval.arxiv_client import search_arxiv, Paper
from ..ranking.pre_ranking import pre_rank_papers, pre_rank_papers_abstract_only, RankedPaper
from ..generation.abstractive_summarizer import summarize_abstractive, _check_ollama_availability
from ..generation.simple_summarizer import simple_summarize

# Import the new fast pipeline
from ..generation.SmartExtractiveResult import (
    run_fast_ollama_pipeline,
    fast_pipeline_extractive_ollama,
    smart_extractive_for_ollama,
    generate_ollama_final_summary
)


@dataclass
class OrchestratedResult:
    paper: Paper
    score: float
    summary_extractive: str
    summary_abstractive: str
    summary_tfidf: str      # Add specific TF-IDF summary
    summary_lsa: str        # Add specific LSA summary
    summary_combined: str   # Add combined summary
    final_response: str     # Human-like response from Mistral meta-processor
    method_used: str        # Method used for primary summary
    # New Ollama-specific fields
    paper_focus: str = ""
    contribution: str = ""
    key_terms: List[str] = None
    ollama_summary: str = ""  # The main Ollama-generated summary


def get_data_dir() -> Path:
    root = os.environ.get('AI_RA_DATA_DIR', None)
    if root is None:
        root = str(Path(__file__).resolve().parents[3] / 'data')
    return Path(root)


def run_pipeline(
    query: str, 
    max_results: int = 20, 
    max_pdfs: int = 3, 
    top_k: int = 5, 
    api: str = 'arxiv', 
    use_pdfs: bool = False, 
    summary_mode: str = "auto", 
    use_mistral_final_response: bool = False,
    use_ollama_fast: bool = True  # New parameter for Ollama fast mode
) -> Dict[str, Any]:
    """
    Enhanced pipeline with Ollama fast mode option
    
    Args:
        use_ollama_fast: If True, use the fast extractive + Ollama pipeline
    """
    
    # If Ollama fast mode is requested, use the new pipeline
    if use_ollama_fast and _check_ollama_availability():
        print("🚀 Using Fast Ollama Pipeline (Extractive + Ollama Generation)")
        return run_fast_ollama_pipeline(
            query=query,
            max_results=max_results,
            top_k=top_k,
            api=api
        )
    
    # Original pipeline logic (fallback or when Ollama not available)
    orig_lang = detect_language(query)
    query_en = translate_to_english(query, orig_lang)
    intent = detect_intent(query_en)
    entities = extract_entities(query_en, lang='en')
    expanded = expand_query(query_en, entities, lang='en')
    search_query = expanded['expanded_query']

    papers = search_arxiv(search_query, max_results=max_results, sort_by='relevance', api=api)
    data_dir = get_data_dir()
    
    # Skip PDF processing for faster results
    if not use_pdfs:
        print(f"Using fast mode (abstracts only) with {summary_mode} summarization for {len(papers)} papers")
        print(f"Mistral final response: {'enabled' if use_mistral_final_response else 'disabled for speed'}")
        ranked: List[RankedPaper] = pre_rank_papers_abstract_only(query_en, papers, top_k=top_k, summary_mode=summary_mode, use_mistral_final_response=use_mistral_final_response)
    else:
        print(f"Using full mode (with PDFs) with {summary_mode} summarization for {min(len(papers), max_pdfs)} papers")
        print(f"Mistral final response: {'enabled' if use_mistral_final_response else 'disabled for speed'}")
        ranked: List[RankedPaper] = pre_rank_papers(query_en, papers, data_dir=data_dir, max_pdfs=max_pdfs, top_k=top_k, summary_mode=summary_mode, use_mistral_final_response=use_mistral_final_response)

    results: List[OrchestratedResult] = []
    for r in ranked:
        # Extract different summary types from the RankedPaper object
        abs_sum = r.summary_abstractive if r.summary_abstractive else "Abstractive summary not available"
        tfidf_sum = r.summary_tfidf
        lsa_sum = r.summary_lsa
        combined_sum = r.summary_combined
        final_resp = r.final_response if r.final_response else "Final response not available"
        
        # For extractive, use the original abstract or a simple extractive summary
        extractive_sum = r.paper.abstract if r.paper.abstract else combined_sum
        
        # If Ollama is available but not using fast mode, still generate Ollama summaries
        ollama_summary = ""
        paper_focus = ""
        contribution = ""
        key_terms = []
        
        if _check_ollama_availability() and not use_ollama_fast:
            try:
                extractive_result = smart_extractive_for_ollama(r.paper.abstract or "", max_sentences=4)
                ollama_summary = generate_ollama_final_summary(
                    title=r.paper.title,
                    authors=r.paper.authors,
                    year=str(r.paper.year),
                    extractive_result=extractive_result,
                    query_context=query_en
                )
                paper_focus = extractive_result.paper_focus
                contribution = extractive_result.contribution
                key_terms = extractive_result.important_terms
            except Exception as e:
                print(f"⚠️ Ollama summary generation failed: {e}")
                ollama_summary = abs_sum  # Fallback to abstractive
        
        results.append(OrchestratedResult(
            paper=r.paper,
            score=r.score,
            summary_extractive=extractive_sum,      # Original abstract or simple extractive
            summary_abstractive=abs_sum,            # AI-generated abstractive summary
            summary_tfidf=tfidf_sum,                # TF-IDF based summary
            summary_lsa=lsa_sum,                    # LSA based summary
            summary_combined=combined_sum,          # Combined summary used for ranking
            final_response=final_resp,              # Human-like response from Mistral meta-processor
            method_used=r.method_used,              # Method used for primary summary
            paper_focus=paper_focus,
            contribution=contribution,
            key_terms=key_terms,
            ollama_summary=ollama_summary or abs_sum
        ))

    response = {
        'query_language': orig_lang,
        'intent': intent,
        'entities': entities,
        'expanded_query': search_query,
        'api_used': api,
        'processing_mode': f"{'pdfs' if use_pdfs else 'abstracts_only'}_{summary_mode}{'_ollama' if _check_ollama_availability() else ''}",
        'summary_method': summary_mode,
        'timestamp': datetime.now().isoformat(),
        'performance': {
            'ollama_available': _check_ollama_availability(),
            'fast_mode_used': False,  # This is the traditional pipeline
            'papers_processed': len(results)
        },
        'results': [
            {
                'paper_id': res.paper.id,
                'title': res.paper.title,
                'link': res.paper.pdf_url or res.paper.entry_url,
                'year': res.paper.year,
                'authors': res.paper.authors,
                'score': res.score,
                'summaries': {
                    'tfidf': res.summary_tfidf,           # Specific TF-IDF summary
                    'lsa': res.summary_lsa,               # Specific LSA summary
                    'abstractive': res.summary_abstractive, # AI-generated summary
                    'combined': res.summary_combined,       # Combined summary
                    'ollama': res.ollama_summary          # NEW: Ollama-generated summary
                },
                'final_response': res.final_response,           # Human-like response from Mistral meta-processor
                'method': res.method_used,                      # Method used for primary summary
                # Legacy fields for backward compatibility - now with proper differentiation
                'abstract_summary': res.summary_extractive,    # Original abstract or simple extractive
                'abstractive_summary': res.ollama_summary if res.ollama_summary else res.summary_abstractive, # Prioritize Ollama
                # New Ollama-specific metadata
                'paper_focus': res.paper_focus,
                'contribution': res.contribution,
                'key_terms': res.key_terms,
                'ollama_summary': res.ollama_summary
            }
            for res in results
        ]
    }
    
    # Save results for evaluation
    _save_results_for_evaluation(query, response, data_dir)
    
    return response


def _save_results_for_evaluation(query: str, response: Dict[str, Any], data_dir: Path) -> None:
    """Save query results to disk for evaluation purposes."""
    try:
        eval_dir = data_dir / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual query result
        query_hash = str(hash(query))[:8]
        result_file = eval_dir / f'query_{query_hash}_{response["timestamp"].replace(":", "-")}.json'
        
        with result_file.open('w', encoding='utf-8') as f:
            json.dump({
                'query': query,
                'response': response
            }, f, ensure_ascii=False, indent=2)
        
        # Append to results log
        log_file = eval_dir / 'all_results.jsonl'
        with log_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps({
                'query': query,
                'timestamp': response['timestamp'],
                'api_used': response['api_used'],
                'processing_mode': response['processing_mode'],
                'result_count': len(response['results']),
                'ranked_papers': [
                    {
                        'paper_id': r['paper_id'],
                        'title': r['title'],
                        'score': r['score']
                    }
                    for r in response['results']
                ]
            }, ensure_ascii=False) + '\n')
            
    except Exception as e:
        print(f"Warning: Could not save results for evaluation: {e}")


# Convenience functions for different modes
def run_fast_pipeline(query: str, max_results: int = 20, top_k: int = 5) -> Dict[str, Any]:
    """Run the fast Ollama pipeline specifically"""
    return run_pipeline(
        query=query,
        max_results=max_results,
        top_k=top_k,
        use_pdfs=False,
        use_ollama_fast=True,
        summary_mode="fast"
    )


def run_traditional_pipeline(query: str, max_results: int = 20, top_k: int = 5, use_pdfs: bool = True) -> Dict[str, Any]:
    """Run the traditional pipeline (with optional Ollama enhancement)"""
    return run_pipeline(
        query=query,
        max_results=max_results,
        top_k=top_k,
        use_pdfs=use_pdfs,
        use_ollama_fast=False,
        summary_mode="auto"
    )