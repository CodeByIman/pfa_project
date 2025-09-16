import argparse
import json

from .orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='AI Research Agent CLI')
    parser.add_argument('query', type=str, help='Research query (English or French)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top papers to return')
    parser.add_argument('--max_results', type=int, default=20, help='Maximum papers to retrieve from search')
    parser.add_argument('--max_pdfs', type=int, default=5, help='Maximum PDFs to process when using full mode')
    parser.add_argument('--api', type=str, default='arxiv', 
                        choices=['arxiv', 'semantic_scholar', 'pubmed', 'crossref'],
                        help='API to use for paper retrieval')
    
    # New parameters from updated orchestrator
    parser.add_argument('--use_pdfs', action='store_true', default=False,
                        help='Use full PDF processing mode (slower but more comprehensive)')
    parser.add_argument('--summary_mode', type=str, default='auto',
                        choices=['auto', 'fast', 'comprehensive', 'tfidf', 'lsa'],
                        help='Summary generation mode')
    parser.add_argument('--use_mistral_final_response', action='store_true', default=False,
                        help='Generate human-like final response using Mistral')
    parser.add_argument('--use_ollama_fast', action='store_true', default=True,
                        help='Use fast Ollama pipeline (extractive + Ollama generation)')
    parser.add_argument('--traditional_mode', action='store_true', default=False,
                        help='Force traditional pipeline instead of Ollama fast mode')
    
    args = parser.parse_args()

    # Call run_pipeline with all the new parameters
    res = run_pipeline(
        query=args.query,
        max_results=args.max_results,
        max_pdfs=args.max_pdfs,
        top_k=args.top_k,
        api=args.api,
        use_pdfs=args.use_pdfs,
        summary_mode=args.summary_mode,
        use_mistral_final_response=args.use_mistral_final_response,
        use_ollama_fast=not args.traditional_mode and args.use_ollama_fast  # Disable if traditional mode is requested
    )
    
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()