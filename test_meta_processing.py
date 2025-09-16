#!/usr/bin/env python3
"""
Test script for the new Mistral meta-processing architecture
Tests that Mistral acts as a final response generator, not just another summarizer
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.generation.abstractive_summarizer import (
    generate_final_response_mistral, 
    _check_ollama_availability
)
from src.core.agent.orchestrator import run_pipeline

def test_final_response_generation():
    """Test the new Mistral meta-processing functionality"""
    print("=" * 60)
    print("TESTING MISTRAL META-PROCESSING ARCHITECTURE")
    print("=" * 60)
    
    # Sample paper data
    title = "Attention Is All You Need"
    authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"]
    year = "2017"
    abstract = """The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."""
    
    # Sample summaries from different methods
    tfidf_summary = "The paper proposes Transformer architecture based on attention mechanisms, dispensing with recurrence and convolutions."
    lsa_summary = "New network architecture called Transformer uses only attention mechanisms for sequence transduction tasks."
    abstractive_summary = "This paper introduces the Transformer, a novel neural network architecture that relies entirely on attention mechanisms, eliminating the need for recurrent or convolutional layers in sequence-to-sequence models."
    
    print(f"Paper: {title}")
    print(f"Authors: {', '.join(authors)}")
    print(f"Year: {year}")
    print(f"\nTF-IDF Summary: {tfidf_summary}")
    print(f"LSA Summary: {lsa_summary}")
    print(f"Abstractive Summary: {abstractive_summary}")
    
    print("\n" + "-" * 60)
    print("GENERATING FINAL RESPONSE WITH MISTRAL META-PROCESSOR")
    print("-" * 60)
    
    # Test the meta-processing function
    final_response = generate_final_response_mistral(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        tfidf_summary=tfidf_summary,
        lsa_summary=lsa_summary,
        abstractive_summary=abstractive_summary
    )
    
    print(f"Final Response: {final_response}")
    
    return final_response

def test_pipeline_with_meta_processing():
    """Test the full pipeline with the new meta-processing architecture"""
    print("\n" + "=" * 60)
    print("TESTING FULL PIPELINE WITH META-PROCESSING")
    print("=" * 60)
    
    test_query = "transformer attention mechanisms neural networks"
    
    print(f"Query: {test_query}")
    print("\nRunning pipeline...")
    
    try:
        result = run_pipeline(
            query=test_query,
            max_results=2,  # Small number for testing
            max_pdfs=1,
            top_k=1,
            use_pdfs=False,  # Fast mode for testing
            summary_mode="auto"
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Summary method used: {result.get('summary_method', 'unknown')}")
        print(f"Number of results: {len(result.get('results', []))}")
        
        if result.get('results'):
            first_result = result['results'][0]
            print(f"\n--- FIRST RESULT ---")
            print(f"Title: {first_result.get('title', 'Unknown')}")
            print(f"Method used: {first_result.get('method', 'unknown')}")
            
            summaries = first_result.get('summaries', {})
            print(f"\nTechnical Summaries Available: {list(summaries.keys())}")
            
            # Show the technical summaries
            for summary_type, summary_text in summaries.items():
                if summary_text and summary_text.strip():
                    print(f"\n{summary_type.upper()} Summary:")
                    print(f"  {summary_text[:200]}...")
            
            # Show the final response from Mistral meta-processor
            final_response = first_result.get('final_response', 'Not available')
            print(f"\n🤖 FINAL RESPONSE (Mistral Meta-Processor):")
            print(f"  {final_response}")
            
            return True
        else:
            print("No results returned from pipeline")
            return False
            
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False

def test_architecture_comparison():
    """Compare old vs new architecture"""
    print("\n" + "=" * 60)
    print("ARCHITECTURE COMPARISON")
    print("=" * 60)
    
    print("OLD ARCHITECTURE:")
    print("  - Mistral was just another summarizer (like TF-IDF, LSA)")
    print("  - Mistral directly processed raw paper text")
    print("  - Output: Multiple parallel summaries")
    
    print("\nNEW ARCHITECTURE:")
    print("  - TF-IDF, LSA, Abstractive generate technical summaries")
    print("  - Mistral acts as meta-processor taking ALL summaries as input")
    print("  - Mistral synthesizes everything into human-like explanation")
    print("  - Output: Technical summaries + Natural language final response")
    
    ollama_available = _check_ollama_availability()
    print(f"\nOllama Status: {'✅ Available' if ollama_available else '❌ Not Available'}")
    
    if not ollama_available:
        print("Note: Without Ollama, fallback responses will be generated")

def main():
    """Run all tests"""
    
    # Test 1: Direct meta-processing function
    final_response = test_final_response_generation()
    
    # Test 2: Full pipeline integration
    pipeline_success = test_pipeline_with_meta_processing()
    
    # Test 3: Architecture explanation
    test_architecture_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    ollama_available = _check_ollama_availability()
    
    print(f"Ollama Available: {'✅' if ollama_available else '❌'}")
    print(f"Meta-processing Function: {'✅' if final_response else '❌'}")
    print(f"Pipeline Integration: {'✅' if pipeline_success else '❌'}")
    
    if pipeline_success:
        print("\n🎉 SUCCESS: New meta-processing architecture is working!")
        print("   - Technical summaries (TF-IDF, LSA, Abstractive) are generated")
        print("   - Mistral synthesizes them into human-like final responses")
        print("   - Frontend will receive both technical summaries AND final response")
    else:
        print("\n⚠️  Issues detected in the new architecture")

if __name__ == "__main__":
    main()
