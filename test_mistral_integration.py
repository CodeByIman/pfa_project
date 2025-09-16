#!/usr/bin/env python3
"""
Test script for Mistral integration with Ollama
Tests the new summarization modes and fallback mechanisms
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.generation.abstractive_summarizer import (
    summarize_mistral, 
    summarize_abstractive, 
    _check_ollama_availability,
    get_available_models
)
from src.core.agent.orchestrator import run_pipeline

def test_ollama_availability():
    """Test if Ollama is available with Mistral model"""
    print("Testing Ollama availability...")
    is_available = _check_ollama_availability()
    print(f"Ollama with Mistral available: {is_available}")
    return is_available

def test_available_models():
    """Test getting available models"""
    print("\nTesting available models...")
    models = get_available_models()
    print(f"Available models: {models}")
    return models

def test_mistral_summarization():
    """Test Mistral summarization with sample text"""
    print("\nTesting Mistral summarization...")
    
    sample_text = """
    This paper presents a novel approach to neural machine translation using transformer architectures.
    The proposed method achieves state-of-the-art results on multiple benchmark datasets including WMT14 and WMT16.
    Our model incorporates attention mechanisms that allow for better handling of long sequences and rare words.
    Experimental results show significant improvements in BLEU scores compared to previous approaches.
    The architecture is also more computationally efficient, requiring 30% less training time than comparable models.
    We demonstrate the effectiveness of our approach across multiple language pairs including English-German,
    English-French, and English-Chinese translations. The model shows particular strength in handling
    technical and scientific texts, which are often challenging for traditional translation systems.
    """
    
    try:
        summary = summarize_mistral(sample_text)
        print(f"Mistral summary: {summary}")
        return True
    except Exception as e:
        print(f"Mistral summarization failed: {e}")
        return False

def test_fallback_mechanism():
    """Test fallback to transformer-based summarization"""
    print("\nTesting fallback mechanism...")
    
    sample_text = """
    Machine learning has revolutionized many fields including computer vision, natural language processing,
    and robotics. Deep learning models, particularly neural networks, have shown remarkable performance
    on various tasks. However, these models often require large amounts of data and computational resources.
    Recent advances in transfer learning and few-shot learning have addressed some of these limitations.
    """
    
    try:
        # This should work regardless of Ollama availability due to fallback
        summary = summarize_abstractive(sample_text)
        print(f"Abstractive summary (with fallback): {summary}")
        return True
    except Exception as e:
        print(f"Fallback mechanism failed: {e}")
        return False

def test_pipeline_integration():
    """Test the full pipeline with different summary modes"""
    print("\nTesting pipeline integration...")
    
    test_query = "transformer neural networks attention mechanisms"
    
    # Test different summary modes
    modes_to_test = ["auto", "mistral", "abstractive", "tfidf", "lsa"]
    
    for mode in modes_to_test:
        print(f"\n--- Testing summary mode: {mode} ---")
        try:
            result = run_pipeline(
                query=test_query,
                max_results=3,  # Small number for testing
                max_pdfs=1,
                top_k=2,
                use_pdfs=False,  # Fast mode for testing
                summary_mode=mode
            )
            
            print(f"Query processed successfully with mode '{mode}'")
            print(f"Summary method used: {result.get('summary_method', 'unknown')}")
            print(f"Number of results: {len(result.get('results', []))}")
            
            # Check if results contain the expected fields
            if result.get('results'):
                first_result = result['results'][0]
                print(f"Method used for first result: {first_result.get('method', 'unknown')}")
                summaries = first_result.get('summaries', {})
                print(f"Available summary types: {list(summaries.keys())}")
            
        except Exception as e:
            print(f"Pipeline test failed for mode '{mode}': {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("MISTRAL INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: Check Ollama availability
    ollama_available = test_ollama_availability()
    
    # Test 2: Check available models
    available_models = test_available_models()
    
    # Test 3: Test Mistral summarization (if available)
    if ollama_available and 'mistral' in available_models:
        mistral_works = test_mistral_summarization()
    else:
        print("\nSkipping Mistral summarization test (Ollama not available)")
        mistral_works = False
    
    # Test 4: Test fallback mechanism
    fallback_works = test_fallback_mechanism()
    
    # Test 5: Test pipeline integration
    test_pipeline_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Ollama available: {ollama_available}")
    print(f"Available models: {available_models}")
    print(f"Mistral summarization: {'✓' if mistral_works else '✗'}")
    print(f"Fallback mechanism: {'✓' if fallback_works else '✗'}")
    
    if ollama_available and mistral_works:
        print("\n✅ Mistral integration is working correctly!")
    elif fallback_works:
        print("\n⚠️  Mistral not available, but fallback is working correctly.")
    else:
        print("\n❌ Integration has issues that need to be addressed.")

if __name__ == "__main__":
    main()
