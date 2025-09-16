#!/usr/bin/env python3
"""
Test script for the optimized API with timeout fixes
"""

import sys
import time
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_api_performance():
    """Test API performance with different modes."""
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Optimized API Performance")
    print("=" * 50)
    
    # Test health endpoint
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return
    
    # Test queries with different modes
    test_queries = [
        {
            "name": "Ultra-fast mode (no abstractive)",
            "params": {
                "query": "deep learning medical imaging",
                "top_k": 2,
                "max_results": 5,
                "api": "arxiv",
                "use_pdfs": False,
                "use_abstractive": False
            }
        },
        {
            "name": "Fast mode (with abstractive)",
            "params": {
                "query": "machine learning computer vision",
                "top_k": 2,
                "max_results": 5,
                "api": "arxiv",
                "use_pdfs": False,
                "use_abstractive": True
            }
        },
        {
            "name": "Full mode (no abstractive)",
            "params": {
                "query": "natural language processing",
                "top_k": 1,
                "max_results": 3,
                "api": "arxiv",
                "use_pdfs": True,
                "max_pdfs": 1,
                "use_abstractive": False
            }
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i+1}️⃣ Testing {test['name']}...")
        print(f"   Query: {test['params']['query']}")
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{base_url}/search",
                json=test['params'],
                timeout=120  # 2 minutes timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                results_count = len(data.get('results', []))
                processing_mode = data.get('processing_mode', 'unknown')
                
                print(f"✅ Success! ({duration:.1f}s)")
                print(f"   Results: {results_count}")
                print(f"   Mode: {processing_mode}")
                
                if results_count > 0:
                    first_result = data['results'][0]
                    print(f"   First paper: {first_result['title'][:60]}...")
                    print(f"   Score: {first_result['score']:.3f}")
                    
                    # Check if summaries are present
                    if 'summaries' in first_result:
                        summaries = first_result['summaries']
                        print(f"   Summaries available: {list(summaries.keys())}")
                    else:
                        print(f"   Legacy summaries: abstract_summary, abstractive_summary")
            else:
                print(f"❌ Failed: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout after 120 seconds")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Performance test completed!")
    print(f"💡 Tips:")
    print(f"   • Use ultra-fast mode for quick results")
    print(f"   • Use fast mode for better quality")
    print(f"   • Use full mode only when needed")

if __name__ == "__main__":
    test_api_performance()
