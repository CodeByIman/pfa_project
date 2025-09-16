#!/usr/bin/env python3
"""
Script de test pour vérifier que l'API FastAPI fonctionne correctement
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test de l'endpoint health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_search():
    """Test de l'endpoint search"""
    try:
        payload = {
            "query": "deep learning medical diagnosis",
            "top_k": 2,
            "max_results": 5,
            "api": "arxiv",
            "use_pdfs": False
        }
        
        print(f"🔍 Testing search with payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"✅ Search test: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Results found: {len(data.get('results', []))}")
            print(f"API used: {data.get('api_used')}")
            print(f"Processing mode: {data.get('processing_mode')}")
            return True
        else:
            print(f"❌ Search failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    print("🧪 Testing AI Research Agent API")
    print("=" * 50)
    
    # Test health
    health_ok = test_health()
    print()
    
    if health_ok:
        # Test search
        search_ok = test_search()
        print()
        
        if search_ok:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("❌ Search test failed. Check the API logs.")
    else:
        print("❌ Health check failed. Make sure the API is running:")
        print("   uvicorn ai_research_agent.src.api.main:app --reload --port 8000")

if __name__ == "__main__":
    main()

