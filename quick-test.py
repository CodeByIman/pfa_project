#!/usr/bin/env python3
"""
Test rapide pour vérifier que le pipeline fonctionne sans erreur
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ai_research_agent.src.core.agent.orchestrator import run_pipeline

def test_pipeline():
    print("🧪 Test rapide du pipeline...")
    
    try:
        # Test simple
        result = run_pipeline(
            query="deep learning medical diagnosis",
            max_results=3,
            max_pdfs=2,
            top_k=2,
            api="arxiv",
            use_pdfs=False
        )
        
        print("✅ Pipeline exécuté avec succès!")
        print(f"📊 Résultats trouvés: {len(result['results'])}")
        print(f"🌐 API utilisée: {result['api_used']}")
        print(f"⚙️ Mode: {result['processing_mode']}")
        
        if result['results']:
            print("\n📚 Premier résultat:")
            first = result['results'][0]
            print(f"  Titre: {first['title'][:80]}...")
            print(f"  Score: {first['score']:.3f}")
            print(f"  Résumé: {first['abstractive_summary'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur dans le pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n🎉 Test réussi! L'API devrait maintenant fonctionner.")
    else:
        print("\n❌ Test échoué. Vérifiez les erreurs ci-dessus.")

