# test_arxiv_summarizer_simple.py
# Script simplifié pour résumer des papers arXiv avec affichage terminal

import os
import sys
import logging
from pathlib import Path
from summarizers.extractive_summarizer_ml import ImprovedExtractiveSummarizer

# Importation des modules de téléchargement et extraction
try:
    from downloaders.arxiv_downloader import ArxivDownloader
    from extractors.pdf_extractor import PDFExtractor
    from extractors.text_cleaner import TextCleaner
    DOWNLOAD_ENABLED = True
except ImportError as e:
    print(f"❌ ERREUR: Modules de téléchargement non disponibles: {e}")
    sys.exit(1)


def process_arxiv_paper(url, output_dir="downloads", num_sentences=10, 
                       min_sentence_length=30, max_sentence_length=500,
                       min_words=5):
    """
    Télécharge un paper arXiv et génère un résumé extractif avec affichage terminal
    
    Args:
        url: URL arXiv du paper
        output_dir: Répertoire de téléchargement
        num_sentences: Nombre de phrases à extraire pour le résumé
        min_sentence_length: Longueur minimale des phrases
        max_sentence_length: Longueur maximale des phrases
        min_words: Nombre minimum de mots significatifs par phrase
    
    Returns:
        dict: Résultats du traitement
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAITEMENT PAPER ARXIV")
    print(f"{'='*80}")
    print(f"📎 URL: {url}")
    
    result = {
        'url': url,
        'success': False,
        'metadata': {},
        'summary_result': {},
        'error': None
    }
    
    try:
        # 1. Initialiser les modules
        print(f"🔧 Initialisation des modules...")
        downloader = ArxivDownloader()
        pdf_extractor = PDFExtractor()
        text_cleaner = TextCleaner()
        summarizer = ImprovedExtractiveSummarizer(
            num_sentences=num_sentences,
            min_sentence_length=min_sentence_length,
            max_sentence_length=max_sentence_length,
            min_words=min_words
        )
        
        # Afficher les infos du modèle
        model_info = summarizer.get_model_info()
        print(f"📊 Modèle LSA chargé:")
        print(f"   Dataset: {model_info.get('dataset', 'N/A')}")
        print(f"   Composantes LSA: {model_info.get('lsa_components', 'N/A')}")
        print(f"   Taille vocabulaire: {model_info.get('vocabulary_size', 'N/A')}")
        
        # 2. Vérifier l'URL
        if not downloader.can_handle(url):
            raise ValueError(f"URL arXiv non valide: {url}")
        print(f"✅ URL arXiv valide")
        
        # 3. Récupérer les métadonnées
        print(f"🔍 Récupération des métadonnées...")
        metadata = downloader.get_metadata(url)
        result['metadata'] = metadata
        print(f"📝 Titre: {metadata.get('title', 'Non trouvé')}")
        print(f"👥 Auteurs: {', '.join(metadata.get('authors', []))}")
        
        # 4. Télécharger le PDF
        print(f"📥 Téléchargement du PDF...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = downloader.generate_filename(url, metadata)
        pdf_path = output_path / filename
        
        download_result = downloader.download(url, str(pdf_path))
        if not download_result['success']:
            raise Exception(f"Échec téléchargement: {download_result.get('error')}")
        
        print(f"✅ PDF téléchargé: {pdf_path}")
        
        # 5. Extraire le texte
        print(f"🔧 Extraction du texte du PDF...")
        extraction_result = pdf_extractor.extract_text(str(pdf_path))
        
        if isinstance(extraction_result, dict):
            if not extraction_result.get('success', True):
                raise Exception(f"Échec extraction: {extraction_result.get('error')}")
            raw_text = extraction_result.get('text', extraction_result.get('content', ''))
            method_used = extraction_result.get('method_used', 'méthode inconnue')
        else:
            raw_text = str(extraction_result)
            method_used = 'extraction directe'
        
        if not raw_text or len(raw_text.strip()) < 100:
            raise Exception("Texte extrait insuffisant")
            
        print(f"✅ Texte extrait ({len(raw_text)} caractères) - Méthode: {method_used}")
        
        # 6. Nettoyer le texte
        print(f"🧹 Nettoyage du texte...")
        cleaned_text = text_cleaner.clean_text(raw_text)
        
        stats_before = text_cleaner.get_text_stats(raw_text)
        stats_after = text_cleaner.get_text_stats(cleaned_text)
        
        print(f"📊 Avant nettoyage: {stats_before['words']} mots")
        print(f"📊 Après nettoyage: {stats_after['words']} mots")
        
        # 7. Générer le résumé extractif amélioré
        print(f"🔍 Génération du résumé extractif...")
        print(f"  📋 Paramètres:")
        print(f"     • {num_sentences} phrases max")
        print(f"     • {min_sentence_length}-{max_sentence_length} caractères par phrase")
        print(f"     • {min_words} mots significatifs minimum")
        
        # Extraire l'abstract si disponible pour améliorer le résumé
        abstract = metadata.get('summary', '')
        if abstract:
            print(f"  📖 Abstract trouvé ({len(abstract)} caractères) - utilisé pour améliorer le scoring")
        
        summary_result = summarizer.summarize(cleaned_text, abstract=abstract)
        result['summary_result'] = summary_result
        
        if 'error' in summary_result:
            raise Exception(f"Erreur génération résumé: {summary_result['error']}")
        
        print(f"✅ Résumé généré avec succès!")
        print(f"  📊 Méthode: {summary_result['method']}")
        print(f"  📊 Confiance: {summary_result['confidence']:.2f}")
        print(f"  📊 Phrases originales: {summary_result['num_original_sentences']}")
        print(f"  📊 Phrases valides: {summary_result['num_valid_sentences']}")
        print(f"  📊 Phrases sélectionnées: {summary_result['num_selected_sentences']}")
        print(f"  📊 Taux de compression: {summary_result['compression_ratio']:.1%}")
        
        # 8. Nettoyer les fichiers temporaires
        try:
            if pdf_path.exists():
                pdf_path.unlink()
                print(f"🗑️ Fichier PDF temporaire supprimé")
        except Exception as e:
            print(f"⚠️ Impossible de supprimer le PDF temporaire: {e}")
        
        result['success'] = True
        print(f"\n🎉 Traitement terminé avec succès!")
        
        return result
        
    except Exception as e:
        error_msg = f"Erreur: {str(e)}"
        print(f"\n❌ {error_msg}")
        logging.error(f"Erreur traitement {url}: {str(e)}", exc_info=True)
        result['error'] = error_msg
        return result


def display_summary_paragraphs(result):
    """Affiche le résumé sous forme de paragraphes dans le terminal"""
    # Vérifier si le traitement a échoué
    if not result.get('success', False) or result.get('error'):
        print(f"❌ Échec du traitement: {result.get('error', 'Erreur inconnue')}")
        return
    
    summary_result = result.get('summary_result', {})
    metadata = result.get('metadata', {})
    
    print("\n" + "="*80)
    print("📋 RÉSUMÉ EXTRACTIF")
    print("="*80)
    
    # Informations du paper
    if metadata:
        print(f"\n📄 PAPER:")
        title = metadata.get('title', 'N/A')
        authors = ', '.join(metadata.get('authors', []))
        date = metadata.get('published_date', 'N/A')
        
        print(f"Titre: {title}")
        print(f"Auteurs: {authors}")
        print(f"Date: {date}")
    
    # Statistiques
    print(f"\n📊 STATISTIQUES:")
    print(f"Phrases originales: {summary_result.get('num_original_sentences', 'N/A')}")
    print(f"Phrases sélectionnées: {summary_result.get('num_selected_sentences', 'N/A')}")
    print(f"Taux de compression: {summary_result.get('compression_ratio', 0):.1%}")
    print(f"Confiance: {summary_result.get('confidence', 0):.2f}")
    
    # Le résumé principal sous forme de paragraphe
    summary_text = summary_result.get('summary', '')
    sentences = summary_result.get('sentences', [])
    
    print(f"\n" + "="*80)
    print("✨ RÉSUMÉ EXTRACTIF")
    print("="*80)
    print()
    
    if sentences:
        # Joindre toutes les phrases en un seul paragraphe
        paragraph = ' '.join(sentences)
        
        # Découper en lignes de longueur raisonnable pour l'affichage
        words = paragraph.split()
        lines = []
        current_line = []
        line_length = 0
        max_line_length = 80
        
        for word in words:
            if line_length + len(word) + 1 > max_line_length and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                line_length = len(word)
            else:
                current_line.append(word)
                line_length += len(word) + (1 if current_line else 0)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Afficher le paragraphe formaté
        for line in lines:
            print(line)
    elif summary_text:
        # Fallback si pas de phrases séparées
        print(summary_text)
    else:
        print("❌ Aucun résumé généré")
    
    print("\n" + "="*80)


def main():
    """Fonction principale simplifiée pour arXiv uniquement"""
    print("🔧 RÉSUMEUR ARXIV SIMPLIFIÉ")
    print("="*40)
    
    # Vérifier que le modèle LSA amélioré est disponible
    try:
        test_summarizer = ImprovedExtractiveSummarizer(num_sentences=3)
        model_info = test_summarizer.get_model_info()
        print("✅ Résumeur amélioré chargé avec succès")
        print(f"   📊 Dataset: {model_info.get('dataset', 'N/A')}")
        print(f"   📊 Composantes LSA: {model_info.get('lsa_components', 'N/A')}")
    except FileNotFoundError:
        print("❌ ERREUR: Modèle LSA introuvable")
        print("💡 Assurez-vous que 'models/summarizer_model.pkl' existe")
        return
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")
        return
    
    while True:
        print(f"\n🌐 TRAITEMENT PAPER ARXIV")
        print("─" * 30)
        
        url = input("URL arXiv du paper (ou 'q' pour quitter): ").strip()
        if url.lower() == 'q':
            print("👋 Au revoir !")
            break
            
        if not url:
            print("❌ URL vide")
            continue
        
        # Configuration rapide avec valeurs par défaut
        print("\n⚙️ Configuration (appuyez sur Entrée pour les valeurs par défaut):")
        
        try:
            num_sentences = input("Nombre de phrases pour le résumé (défaut: 4): ").strip()
            num_sentences = int(num_sentences) if num_sentences else 10
        except ValueError:
            num_sentences = 10
        
        try:
            min_length = input("Longueur minimale des phrases (défaut: 30): ").strip()
            min_length = int(min_length) if min_length else 30
        except ValueError:
            min_length = 30
            
        try:
            max_length = input("Longueur maximale des phrases (défaut: 500): ").strip()
            max_length = int(max_length) if max_length else 500
        except ValueError:
            max_length = 500
            
        try:
            min_words = input("Mots significatifs minimum (défaut: 5): ").strip()
            min_words = int(min_words) if min_words else 5
        except ValueError:
            min_words = 5
        
        # Traitement
        result = process_arxiv_paper(
            url=url,
            output_dir="temp_downloads",  # Dossier temporaire
            num_sentences=num_sentences,
            min_sentence_length=min_length,
            max_sentence_length=max_length,
            min_words=min_words
        )
        
        # Affichage du résumé sous forme de paragraphe
        display_summary_paragraphs(result)
        
        print(f"\n{'─'*80}")
        continuer = input("Voulez-vous traiter un autre paper ? (o/N): ").strip().lower()
        if continuer not in ['o', 'oui', 'y', 'yes']:
            print("👋 Au revoir !")
            break


if __name__ == "__main__":
    main()