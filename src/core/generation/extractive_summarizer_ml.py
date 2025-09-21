# extractive_summarizer_improved.py
# Version améliorée du résumeur extractif

import os
import pickle
import logging
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize 

# Télécharger les données NLTK si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedExtractiveSummarizer:
    """
    Résumeur extractif amélioré avec meilleure segmentation et scoring
    """
    
    def __init__(self, 
                 model_path: str = "models/summarizer_model.pkl",
                 num_sentences: int = 3,
                 min_sentence_length: int = 30,  # Augmenté
                 max_sentence_length: int = 500,  # Augmenté
                 min_words: int = 5):  # Nouveau paramètre
        
        self.model_path = model_path
        self.num_sentences = num_sentences
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.min_words = min_words
        
        # Modèles chargés
        self.tfidf_vectorizer = None
        self.lsa_model = None
        self.stop_words = None
        self.model_info = {}
        
        # Mots vides étendus pour textes scientifiques
        self.scientific_stopwords = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'thus', 'hence', 'meanwhile', 'nevertheless',
            'nonetheless', 'whereas', 'although', 'though', 'since', 'because',
            'due', 'according', 'regarding', 'concerning', 'respect', 'related',
            'based', 'using', 'used', 'shown', 'shows', 'show', 'demonstrated',
            'results', 'result', 'findings', 'finding', 'conclusion', 'conclusions',
            'paper', 'study', 'research', 'work', 'approach', 'method', 'methods',
            'technique', 'techniques', 'algorithm', 'algorithms', 'model', 'models'
        }
        
        self._load_model()
    
    def _load_model(self):
        """Charge le modèle pré-entraîné avec gestion d'erreur améliorée"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        try:
            logger.info(f"Chargement du modèle: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.lsa_model = model_data['lsa_model']
            
            # Mots vides étendus - conversion en set si nécessaire
            base_stopwords = model_data.get('stop_words', set())
            if isinstance(base_stopwords, list):
                base_stopwords = set(base_stopwords)
            
            french_stopwords = {
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour',
                'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout',
                'plus', 'par', 'grand', 'lui', 'bien', 'autre', 'depuis', 'du', 'état',
                'moins', 'nous', 'vous', 'ils', 'elle', 'elles', 'cette', 'ces',
                'la', 'les', 'des', 'ses', 'mes', 'tes', 'nos', 'vos', 'leurs'
            }
            
            self.stop_words = base_stopwords.union(french_stopwords).union(self.scientific_stopwords)
            
            self.model_info = {
                'dataset': model_data.get('dataset_name', 'unknown'),
                'training_date': model_data.get('training_date', 'unknown'),
                'vocabulary_size': model_data.get('vocabulary_size', 0),
                'lsa_components': model_data.get('lsa_components_actual', 0),
                'explained_variance': model_data.get('explained_variance_ratio', 0.0),
                'training_samples': model_data.get('max_training_samples', 0)
            }
            
            logger.info(f"✓ Modèle chargé avec {len(self.stop_words)} mots vides")
            
        except Exception as e:
            logger.error(f"Erreur chargement modèle: {str(e)}")
            raise
    
    def summarize(self, text: str, abstract: str = None) -> Dict[str, any]:
        """
        Génère un résumé extractif 
        """
        try:
            if not text or not text.strip():
                return self._empty_result("Texte vide")
            
            # Segmentation améliorée
            sentences = self._split_into_sentences_improved(text)
            if not sentences:
                return self._empty_result("Aucune phrase trouvée")
            
            logger.info(f"Phrases détectées: {len(sentences)}")
            
            # Filtrage amélioré
            valid_sentences = self._filter_sentences_improved(sentences)
            if not valid_sentences:
                return self._empty_result("Aucune phrase valide")
            
            logger.info(f"Phrases valides: {len(valid_sentences)}")
            
            if len(valid_sentences) <= self.num_sentences:
                selected_sentences = valid_sentences
                confidence = 0.8
            else:
                # Scoring multi-critères
                sentence_scores = self._calculate_improved_scores(valid_sentences, abstract, sentences)
                selected_sentences = self._select_diverse_sentences(valid_sentences, sentence_scores)
                confidence = self._calculate_confidence(sentence_scores, selected_sentences)
            
            # Post-traitement
            ordered_sentences = self._post_process_summary(selected_sentences, sentences)
            summary_text = self._create_coherent_summary(ordered_sentences)
            
            return {
                'summary': summary_text,
                'sentences': ordered_sentences,
                'confidence': confidence,
                'method': 'extractive_improved',
                'model_info': self.model_info,
                'num_original_sentences': len(sentences),
                'num_valid_sentences': len(valid_sentences),
                'num_selected_sentences': len(selected_sentences),
                'compression_ratio': len(summary_text) / len(text)
            }
            
        except Exception as e:
            logger.error(f"Erreur pendant le résumé: {str(e)}")
            return self._empty_result(f"Erreur: {str(e)}")
    
    def _split_into_sentences_improved(self, text: str) -> List[str]:
        """
        Segmentation améliorée utilisant NLTK + regex personnalisées
        """
        # Nettoyer le texte d'abord
        text = self._clean_text_for_segmentation(text)
        
        # Utiliser NLTK comme base
        try:
            sentences = sent_tokenize(text)
        except:
            # Fallback vers regex si NLTK échoue
            sentences = self._fallback_sentence_split(text)
        
        # Post-traitement des phrases
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Corriger les phrases mal segmentées
                sentence = self._fix_sentence_boundaries(sentence)
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _clean_text_for_segmentation(self, text: str) -> str:
        """
        Nettoie le texte pour améliorer la segmentation
        """
        # Corriger les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Corriger les points collés aux mots
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)
        
        # Gérer les abréviations courantes
        abbreviations = ['Dr', 'Prof', 'Fig', 'Tab', 'Eq', 'vs', 'etc', 'i.e', 'e.g']
        for abbr in abbreviations:
            text = text.replace(f'{abbr}.', f'{abbr}[DOT]')
        
        # Remettre les points après traitement
        text = text.replace('[DOT]', '.')
        
        return text.strip()
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """
        Segmentation de secours si NLTK échoue
        """
        # Pattern amélioré pour la segmentation
        patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Pattern de base
            r'(?<=[.!?])\s+(?=\d)',     # Phrases commençant par un chiffre
            r'(?<=\.)\s*\n\s*(?=[A-Z])', # Nouvelle ligne après point
        ]
        
        sentences = [text]
        for pattern in patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _fix_sentence_boundaries(self, sentence: str) -> str:
        """
        Corrige les limites de phrases mal détectées
        """
        # Supprimer les débuts/fins bizarres
        sentence = re.sub(r'^[^\w\s]+', '', sentence)
        sentence = re.sub(r'[^\w\s.!?]+$', '', sentence)
        
        # Assurer qu'elle se termine par une ponctuation
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        return sentence.strip()
    
    def _filter_sentences_improved(self, sentences: List[str]) -> List[str]:
        """
        Filtrage amélioré des phrases
        """
        valid_sentences = []
        
        for sentence in sentences:
            # Critères de base
            if len(sentence) < self.min_sentence_length:
                continue
            if len(sentence) > self.max_sentence_length:
                continue
            
            # Compter les mots significatifs
            words = self._extract_meaningful_words(sentence)
            if len(words) < self.min_words:
                continue
            
            # Éviter les phrases trop répétitives
            if self._is_too_repetitive(sentence):
                continue
            
            # Éviter les phrases incomplètes
            if self._is_incomplete_sentence(sentence):
                continue
            
            valid_sentences.append(sentence)
        
        return valid_sentences
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """
        Extrait les mots significatifs (non stop-words)
        """
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower())
        return [w for w in words if len(w) > 2 and w not in self.stop_words]
    
    def _is_too_repetitive(self, sentence: str) -> bool:
        """
        Détecte si une phrase est trop répétitive
        """
        words = sentence.lower().split()
        if len(words) < 5:
            return False
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Si plus de 30% des mots sont répétés
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        return repeated_words / len(words) > 0.3
    
    def _is_incomplete_sentence(self, sentence: str) -> bool:
        """
        Détecte les phrases incomplètes ou mal formées
        """
        # Phrases trop courtes en mots
        if len(sentence.split()) < 4:
            return True
        
        # Phrases qui commencent par des mots de liaison
        sentence_lower = sentence.lower().strip()
        bad_starts = ['and', 'but', 'or', 'so', 'because', 'since', 'although', 'et', 'mais', 'ou', 'car', 'donc']
        if any(sentence_lower.startswith(start + ' ') for start in bad_starts):
            return True
        
        # Phrases sans verbe principal (approximation)
        words = sentence.lower().split()
        has_potential_verb = any(
            word.endswith(('ed', 'ing', 'es', 's', 'er', 'est', 'ent', 'er', 'ir', 're')) 
            for word in words
        )
        
        return not has_potential_verb
    
    def _calculate_improved_scores(self, sentences: List[str], abstract: str, 
                                 all_sentences: List[str]) -> Dict[str, float]:
        """
        Calcul de scores multi-critères amélioré
        """
        scores = {}
        
        try:
            # Scores LSA de base
            base_scores = self._calculate_lsa_scores(sentences)
            
            for i, sentence in enumerate(sentences):
                base_score = base_scores.get(sentence, 0.0)
                
                # 1. Score de position (début et fin importants)
                position_in_all = self._find_sentence_position(sentence, all_sentences)
                total_sentences = len(all_sentences)
                
                if position_in_all < total_sentences * 0.2:  # 20% du début
                    position_score = 1.0
                elif position_in_all > total_sentences * 0.8:  # 20% de la fin
                    position_score = 0.8
                else:
                    position_score = 0.5
                
                # 2. Score de longueur (phrases moyennes favorisées)
                length_score = self._calculate_length_score(sentence)
                
                # 3. Score de mots-clés
                keyword_score = self._calculate_keyword_score(sentence)
                
                # 4. Score de cohésion avec abstract
                abstract_score = self._calculate_abstract_bonus(sentence, abstract, None, 0) if abstract else 0.0
                
                # 5. Score de diversité lexicale
                diversity_score = self._calculate_diversity_score(sentence)
                
                # Combinaison pondérée
                final_score = (
                    base_score * 0.4 +           # LSA centrality
                    position_score * 0.2 +       # Position
                    length_score * 0.15 +        # Longueur
                    keyword_score * 0.1 +        # Mots-clés
                    abstract_score * 0.1 +       # Abstract similarity
                    diversity_score * 0.05       # Diversité
                )
                
                scores[sentence] = final_score
            
            return scores
            
        except Exception as e:
            logger.warning(f"Erreur calcul scores améliorés: {str(e)}")
            # Fallback simple
            return {sentence: 1.0 - (i / len(sentences)) for i, sentence in enumerate(sentences)}
    
    def _calculate_lsa_scores(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calcule les scores LSA de base
        """
        try:
            preprocessed = [self._preprocess_sentence_improved(s) for s in sentences]
            tfidf_matrix = self.tfidf_vectorizer.transform(preprocessed)
            lsa_matrix = self.lsa_model.transform(tfidf_matrix)
            similarity_matrix = cosine_similarity(lsa_matrix)
            
            scores = {}
            for i, sentence in enumerate(sentences):
                centrality_score = np.mean(similarity_matrix[i])
                scores[sentence] = centrality_score
            
            return scores
        except:
            return {sentence: 0.5 for sentence in sentences}
    
    def _find_sentence_position(self, sentence: str, all_sentences: List[str]) -> int:
        """
        Trouve la position d'une phrase dans le texte original
        """
        try:
            return all_sentences.index(sentence)
        except ValueError:
            return len(all_sentences) // 2  # Position médiane par défaut
    
    def _calculate_length_score(self, sentence: str) -> float:
        """
        Score basé sur la longueur (phrases moyennes favorisées)
        """
        words = len(sentence.split())
        if 10 <= words <= 25:  # Longueur idéale
            return 1.0
        elif 8 <= words <= 30:  # Acceptable
            return 0.8
        elif 5 <= words <= 40:  # Limite
            return 0.6
        else:
            return 0.3
    
    def _calculate_keyword_score(self, sentence: str) -> float:
        """
        Score basé sur la présence de mots-clés importants
        """
        # Mots-clés techniques/scientifiques importants
        important_keywords = {
            'method', 'approach', 'algorithm', 'model', 'result', 'conclusion',
            'finding', 'discovery', 'analysis', 'evaluation', 'performance',
            'improvement', 'novel', 'significant', 'important', 'main', 'key',
            'propose', 'introduce', 'demonstrate', 'show', 'prove', 'achieve'
        }
        
        words = set(word.lower() for word in re.findall(r'\b\w+\b', sentence))
        keyword_count = len(words.intersection(important_keywords))
        
        return min(keyword_count * 0.2, 1.0)
    
    def _calculate_diversity_score(self, sentence: str) -> float:
        """
        Score de diversité lexicale de la phrase
        """
        words = re.findall(r'\b\w+\b', sentence.lower())
        if len(words) < 5:
            return 0.5
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return diversity_ratio
    
    def _preprocess_sentence_improved(self, sentence: str) -> str:
        """
        Préprocessing amélioré qui préserve plus d'informations
        """
        # Garder la ponctuation importante
        sentence = re.sub(r'[^\w\s.!?(),-]', ' ', sentence)
        # Normaliser les espaces sans être trop agressif
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence.strip()
    
    def _select_diverse_sentences(self, sentences: List[str], scores: Dict[str, float]) -> List[str]:
        """
        Sélection diversifiée des phrases pour éviter la redondance
        """
        # Trier par score
        sorted_sentences = sorted(sentences, key=lambda s: scores.get(s, 0), reverse=True)
        
        selected = []
        for sentence in sorted_sentences:
            if len(selected) >= self.num_sentences:
                break
            
            # Vérifier la diversité avec les phrases déjà sélectionnées
            if not selected or self._is_diverse_enough(sentence, selected):
                selected.append(sentence)
        
        # Si pas assez de phrases diverses, compléter avec les meilleures
        while len(selected) < self.num_sentences and len(selected) < len(sentences):
            for sentence in sorted_sentences:
                if sentence not in selected:
                    selected.append(sentence)
                    break
        
        return selected
    
    def _is_diverse_enough(self, sentence: str, selected_sentences: List[str]) -> bool:
        """
        Vérifie si une phrase est suffisamment différente des déjà sélectionnées
        """
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        for selected in selected_sentences:
            selected_words = set(re.findall(r'\b\w+\b', selected.lower()))
            
            # Calculer la similarité Jaccard
            intersection = len(sentence_words.intersection(selected_words))
            union = len(sentence_words.union(selected_words))
            
            if union > 0:
                jaccard_similarity = intersection / union
                if jaccard_similarity > 0.4:  # Trop similaire
                    return False
        
        return True
    
    def _post_process_summary(self, selected_sentences: List[str], 
                            original_sentences: List[str]) -> List[str]:
        """
        Post-traitement pour améliorer la cohérence
        """
        # Remettre dans l'ordre original
        sentence_positions = {
            sentence: i for i, sentence in enumerate(original_sentences)
        }
        
        ordered = sorted(
            selected_sentences, 
            key=lambda s: sentence_positions.get(s, float('inf'))
        )
        
        # Nettoyer les phrases
        cleaned = []
        for sentence in ordered:
            cleaned_sentence = self._clean_final_sentence(sentence)
            if cleaned_sentence:
                cleaned.append(cleaned_sentence)
        
        return cleaned
    
    def _clean_final_sentence(self, sentence: str) -> str:
        """
        Nettoyage final d'une phrase
        """
        # Supprimer les débuts/fins inappropriés
        sentence = sentence.strip()
        
        # Assurer une ponctuation correcte
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'
        
        # Capitaliser le début si nécessaire
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _create_coherent_summary(self, sentences: List[str]) -> str:
        """
        Crée un résumé cohérent à partir des phrases sélectionnées
        """
        if not sentences:
            return ""
        
        # Joindre avec des espaces appropriés
        summary = ' '.join(sentences)
        
        # Normaliser les espaces
        summary = re.sub(r'\s+', ' ', summary)
        
        return summary.strip()
    
    # Méthodes héritées avec signatures compatibles
    def _calculate_abstract_bonus(self, sentence: str, abstract: str, 
                                 lsa_matrix, sentence_idx: int) -> float:
        """Version compatible de la méthode abstract bonus"""
        if not abstract:
            return 0.0
        
        try:
            clean_abstract = self._preprocess_sentence_improved(abstract)
            clean_sentence = self._preprocess_sentence_improved(sentence)
            
            # Similarité simple basée sur les mots
            abstract_words = set(re.findall(r'\b\w+\b', clean_abstract.lower()))
            sentence_words = set(re.findall(r'\b\w+\b', clean_sentence.lower()))
            
            if not abstract_words or not sentence_words:
                return 0.0
            
            intersection = len(abstract_words.intersection(sentence_words))
            union = len(abstract_words.union(sentence_words))
            
            similarity = intersection / union if union > 0 else 0.0
            return min(similarity * 0.5, 0.5)
            
        except Exception:
            return 0.0
    
    def _calculate_confidence(self, scores: Dict[str, float], 
                            selected_sentences: List[str]) -> float:
        """Calcule la confiance du résumé"""
        if not scores or not selected_sentences:
            return 0.0
        
        selected_scores = [scores.get(sentence, 0) for sentence in selected_sentences]
        all_scores = list(scores.values())
        
        if not all_scores:
            return 0.0
        
        avg_selected = sum(selected_scores) / len(selected_scores)
        avg_all = sum(all_scores) / len(all_scores)
        
        confidence = min(avg_selected / max(avg_all, 0.1), 1.0)
        return max(confidence, 0.0)
    
    def _empty_result(self, error_message: str) -> Dict[str, any]:
        """Retourne un résultat vide"""
        return {
            'summary': '',
            'sentences': [],
            'confidence': 0.0,
            'method': 'extractive_improved',
            'model_info': self.model_info,
            'error': error_message,
            'num_original_sentences': 0,
            'num_valid_sentences': 0,
            'num_selected_sentences': 0,
            'compression_ratio': 0.0
        }

    def get_model_info(self) -> Dict[str, any]:
        """Retourne les informations du modèle"""
        return self.model_info.copy()


# Test de la version améliorée
def main():
    """Test du résumeur amélioré"""
    
    # Texte scientifique complexe pour tester
    sample_text = """
    Deep learning has revolutionized many fields of artificial intelligence. 
    Traditional machine learning approaches often required extensive feature engineering. 
    However, deep neural networks can automatically learn hierarchical representations from raw data.
    Convolutional Neural Networks (CNNs) have been particularly successful in computer vision tasks.
    They use shared weights and local connectivity to process images efficiently.
    Recurrent Neural Networks (RNNs) excel at sequential data processing.
    Long Short-Term Memory (LSTM) networks address the vanishing gradient problem in RNNs.
    Transformer architectures have achieved state-of-the-art results in natural language processing.
    The attention mechanism allows models to focus on relevant parts of the input.
    GPT and BERT are prominent examples of transformer-based language models.
    These models demonstrate impressive capabilities in text generation and understanding.
    Nevertheless, they require substantial computational resources for training.
    Furthermore, concerns about bias and interpretability remain significant challenges.
    Future research directions include improving efficiency and addressing ethical considerations.
    """
    
    try:
        print("=" * 60)
        print("TEST DU RÉSUMEUR EXTRACTIF AMÉLIORÉ")
        print("=" * 60)
        
        # Tester avec un modèle fictif pour la démonstration
        model_path = "models/summarizer_model.pkl"
        if not os.path.exists(model_path):
            print(f"⚠️ Modèle non trouvé à {model_path}")
            print("Ce test nécessite un modèle pré-entraîné.")
            return
        
        summarizer = ImprovedExtractiveSummarizer(
            model_path=model_path,
            num_sentences=3,
            min_sentence_length=30,
            max_sentence_length=500
        )
        
        print(f"\n📖 TEXTE ORIGINAL ({len(sample_text)} caractères):")
        print(sample_text.strip())
        
        result = summarizer.summarize(sample_text)
        
        print(f"\n✅ RÉSUMÉ GÉNÉRÉ (confiance: {result['confidence']:.2f}):")
        print(f"{'='*50}")
        print(result['summary'])
        print(f"{'='*50}")
        
        print(f"\n📈 STATISTIQUES:")
        print(f"  Phrases originales: {result['num_original_sentences']}")
        print(f"  Phrases valides: {result['num_valid_sentences']}")
        print(f"  Phrases sélectionnées: {result['num_selected_sentences']}")
        print(f"  Taux de compression: {result['compression_ratio']:.1%}")
        
        if 'error' in result:
            print(f"  ⚠️ Erreur: {result['error']}")
        
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")

if __name__ == "__main__":
    main()