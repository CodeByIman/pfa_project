import functools
import re
from typing import Literal, Optional, List

# Primary language detection with langid (more stable than langdetect)
try:
    import langid
    langid.set_languages(['en', 'fr'])  # Restrict to English and French for better accuracy
    detect = langid.classify
except Exception:
    detect = None

# Fallback to langdetect if langid unavailable
if detect is None:
    try:
        from langdetect import detect as langdetect_detect, DetectorFactory
        # Set seed for consistent results
        DetectorFactory.seed = 0
        detect = langdetect_detect
    except Exception:
        detect = None

try:
    from transformers import MarianMTModel, MarianTokenizer
    exists_transformers = True
except Exception:
    exists_transformers = False


LanguageCode = Literal['en', 'fr', 'unknown']

# Confidence threshold for language detection
CONFIDENCE_THRESHOLD = 0.7
MIN_TEXT_LENGTH = 3


def _get_enhanced_heuristic_markers():
    """Enhanced heuristic markers for French/English detection in research contexts."""
    french_markers = {
        # Research-specific terms
        "état de l'art", 'apprentissage', 'diagnostic', 'médical', 'réseau', 'données',
        'recherche', 'algorithme', 'modèle', 'analyse', 'étude', 'méthode', 'résultats',
        'performance', 'évaluation', 'classification', 'prédiction', 'optimisation',
        'traitement', 'système', 'intelligence artificielle', 'apprentissage automatique',
        
        # Common French words/phrases
        'avec', 'dans', 'pour', 'sur', 'par', 'une', 'des', 'les', 'est', 'sont',
        'peut', 'plus', 'très', 'aussi', 'bien', 'comme', 'même', 'sans', 'sous',
        'entre', 'contre', 'depuis', 'pendant', 'selon', 'vers', 'chez',
        
        # French articles and prepositions
        'du', 'de la', 'de l\'', 'au', 'aux', 'à la', 'à l\'',
        
        # French question words
        'qu\'est-ce que', 'comment', 'pourquoi', 'quand', 'où', 'qui', 'que', 'quoi'
    }
    
    english_markers = {
        # Research-specific terms
        'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
        'state of the art', 'performance', 'evaluation', 'classification', 'prediction',
        'optimization', 'algorithm', 'model', 'analysis', 'study', 'method', 'results',
        'research', 'medical', 'diagnostic', 'network', 'data', 'system',
        
        # Common English words
        'the', 'and', 'for', 'with', 'from', 'this', 'that', 'these', 'those',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom',
        'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might',
        'have', 'has', 'had', 'been', 'being', 'are', 'is', 'was', 'were',
        
        # English prepositions and conjunctions
        'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
        'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
        'during', 'except', 'inside', 'into', 'like', 'near', 'outside', 'over',
        'since', 'through', 'throughout', 'under', 'until', 'upon', 'within', 'without'
    }
    
    return french_markers, english_markers


def _clean_text_for_detection(text: str) -> str:
    """Clean and prepare text for language detection."""
    # Remove URLs, emails, and special characters that might confuse detection
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _heuristic_language_detection(text: str) -> tuple[LanguageCode, float]:
    """Enhanced heuristic-based language detection with confidence score."""
    french_markers, english_markers = _get_enhanced_heuristic_markers()
    
    lowered = text.lower()
    cleaned = _clean_text_for_detection(lowered)
    words = cleaned.split()
    
    if len(words) == 0:
        return 'unknown', 0.0
    
    french_score = 0
    english_score = 0
    total_words = len(words)
    
    # Score based on marker presence
    for marker in french_markers:
        if marker in lowered:
            # Longer markers get higher weight
            weight = len(marker.split()) * 2
            french_score += weight
    
    for marker in english_markers:
        if marker in lowered:
            weight = len(marker.split()) * 2
            english_score += weight
    
    # Additional scoring for individual words
    for word in words:
        if word in french_markers:
            french_score += 1
        if word in english_markers:
            english_score += 1
    
    # Calculate confidence based on total score vs text length
    max_score = max(french_score, english_score)
    total_score = french_score + english_score
    
    if total_score == 0:
        return 'unknown', 0.0
    
    confidence = min(max_score / (total_words + 1), 1.0)
    
    if french_score > english_score:
        return 'fr', confidence
    elif english_score > french_score:
        return 'en', confidence
    else:
        return 'unknown', 0.0


def detect_language(text: str) -> LanguageCode:
    """
    Detect language with improved accuracy for short research queries.
    Uses langid (preferred) or langdetect with heuristic fallback.
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return 'unknown'
    
    cleaned_text = _clean_text_for_detection(text)
    if len(cleaned_text.strip()) < MIN_TEXT_LENGTH:
        return 'unknown'
    
    # Try primary detection methods first
    if detect is not None:
        try:
            if hasattr(detect, '__name__') and 'langid' in str(detect):
                # Using langid - returns (language, confidence)
                lang, confidence = detect(text)
                if confidence >= CONFIDENCE_THRESHOLD:
                    if lang == 'fr':
                        return 'fr'
                    elif lang == 'en':
                        return 'en'
            else:
                # Using langdetect - returns language string
                from langdetect import detect_langs
                detections = detect_langs(text)
                for detection in detections:
                    if detection.prob >= CONFIDENCE_THRESHOLD:
                        lang = detection.lang.lower()
                        if lang.startswith('fr'):
                            return 'fr'
                        elif lang.startswith('en'):
                            return 'en'
        except Exception:
            pass  # Fall through to heuristics
    
    # Fallback to enhanced heuristics
    heuristic_lang, heuristic_confidence = _heuristic_language_detection(text)
    if heuristic_confidence >= CONFIDENCE_THRESHOLD:
        return heuristic_lang
    
    # For very short text or low confidence, use relaxed heuristics
    if len(cleaned_text.split()) <= 5:
        if heuristic_confidence > 0.3:  # Lower threshold for short text
            return heuristic_lang
    
    # Final fallback - assume English for research contexts if unsure
    return 'en' if heuristic_lang == 'unknown' else heuristic_lang


@functools.lru_cache(maxsize=1)
def _load_marian_fr_en():
    """Load French-to-English translation model with better error handling."""
    if not exists_transformers:
        return None, None
    
    try:
        model_name = 'Helsinki-NLP/opus-mt-fr-en'
        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            clean_up_tokenization_spaces=True
        )
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        # Log error if logging is available
        if hasattr(__builtins__, 'print'):
            print(f"Warning: Failed to load translation model: {e}")
        return None, None


def _preprocess_for_translation(text: str) -> str:
    """Pr eprocess text for better translation quality."""
    # Handle common research abbreviations and terms
    text = text.strip()
    
    # Don't translate if text is mostly English already (mixed language)
    english_ratio = len(re.findall(r'\b(the|and|of|for|with|in|on|at|by|from|to|a|an)\b', text.lower())) / max(len(text.split()), 1)
    if english_ratio > 0.3:  # High English content
        return text
    
    return text


def _postprocess_translation(original: str, translated: str) -> str:
    """Post-process translated text for better quality."""
    if not translated or translated.strip() == original.strip():
        return original
    
    # Clean up common translation artifacts
    translated = translated.strip()
    
    # Remove redundant spaces
    translated = re.sub(r'\s+', ' ', translated)
    
    # Fix common punctuation issues
    translated = re.sub(r'\s+([.!?,:;])', r'\1', translated)
    translated = re.sub(r'([.!?])\s*$', r'\1', translated)
    
    return translated


def translate_to_english(text: str, src_lang: LanguageCode) -> str:
    """
    Translate French text to English with improved preprocessing and error handling.
    Supports batch processing internally for efficiency.
    """
    if not text or src_lang != 'fr':
        return text
    
    if not exists_transformers:
        return text
    
    try:
        tokenizer, model = _load_marian_fr_en()
        if tokenizer is None or model is None:
            return text
        
        # Preprocess text
        processed_text = _preprocess_for_translation(text)
        if processed_text != text and len(re.findall(r'[a-zA-Z]', processed_text)) < len(re.findall(r'[a-zA-Z]', text)) * 0.5:
            # If preprocessing removed too much, use original
            processed_text = text
        
        # Translate with improved parameters
        inputs = tokenizer(
            [processed_text], 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Generate with better parameters for research text
        outputs = model.generate(
            **inputs,
            max_new_tokens=min(256, len(processed_text.split()) * 2),
            num_beams=4,  # Better quality for research content
            do_sample=False,  # Deterministic for consistency
            early_stopping=True
        )
        
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Post-process the translation
        final_translation = _postprocess_translation(text, translated)
        
        # Sanity check - if translation is suspiciously short or empty, return original
        if len(final_translation.strip()) < len(text.strip()) * 0.3:
            return text
            
        return final_translation
        
    except Exception as e:
        # Fail open - return original text
        if hasattr(__builtins__, 'print'):
            print(f"Translation failed: {e}")
        return text


# Helper function for batch processing (if needed in the future)
def _batch_translate_to_english(texts: List[str], src_langs: List[LanguageCode]) -> List[str]:
    """
    Internal helper for batch translation (not part of the required interface).
    Can be used to optimize multiple translations.
    """
    if not exists_transformers:
        return texts
    
    try:
        tokenizer, model = _load_marian_fr_en()
        if tokenizer is None or model is None:
            return texts
        
        # Filter French texts for translation
        french_texts = []
        french_indices = []
        
        for i, (text, lang) in enumerate(zip(texts, src_langs)):
            if lang == 'fr' and text.strip():
                french_texts.append(_preprocess_for_translation(text))
                french_indices.append(i)
        
        if not french_texts:
            return texts
        
        # Batch translate
        inputs = tokenizer(
            french_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )
        
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Merge results back
        result = texts.copy()
        for i, translation in zip(french_indices, translations):
            processed = _postprocess_translation(texts[i], translation)
            if len(processed.strip()) >= len(texts[i].strip()) * 0.3:
                result[i] = processed
        
        return result
        
    except Exception:
        return texts