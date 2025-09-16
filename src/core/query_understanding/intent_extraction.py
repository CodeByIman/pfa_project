from typing import Literal

Intent = Literal['survey', 'methodology', 'application', 'comparison', 'recent', 'general']


def detect_intent(text: str) -> Intent:
    q = (text or '').lower()
    
    # Score-based approach to handle overlapping keywords
    scores = {
        'survey': 0,
        'methodology': 0,
        'application': 0,
        'comparison': 0,
        'recent': 0
    }
    
    # Survey keywords (higher weight for specific terms)
    survey_keywords = {
        'survey': 2, 'state of the art': 2, "état de l'art": 2, 'overview': 2, 'review': 1,
        'literature review': 3, 'systematic review': 3, 'meta-analysis': 2, 'comprehensive review': 2
    }
    for keyword, weight in survey_keywords.items():
        if keyword in q:
            scores['survey'] += weight
    
    # Methodology keywords
    methodology_keywords = {
        'method': 1, 'methodology': 2, 'approach': 1, 'algorithm': 2, 'architecture': 2,
        'technique': 1, 'framework': 1, 'model': 1, 'implementation': 1, 'pipeline': 1
    }
    for keyword, weight in methodology_keywords.items():
        if keyword in q:
            scores['methodology'] += weight
    
    # Application keywords
    application_keywords = {
        'application': 2, 'use case': 2, 'case study': 2, 'diagnosis': 1, 'classification': 1,
        'segmentation': 1, 'detection': 1, 'deployment': 1, 'real-world': 1, 'practical': 1
    }
    for keyword, weight in application_keywords.items():
        if keyword in q:
            scores['application'] += weight
    
    # Comparison keywords
    comparison_keywords = {
        'compare': 2, 'comparison': 2, 'vs': 2, 'versus': 2, 'benchmark': 2,
        'evaluation': 1, 'performance': 1, 'comparative': 2
    }
    for keyword, weight in comparison_keywords.items():
        if keyword in q:
            scores['comparison'] += weight
    
    # Recent keywords (updated years)
    recent_keywords = {
        'recent': 2, '2023': 1, '2024': 2, '2025': 3, 'latest': 2, 'new': 1,
        'current': 1, 'modern': 1, 'contemporary': 1
    }
    for keyword, weight in recent_keywords.items():
        if keyword in q:
            scores['recent'] += weight
    
    # Return intent with highest score, or 'general' if no matches
    max_score = max(scores.values())
    if max_score > 0:
        return max(scores, key=scores.get)
    return 'general'