"""
Structured PDF Summarization Module

This module provides comprehensive PDF analysis with structured extractive summaries
organized into clear sections, plus abstractive rewriting via Ollama/Mistral.
"""

import re
import requests
import logging
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..pdf_processing.extract_pdf import extract_text_from_pdf
from ..pdf_processing.preprocess import clean_scientific_text
from .tfidf_summarizer import summarize_tfidf
from .lsa_summarizer import summarize_lsa
from .abstractive_summarizer import _check_ollama_availability

logger = logging.getLogger(__name__)

@dataclass
class StructuredSummary:
    """Structured summary with organized sections"""
    contributions: str
    methodology: str
    results: str
    limitations: str
    future_work: str
    short_overview: str
    abstractive_summary: str = ""


@dataclass
class SectionBoundary:
    """Represents a detected section boundary with confidence score"""
    start_idx: int
    end_idx: int
    section_type: str
    confidence: float
    header_text: str


class SectionDetector:
    """Intelligent section detector with multi-pattern matching and scoring"""
    
    def __init__(self):
        # Comprehensive section patterns with variations
        self.section_patterns = {
            'abstract': [
                r'(?i)^\s*(?:abstract|summary|résumé)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?abstract\s*$',
            ],
            'introduction': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:introduction|background|motivation|overview)\s*$',
                r'(?i)^\s*(?:i\.?\s*)?introduction\s*$',
                r'(?i)^\s*(?:chapter\s+\d+\s*:?\s*)?introduction\s*$',
                r'(?i)^\s*(?:section\s+\d+\s*:?\s*)?(?:introduction|background)\s*$',
            ],
            'related_work': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:related\s+work|literature\s+review|prior\s+work|previous\s+work)\s*$',
                r'(?i)^\s*(?:ii\.?\s*)?related\s+work\s*$',
            ],
            'methodology': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:method(?:ology)?|approach|technique|algorithm|implementation|model|framework)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:materials?\s+and\s+methods?|experimental\s+setup|system\s+design)\s*$',
                r'(?i)^\s*(?:iii\.?\s*)?(?:method(?:ology)?|approach)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:proposed\s+method|our\s+approach|solution)\s*$',
            ],
            'results': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:results?|findings?|experiments?|evaluation|performance|analysis)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:experimental\s+results?|empirical\s+evaluation|performance\s+analysis)\s*$',
                r'(?i)^\s*(?:iv\.?\s*)?(?:results?|evaluation)\s*$',
            ],
            'discussion': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:discussion|interpretation|implications|analysis)\s*$',
                r'(?i)^\s*(?:v\.?\s*)?discussion\s*$',
            ],
            'conclusion': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:conclusion|summary|final\s+remarks?|closing)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:future\s+work|limitations?\s+and\s+future\s+work)\s*$',
                r'(?i)^\s*(?:vi\.?\s*)?(?:conclusion|summary)\s*$',
            ]
        }
        
        # Position-based section mapping (as fallback)
        self.position_mapping = {
            (0.0, 0.15): 'introduction',    # First 15%
            (0.15, 0.35): 'methodology',   # 15-35%
            (0.35, 0.65): 'results',       # 35-65%
            (0.65, 0.85): 'discussion',    # 65-85%
            (0.85, 1.0): 'conclusion'      # Last 15%
        }
    
    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect sections using multi-level approach with fallbacks
        """
        lines = text.split('\n')
        
        # Step 1: Pattern-based detection
        boundaries = self._detect_pattern_boundaries(lines)
        
        # Step 2: Validate and resolve conflicts
        boundaries = self._validate_boundaries(boundaries, len(lines))
        
        # Step 3: Position-based fallback if insufficient sections found
        if len(boundaries) < 3:
            boundaries = self._position_based_fallback(lines)
        
        # Step 4: Extract content between boundaries
        sections = self._extract_section_content(lines, boundaries)
        
        # Step 5: Final fallback - equal division
        if len(sections) < 3:
            sections = self._equal_division_fallback(text)
        
        return sections
    
    def _detect_pattern_boundaries(self, lines: List[str]) -> List[SectionBoundary]:
        """Detect section boundaries using regex patterns"""
        boundaries = []
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) > 100:  # Skip very long lines
                continue
            
            # Try each section type
            for section_type, patterns in self.section_patterns.items():
                max_confidence = 0.0
                
                for pattern in patterns:
                    if re.match(pattern, line_clean):
                        # Calculate confidence based on pattern specificity
                        confidence = self._calculate_pattern_confidence(pattern, line_clean)
                        max_confidence = max(max_confidence, confidence)
                
                if max_confidence > 0.5:  # Threshold for acceptance
                    boundaries.append(SectionBoundary(
                        start_idx=i,
                        end_idx=-1,  # Will be set later
                        section_type=section_type,
                        confidence=max_confidence,
                        header_text=line_clean
                    ))
                    break  # Don't match multiple patterns for same line
        
        return boundaries
    
    def _calculate_pattern_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.7
        
        # Boost confidence for numbered sections
        if re.search(r'\d+\.?\s*', text):
            base_confidence += 0.2
        
        # Boost for exact matches
        if len(text.split()) <= 3:  # Short, precise headers
            base_confidence += 0.1
        
        # Reduce for very generic patterns
        if 'analysis' in text.lower() or 'summary' in text.lower():
            base_confidence -= 0.1
        
        return min(base_confidence, 1.0)
    
    def _validate_boundaries(self, boundaries: List[SectionBoundary], total_lines: int) -> List[SectionBoundary]:
        """Validate and resolve conflicts in detected boundaries"""
        if not boundaries:
            return boundaries
        
        # Sort by start index
        boundaries.sort(key=lambda x: x.start_idx)
        
        # Remove duplicates and conflicts
        validated = []
        for boundary in boundaries:
            # Check if too close to previous boundary
            if validated and boundary.start_idx - validated[-1].start_idx < 10:
                # Keep the one with higher confidence
                if boundary.confidence > validated[-1].confidence:
                    validated[-1] = boundary
            else:
                validated.append(boundary)
        
        # Set end indices
        for i, boundary in enumerate(validated):
            if i < len(validated) - 1:
                boundary.end_idx = validated[i + 1].start_idx
            else:
                boundary.end_idx = total_lines
        
        return validated
    
    def _position_based_fallback(self, lines: List[str]) -> List[SectionBoundary]:
        """Fallback to position-based section detection"""
        logger.info("Using position-based fallback for section detection")
        
        total_lines = len(lines)
        boundaries = []
        
        for (start_pct, end_pct), section_type in self.position_mapping.items():
            start_idx = int(total_lines * start_pct)
            end_idx = int(total_lines * end_pct)
            
            boundaries.append(SectionBoundary(
                start_idx=start_idx,
                end_idx=end_idx,
                section_type=section_type,
                confidence=0.3,  # Low confidence for position-based
                header_text=f"Position-based {section_type}"
            ))
        
        return boundaries
    
    def _extract_section_content(self, lines: List[str], boundaries: List[SectionBoundary]) -> Dict[str, str]:
        """Extract content between section boundaries"""
        sections = {}
        
        for boundary in boundaries:
            content_lines = lines[boundary.start_idx + 1:boundary.end_idx]  # Skip header line
            content = '\n'.join(line.strip() for line in content_lines if line.strip())
            
            if content and len(content) > 50:  # Minimum content threshold
                sections[boundary.section_type] = content
        
        return sections
    
    def _equal_division_fallback(self, text: str) -> Dict[str, str]:
        """Final fallback: divide text into equal parts"""
        logger.warning("Using equal division fallback for section detection")
        
        lines = text.split('\n')
        total_lines = len(lines)
        section_size = total_lines // 5
        
        sections = {}
        section_types = ['introduction', 'methodology', 'results', 'discussion', 'conclusion']
        
        for i, section_type in enumerate(section_types):
            start_idx = i * section_size
            end_idx = (i + 1) * section_size if i < 4 else total_lines
            
            content_lines = lines[start_idx:end_idx]
            content = '\n'.join(line.strip() for line in content_lines if line.strip())
            
            if content:
                sections[section_type] = content
        
        return sections


def _split_into_sections(text: str) -> Dict[str, str]:
    """
    Enhanced section detection with robust multi-pattern matching and fallbacks.
    
    Args:
        text: Full PDF text content
        
    Returns:
        Dictionary with section names as keys and content as values
    """
    if not text or not text.strip():
        return {}
    
    detector = SectionDetector()
    sections = detector.detect_sections(text)
    
    # Always try to extract abstract separately (often at the beginning)
    abstract_content = _extract_abstract_separately(text)
    if abstract_content:
        sections['abstract'] = abstract_content
    
    logger.info(f"Detected sections: {list(sections.keys())}")
    return sections


def _extract_abstract_separately(text: str) -> Optional[str]:
    """Extract abstract using dedicated logic"""
    lines = text.split('\n')
    abstract_content = []
    in_abstract = False
    
    for i, line in enumerate(lines[:100]):  # Check first 100 lines
        line_clean = line.strip()
        if not line_clean:
            continue
        
        # Start of abstract
        if re.match(r'(?i)^\s*(?:abstract|summary|résumé)\s*$', line_clean):
            in_abstract = True
            continue
        
        # End of abstract (next section or keywords)
        if in_abstract:
            if (re.match(r'(?i)^\s*(?:\d+\.?\s*)?(?:introduction|keywords?|key\s+words)', line_clean) or
                len(abstract_content) > 30):  # Reasonable length limit
                break
            abstract_content.append(line_clean)
    
    if abstract_content and len(abstract_content) >= 3:  # Minimum abstract length
        return '\n'.join(abstract_content)
    
    return None


class ContentExtractor:
    """Advanced content extractor with contextual analysis and cross-validation"""
    
    def __init__(self):
        # Enhanced patterns with contextual markers
        self.contribution_patterns = [
            # Direct contribution statements
            r'(?i)(?:our|the|main|key|primary|novel)\s+(?:contribution|novelty|innovation|advance)s?\s+(?:is|are|include)',
            r'(?i)(?:we|this\s+(?:paper|work|study))\s+(?:propose|present|introduce|contribute|develop|make)',
            r'(?i)(?:novel|new|innovative|original|unique)\s+(?:approach|method|technique|algorithm|framework|model|solution)',
            r'(?i)(?:first|pioneering)\s+(?:to|work|study|approach|attempt)',
            r'(?i)(?:significantly|substantially|dramatically)\s+(?:improve|enhance|outperform|exceed|surpass)',
            r'(?i)(?:state-of-the-art|sota)\s+(?:performance|results|accuracy|precision)',
            # Numbered contributions
            r'(?i)(?:first|second|third|1\.|2\.|3\.)\s*,?\s*(?:we|our|this)',
            # Achievement statements
            r'(?i)(?:achieve|obtain|reach|attain)\s+(?:better|superior|improved|higher)'
        ]
        
        self.methodology_patterns = [
            # Method descriptions
            r'(?i)(?:we|our)\s+(?:use|employ|apply|implement|develop|design|adopt|utilize)',
            r'(?i)(?:algorithm|model|framework|approach|method|technique|system)\s+(?:is|was|consists|works|operates)',
            r'(?i)(?:based\s+on|using|utilizing|leveraging|employing|building\s+on)',
            r'(?i)(?:neural\s+network|machine\s+learning|deep\s+learning|transformer|attention|convolution)',
            r'(?i)(?:training|optimization|learning\s+rate|batch\s+size|epochs?|iterations?)',
            # Architecture descriptions
            r'(?i)(?:architecture|network|model)\s+(?:consists|comprises|includes|contains)',
            r'(?i)(?:input|output|hidden)\s+(?:layer|dimension|size|feature)'
        ]
        
        self.results_patterns = [
            # Performance metrics
            r'(?i)(?:accuracy|precision|recall|f1|score|performance)\s+(?:of|is|was|reaches?|achieves?)',
            r'(?i)(?:outperform|exceed|surpass|beat)\s+(?:baseline|previous|existing|state-of-the-art)',
            r'(?i)(?:improvement|gain|increase|boost)\s+(?:of|by|in)\s+[\d.]+%?',
            r'(?i)(?:results?|experiments?)\s+(?:show|demonstrate|indicate|reveal|suggest)',
            r'(?i)(?:compared\s+to|versus|vs\.?)\s+(?:baseline|previous|existing)',
            # Statistical significance
            r'(?i)(?:statistically\s+)?significant\s+(?:improvement|difference|gain)'
        ]
        
        self.limitations_patterns = [
            # Direct limitation statements
            r'(?i)(?:limitation|drawback|weakness|shortcoming|constraint)s?\s+(?:of|include|are)',
            r'(?i)(?:however|although|despite|unfortunately|nevertheless)',
            r'(?i)(?:does\s+not|cannot|unable\s+to|fails\s+to|limited\s+to)',
            r'(?i)(?:future\s+work|further\s+research|next\s+steps?)\s+(?:should|will|could|might)',
            r'(?i)(?:challenging|difficult|hard)\s+(?:to|for)'
        ]
        
        self.future_work_patterns = [
            r'(?i)(?:future\s+work|future\s+research|next\s+steps?|further\s+investigation)',
            r'(?i)(?:plan\s+to|intend\s+to|will|would\s+like\s+to|aim\s+to)',
            r'(?i)(?:extension|improvement|enhancement)\s+(?:of|to|could)',
            r'(?i)(?:explore|investigate|study|examine|consider)\s+(?:further|more|additional)',
            r'(?i)(?:potential|possible|promising)\s+(?:direction|avenue|approach)'
        ]
    
    def extract_contributions(self, sections: Dict[str, str]) -> str:
        """Extract contributions with contextual analysis and scoring"""
        candidates = []
        
        # Multi-section search with priority weighting
        search_sections = [
            ('abstract', sections.get('abstract', ''), 1.0),
            ('introduction', sections.get('introduction', ''), 0.8),
            ('conclusion', sections.get('conclusion', ''), 0.9),
            ('related_work', sections.get('related_work', ''), 0.3)
        ]
        
        for section_name, section_text, weight in search_sections:
            if not section_text:
                continue
            
            section_candidates = self._extract_with_patterns(
                section_text, self.contribution_patterns, section_name, weight
            )
            candidates.extend(section_candidates)
        
        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates, 'contribution')
        
        # Select top candidates with diversity
        selected = self._select_diverse_candidates(scored_candidates, max_count=3)
        
        if selected:
            return '. '.join([c['text'] for c in selected]) + '.'
        
        # Enhanced fallback with LSA
        return self._fallback_extraction(sections, ['abstract', 'introduction'], 'contribution')
    
    def extract_methodology(self, sections: Dict[str, str]) -> str:
        """Extract methodology with technical detail focus"""
        candidates = []
        
        search_sections = [
            ('methodology', sections.get('methodology', ''), 1.0),
            ('introduction', sections.get('introduction', ''), 0.6),
            ('results', sections.get('results', ''), 0.4),
            ('other', sections.get('other', ''), 0.7)
        ]
        
        for section_name, section_text, weight in search_sections:
            if not section_text or len(section_text) < 100:
                continue
            
            section_candidates = self._extract_with_patterns(
                section_text, self.methodology_patterns, section_name, weight
            )
            candidates.extend(section_candidates)
        
        scored_candidates = self._score_candidates(candidates, 'methodology')
        selected = self._select_diverse_candidates(scored_candidates, max_count=4)
        
        if selected:
            return '. '.join([c['text'] for c in selected]) + '.'
        
        return self._fallback_extraction(sections, ['methodology', 'other'], 'methodology')
    
    def extract_results(self, sections: Dict[str, str]) -> str:
        """Extract results with performance metrics focus"""
        candidates = []
        
        search_sections = [
            ('results', sections.get('results', ''), 1.0),
            ('conclusion', sections.get('conclusion', ''), 0.7),
            ('abstract', sections.get('abstract', ''), 0.6),
            ('discussion', sections.get('discussion', ''), 0.8)
        ]
        
        for section_name, section_text, weight in search_sections:
            if not section_text:
                continue
            
            section_candidates = self._extract_with_patterns(
                section_text, self.results_patterns, section_name, weight
            )
            candidates.extend(section_candidates)
        
        scored_candidates = self._score_candidates(candidates, 'results')
        selected = self._select_diverse_candidates(scored_candidates, max_count=4)
        
        if selected:
            return '. '.join([c['text'] for c in selected]) + '.'
        
        return self._fallback_extraction(sections, ['results', 'discussion'], 'results')
    
    def extract_limitations(self, sections: Dict[str, str]) -> str:
        """Extract limitations and challenges"""
        candidates = []
        
        search_sections = [
            ('discussion', sections.get('discussion', ''), 1.0),
            ('conclusion', sections.get('conclusion', ''), 0.8),
            ('results', sections.get('results', ''), 0.6),
            ('methodology', sections.get('methodology', ''), 0.5)
        ]
        
        for section_name, section_text, weight in search_sections:
            if not section_text:
                continue
            
            section_candidates = self._extract_with_patterns(
                section_text, self.limitations_patterns, section_name, weight
            )
            candidates.extend(section_candidates)
        
        scored_candidates = self._score_candidates(candidates, 'limitations')
        selected = self._select_diverse_candidates(scored_candidates, max_count=3)
        
        if selected:
            return '. '.join([c['text'] for c in selected]) + '.'
        
        return "Limitations not explicitly discussed in available sections."
    
    def extract_future_work(self, sections: Dict[str, str]) -> str:
        """Extract future work and research directions"""
        candidates = []
        
        search_sections = [
            ('conclusion', sections.get('conclusion', ''), 1.0),
            ('discussion', sections.get('discussion', ''), 0.8),
            ('results', sections.get('results', ''), 0.5)
        ]
        
        for section_name, section_text, weight in search_sections:
            if not section_text:
                continue
            
            section_candidates = self._extract_with_patterns(
                section_text, self.future_work_patterns, section_name, weight
            )
            candidates.extend(section_candidates)
        
        scored_candidates = self._score_candidates(candidates, 'future_work')
        selected = self._select_diverse_candidates(scored_candidates, max_count=3)
        
        if selected:
            return '. '.join([c['text'] for c in selected]) + '.'
        
        return "Future work directions not explicitly mentioned."
    
    def _extract_with_patterns(self, text: str, patterns: List[str], section_name: str, weight: float) -> List[Dict]:
        """Extract candidates using pattern matching with context"""
        candidates = []
        sentences = self._robust_sentence_split(text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 25 or len(sentence) > 300:  # Quality filter
                continue
            
            for pattern in patterns:
                if re.search(pattern, sentence):
                    # Calculate base score
                    score = weight * 0.7  # Base score from section weight
                    
                    # Position bonus (earlier sentences often more important)
                    position_bonus = max(0, (len(sentences) - i) / len(sentences) * 0.2)
                    score += position_bonus
                    
                    # Length bonus (moderate length preferred)
                    length_score = min(1.0, len(sentence) / 150) * 0.1
                    score += length_score
                    
                    candidates.append({
                        'text': sentence,
                        'score': score,
                        'section': section_name,
                        'position': i,
                        'pattern': pattern
                    })
                    break
        
        return candidates
    
    def _score_candidates(self, candidates: List[Dict], content_type: str) -> List[Dict]:
        """Score candidates with content-specific criteria"""
        if not candidates:
            return []
        
        # Content-specific keyword bonuses
        keyword_bonuses = {
            'contribution': ['novel', 'new', 'first', 'innovative', 'significant', 'improve'],
            'methodology': ['algorithm', 'model', 'framework', 'approach', 'technique', 'implement'],
            'results': ['accuracy', 'performance', 'outperform', 'improvement', 'significant'],
            'limitations': ['limitation', 'however', 'challenge', 'difficult', 'cannot'],
            'future_work': ['future', 'next', 'plan', 'explore', 'investigate', 'extend']
        }
        
        keywords = keyword_bonuses.get(content_type, [])
        
        for candidate in candidates:
            # Keyword bonus
            keyword_count = sum(1 for kw in keywords if kw in candidate['text'].lower())
            candidate['score'] += keyword_count * 0.05
            
            # Avoid repetitive content
            if len(set(candidate['text'].lower().split())) < len(candidate['text'].split()) * 0.7:
                candidate['score'] *= 0.8
        
        return sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    def _select_diverse_candidates(self, candidates: List[Dict], max_count: int) -> List[Dict]:
        """Select diverse candidates avoiding redundancy"""
        if not candidates:
            return []
        
        selected = []
        used_sections = set()
        
        for candidate in candidates:
            if len(selected) >= max_count:
                break
            
            # Check for diversity
            if candidate['section'] in used_sections and len(selected) > 0:
                # Allow if significantly higher score
                if candidate['score'] < selected[-1]['score'] * 1.2:
                    continue
            
            # Check for content similarity
            is_similar = False
            for selected_candidate in selected:
                similarity = self._calculate_similarity(candidate['text'], selected_candidate['text'])
                if similarity > 0.6:
                    is_similar = True
                    break
            
            if not is_similar:
                selected.append(candidate)
                used_sections.add(candidate['section'])
        
        return selected
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _robust_sentence_split(self, text: str) -> List[str]:
        """Robust sentence splitting handling abbreviations and citations"""
        # Handle common abbreviations
        text = re.sub(r'\b(?:Dr|Prof|Fig|Table|Eq|et\s+al)\.', lambda m: m.group().replace('.', '<!DOT!>'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<!DOT!>', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _fallback_extraction(self, sections: Dict[str, str], section_names: List[str], content_type: str) -> str:
        """Enhanced fallback using LSA summarization"""
        fallback_text = ''
        for section_name in section_names:
            if section_name in sections and sections[section_name]:
                fallback_text += sections[section_name] + ' '
        
        fallback_text = fallback_text.strip()
        if fallback_text and len(fallback_text) > 100:
            try:
                from .enhanced_lsa_summarizer import summarize_lsa_enhanced
                summary = summarize_lsa_enhanced(fallback_text, max_sentences=3)
                if summary and len(summary) > 50:
                    return summary
            except ImportError:
                # Fallback to basic LSA
                summary = summarize_lsa(fallback_text, max_sentences=3)
                if summary and len(summary) > 50:
                    return summary
        
        return f"{content_type.title()} not clearly identified in available sections."


def _extract_contributions(text: str, sections: Dict[str, str]) -> str:
    """Extract main contributions using enhanced content extractor"""
    extractor = ContentExtractor()
    return extractor.extract_contributions(sections)


def _extract_methodology(sections: Dict[str, str]) -> str:
    """Extract methodology using enhanced content extractor"""
    extractor = ContentExtractor()
    return extractor.extract_methodology(sections)


def _extract_results(sections: Dict[str, str]) -> str:
    """Extract results using enhanced content extractor"""
    extractor = ContentExtractor()
    return extractor.extract_results(sections)


def _extract_limitations(sections: Dict[str, str]) -> str:
    """Extract limitations using enhanced content extractor"""
    extractor = ContentExtractor()
    return extractor.extract_limitations(sections)


def _extract_future_work(sections: Dict[str, str]) -> str:
    """Extract future work using enhanced content extractor"""
    extractor = ContentExtractor()
    return extractor.extract_future_work(sections)


def _extract_limitations_and_future_work(sections: Dict[str, str]) -> Tuple[str, str]:
    """Extract limitations and future work using enhanced extractors"""
    limitations = _extract_limitations(sections)
    future_work = _extract_future_work(sections)
    return limitations, future_work


class PaperTypeDetector:
    """Detects paper type and research domain for adaptive processing"""
    
    def __init__(self):
        # Paper type patterns
        self.paper_type_patterns = {
            'theoretical': [
                r'(?i)\b(?:theorem|proof|lemma|proposition|corollary)\b',
                r'(?i)\b(?:mathematical|formal|theoretical)\s+(?:analysis|framework|model)\b',
                r'(?i)\b(?:complexity|convergence|optimality)\s+(?:analysis|proof)\b'
            ],
            'empirical': [
                r'(?i)\b(?:experiment|evaluation|benchmark|dataset|corpus)\b',
                r'(?i)\b(?:accuracy|precision|recall|f1|performance)\s+(?:score|metric|evaluation)\b',
                r'(?i)\b(?:training|testing|validation)\s+(?:set|data|phase)\b'
            ],
            'survey': [
                r'(?i)\b(?:survey|review|overview|taxonomy|classification)\b',
                r'(?i)\b(?:comprehensive|systematic)\s+(?:review|analysis|study)\b',
                r'(?i)\b(?:state-of-the-art|existing|current)\s+(?:approaches|methods|techniques)\b'
            ],
            'position': [
                r'(?i)\b(?:position|opinion|perspective|viewpoint)\s+(?:paper|article)\b',
                r'(?i)\b(?:argue|claim|advocate|propose)\s+(?:that|for)\b',
                r'(?i)\b(?:vision|manifesto|call\s+for)\b'
            ]
        }
        
        # Research domain patterns
        self.domain_patterns = {
            'machine_learning': [
                r'(?i)\b(?:machine\s+learning|deep\s+learning|neural\s+network)\b',
                r'(?i)\b(?:supervised|unsupervised|reinforcement)\s+learning\b',
                r'(?i)\b(?:gradient|backpropagation|optimization|training)\b'
            ],
            'natural_language_processing': [
                r'(?i)\b(?:natural\s+language|nlp|text\s+processing)\b',
                r'(?i)\b(?:tokenization|parsing|sentiment|translation)\b',
                r'(?i)\b(?:transformer|bert|gpt|attention)\s+(?:model|mechanism)\b'
            ],
            'computer_vision': [
                r'(?i)\b(?:computer\s+vision|image\s+processing|visual)\b',
                r'(?i)\b(?:convolutional|cnn|object\s+detection|segmentation)\b',
                r'(?i)\b(?:pixel|feature\s+extraction|classification)\b'
            ],
            'systems': [
                r'(?i)\b(?:distributed|parallel|concurrent|scalable)\b',
                r'(?i)\b(?:database|storage|network|protocol)\b',
                r'(?i)\b(?:performance|throughput|latency|optimization)\b'
            ],
            'security': [
                r'(?i)\b(?:security|privacy|cryptography|encryption)\b',
                r'(?i)\b(?:attack|vulnerability|defense|protection)\b',
                r'(?i)\b(?:authentication|authorization|access\s+control)\b'
            ],
            'theory': [
                r'(?i)\b(?:algorithm|complexity|computational)\b',
                r'(?i)\b(?:graph\s+theory|combinatorics|discrete)\b',
                r'(?i)\b(?:polynomial|exponential|logarithmic)\s+(?:time|space)\b'
            ]
        }
    
    def detect_paper_type(self, sections: Dict[str, str]) -> Tuple[str, float]:
        """Detect the type of research paper"""
        # Combine abstract and introduction for analysis
        analysis_text = ''
        for section in ['abstract', 'introduction', 'methodology']:
            if section in sections:
                analysis_text += sections[section] + ' '
        
        if not analysis_text.strip():
            return 'unknown', 0.0
        
        type_scores = {}
        
        for paper_type, patterns in self.paper_type_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, analysis_text))
                score += matches * 0.1  # Each match adds to score
            
            # Normalize by text length
            score = score / max(1, len(analysis_text.split()) / 100)
            type_scores[paper_type] = score
        
        # Return type with highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            confidence = min(1.0, type_scores[best_type])
            return best_type, confidence
        
        return 'unknown', 0.0
    
    def detect_domain(self, sections: Dict[str, str]) -> Tuple[str, float]:
        """Detect the research domain/field"""
        # Use title, abstract, and keywords for domain detection
        analysis_text = ''
        for section in ['abstract', 'introduction']:
            if section in sections:
                analysis_text += sections[section] + ' '
        
        if not analysis_text.strip():
            return 'unknown', 0.0
        
        domain_scores = {}
        
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, analysis_text))
                score += matches * 0.15  # Domain keywords are weighted higher
            
            # Normalize by text length
            score = score / max(1, len(analysis_text.split()) / 100)
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(1.0, domain_scores[best_domain])
            return best_domain, confidence
        
        return 'unknown', 0.0
    
    def get_adaptive_extraction_config(self, paper_type: str, domain: str) -> Dict[str, Any]:
        """Get adaptive configuration based on paper type and domain"""
        config = {
            'max_contributions': 3,
            'max_methodology': 4,
            'max_results': 4,
            'max_limitations': 3,
            'max_future_work': 3,
            'focus_sections': ['abstract', 'introduction', 'conclusion']
        }
        
        # Adjust based on paper type
        if paper_type == 'theoretical':
            config['max_contributions'] = 2
            config['max_methodology'] = 5  # More focus on theoretical approach
            config['focus_sections'].append('methodology')
        elif paper_type == 'empirical':
            config['max_results'] = 5  # More focus on experimental results
            config['focus_sections'].extend(['results', 'methodology'])
        elif paper_type == 'survey':
            config['max_contributions'] = 4  # Surveys often have multiple contributions
            config['max_results'] = 2  # Less focus on specific results
            config['focus_sections'].extend(['related_work', 'discussion'])
        
        # Adjust based on domain
        if domain == 'machine_learning':
            config['focus_sections'].extend(['methodology', 'results'])
        elif domain == 'theory':
            config['max_methodology'] = 5
            config['focus_sections'].append('methodology')
        
        return config


class QualityValidator:
    """Validates and scores the quality of extracted summaries"""
    
    def __init__(self):
        self.min_section_length = 30
        self.max_section_length = 1000
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
    
    def validate_structured_summary(self, summary: StructuredSummary, sections: Dict[str, str]) -> Dict[str, Any]:
        """Comprehensive validation of structured summary quality"""
        validation_results = {
            'overall_score': 0.0,
            'section_scores': {},
            'quality_level': 'poor',
            'issues': [],
            'recommendations': []
        }
        
        section_scores = []
        
        # Validate each section
        for field_name in ['contributions', 'methodology', 'results', 'limitations', 'future_work']:
            content = getattr(summary, field_name, '')
            score = self._validate_section_content(content, field_name, sections)
            validation_results['section_scores'][field_name] = score
            section_scores.append(score)
        
        # Calculate overall score
        validation_results['overall_score'] = sum(section_scores) / len(section_scores)
        
        # Determine quality level
        overall_score = validation_results['overall_score']
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                validation_results['quality_level'] = level
                break
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        return validation_results
    
    def _validate_section_content(self, content: str, section_type: str, original_sections: Dict[str, str]) -> float:
        """Validate individual section content quality"""
        if not content or content.strip() in [
            f"{section_type.title()} not clearly identified in available sections.",
            f"{section_type.title()} not explicitly discussed in available sections.",
            f"{section_type.title()} directions not explicitly mentioned."
        ]:
            return 0.1  # Very low score for default messages
        
        score = 0.0
        
        # Length validation (30% of score)
        length_score = self._score_content_length(content)
        score += length_score * 0.3
        
        # Content quality validation (40% of score)
        quality_score = self._score_content_quality(content, section_type)
        score += quality_score * 0.4
        
        # Relevance validation (30% of score)
        relevance_score = self._score_content_relevance(content, section_type, original_sections)
        score += relevance_score * 0.3
        
        return min(1.0, score)
    
    def _score_content_length(self, content: str) -> float:
        """Score content based on appropriate length"""
        length = len(content.strip())
        
        if length < self.min_section_length:
            return length / self.min_section_length  # Linear scaling up to minimum
        elif length > self.max_section_length:
            return max(0.5, 1.0 - (length - self.max_section_length) / self.max_section_length)
        else:
            return 1.0  # Optimal length range
    
    def _score_content_quality(self, content: str, section_type: str) -> float:
        """Score content quality based on section-specific criteria"""
        quality_indicators = {
            'contributions': [
                r'(?i)\b(?:novel|new|innovative|first|significant|improve)\b',
                r'(?i)\b(?:propose|present|introduce|develop|achieve)\b',
                r'(?i)\b(?:contribution|advance|breakthrough|innovation)\b'
            ],
            'methodology': [
                r'(?i)\b(?:algorithm|model|framework|approach|method)\b',
                r'(?i)\b(?:implement|design|develop|use|employ)\b',
                r'(?i)\b(?:training|optimization|evaluation|experiment)\b'
            ],
            'results': [
                r'(?i)\b(?:accuracy|performance|improvement|outperform)\b',
                r'(?i)\b(?:significant|better|superior|higher|lower)\b',
                r'(?i)\b(?:experiment|evaluation|benchmark|comparison)\b'
            ],
            'limitations': [
                r'(?i)\b(?:limitation|constraint|challenge|difficulty)\b',
                r'(?i)\b(?:however|although|despite|unfortunately)\b',
                r'(?i)\b(?:cannot|unable|limited|restricted)\b'
            ],
            'future_work': [
                r'(?i)\b(?:future|next|further|additional|extend)\b',
                r'(?i)\b(?:plan|intend|explore|investigate|improve)\b',
                r'(?i)\b(?:direction|opportunity|potential|possibility)\b'
            ]
        }
        
        patterns = quality_indicators.get(section_type, [])
        if not patterns:
            return 0.5  # Default score for unknown section types
        
        matches = 0
        for pattern in patterns:
            matches += len(re.findall(pattern, content))
        
        # Normalize by content length and number of patterns
        normalized_score = matches / max(1, len(content.split()) / 20)  # Per 20 words
        return min(1.0, normalized_score)
    
    def _score_content_relevance(self, content: str, section_type: str, original_sections: Dict[str, str]) -> float:
        """Score how well content matches the expected section type"""
        # Check if content contains section-specific keywords
        section_keywords = {
            'contributions': ['contribution', 'novel', 'propose', 'present', 'new'],
            'methodology': ['method', 'approach', 'algorithm', 'model', 'framework'],
            'results': ['results', 'performance', 'accuracy', 'experiment', 'evaluation'],
            'limitations': ['limitation', 'challenge', 'constraint', 'however', 'difficult'],
            'future_work': ['future', 'next', 'further', 'plan', 'explore']
        }
        
        keywords = section_keywords.get(section_type, [])
        if not keywords:
            return 0.5
        
        keyword_matches = sum(1 for kw in keywords if kw in content.lower())
        relevance_score = keyword_matches / len(keywords)
        
        # Bonus for coherent sentences
        sentences = content.split('.')
        coherent_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
        coherence_bonus = min(0.3, coherent_sentences / max(1, len(sentences)))
        
        return min(1.0, relevance_score + coherence_bonus)
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on validation results"""
        recommendations = []
        
        overall_score = validation_results['overall_score']
        section_scores = validation_results['section_scores']
        
        # Overall recommendations
        if overall_score < 0.4:
            recommendations.append("Consider using a different PDF or improving text extraction quality")
        elif overall_score < 0.6:
            recommendations.append("Summary quality is fair - consider manual review and editing")
        
        # Section-specific recommendations
        for section, score in section_scores.items():
            if score < 0.3:
                recommendations.append(f"The {section} section needs significant improvement - consider manual extraction")
            elif score < 0.5:
                recommendations.append(f"The {section} section could benefit from additional context or refinement")
        
        # Check for missing critical sections
        critical_sections = ['contributions', 'methodology', 'results']
        missing_critical = [s for s in critical_sections if section_scores.get(s, 0) < 0.2]
        if missing_critical:
            recommendations.append(f"Critical sections missing or inadequate: {', '.join(missing_critical)}")
        
        return recommendations


def _generate_short_overview(structured_summary: StructuredSummary) -> str:
    """Generate a concise 3-5 sentence overview"""
    # Combine key points from each section
    key_points = []
    
    if structured_summary.contributions:
        # Take first sentence of contributions
        first_contrib = structured_summary.contributions.split('.')[0] + '.'
        key_points.append(first_contrib)
    
    if structured_summary.methodology:
        # Take first sentence of methodology
        first_method = structured_summary.methodology.split('.')[0] + '.'
        key_points.append(first_method)
    
    if structured_summary.results:
        # Take first sentence of results
        first_result = structured_summary.results.split('.')[0] + '.'
        key_points.append(first_result)
    
    return ' '.join(key_points[:4])  # Limit to 4 sentences max


def _generate_ollama_abstractive_summary(structured_summary: StructuredSummary, 
                                       title: str, authors: List[str], year: str) -> str:
    """
    Generate human-readable abstractive summary using Ollama/Mistral
    """
    if not _check_ollama_availability():
        return "Ollama not available for abstractive summarization."
    
    # Prepare structured input for Ollama
    authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
    
    prompt = f"""Based on this structured analysis of a research paper, write a clear, fluent summary in 2-3 paragraphs:

Title: {title}
Authors: {authors_str} ({year})

CONTRIBUTIONS: {structured_summary.contributions[:300]}...

METHODOLOGY: {structured_summary.methodology[:300]}...

RESULTS: {structured_summary.results[:300]}...

LIMITATIONS: {structured_summary.limitations[:200]}...

FUTURE WORK: {structured_summary.future_work[:200]}...

Write a natural, human-readable summary that flows well and explains what this research is about, how it was done, what was found, and why it matters. Keep it concise but comprehensive."""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.5,
                    'top_p': 0.9,
                    'num_predict': 400,
                    'stop': ['\n\n\n', 'Title:', 'Paper:']
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            if result and len(result) > 100:
                return result
                
    except Exception as e:
        logger.warning(f"Ollama abstractive summary failed: {e}")
    
    # Fallback: combine structured sections into readable text
    return f"""This paper presents {structured_summary.contributions[:200]}... The methodology involves {structured_summary.methodology[:200]}... The results show {structured_summary.results[:200]}... {structured_summary.limitations[:100]}..."""


def process_pdf_structured_summary(pdf_path: str, title: str, authors: List[str], 
                                 year: str, max_chars: int = 50000) -> Tuple[StructuredSummary, Dict[str, Any]]:
    """
    Enhanced PDF processing with adaptive extraction and quality validation.
    
    Args:
        pdf_path: Path to PDF file
        title: Paper title
        authors: List of authors
        year: Publication year
        max_chars: Maximum characters to process (for memory management)
        
    Returns:
        Tuple of (StructuredSummary object, validation results)
    """
    logger.info(f"Processing PDF for structured summary: {title[:50]}...")
    
    try:
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_path)
        
        if not full_text or len(full_text) < 500:
            raise ValueError("PDF text extraction failed or insufficient content")
        
        # Clean and limit text size
        cleaned_text = clean_scientific_text(full_text)
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        # Enhanced section detection
        sections = _split_into_sections(cleaned_text)
        logger.info(f"Identified sections: {list(sections.keys())}")
        
        # Detect paper type and domain for adaptive processing
        type_detector = PaperTypeDetector()
        paper_type, type_confidence = type_detector.detect_paper_type(sections)
        domain, domain_confidence = type_detector.detect_domain(sections)
        
        logger.info(f"Detected paper type: {paper_type} (confidence: {type_confidence:.2f})")
        logger.info(f"Detected domain: {domain} (confidence: {domain_confidence:.2f})")
        
        # Get adaptive configuration
        config = type_detector.get_adaptive_extraction_config(paper_type, domain)
        
        # Enhanced content extraction with contextual analysis
        extractor = ContentExtractor()
        contributions = extractor.extract_contributions(sections)
        methodology = extractor.extract_methodology(sections)
        results = extractor.extract_results(sections)
        limitations = extractor.extract_limitations(sections)
        future_work = extractor.extract_future_work(sections)
        
        # Create structured summary
        structured_summary = StructuredSummary(
            contributions=contributions,
            methodology=methodology,
            results=results,
            limitations=limitations,
            future_work=future_work,
            short_overview=""  # Will be filled next
        )
        
        # Generate short overview
        structured_summary.short_overview = _generate_short_overview(structured_summary)
        
        # Quality validation
        validator = QualityValidator()
        validation_results = validator.validate_structured_summary(structured_summary, sections)
        
        logger.info(f"Summary quality: {validation_results['quality_level']} (score: {validation_results['overall_score']:.2f})")
        
        # Generate abstractive summary with Ollama
        structured_summary.abstractive_summary = _generate_ollama_abstractive_summary(
            structured_summary, title, authors, year
        )
        
        # Add metadata to validation results
        validation_results.update({
            'paper_type': paper_type,
            'type_confidence': type_confidence,
            'domain': domain,
            'domain_confidence': domain_confidence,
            'sections_detected': list(sections.keys()),
            'adaptive_config': config
        })
        
        logger.info("✅ Enhanced structured PDF summary completed successfully")
        return structured_summary, validation_results
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        # Return error summary with validation
        error_summary = StructuredSummary(
            contributions=f"Error processing PDF: {str(e)}",
            methodology="Unable to extract methodology",
            results="Unable to extract results",
            limitations="Unable to extract limitations",
            future_work="Unable to extract future work",
            short_overview="PDF processing failed",
            abstractive_summary="Abstractive summary unavailable due to processing error"
        )
        
        error_validation = {
            'overall_score': 0.0,
            'quality_level': 'error',
            'section_scores': {},
            'issues': [f"Processing error: {str(e)}"],
            'recommendations': ["Check PDF file integrity and try again"],
            'paper_type': 'unknown',
            'domain': 'unknown'
        }
        
        return error_summary, error_validation


def download_and_process_pdf(pdf_url: str, title: str, authors: List[str], 
                           year: str, cache_dir: Optional[Path] = None) -> StructuredSummary:
    """
    Download PDF from URL and process it for structured summary.
    
    Args:
        pdf_url: URL to download PDF from
        title: Paper title
        authors: List of authors  
        year: Publication year
        cache_dir: Directory to cache downloaded PDFs
        
    Returns:
        StructuredSummary object
    """
    if not cache_dir:
        cache_dir = Path("data/cache/pdfs")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
    pdf_filename = f"{safe_title}_{year}.pdf"
    pdf_path = cache_dir / pdf_filename
    
    try:
        # Download PDF if not cached
        if not pdf_path.exists():
            logger.info(f"Downloading PDF: {pdf_url}")
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF cached at: {pdf_path}")
        else:
            logger.info(f"Using cached PDF: {pdf_path}")
        
        # Process the PDF
        return process_pdf_structured_summary(pdf_path, title, authors, year)
        
    except Exception as e:
        logger.error(f"PDF download/processing failed: {e}")
        return StructuredSummary(
            contributions=f"Failed to download/process PDF: {str(e)}",
            methodology="PDF unavailable",
            results="PDF unavailable", 
            limitations="PDF unavailable",
            future_work="PDF unavailable",
            short_overview="PDF processing failed",
            abstractive_summary="Unable to generate summary due to PDF access issues"
        )
