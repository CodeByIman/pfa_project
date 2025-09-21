def _sanitize_pdf_text(raw: str) -> str:
    """Lightly sanitize raw PDF text to reduce noise before sectioning.
    - Remove LaTeX math blocks and inline math
    - Drop URLs and image/figure/table references
    - Collapse excessive whitespace while preserving paragraphs
    """
    if not raw:
        return raw
    text = raw
    # Remove LaTeX math and inline formulas
    text = re.sub(r"\$[^$\n]{1,200}\$", " ", text)
    text = re.sub(r"\\\([\s\S]{1,300}?\\\)", " ", text)
    text = re.sub(r"\\\[[\s\S]{1,400}?\\\]", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove figure/table/page references lines
    text = re.sub(r"^(?:figure|fig\.|table|tab\.|page)\s+\d+.*$", " ", text, flags=re.IGNORECASE|re.MULTILINE)
    # Remove citation brackets
    text = re.sub(r"\[(?:\d+[ ,;-]?)+\]", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()

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

from ..pdf_processing.extractor import extract_text_from_pdf
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

"""
Enhanced Section Detection for Academic Papers
Addresses the issues identified in your test results
"""

import re
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedSectionDetector:
    """
    Improved section detector with multiple detection strategies
    and better handling of academic paper formats
    """
    
    def __init__(self):
        # Enhanced patterns with more variations and flexibility
        self.section_patterns = {
            'abstract': [
                r'(?i)^\s*(?:\d*\.?\s*)?(?:abstract|summary|résumé)\s*:?\s*$',
                r'(?i)^\s*(?:[IVX]+\.?\s*)?(?:abstract|summary)\s*$',
                r'(?i)^abstract\s*$',
            ],
            'introduction': [
                r'(?i)^\s*(?:\d+\.?\s*)?introduction\s*:?\s*$',
                r'(?i)^\s*(?:[IVX]+\.?\s*)?introduction\s*$',
                r'(?i)^\s*(?:1\.?\s*)?introduction\s*$',
                r'(?i)^introduction\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?background\s*$',
            ],
            'methodology': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:methodology|methods?|approach|model|technique)\s*:?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?experimental\s+(?:setup|design|method|procedure)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:materials?\s+and\s+)?methods?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:implementation|algorithm|framework)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?proposed\s+(?:method|approach|model)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?system\s+(?:design|architecture|overview)\s*$',
            ],
            'results': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:results?|findings?|experiments?|evaluation)\s*:?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?experimental\s+(?:results?|evaluation)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?performance\s+(?:evaluation|analysis|results?)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:empirical\s+)?(?:analysis|study)\s*$',
            ],
            'discussion': [
                r'(?i)^\s*(?:\d+\.?\s*)?discussion\s*:?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:analysis|interpretation)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?results?\s+and\s+discussion\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?discussion\s+of\s+results?\s*$',
            ],
            'conclusion': [
                r'(?i)^\s*(?:\d+\.?\s*)?conclusions?\s*:?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?concluding\s+(?:remarks?|thoughts?)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?summary\s+and\s+conclusions?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?final\s+(?:remarks?|thoughts?)\s*$',
            ],
            'related_work': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:related\s+work|prior\s+work|previous\s+work)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:background|literature\s+review)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?state\s+of\s+the\s+art\s*$',
            ],
            'future_work': [
                r'(?i)^\s*(?:\d+\.?\s*)?(?:future\s+work|future\s+directions?)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:next\s+steps?|future\s+research)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:open\s+problems?|challenges?)\s*$',
            ],
            'limitations': [
                r'(?i)^\s*(?:\d+\.?\s*)?limitations?\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?(?:challenges?|drawbacks?)\s*$',
                r'(?i)^\s*(?:\d+\.?\s*)?threats?\s+to\s+validity\s*$',
            ]
        }
        
        # Enhanced content-based indicators with more comprehensive patterns
        self.content_indicators = {
            'abstract': [
                r'(?i)(?:this\s+(?:paper|work|study|research))\s+(?:presents?|proposes?|introduces?|describes?|addresses?)',
                r'(?i)(?:we|the\s+authors?)\s+(?:present|propose|introduce|develop|investigate)',
                r'(?i)in\s+this\s+(?:paper|work|study|article)',
                r'(?i)(?:main\s+)?(?:contribution|result|finding)s?\s+(?:of\s+this\s+work\s+)?(?:are?|include)',
            ],
            'introduction': [
                r'(?i)in\s+recent\s+(?:years?|decades?)',
                r'(?i)(?:the\s+)?(?:field\s+of|area\s+of|domain\s+of|problem\s+of)',
                r'(?i)(?:traditional|existing|current|conventional)\s+(?:approaches?|methods?|techniques?)',
                r'(?i)(?:motivation|rationale)\s+(?:for\s+this\s+work|behind)',
                r'(?i)(?:the\s+)?(?:main\s+)?(?:research\s+)?(?:question|problem|challenge)',
            ],
            'methodology': [
                r'(?i)(?:our|the\s+proposed)\s+(?:approach|method|algorithm|framework|model|technique)',
                r'(?i)(?:we|the\s+algorithm)\s+(?:use|employ|implement|develop|design|adopt|utilize)',
                r'(?i)(?:the\s+)?(?:model|algorithm|system|framework)\s+(?:consists?|works?|operates?)',
                r'(?i)(?:based\s+on|using|utilizing|leveraging|employing|building\s+upon)',
                r'(?i)(?:training|optimization|learning)\s+(?:procedure|process|phase)',
                r'(?i)(?:architecture|design|structure)\s+(?:of\s+)?(?:the\s+)?(?:model|system|network)',
            ],
            'results': [
                r'(?i)(?:experimental\s+)?results?\s+(?:show|demonstrate|indicate|reveal|suggest)',
                r'(?i)(?:our\s+)?(?:experiments?|evaluation|study)\s+(?:show|demonstrate|reveal)',
                r'(?i)(?:performance|accuracy|precision|recall|f1-score)\s+(?:of|is|was|reaches?)',
                r'(?i)(?:outperform|exceed|surpass|beat|improve\s+(?:over|upon))',
                r'(?i)(?:compared\s+(?:to|with)|versus|vs\.?)\s+(?:baseline|previous|existing)',
                r'(?i)(?:significant|substantial|dramatic)\s+(?:improvement|gain|increase)',
            ],
            'discussion': [
                r'(?i)(?:these\s+)?results?\s+(?:suggest|indicate|imply|show)',
                r'(?i)(?:the\s+)?(?:findings|results?|observations?)\s+(?:indicate|suggest|show)',
                r'(?i)(?:possible\s+)?(?:explanation|reason|cause)\s+(?:for|of)',
                r'(?i)(?:implications?\s+of|impact\s+of)',
            ],
            'conclusion': [
                r'(?i)in\s+(?:conclusion|summary|closing)',
                r'(?i)(?:we\s+)?(?:conclude|summarize|find)\s+that',
                r'(?i)this\s+(?:work|paper|study)\s+has\s+(?:shown|demonstrated|presented)',
                r'(?i)(?:to\s+)?(?:conclude|summarize)',
                r'(?i)(?:overall|in\s+general),\s*(?:the|our|this)',
            ],
            'limitations': [
                r'(?i)(?:limitation|drawback|weakness|constraint)s?\s+(?:of|include|are)',
                r'(?i)(?:however|although|despite|unfortunately|nevertheless)',
                r'(?i)(?:one|a|the)\s+(?:main\s+)?(?:limitation|drawback|issue|problem)',
                r'(?i)(?:does\s+not|cannot|unable\s+to|fails\s+to|limited\s+to)',
            ],
            'future_work': [
                r'(?i)(?:future\s+work|future\s+research|further\s+work|next\s+steps?)',
                r'(?i)(?:plan\s+to|intend\s+to|will|would\s+like\s+to|aim\s+to)',
                r'(?i)(?:potential\s+)?(?:extension|improvement|enhancement)',
                r'(?i)(?:interesting\s+)?(?:direction|avenue|area)\s+for\s+future',
            ]
        }

    def detect_sections_enhanced(self, text: str) -> Dict[str, str]:
        """
        Enhanced section detection using multiple strategies with improved logic
        """
        if not text or not text.strip():
            return {}
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Strategy 1: Enhanced header-based detection
        header_sections = self._detect_by_headers_enhanced(text)
        if len(header_sections) >= 2 and self._validate_sections(header_sections):
            logger.info(f"Header-based detection found {len(header_sections)} valid sections")
            return header_sections
        
        # Strategy 2: Content-based detection with context
        content_sections = self._detect_by_content_enhanced(text)
        if len(content_sections) >= 2 and self._validate_sections(content_sections):
            logger.info(f"Content-based detection found {len(content_sections)} valid sections")
            return content_sections
        
        # Strategy 3: Hybrid approach combining headers and content
        hybrid_sections = self._hybrid_detection(text)
        if len(hybrid_sections) >= 2 and self._validate_sections(hybrid_sections):
            logger.info(f"Hybrid detection found {len(hybrid_sections)} valid sections")
            return hybrid_sections
        
        # Strategy 4: Intelligent paragraph-based splitting
        paragraph_sections = self._detect_by_paragraphs_enhanced(text)
        if len(paragraph_sections) >= 2:
            logger.info(f"Enhanced paragraph-based detection found {len(paragraph_sections)} sections")
            return paragraph_sections
        
        # Final fallback with better content preservation
        logger.warning("All detection strategies failed, using enhanced fallback")
        return self._enhanced_fallback_detection(text)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better processing"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Normalize section headers
        text = re.sub(r'^\s*([IVX]+)\.\s*([A-Z])', r'\1. \2', text, flags=re.MULTILINE)
        # Fix common OCR issues
        text = re.sub(r'(?<=\w)(?=[A-Z][a-z])', ' ', text)
        return text.strip()

    def _detect_by_headers_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced header detection with better boundary identification"""
        lines = text.split('\n')
        sections = {}
        current_section = None
        current_content = []
        header_positions = []
        
        # First pass: identify all potential headers
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean or len(line_clean) > 100:  # Skip very long lines
                continue
            
            detected_section = self._identify_section_header(line_clean)
            if detected_section:
                header_positions.append((i, detected_section, line_clean))
        
        # Second pass: build sections with better content boundaries
        current_header_idx = 0
        for i, line in enumerate(lines):
            # Check if this line is a section header
            if current_header_idx < len(header_positions) and i == header_positions[current_header_idx][0]:
                # Save previous section
                if current_section and current_content:
                    content = self._clean_section_content('\n'.join(current_content))
                    if content and len(content) > 20:  # Minimum content length
                        sections[current_section] = content
                
                # Start new section
                current_section = header_positions[current_header_idx][1]
                current_content = []
                current_header_idx += 1
            else:
                # Add to current section
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            content = self._clean_section_content('\n'.join(current_content))
            if content and len(content) > 20:
                sections[current_section] = content
        
        return sections

    def _identify_section_header(self, line: str) -> str:
        """Identify section type from header line"""
        line_clean = line.strip()
        
        # Check each section type
        for section_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, line_clean):
                    return section_type
        
        return None

    def _detect_by_content_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced content-based detection with better context analysis"""
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) < 3:
            return {}
        
        sections = {}
        paragraph_assignments = {}
        
        # Analyze each paragraph for section indicators
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            best_section = None
            best_score = 0
            
            for section_type, indicators in self.content_indicators.items():
                score = self._calculate_content_score(paragraph, indicators)
                if score > best_score:
                    best_score = score
                    best_section = section_type
            
            if best_section and best_score > 0.5:  # Minimum confidence threshold
                paragraph_assignments[i] = (best_section, best_score)
        
        # Group consecutive paragraphs of the same type
        current_section = None
        current_content = []
        
        for i, paragraph in enumerate(paragraphs):
            if i in paragraph_assignments:
                assigned_section = paragraph_assignments[i][0]
                
                if assigned_section != current_section:
                    # Save previous section
                    if current_section and current_content:
                        content = '\n\n'.join(current_content)
                        if len(content) > 50:
                            sections[current_section] = content
                    
                    # Start new section
                    current_section = assigned_section
                    current_content = [paragraph]
                else:
                    current_content.append(paragraph)
            else:
                # Add unassigned paragraphs to current section
                if current_section and current_content:
                    current_content.append(paragraph)
        
        # Save final section
        if current_section and current_content:
            content = '\n\n'.join(current_content)
            if len(content) > 50:
                sections[current_section] = content
        
        return sections

    def _hybrid_detection(self, text: str) -> Dict[str, str]:
        """Hybrid approach combining header and content detection"""
        header_sections = self._detect_by_headers_enhanced(text)
        content_sections = self._detect_by_content_enhanced(text)
        
        if not header_sections and not content_sections:
            return {}
        
        # Merge strategies, preferring header-based when available
        merged_sections = {}
        
        # Start with header-based sections
        for section_type, content in header_sections.items():
            merged_sections[section_type] = content
        
        # Add content-based sections that don't conflict
        for section_type, content in content_sections.items():
            if section_type not in merged_sections:
                merged_sections[section_type] = content
            elif len(content) > len(merged_sections[section_type]):
                # Use longer content if significantly better
                merged_sections[section_type] = content
        
        return merged_sections

    def _detect_by_paragraphs_enhanced(self, text: str) -> Dict[str, str]:
        """Enhanced paragraph-based splitting with better content analysis"""
        paragraphs = self._split_into_paragraphs(text)
        
        if len(paragraphs) < 4:
            return {}
        
        sections = {}
        
        # More intelligent paragraph assignment based on position and content
        n_paragraphs = len(paragraphs)
        
        # First paragraph - likely abstract or introduction
        first_para = paragraphs[0]
        if len(first_para) > 100 and any(pattern in first_para.lower() 
                                        for pattern in ['this paper', 'we present', 'this work']):
            sections['abstract'] = first_para
            start_idx = 1
        else:
            sections['introduction'] = first_para
            start_idx = 1
        
        # Distribute remaining paragraphs more intelligently
        remaining = paragraphs[start_idx:]
        n_remaining = len(remaining)
        
        if n_remaining >= 6:
            # Full structure: intro/related, methodology, results, discussion, conclusion
            sections['methodology'] = '\n\n'.join(remaining[:n_remaining//3])
            sections['results'] = '\n\n'.join(remaining[n_remaining//3:2*n_remaining//3])
            sections['conclusion'] = '\n\n'.join(remaining[2*n_remaining//3:])
        elif n_remaining >= 3:
            # Basic structure: methodology, results, conclusion
            sections['methodology'] = '\n\n'.join(remaining[:n_remaining//3])
            sections['results'] = '\n\n'.join(remaining[n_remaining//3:2*n_remaining//3])
            sections['conclusion'] = '\n\n'.join(remaining[2*n_remaining//3:])
        else:
            # Minimal structure
            sections['other'] = '\n\n'.join(remaining)
        
        return sections

    def _enhanced_fallback_detection(self, text: str) -> Dict[str, str]:
        """Enhanced fallback with better content analysis"""
        sections = {}
        
        # Try to identify abstract at the beginning
        abstract_content = self._extract_abstract_fallback(text)
        if abstract_content:
            sections['abstract'] = abstract_content
            remaining_text = text.replace(abstract_content, '', 1).strip()
        else:
            remaining_text = text
        
        # Split remaining text more intelligently
        text_length = len(remaining_text)
        if text_length > 1000:
            # Find natural break points
            break_points = self._find_natural_breaks(remaining_text)
            
            if len(break_points) >= 2:
                sections['introduction'] = remaining_text[:break_points[0]].strip()
                sections['methodology'] = remaining_text[break_points[0]:break_points[1]].strip()
                sections['results'] = remaining_text[break_points[1]:].strip()
            else:
                # Simple split
                mid_point = text_length // 2
                sections['introduction'] = remaining_text[:mid_point].strip()
                sections['results'] = remaining_text[mid_point:].strip()
        else:
            sections['other'] = remaining_text
        
        return {k: v for k, v in sections.items() if v and len(v) > 20}

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split on double newlines
        paragraphs = text.split('\n\n')
        # Clean and filter
        cleaned = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 30:  # Minimum paragraph length
                cleaned.append(para)
        return cleaned

    def _calculate_content_score(self, paragraph: str, indicators: List[str]) -> float:
        """Calculate content score based on indicators"""
        paragraph_lower = paragraph.lower()
        score = 0.0
        
        for indicator in indicators:
            matches = re.finditer(indicator, paragraph_lower)
            for match in matches:
                # Position bonus (earlier matches score higher)
                position_factor = 1.0 - (match.start() / len(paragraph_lower)) * 0.3
                score += position_factor
        
        # Normalize by paragraph length
        return min(score / max(len(paragraph) / 100, 1), 2.0)

    def _clean_section_content(self, content: str) -> str:
        """Clean section content"""
        content = content.strip()
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        # Remove leading/trailing empty lines
        content = re.sub(r'^\n+|\n+$', '', content)
        return content

    def _validate_sections(self, sections: Dict[str, str]) -> bool:
        """Validate that sections contain meaningful content"""
        if not sections:
            return False
        
        # Check that sections have reasonable content
        total_length = sum(len(content) for content in sections.values())
        if total_length < 200:  # Too little content
            return False
        
        # Check for minimum section diversity
        if len(sections) == 1 and 'other' in sections:
            return False
        
        return True

    def _extract_abstract_fallback(self, text: str) -> str:
        """Extract abstract when no clear header is found"""
        lines = text.split('\n')
        
        # Look for abstract indicators in first 30 lines
        for i, line in enumerate(lines[:30]):
            line_clean = line.strip().lower()
            if 'abstract' in line_clean and len(line_clean) < 20:
                # Found abstract header, collect following content
                content_lines = []
                for j in range(i + 1, min(i + 20, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if any(keyword in next_line.lower() for keyword in ['introduction', 'keywords', '1.', 'i.']):
                        break
                    content_lines.append(next_line)
                
                if content_lines and len(' '.join(content_lines)) > 100:
                    return '\n'.join(content_lines)
        
        return None

    def _find_natural_breaks(self, text: str) -> List[int]:
        """Find natural break points in text"""
        break_points = []
        
        # Look for paragraph boundaries with topic shifts
        paragraphs = text.split('\n\n')
        current_pos = 0
        
        for i, para in enumerate(paragraphs[:-1]):  # Exclude last paragraph
            current_pos += len(para) + 2  # +2 for \n\n
            
            # Look for topic shift indicators
            next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
            
            if self._is_topic_shift(para, next_para):
                break_points.append(current_pos)
        
        return break_points

    def _is_topic_shift(self, para1: str, para2: str) -> bool:
        """Detect if there's a topic shift between paragraphs"""
        if not para1 or not para2:
            return False
        
        # Look for transition indicators
        transition_words = ['however', 'moreover', 'furthermore', 'in contrast', 
                          'on the other hand', 'meanwhile', 'next', 'then']
        
        para2_start = para2.lower()[:100]
        
        return any(word in para2_start for word in transition_words)


def improved_split_into_sections(text: str) -> Dict[str, str]:
    """
    Improved section splitting function with better academic paper understanding
    """
    detector = EnhancedSectionDetector()
    sections = detector.detect_sections_enhanced(text)
    
    logger.info(f"Enhanced detection found sections: {list(sections.keys())}")
    
    # Post-processing to ensure quality
    sections = _post_process_sections(sections, text)
    
    return sections


def _post_process_sections(sections: Dict[str, str], original_text: str) -> Dict[str, str]:
    """Post-process sections to improve quality and coverage"""
    
    if not sections:
        return {'other': original_text}
    
    # Ensure minimum content coverage
    total_section_length = sum(len(content) for content in sections.values())
    original_length = len(original_text)
    coverage_ratio = total_section_length / original_length
    
    if coverage_ratio < 0.7:  # Less than 70% coverage
        logger.warning(f"Low section coverage: {coverage_ratio:.2f}")
        
        # Add missing content as 'other' section
        all_section_content = '\n\n'.join(sections.values())
        missing_parts = []
        
        # Simple approach: find parts not in any section
        for paragraph in original_text.split('\n\n'):
            paragraph = paragraph.strip()
            if len(paragraph) > 50 and paragraph not in all_section_content:
                missing_parts.append(paragraph)
        
        if missing_parts:
            if 'other' in sections:
                sections['other'] += '\n\n' + '\n\n'.join(missing_parts)
            else:
                sections['other'] = '\n\n'.join(missing_parts)
    
    # Remove sections that are too short
    min_length = 30
    sections = {k: v for k, v in sections.items() if len(v.strip()) >= min_length}
    
    return sections


# Enhanced utility functions remain the same but with improved extraction
def extract_key_sentences(text: str, keywords: List[str], max_sentences: int = 3) -> str:
    """
    Enhanced key sentence extraction with better context preservation
    """
    if not text or not keywords:
        return "Content not available."
    
    sentences = re.split(r'[.!?]+', text)
    scored_sentences = []
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        
        score = 0
        sentence_lower = sentence.lower()
        
        # Enhanced scoring
        for keyword in keywords:
            if keyword.lower() in sentence_lower:
                # Context bonus for keywords at sentence beginning
                if sentence_lower.startswith(keyword.lower()):
                    score += 1.5
                else:
                    score += 1
        
        # Position bonus (earlier sentences often more important)
        position_bonus = max(0, (len(sentences) - i) / len(sentences) * 0.3)
        score += position_bonus
        
        # Length bonus (prefer moderate length)
        if 50 <= len(sentence) <= 200:
            score += 0.3
        elif len(sentence) > 200:
            score += 0.1
        
        # Penalty for too many citations
        citations = len(re.findall(r'\[[^\]]+\]|\([^)]+\)', sentence))
        if citations > 3:
            score -= 0.2
        
        # Bonus for sentences with numbers/metrics (often important results)
        if re.search(r'\d+(?:\.\d+)?%?', sentence):
            score += 0.2
        
        if score > 0.5:  # Minimum threshold
            scored_sentences.append((sentence, score))
    
    # Sort by score and select diverse sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Select diverse sentences (avoid too similar content)
    selected_sentences = []
    for sentence, score in scored_sentences:
        if len(selected_sentences) >= max_sentences:
            break
        
        # Check similarity with already selected
        is_similar = False
        for selected in selected_sentences:
            words1 = set(sentence.lower().split())
            words2 = set(selected.lower().split())
            if len(words1.intersection(words2)) / len(words1.union(words2)) > 0.6:
                is_similar = True
                break
        
        if not is_similar:
            selected_sentences.append(sentence)
    
    if selected_sentences:
        return '. '.join(selected_sentences) + '.'
    else:
        # Fallback: return first substantial sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if 30 <= len(sentence) <= 300:
                return sentence + '.'
        return "Relevant content not clearly identified."



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
            if len(sentence) < 25 or len(sentence) > 350:  # Quality filter (allow slightly longer)
                continue
            # Skip noisy/equation-like sentences
            if re.search(r"https?://|\\\(|\\\[|\$.*\$", sentence):
                continue
            # Skip sentences dominated by symbols/digits
            total = len(sentence)
            sym_ratio = sum(1 for ch in sentence if ch in "=+*/^_|<>∑√≈≃≤≥δλξπϕμσ∆∂→↦·•◦□{}[]()") / max(1, total)
            dig_ratio = sum(1 for ch in sentence if ch.isdigit()) / max(1, total)
            if sym_ratio > 0.25 or dig_ratio > 0.5:
                continue
            # Skip obvious section labels
            if re.match(r"^(figure|fig\.|table|tab\.|algorithm|theorem|lemma)\b", sentence, flags=re.IGNORECASE):
                continue
        
            for pattern in patterns:
                if re.search(pattern, sentence):
                    # Calculate base score
                    score = 1.0 * weight  # Base score from section weight
                    
                    # Position bonus (earlier sentences often more important)
                    position_bonus = max(0, (len(sentences) - i) / len(sentences) * 0.2)
                    score += position_bonus
                    
                    # Length bonus (moderate length preferred)
                    length_score = min(1.0, len(sentence) / 180) * 0.1
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
    
    # Prepare structured input for Ollama: combine cleaned sections into one extractive block
    authors_str = ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
    sections_block = f"""
CONTRIBUTIONS:
{structured_summary.contributions}

METHODOLOGY:
{structured_summary.methodology}

RESULTS:
{structured_summary.results}

LIMITATIONS:
{structured_summary.limitations}

FUTURE_WORK:
{structured_summary.future_work}
""".strip()
    # Light sanitize for LLM
    sections_block = _sanitize_pdf_text(sections_block)
    
    prompt = f"""
Using ONLY the following structured extractive content, write a fluent, human-like summary (8–12 sentences, 2–3 paragraphs, no bullets) that starts with "This paper" and explains the problem/context, methodology, key results, and implications.

Title: {title}
Authors: {authors_str} ({year})

EXTRACTIVE CONTENT:
<<<
{sections_block}
>>>

Requirements:
- Paraphrase; do NOT copy equations, variables, or proof snippets.
- No figure/table/page references. No citations. No bullet points.
- Keep it coherent and readable for a general research audience.
"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'num_predict': 12000,
                    'num_ctx': 8192,
                    'repeat_penalty': 1.1,
                    'presence_penalty': 0.1
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            if result and len(result) > 300:
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
        
        # Sanitize then clean for NLP
        text_sanitized = _sanitize_pdf_text(full_text)
        text_clean = clean_scientific_text(text_sanitized)
        if len(text_clean) > max_chars:
            text_clean = text_clean[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        # Save cleaned text next to the PDF for caching and reuse
        try:
            clean_text_path = Path(pdf_path).with_suffix('.clean.txt')
            with clean_text_path.open('w', encoding='utf-8') as f:
                f.write(text_clean)
        except Exception as e:
            logger.warning(f"Failed to save cleaned text: {e}")
        
        # Enhanced section detection
        sections = improved_split_into_sections(text_clean)
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
            'sections_text': sections,  # expose raw sections for frontend
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
        cache_dir = Path("data/pdfs")
    
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
        structured_summary, validation_results = process_pdf_structured_summary(pdf_path, title, authors, year)
        return structured_summary
        
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


def download_and_process_pdf_with_details(pdf_url: str, title: str, authors: List[str], 
                           year: str, cache_dir: Optional[Path] = None) -> Tuple[StructuredSummary, Dict[str, Any]]:
    """
    Same as download_and_process_pdf, but also returns validation results (includes sections_text).
    """
    if not cache_dir:
        cache_dir = Path("data/pdfs")
    
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
        structured_summary, validation_results = process_pdf_structured_summary(pdf_path, title, authors, year)
        return structured_summary, validation_results
        
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
        ), {
            'overall_score': 0.0,
            'quality_level': 'error',
            'section_scores': {},
            'issues': [f"PDF error: {str(e)}"],
            'recommendations': ["Verify the PDF URL or network connectivity"],
            'sections_detected': [],
            'sections_text': {},
        }
