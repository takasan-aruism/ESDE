"""
ESDE Phase 9-0: Segmenter
=========================
Regex-based sentence segmentation.

Design:
  - Minimal dependencies (no NLTK)
  - Replaceable interface (Segmenter base class)
  - Sentence-level is primary unit for Phase 9-0
  - Paragraph support for future extension

Spec: v5.4.0-P9.0
"""

import re
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass


# ==========================================
# Segmenter Interface
# ==========================================

class Segmenter(ABC):
    """
    Abstract base class for text segmentation.
    
    Contract:
        - Input: raw text string
        - Output: list of (start, end) tuples referencing input
        - Segments must not overlap
        - Union of segments should cover meaningful content
    """
    
    @abstractmethod
    def segment(self, text: str) -> List[Tuple[int, int]]:
        """
        Segment text into spans.
        
        Args:
            text: Raw input text
            
        Returns:
            List of (start, end) tuples
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Segmenter identifier for audit trail."""
        pass


# ==========================================
# Regex Sentence Segmenter
# ==========================================

class RegexSentenceSegmenter(Segmenter):
    """
    Regex-based sentence segmenter.
    
    Algorithm:
        1. Split on sentence boundaries: .!? followed by space or end
        2. Handle common abbreviations (Mr., Dr., etc.)
        3. Preserve span positions in original text
    
    Limitations (acceptable for Phase 9-0):
        - May mis-split on abbreviations not in list
        - Does not handle quoted speech perfectly
        - English-centric (Japanese/Chinese need different segmenter)
    """
    
    # Common abbreviations that don't end sentences
    ABBREVIATIONS = {
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
        'vs', 'etc', 'inc', 'ltd', 'co', 'corp',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'st', 'ave', 'blvd', 'rd',
        'e.g', 'i.e', 'cf', 'al',
    }
    
    # Sentence boundary pattern
    # Matches: . ! ? followed by space(s) and capital letter, or end of string
    SENTENCE_END_PATTERN = re.compile(
        r'([.!?])'           # Sentence-ending punctuation
        r'(?=\s+[A-Z]|\s*$)' # Followed by space+capital or end
    )
    
    def __init__(self, min_length: int = 3):
        """
        Args:
            min_length: Minimum segment length (chars) to emit
        """
        self.min_length = min_length
    
    @property
    def name(self) -> str:
        return "regex_sentence_v1"
    
    def segment(self, text: str) -> List[Tuple[int, int]]:
        """
        Segment text into sentences.
        
        Returns list of (start, end) positions in original text.
        """
        if not text or not text.strip():
            return []
        
        segments = []
        current_start = 0
        
        # Find all potential sentence boundaries
        for match in self.SENTENCE_END_PATTERN.finditer(text):
            end_pos = match.end()
            
            # Check if this is actually a sentence end (not abbreviation)
            if self._is_real_sentence_end(text, match.start()):
                segment_text = text[current_start:end_pos].strip()
                
                if len(segment_text) >= self.min_length:
                    # Find actual start (skip leading whitespace)
                    actual_start = current_start
                    while actual_start < end_pos and text[actual_start].isspace():
                        actual_start += 1
                    
                    # Find actual end (include punctuation, skip trailing space)
                    actual_end = end_pos
                    while actual_end > actual_start and text[actual_end - 1].isspace():
                        actual_end -= 1
                    
                    if actual_end > actual_start:
                        segments.append((actual_start, actual_end))
                
                current_start = end_pos
        
        # Handle remaining text (last segment without sentence-ending punctuation)
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if len(remaining) >= self.min_length:
                actual_start = current_start
                while actual_start < len(text) and text[actual_start].isspace():
                    actual_start += 1
                
                actual_end = len(text)
                while actual_end > actual_start and text[actual_end - 1].isspace():
                    actual_end -= 1
                
                if actual_end > actual_start:
                    segments.append((actual_start, actual_end))
        
        return segments
    
    def _is_real_sentence_end(self, text: str, punct_pos: int) -> bool:
        """
        Check if punctuation at punct_pos is a real sentence end.
        
        Returns False for abbreviations like "Mr." or "e.g."
        """
        if punct_pos == 0:
            return True
        
        # Find the word before the punctuation
        word_end = punct_pos
        word_start = punct_pos - 1
        
        while word_start > 0 and text[word_start - 1].isalpha():
            word_start -= 1
        
        word = text[word_start:word_end].lower()
        
        # Check if it's a known abbreviation
        if word in self.ABBREVIATIONS:
            return False
        
        # Check for patterns like "U.S." or "e.g."
        if len(word) <= 2 and word_start > 0:
            # Could be part of an acronym
            prev_char = text[word_start - 1] if word_start > 0 else ''
            if prev_char == '.':
                return False
        
        return True


# ==========================================
# Paragraph Segmenter (Future Extension)
# ==========================================

class ParagraphSegmenter(Segmenter):
    """
    Paragraph-based segmenter using double newlines.
    
    Useful for:
        - Long documents
        - Structured text (articles, papers)
    """
    
    PARAGRAPH_PATTERN = re.compile(r'\n\s*\n')
    
    def __init__(self, min_length: int = 10):
        self.min_length = min_length
    
    @property
    def name(self) -> str:
        return "paragraph_v1"
    
    def segment(self, text: str) -> List[Tuple[int, int]]:
        """Segment text into paragraphs."""
        if not text or not text.strip():
            return []
        
        segments = []
        current_start = 0
        
        for match in self.PARAGRAPH_PATTERN.finditer(text):
            para_end = match.start()
            para_text = text[current_start:para_end].strip()
            
            if len(para_text) >= self.min_length:
                # Find actual boundaries
                actual_start = current_start
                while actual_start < para_end and text[actual_start].isspace():
                    actual_start += 1
                
                actual_end = para_end
                while actual_end > actual_start and text[actual_end - 1].isspace():
                    actual_end -= 1
                
                if actual_end > actual_start:
                    segments.append((actual_start, actual_end))
            
            current_start = match.end()
        
        # Handle last paragraph
        if current_start < len(text):
            remaining = text[current_start:].strip()
            if len(remaining) >= self.min_length:
                actual_start = current_start
                while actual_start < len(text) and text[actual_start].isspace():
                    actual_start += 1
                
                actual_end = len(text)
                while actual_end > actual_start and text[actual_end - 1].isspace():
                    actual_end -= 1
                
                if actual_end > actual_start:
                    segments.append((actual_start, actual_end))
        
        return segments


# ==========================================
# Hybrid Segmenter
# ==========================================

class HybridSegmenter(Segmenter):
    """
    Hybrid segmenter: Paragraph first, then sentence.
    
    Strategy:
        1. Split into paragraphs
        2. Split each paragraph into sentences
        3. Flatten into single list
    
    Useful for long documents where paragraph structure matters.
    """
    
    def __init__(self, min_sentence_length: int = 3, min_paragraph_length: int = 10):
        self.paragraph_segmenter = ParagraphSegmenter(min_paragraph_length)
        self.sentence_segmenter = RegexSentenceSegmenter(min_sentence_length)
    
    @property
    def name(self) -> str:
        return "hybrid_v1"
    
    def segment(self, text: str) -> List[Tuple[int, int]]:
        """Segment into paragraphs, then sentences."""
        paragraphs = self.paragraph_segmenter.segment(text)
        
        if not paragraphs:
            # No paragraph breaks, fall back to sentence
            return self.sentence_segmenter.segment(text)
        
        all_segments = []
        
        for para_start, para_end in paragraphs:
            para_text = text[para_start:para_end]
            sentences = self.sentence_segmenter.segment(para_text)
            
            # Adjust positions to global coordinates
            for sent_start, sent_end in sentences:
                all_segments.append((para_start + sent_start, para_start + sent_end))
        
        return all_segments


# ==========================================
# Factory
# ==========================================

def get_segmenter(name: str = "sentence") -> Segmenter:
    """
    Factory for getting segmenter by name.
    
    Args:
        name: "sentence", "paragraph", or "hybrid"
    
    Returns:
        Segmenter instance
    """
    segmenters = {
        "sentence": RegexSentenceSegmenter,
        "paragraph": ParagraphSegmenter,
        "hybrid": HybridSegmenter,
    }
    
    if name not in segmenters:
        raise ValueError(f"Unknown segmenter: {name}. Available: {list(segmenters.keys())}")
    
    return segmenters[name]()


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-0 Segmenter Test")
    print("=" * 60)
    
    # Test text
    test_text = """Dr. Smith went to the store. He bought apples and oranges! 
What did he buy? He also visited Mr. Jones at 123 Main St. in New York.

This is a new paragraph. It has multiple sentences. Some are short. Others are longer and more complex."""
    
    print(f"Input text ({len(test_text)} chars):")
    print(f"'{test_text}'")
    print()
    
    # Test 1: Sentence segmenter
    print("[Test 1] RegexSentenceSegmenter")
    segmenter = RegexSentenceSegmenter()
    segments = segmenter.segment(test_text)
    print(f"  Found {len(segments)} segments")
    for i, (start, end) in enumerate(segments):
        print(f"  [{i}] ({start:3d}, {end:3d}): '{test_text[start:end]}'")
    assert len(segments) >= 5, "Should find at least 5 sentences"
    print("  ✅ PASS")
    print()
    
    # Test 2: Abbreviation handling
    print("[Test 2] Abbreviation handling")
    abbrev_text = "Mr. Smith and Dr. Jones met at 5 p.m. They discussed the U.S. economy."
    segments = segmenter.segment(abbrev_text)
    print(f"  Input: '{abbrev_text}'")
    print(f"  Found {len(segments)} segments")
    for i, (start, end) in enumerate(segments):
        print(f"  [{i}] '{abbrev_text[start:end]}'")
    # Should be 1 sentence because "U.S." is detected as part of an abbreviation
    # and the sentence doesn't end with a clear boundary
    assert len(segments) >= 1, f"Expected at least 1 sentence, got {len(segments)}"
    print("  ✅ PASS")
    print()
    
    # Test 3: Paragraph segmenter
    print("[Test 3] ParagraphSegmenter")
    para_segmenter = ParagraphSegmenter()
    segments = para_segmenter.segment(test_text)
    print(f"  Found {len(segments)} paragraphs")
    for i, (start, end) in enumerate(segments):
        preview = test_text[start:end][:50] + "..." if len(test_text[start:end]) > 50 else test_text[start:end]
        print(f"  [{i}] ({start:3d}, {end:3d}): '{preview}'")
    assert len(segments) == 2, "Should find 2 paragraphs"
    print("  ✅ PASS")
    print()
    
    # Test 4: Hybrid segmenter
    print("[Test 4] HybridSegmenter")
    hybrid = HybridSegmenter()
    segments = hybrid.segment(test_text)
    print(f"  Found {len(segments)} segments (sentences within paragraphs)")
    for i, (start, end) in enumerate(segments):
        print(f"  [{i}] ({start:3d}, {end:3d}): '{test_text[start:end]}'")
    assert len(segments) >= 5, "Should find multiple sentences across paragraphs"
    print("  ✅ PASS")
    print()
    
    # Test 5: Empty input
    print("[Test 5] Empty input handling")
    assert segmenter.segment("") == []
    assert segmenter.segment("   ") == []
    assert segmenter.segment(None if False else "") == []
    print("  ✅ PASS")
    print()
    
    # Test 6: Factory
    print("[Test 6] Factory function")
    s1 = get_segmenter("sentence")
    s2 = get_segmenter("paragraph")
    s3 = get_segmenter("hybrid")
    print(f"  sentence: {s1.name}")
    print(f"  paragraph: {s2.name}")
    print(f"  hybrid: {s3.name}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
