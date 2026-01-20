"""
ESDE Phase 9-1: W1 Tokenizer
============================
Tokenization for W1 statistics collection.

Design:
  - English: Word-based (similar to existing utils.py but preserves case)
  - Japanese/CJK: Character bigram (MeCab deferred to Phase 9-2+)
  - No semantic inference, just segmentation

Spec: v5.4.1-P9.1
"""

import re
import unicodedata
from typing import List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ==========================================
# Token Result
# ==========================================

@dataclass
class TokenResult:
    """Result of tokenization: surface form and position."""
    surface: str        # Original surface form (before normalization)
    start: int          # Start position in input text
    end: int            # End position in input text


# ==========================================
# Base Tokenizer Interface
# ==========================================

class W1Tokenizer(ABC):
    """Abstract base class for W1 tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[TokenResult]:
        """
        Tokenize text into tokens with positions.
        
        Args:
            text: Input text
            
        Returns:
            List of TokenResult (surface, start, end)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tokenizer identifier for audit trail."""
        pass


# ==========================================
# English Word Tokenizer
# ==========================================

class EnglishWordTokenizer(W1Tokenizer):
    """
    English word tokenizer.
    
    Algorithm:
      1. Find word boundaries using regex
      2. Extract words (alphanumeric + apostrophe for contractions)
      3. Preserve original case (normalization is separate step)
    
    Based on existing esde_engine/utils.py tokenize() but:
      - Preserves case (for surface_forms tracking)
      - Returns positions (for traceability)
    """
    
    # Word pattern: alphanumeric sequences, may contain apostrophe
    WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?")
    
    def __init__(self, min_length: int = 1):
        """
        Args:
            min_length: Minimum token length (default: 1, filtering is separate)
        """
        self.min_length = min_length
    
    @property
    def name(self) -> str:
        return "english_word_v1"
    
    def tokenize(self, text: str) -> List[TokenResult]:
        """Tokenize English text into words."""
        results = []
        
        for match in self.WORD_PATTERN.finditer(text):
            surface = match.group()
            if len(surface) >= self.min_length:
                results.append(TokenResult(
                    surface=surface,
                    start=match.start(),
                    end=match.end(),
                ))
        
        return results


# ==========================================
# CJK Character Bigram Tokenizer
# ==========================================

class CJKBigramTokenizer(W1Tokenizer):
    """
    CJK (Chinese/Japanese/Korean) character bigram tokenizer.
    
    Algorithm:
      1. Extract CJK character sequences
      2. Generate overlapping bigrams
      3. Single chars at boundaries become unigrams
    
    Note: MeCab integration deferred to Phase 9-2+
    """
    
    # CJK Unicode ranges
    CJK_PATTERN = re.compile(
        r'[\u4e00-\u9fff'      # CJK Unified Ideographs
        r'\u3040-\u309f'       # Hiragana
        r'\u30a0-\u30ff'       # Katakana
        r'\uac00-\ud7af'       # Hangul Syllables
        r'\u3400-\u4dbf'       # CJK Extension A
        r']+'
    )
    
    @property
    def name(self) -> str:
        return "cjk_bigram_v1"
    
    def tokenize(self, text: str) -> List[TokenResult]:
        """Tokenize CJK text into character bigrams."""
        results = []
        
        for match in self.CJK_PATTERN.finditer(text):
            cjk_seq = match.group()
            seq_start = match.start()
            
            if len(cjk_seq) == 1:
                # Single character - emit as unigram
                results.append(TokenResult(
                    surface=cjk_seq,
                    start=seq_start,
                    end=seq_start + 1,
                ))
            else:
                # Generate bigrams
                for i in range(len(cjk_seq) - 1):
                    bigram = cjk_seq[i:i+2]
                    results.append(TokenResult(
                        surface=bigram,
                        start=seq_start + i,
                        end=seq_start + i + 2,
                    ))
        
        return results


# ==========================================
# Hybrid Tokenizer (English + CJK)
# ==========================================

class HybridTokenizer(W1Tokenizer):
    """
    Hybrid tokenizer that handles both English and CJK text.
    
    Algorithm:
      1. Scan text for character types
      2. Apply EnglishWordTokenizer to ASCII/Latin portions
      3. Apply CJKBigramTokenizer to CJK portions
      4. Merge results sorted by position
    """
    
    def __init__(self, min_english_length: int = 1):
        self.english_tokenizer = EnglishWordTokenizer(min_length=min_english_length)
        self.cjk_tokenizer = CJKBigramTokenizer()
    
    @property
    def name(self) -> str:
        return "hybrid_v1"
    
    def tokenize(self, text: str) -> List[TokenResult]:
        """Tokenize mixed English/CJK text."""
        results = []
        
        # Get tokens from both tokenizers
        english_tokens = self.english_tokenizer.tokenize(text)
        cjk_tokens = self.cjk_tokenizer.tokenize(text)
        
        # Merge and sort by position
        results = english_tokens + cjk_tokens
        results.sort(key=lambda t: (t.start, t.end))
        
        return results


# ==========================================
# Factory
# ==========================================

def get_w1_tokenizer(name: str = "hybrid") -> W1Tokenizer:
    """
    Factory for getting W1 tokenizer by name.
    
    Args:
        name: "english", "cjk", or "hybrid" (default)
    
    Returns:
        W1Tokenizer instance
    """
    tokenizers = {
        "english": EnglishWordTokenizer,
        "cjk": CJKBigramTokenizer,
        "hybrid": HybridTokenizer,
    }
    
    if name not in tokenizers:
        raise ValueError(f"Unknown tokenizer: {name}. Available: {list(tokenizers.keys())}")
    
    return tokenizers[name]()


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-1 Tokenizer Test")
    print("=" * 60)
    
    # Test 1: English tokenizer
    print("\n[Test 1] EnglishWordTokenizer")
    eng_tok = EnglishWordTokenizer()
    
    text = "Hello World! It's a beautiful day."
    tokens = eng_tok.tokenize(text)
    
    print(f"  Input: '{text}'")
    print(f"  Tokens: {[t.surface for t in tokens]}")
    
    assert len(tokens) == 6, f"Expected 6 tokens, got {len(tokens)}"
    assert tokens[0].surface == "Hello"
    assert tokens[2].surface == "It's"  # Contraction preserved
    print("  ✅ PASS")
    
    # Test 2: CJK bigram tokenizer
    print("\n[Test 2] CJKBigramTokenizer")
    cjk_tok = CJKBigramTokenizer()
    
    text = "日本語テスト"
    tokens = cjk_tok.tokenize(text)
    
    print(f"  Input: '{text}'")
    print(f"  Tokens: {[t.surface for t in tokens]}")
    
    # "日本語テスト" (6 chars) -> 5 bigrams
    assert len(tokens) == 5, f"Expected 5 bigrams, got {len(tokens)}"
    assert tokens[0].surface == "日本"
    assert tokens[1].surface == "本語"
    print("  ✅ PASS")
    
    # Test 3: Single CJK character
    print("\n[Test 3] Single CJK character (unigram)")
    tokens = cjk_tok.tokenize("愛")
    
    print(f"  Input: '愛'")
    print(f"  Tokens: {[t.surface for t in tokens]}")
    
    assert len(tokens) == 1
    assert tokens[0].surface == "愛"
    print("  ✅ PASS")
    
    # Test 4: Hybrid tokenizer
    print("\n[Test 4] HybridTokenizer (mixed English/Japanese)")
    hybrid = HybridTokenizer()
    
    text = "I love 日本語 and Python!"
    tokens = hybrid.tokenize(text)
    
    print(f"  Input: '{text}'")
    print(f"  Tokens: {[t.surface for t in tokens]}")
    
    surfaces = [t.surface for t in tokens]
    assert "I" in surfaces
    assert "love" in surfaces
    assert "日本" in surfaces  # Japanese bigram
    assert "Python" in surfaces
    print("  ✅ PASS")
    
    # Test 5: Position tracking
    print("\n[Test 5] Position tracking")
    text = "Hello 世界"
    tokens = hybrid.tokenize(text)
    
    for t in tokens:
        extracted = text[t.start:t.end]
        print(f"  '{t.surface}' at [{t.start}:{t.end}] -> '{extracted}'")
        assert extracted == t.surface, f"Position mismatch: {extracted} != {t.surface}"
    print("  ✅ PASS")
    
    # Test 6: Factory
    print("\n[Test 6] Factory function")
    t1 = get_w1_tokenizer("english")
    t2 = get_w1_tokenizer("cjk")
    t3 = get_w1_tokenizer("hybrid")
    
    print(f"  english: {t1.name}")
    print(f"  cjk: {t2.name}")
    print(f"  hybrid: {t3.name}")
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
