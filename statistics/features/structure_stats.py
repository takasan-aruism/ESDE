#!/usr/bin/env python3
"""
ESDE Phase 9: Structure Statistics
===================================

Computes article/section structure statistics from TokenFeature data.

These metrics capture "author fingerprint" - unconscious writing patterns:
  - Sentence length (avg, std, min, max)
  - Paragraph structure
  - Section composition

Spec: Phase 9 Structure Statistics v1.0
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict


@dataclass
class StructureStats:
    """
    Structure statistics for an article or section.
    
    Captures the "shape" of writing independent of content.
    """
    
    # Identification
    article_id: str = ""
    section_name: str = ""
    
    # Token level
    total_tokens: int = 0
    
    # Sentence level
    total_sentences: int = 0
    sentence_lengths: List[int] = field(default_factory=list)
    
    @property
    def avg_sentence_length(self) -> float:
        if not self.sentence_lengths:
            return 0.0
        return sum(self.sentence_lengths) / len(self.sentence_lengths)
    
    @property
    def std_sentence_length(self) -> float:
        if len(self.sentence_lengths) < 2:
            return 0.0
        avg = self.avg_sentence_length
        variance = sum((x - avg) ** 2 for x in self.sentence_lengths) / len(self.sentence_lengths)
        return math.sqrt(variance)
    
    @property
    def min_sentence_length(self) -> int:
        return min(self.sentence_lengths) if self.sentence_lengths else 0
    
    @property
    def max_sentence_length(self) -> int:
        return max(self.sentence_lengths) if self.sentence_lengths else 0
    
    # Paragraph level
    total_paragraphs: int = 0
    sentences_per_paragraph: List[int] = field(default_factory=list)
    
    @property
    def avg_sentences_per_paragraph(self) -> float:
        if not self.sentences_per_paragraph:
            return 0.0
        return sum(self.sentences_per_paragraph) / len(self.sentences_per_paragraph)
    
    # Section level
    total_sections: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "article_id": self.article_id,
            "section_name": self.section_name,
            "total_tokens": self.total_tokens,
            "total_sentences": self.total_sentences,
            "total_paragraphs": self.total_paragraphs,
            "total_sections": self.total_sections,
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "std_sentence_length": round(self.std_sentence_length, 2),
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "avg_sentences_per_paragraph": round(self.avg_sentences_per_paragraph, 2),
            "sentence_lengths": self.sentence_lengths,
            "sentences_per_paragraph": self.sentences_per_paragraph,
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        name = self.section_name or self.article_id or "Unknown"
        lines.append(f"[{name}]")
        lines.append(f"  Tokens: {self.total_tokens}")
        lines.append(f"  Sentences: {self.total_sentences}")
        lines.append(f"  Paragraphs: {self.total_paragraphs}")
        lines.append(f"  Avg sentence length: {self.avg_sentence_length:.1f} (std: {self.std_sentence_length:.1f})")
        lines.append(f"  Sentence range: {self.min_sentence_length} - {self.max_sentence_length}")
        lines.append(f"  Avg sentences/paragraph: {self.avg_sentences_per_paragraph:.1f}")
        return "\n".join(lines)


def compute_structure_stats(
    features: List[Any],  # List[TokenFeature]
    article_id: str = "",
    section_name: str = "",
) -> StructureStats:
    """
    Compute structure statistics from TokenFeature list.
    
    Args:
        features: List of TokenFeature (must have structure metadata)
        article_id: Optional article identifier
        section_name: Optional section name
        
    Returns:
        StructureStats with computed metrics
        
    Invariants (must hold):
        - total_sentences == len(sentence_lengths)
        - total_paragraphs == len(sentences_per_paragraph)
        - sum(sentences_per_paragraph) == total_sentences
    """
    if not features:
        return StructureStats(article_id=article_id, section_name=section_name)
    
    stats = StructureStats(
        article_id=article_id,
        section_name=section_name,
        total_tokens=len(features),
    )
    
    # Collect unique sentence lengths
    # Key: (section_idx, paragraph_idx, sentence_idx) to handle per-section resets
    seen_sentences = {}
    for f in features:
        sec_idx = getattr(f, 'section_idx', 0)
        para_idx = getattr(f, 'paragraph_idx', 0)
        sent_idx = getattr(f, 'sentence_idx', 0)
        sent_len = getattr(f, 'sentence_length', 0)
        
        key = (sec_idx, para_idx, sent_idx)
        if key not in seen_sentences and sent_len > 0:
            seen_sentences[key] = sent_len
    
    stats.sentence_lengths = list(seen_sentences.values())
    stats.total_sentences = len(stats.sentence_lengths)  # Derived from sentence_lengths
    
    # Collect sentences per paragraph
    # Key: (section_idx, paragraph_idx) to handle per-section resets
    para_sentences = defaultdict(set)
    for f in features:
        sec_idx = getattr(f, 'section_idx', 0)
        para_idx = getattr(f, 'paragraph_idx', 0)
        sent_idx = getattr(f, 'sentence_idx', 0)
        
        para_key = (sec_idx, para_idx)
        sent_key = (sec_idx, para_idx, sent_idx)
        para_sentences[para_key].add(sent_key)
    
    stats.sentences_per_paragraph = [len(sents) for _, sents in sorted(para_sentences.items())]
    stats.total_paragraphs = len(stats.sentences_per_paragraph)  # Derived from sentences_per_paragraph
    
    # Section count from unique section indices
    section_indices = set(getattr(f, 'section_idx', 0) for f in features)
    stats.total_sections = len(section_indices)
    
    # Invariant checks (debug)
    assert stats.total_sentences == len(stats.sentence_lengths), \
        f"Invariant violation: total_sentences({stats.total_sentences}) != len(sentence_lengths)({len(stats.sentence_lengths)})"
    assert stats.total_paragraphs == len(stats.sentences_per_paragraph), \
        f"Invariant violation: total_paragraphs({stats.total_paragraphs}) != len(sentences_per_paragraph)({len(stats.sentences_per_paragraph)})"
    assert sum(stats.sentences_per_paragraph) == stats.total_sentences, \
        f"Invariant violation: sum(sentences_per_paragraph)({sum(stats.sentences_per_paragraph)}) != total_sentences({stats.total_sentences})"
    
    return stats


def compute_article_stats(
    article_features: Dict[str, List[Any]],  # article_id -> List[TokenFeature]
) -> Dict[str, StructureStats]:
    """
    Compute structure statistics for multiple articles.
    
    Args:
        article_features: Dict mapping article_id to TokenFeature list
        
    Returns:
        Dict mapping article_id to StructureStats
    """
    results = {}
    for article_id, features in article_features.items():
        results[article_id] = compute_structure_stats(
            features, 
            article_id=article_id
        )
    return results


def compare_articles(
    stats_dict: Dict[str, StructureStats],
) -> str:
    """
    Generate comparison table of multiple articles.
    
    Args:
        stats_dict: Dict of article_id -> StructureStats
        
    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Article Structure Comparison")
    lines.append("=" * 80)
    lines.append("")
    
    # Header
    header = f"{'Article':<25} {'Tokens':>8} {'Sents':>6} {'Paras':>6} {'AvgLen':>8} {'StdLen':>8} {'Range':>12}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Data rows
    for article_id, stats in sorted(stats_dict.items()):
        range_str = f"{stats.min_sentence_length}-{stats.max_sentence_length}"
        row = f"{article_id:<25} {stats.total_tokens:>8} {stats.total_sentences:>6} {stats.total_paragraphs:>6} {stats.avg_sentence_length:>8.1f} {stats.std_sentence_length:>8.1f} {range_str:>12}"
        lines.append(row)
    
    lines.append("-" * 80)
    
    # Summary statistics
    all_avgs = [s.avg_sentence_length for s in stats_dict.values()]
    all_stds = [s.std_sentence_length for s in stats_dict.values()]
    
    if all_avgs:
        overall_avg = sum(all_avgs) / len(all_avgs)
        overall_std = sum(all_stds) / len(all_stds)
        lines.append(f"{'Overall average:':<25} {'':>8} {'':>6} {'':>6} {overall_avg:>8.1f} {overall_std:>8.1f}")
    
    return "\n".join(lines)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Structure Statistics Test")
    print("=" * 60)
    
    # Create mock TokenFeature-like objects
    @dataclass
    class MockFeature:
        token: str
        sentence_idx: int
        sentence_length: int
        paragraph_idx: int
        paragraph_count: int
        total_sentences: int
        total_sections: int
    
    # Simulate article with varying sentence lengths
    features = []
    
    # Paragraph 1: 2 sentences (lengths 5, 8)
    for i in range(5):
        features.append(MockFeature(f"word{i}", 0, 5, 0, 2, 4, 1))
    for i in range(8):
        features.append(MockFeature(f"word{i}", 1, 8, 0, 2, 4, 1))
    
    # Paragraph 2: 2 sentences (lengths 12, 3)
    for i in range(12):
        features.append(MockFeature(f"word{i}", 2, 12, 1, 2, 4, 1))
    for i in range(3):
        features.append(MockFeature(f"word{i}", 3, 3, 1, 2, 4, 1))
    
    stats = compute_structure_stats(features, article_id="test_article")
    
    print(stats.summary())
    print()
    print("Raw data:")
    print(f"  Sentence lengths: {stats.sentence_lengths}")
    print(f"  Sentences/para: {stats.sentences_per_paragraph}")
    print()
    print("✅ Structure statistics working!")
