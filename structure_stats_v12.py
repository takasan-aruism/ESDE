#!/usr/bin/env python3
"""
ESDE Phase 9: Structure Statistics
===================================

Computes article/section structure statistics from TokenFeature data.

These metrics capture "author fingerprint" - unconscious writing patterns:
  - Sentence length (avg, std, min, max, median, CV, IQR, skewness, kurtosis)
  - Paragraph structure
  - Section composition
  - Outlier sentences (longest/shortest)

Spec: Phase 9 Structure Statistics v1.2
  - v1.0: Basic metrics
  - v1.1: Added CV, median, IQR, skewness, kurtosis
  - v1.2: Added outlier sentence extraction
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict


# ==========================================
# Configuration
# ==========================================

# Outlier extraction settings
TOP_LONGEST = 5
BOTTOM_SHORTEST = 3


# ==========================================
# Data Classes
# ==========================================

@dataclass
class OutlierSentence:
    """A sentence identified as an outlier (longest or shortest)."""
    rank: int                    # 1 = longest/shortest
    length: int                  # Token count
    section_name: str            # Which section
    section_idx: int             # Section index
    sentence_idx: int            # Sentence index within text
    text: str                    # Reconstructed sentence text
    category: str                # "longest" or "shortest"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "length": self.length,
            "section_name": self.section_name,
            "section_idx": self.section_idx,
            "sentence_idx": self.sentence_idx,
            "text": self.text,
            "category": self.category,
        }


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
    
    # Outlier sentences (v1.2)
    outlier_sentences: List[OutlierSentence] = field(default_factory=list)
    
    # ─────────────────────────────────────
    # Basic Properties (v1.0)
    # ─────────────────────────────────────
    
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
    
    # ─────────────────────────────────────
    # Extended Properties (v1.1)
    # ─────────────────────────────────────
    
    @property
    def median_sentence_length(self) -> float:
        """Median sentence length (robust to outliers)."""
        if not self.sentence_lengths:
            return 0.0
        sorted_lens = sorted(self.sentence_lengths)
        n = len(sorted_lens)
        if n % 2 == 1:
            return float(sorted_lens[n // 2])
        else:
            return (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2.0
    
    @property
    def cv_sentence_length(self) -> float:
        """Coefficient of Variation (std/mean). Scale-independent variability."""
        avg = self.avg_sentence_length
        if avg == 0:
            return 0.0
        return self.std_sentence_length / avg
    
    @property
    def q25_sentence_length(self) -> float:
        """25th percentile (Q1)."""
        return self._percentile(25)
    
    @property
    def q75_sentence_length(self) -> float:
        """75th percentile (Q3)."""
        return self._percentile(75)
    
    @property
    def iqr_sentence_length(self) -> float:
        """Interquartile Range (Q3 - Q1). Robust spread measure."""
        return self.q75_sentence_length - self.q25_sentence_length
    
    @property
    def skewness_sentence_length(self) -> float:
        """
        Skewness (Fisher's definition).
        Positive = right tail (short sentences + occasional long ones)
        Negative = left tail (long sentences + occasional short ones)
        """
        if len(self.sentence_lengths) < 3:
            return 0.0
        avg = self.avg_sentence_length
        std = self.std_sentence_length
        if std == 0:
            return 0.0
        n = len(self.sentence_lengths)
        m3 = sum((x - avg) ** 3 for x in self.sentence_lengths) / n
        return m3 / (std ** 3)
    
    @property
    def kurtosis_sentence_length(self) -> float:
        """
        Excess Kurtosis (Fisher's definition, normal = 0).
        Positive = peaked distribution (concentrated rhythm)
        Negative = flat distribution (diverse rhythm)
        """
        if len(self.sentence_lengths) < 4:
            return 0.0
        avg = self.avg_sentence_length
        std = self.std_sentence_length
        if std == 0:
            return 0.0
        n = len(self.sentence_lengths)
        m4 = sum((x - avg) ** 4 for x in self.sentence_lengths) / n
        return (m4 / (std ** 4)) - 3.0
    
    def _percentile(self, p: float) -> float:
        """Calculate percentile using linear interpolation."""
        if not self.sentence_lengths:
            return 0.0
        sorted_lens = sorted(self.sentence_lengths)
        n = len(sorted_lens)
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(sorted_lens[int(k)])
        return sorted_lens[int(f)] * (c - k) + sorted_lens[int(c)] * (k - f)
    
    # ─────────────────────────────────────
    # Paragraph level
    # ─────────────────────────────────────
    
    total_paragraphs: int = 0
    sentences_per_paragraph: List[int] = field(default_factory=list)
    
    @property
    def avg_sentences_per_paragraph(self) -> float:
        if not self.sentences_per_paragraph:
            return 0.0
        return sum(self.sentences_per_paragraph) / len(self.sentences_per_paragraph)
    
    # Section level
    total_sections: int = 0
    
    # ─────────────────────────────────────
    # Export
    # ─────────────────────────────────────
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        result = {
            "article_id": self.article_id,
            "section_name": self.section_name,
            "total_tokens": self.total_tokens,
            "total_sentences": self.total_sentences,
            "total_paragraphs": self.total_paragraphs,
            "total_sections": self.total_sections,
            # v1.0 metrics
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "std_sentence_length": round(self.std_sentence_length, 2),
            "min_sentence_length": self.min_sentence_length,
            "max_sentence_length": self.max_sentence_length,
            "avg_sentences_per_paragraph": round(self.avg_sentences_per_paragraph, 2),
            "sentence_lengths": self.sentence_lengths,
            "sentences_per_paragraph": self.sentences_per_paragraph,
            # v1.1 metrics
            "median_sentence_length": round(self.median_sentence_length, 1),
            "cv_sentence_length": round(self.cv_sentence_length, 3),
            "q25_sentence_length": round(self.q25_sentence_length, 1),
            "q75_sentence_length": round(self.q75_sentence_length, 1),
            "iqr_sentence_length": round(self.iqr_sentence_length, 1),
            "skewness_sentence_length": round(self.skewness_sentence_length, 2),
            "kurtosis_sentence_length": round(self.kurtosis_sentence_length, 2),
            # v1.2 outliers
            "outlier_sentences": [o.to_dict() for o in self.outlier_sentences],
        }
        return result
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        name = self.section_name or self.article_id or "Unknown"
        lines.append(f"[{name}]")
        lines.append(f"  Tokens: {self.total_tokens}")
        lines.append(f"  Sentences: {self.total_sentences}")
        lines.append(f"  Paragraphs: {self.total_paragraphs}")
        lines.append(f"  Avg sentence length: {self.avg_sentence_length:.1f} (std: {self.std_sentence_length:.1f})")
        lines.append(f"  Median: {self.median_sentence_length:.1f}, CV: {self.cv_sentence_length:.3f}")
        lines.append(f"  IQR: {self.iqr_sentence_length:.1f} (Q1={self.q25_sentence_length:.1f}, Q3={self.q75_sentence_length:.1f})")
        lines.append(f"  Skewness: {self.skewness_sentence_length:+.2f}, Kurtosis: {self.kurtosis_sentence_length:+.2f}")
        lines.append(f"  Sentence range: {self.min_sentence_length} - {self.max_sentence_length}")
        lines.append(f"  Avg sentences/paragraph: {self.avg_sentences_per_paragraph:.1f}")
        
        # Outliers summary
        longest = [o for o in self.outlier_sentences if o.category == "longest"]
        shortest = [o for o in self.outlier_sentences if o.category == "shortest"]
        if longest:
            lines.append(f"  Top {len(longest)} longest: {[o.length for o in longest]}")
        if shortest:
            lines.append(f"  Top {len(shortest)} shortest: {[o.length for o in shortest]}")
        
        return "\n".join(lines)


# ==========================================
# Core Function
# ==========================================

def compute_structure_stats(
    features: List[Any],  # List[TokenFeature]
    article_id: str = "",
    section_name: str = "",
    top_longest: int = TOP_LONGEST,
    bottom_shortest: int = BOTTOM_SHORTEST,
) -> StructureStats:
    """
    Compute structure statistics from TokenFeature list.
    
    Args:
        features: List of TokenFeature (must have structure metadata)
        article_id: Optional article identifier
        section_name: Optional section name
        top_longest: Number of longest sentences to extract
        bottom_shortest: Number of shortest sentences to extract
        
    Returns:
        StructureStats with computed metrics and outlier sentences
        
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
    
    # ─────────────────────────────────────
    # Collect sentence info (length + tokens)
    # ─────────────────────────────────────
    
    # Key: (section_idx, paragraph_idx, sentence_idx)
    # Value: {"length": int, "tokens": List[str], "section_name": str, "section_idx": int, "sentence_idx": int}
    sentence_data: Dict[tuple, Dict[str, Any]] = {}
    
    for f in features:
        sec_idx = getattr(f, 'section_idx', 0)
        para_idx = getattr(f, 'paragraph_idx', 0)
        sent_idx = getattr(f, 'sentence_idx', 0)
        sent_len = getattr(f, 'sentence_length', 0)
        sec_name = getattr(f, 'section_name', '')
        token = getattr(f, 'token', '')
        
        key = (sec_idx, para_idx, sent_idx)
        
        if key not in sentence_data and sent_len > 0:
            sentence_data[key] = {
                "length": sent_len,
                "tokens": [],
                "section_name": sec_name,
                "section_idx": sec_idx,
                "sentence_idx": sent_idx,
            }
        
        if key in sentence_data:
            sentence_data[key]["tokens"].append(token)
    
    # Extract sentence lengths
    stats.sentence_lengths = [d["length"] for d in sentence_data.values()]
    stats.total_sentences = len(stats.sentence_lengths)
    
    # ─────────────────────────────────────
    # Extract outlier sentences
    # ─────────────────────────────────────
    
    if sentence_data:
        # Sort by length descending for longest
        sorted_by_length = sorted(
            sentence_data.items(),
            key=lambda x: x[1]["length"],
            reverse=True
        )
        
        # Top N longest
        for rank, (key, data) in enumerate(sorted_by_length[:top_longest], start=1):
            text = " ".join(data["tokens"])
            stats.outlier_sentences.append(OutlierSentence(
                rank=rank,
                length=data["length"],
                section_name=data["section_name"],
                section_idx=data["section_idx"],
                sentence_idx=data["sentence_idx"],
                text=text,
                category="longest",
            ))
        
        # Bottom N shortest (from the other end, but re-rank)
        # Filter out sentences already in longest (edge case: very few sentences)
        longest_keys = set(k for k, _ in sorted_by_length[:top_longest])
        shortest_candidates = [(k, d) for k, d in sorted_by_length if k not in longest_keys]
        shortest_candidates.reverse()  # Now ascending order
        
        for rank, (key, data) in enumerate(shortest_candidates[:bottom_shortest], start=1):
            text = " ".join(data["tokens"])
            stats.outlier_sentences.append(OutlierSentence(
                rank=rank,
                length=data["length"],
                section_name=data["section_name"],
                section_idx=data["section_idx"],
                sentence_idx=data["sentence_idx"],
                text=text,
                category="shortest",
            ))
    
    # ─────────────────────────────────────
    # Collect sentences per paragraph
    # ─────────────────────────────────────
    
    para_sentences = defaultdict(set)
    for f in features:
        sec_idx = getattr(f, 'section_idx', 0)
        para_idx = getattr(f, 'paragraph_idx', 0)
        sent_idx = getattr(f, 'sentence_idx', 0)
        
        para_key = (sec_idx, para_idx)
        sent_key = (sec_idx, para_idx, sent_idx)
        para_sentences[para_key].add(sent_key)
    
    stats.sentences_per_paragraph = [len(sents) for _, sents in sorted(para_sentences.items())]
    stats.total_paragraphs = len(stats.sentences_per_paragraph)
    
    # Section count from unique section indices
    section_indices = set(getattr(f, 'section_idx', 0) for f in features)
    stats.total_sections = len(section_indices)
    
    # ─────────────────────────────────────
    # Invariant checks
    # ─────────────────────────────────────
    
    assert stats.total_sentences == len(stats.sentence_lengths), \
        f"Invariant violation: total_sentences({stats.total_sentences}) != len(sentence_lengths)({len(stats.sentence_lengths)})"
    assert stats.total_paragraphs == len(stats.sentences_per_paragraph), \
        f"Invariant violation: total_paragraphs({stats.total_paragraphs}) != len(sentences_per_paragraph)({len(stats.sentences_per_paragraph)})"
    assert sum(stats.sentences_per_paragraph) == stats.total_sentences, \
        f"Invariant violation: sum(sentences_per_paragraph)({sum(stats.sentences_per_paragraph)}) != total_sentences({stats.total_sentences})"
    
    return stats


# ==========================================
# Multi-Article Functions
# ==========================================

def compute_article_stats(
    article_features: Dict[str, List[Any]],  # article_id -> List[TokenFeature]
    top_longest: int = TOP_LONGEST,
    bottom_shortest: int = BOTTOM_SHORTEST,
) -> Dict[str, StructureStats]:
    """
    Compute structure statistics for multiple articles.
    
    Args:
        article_features: Dict mapping article_id to TokenFeature list
        top_longest: Number of longest sentences to extract per article
        bottom_shortest: Number of shortest sentences to extract per article
        
    Returns:
        Dict mapping article_id to StructureStats
    """
    results = {}
    for article_id, features in article_features.items():
        results[article_id] = compute_structure_stats(
            features, 
            article_id=article_id,
            top_longest=top_longest,
            bottom_shortest=bottom_shortest,
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
        Formatted comparison string (v1.1 layout with CV)
    """
    lines = []
    lines.append("=" * 100)
    lines.append("Article Structure Comparison (v1.2)")
    lines.append("=" * 100)
    lines.append("")
    
    # Header (v1.1 layout)
    header = f"{'Article':<25} {'Tokens':>8} {'Sents':>6} {'Paras':>6} {'Avg':>7} {'Median':>7} {'Std':>7} {'CV':>6}"
    lines.append(header)
    lines.append("-" * 100)
    
    # Data rows
    for article_id, stats in sorted(stats_dict.items()):
        row = (
            f"{article_id:<25} "
            f"{stats.total_tokens:>8} "
            f"{stats.total_sentences:>6} "
            f"{stats.total_paragraphs:>6} "
            f"{stats.avg_sentence_length:>7.1f} "
            f"{stats.median_sentence_length:>7.1f} "
            f"{stats.std_sentence_length:>7.1f} "
            f"{stats.cv_sentence_length:>6.3f}"
        )
        lines.append(row)
    
    lines.append("-" * 100)
    
    # Summary statistics
    all_avgs = [s.avg_sentence_length for s in stats_dict.values()]
    all_cvs = [s.cv_sentence_length for s in stats_dict.values()]
    
    if all_avgs:
        overall_avg = sum(all_avgs) / len(all_avgs)
        overall_cv = sum(all_cvs) / len(all_cvs)
        lines.append(f"{'Mean across articles:':<25} {'':>8} {'':>6} {'':>6} {overall_avg:>7.1f} {'':>7} {'':>7} {overall_cv:>6.3f}")
    
    return "\n".join(lines)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Structure Statistics v1.2 Test (with outlier extraction)")
    print("=" * 70)
    
    from dataclasses import dataclass as dc
    
    @dc
    class MockFeature:
        token: str
        sentence_idx: int
        sentence_length: int
        paragraph_idx: int
        paragraph_count: int
        total_sentences: int
        total_sections: int
        section_idx: int = 0
        section_name: str = ""
    
    # Simulate article with varying sentence lengths
    features = []
    
    # Section 0: "Early Life"
    # Sentence 0: 5 tokens (short)
    for i, word in enumerate(["He", "was", "born", "in", "1534."]):
        features.append(MockFeature(word, 0, 5, 0, 2, 6, 1, 0, "Early Life"))
    
    # Sentence 1: 15 tokens (medium)
    words = "Nobunaga was the second son of Oda Nobuhide a minor lord in Owari".split()
    for i, word in enumerate(words):
        features.append(MockFeature(word, 1, len(words), 0, 2, 6, 1, 0, "Early Life"))
    
    # Section 1: "Military Campaigns"
    # Sentence 2: 3 tokens (very short)
    for i, word in enumerate(["War", "broke", "out."]):
        features.append(MockFeature(word, 2, 3, 1, 2, 6, 1, 1, "Military Campaigns"))
    
    # Sentence 3: 25 tokens (long)
    words = "The battle of Okehazama was a decisive engagement in which Nobunaga defeated the much larger forces of Imagawa Yoshimoto establishing his reputation as a brilliant tactician".split()
    for i, word in enumerate(words):
        features.append(MockFeature(word, 3, len(words), 1, 2, 6, 1, 1, "Military Campaigns"))
    
    # Sentence 4: 8 tokens (medium)
    words = "He then expanded his territory through diplomacy".split()
    for i, word in enumerate(words):
        features.append(MockFeature(word, 4, len(words), 1, 2, 6, 1, 1, "Military Campaigns"))
    
    # Sentence 5: 2 tokens (shortest)
    for i, word in enumerate(["Victory", "followed."]):
        features.append(MockFeature(word, 5, 2, 1, 2, 6, 1, 1, "Military Campaigns"))
    
    stats = compute_structure_stats(
        features, 
        article_id="test_nobunaga",
        top_longest=3,
        bottom_shortest=2,
    )
    
    print(stats.summary())
    print()
    
    print("Outlier Sentences:")
    print("-" * 70)
    for o in stats.outlier_sentences:
        preview = o.text[:60] + "..." if len(o.text) > 60 else o.text
        print(f"  [{o.category}#{o.rank}] {o.length} tokens in '{o.section_name}'")
        print(f"    \"{preview}\"")
    print()
    
    print("JSON export (outlier_sentences field):")
    import json
    export = stats.to_dict()
    print(json.dumps(export["outlier_sentences"], indent=2))
    
    print()
    print("✅ Structure statistics v1.2 working!")
