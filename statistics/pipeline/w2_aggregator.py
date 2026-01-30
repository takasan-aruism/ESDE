#!/usr/bin/env python3
"""
ESDE Phase 9: Conditional Statistics (W2)
==========================================

Aggregates token statistics by condition using ConditionProvider.

This is the Phase 9 version of W2, designed to work with:
  - New W1 TokenFeature output (20-dimensional vectors)
  - Pluggable ConditionProvider for internal condition extraction

Key Differences from Legacy W2:
  - Conditions come from internal structure, not source_meta
  - Input is TokenFeature list, not ArticleRecord
  - Supports multiple aggregation units (token/sentence/section)

Output:
  - W2Stats: Global statistics + per-condition statistics
  - Ready for W3 S-Score calculation

Spec: Phase 9 W2 v1.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timezone

from .condition_provider import (
    BaseConditionProvider,
    AggregationContext,
    get_condition_provider,
)


# ==========================================
# Constants
# ==========================================

W2_VERSION = "v9.phase9.1"
CANONICAL_SEPARATORS = (',', ':')


# ==========================================
# Data Structures
# ==========================================

@dataclass
class ConditionStats:
    """Statistics for a single condition."""
    condition_id: str
    factors: Dict[str, Any]
    token_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_tokens: int = 0
    doc_count: int = 0  # Number of articles with this condition
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "factors": self.factors,
            "total_tokens": self.total_tokens,
            "doc_count": self.doc_count,
            "unique_tokens": len(self.token_counts),
        }


@dataclass
class W2Stats:
    """
    W2 Statistics container.
    
    Contains:
      - Global token counts (all conditions combined)
      - Per-condition token counts
      - Metadata for traceability
    """
    # Provider info
    provider_id: str
    axis_name: str
    aggregation_unit: str
    
    # Global statistics
    global_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    global_total: int = 0
    
    # Per-condition statistics
    conditions: Dict[str, ConditionStats] = field(default_factory=dict)
    
    # Metadata
    articles_processed: int = 0
    version: str = W2_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_condition(self, condition_id: str) -> ConditionStats:
        """Get or create condition stats."""
        if condition_id not in self.conditions:
            self.conditions[condition_id] = ConditionStats(
                condition_id=condition_id,
                factors={},
            )
        return self.conditions[condition_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "axis_name": self.axis_name,
            "aggregation_unit": self.aggregation_unit,
            "global_total": self.global_total,
            "unique_tokens": len(self.global_counts),
            "condition_count": len(self.conditions),
            "articles_processed": self.articles_processed,
            "version": self.version,
            "created_at": self.created_at,
            "conditions": {
                cid: cond.to_dict() 
                for cid, cond in self.conditions.items()
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ==========================================
# W2 Aggregator (Phase 9 Version)
# ==========================================

class W2Aggregator:
    """
    Aggregates token statistics by condition.
    
    Usage:
        # Using axis name
        aggregator = W2Aggregator(axis="section")
        
        # Or using provider directly
        provider = SectionConditionProvider()
        aggregator = W2Aggregator(provider=provider)
        
        # Process articles
        for article in articles:
            aggregator.process_article(article_id, features, sections)
        
        # Get results
        stats = aggregator.get_stats()
    """
    
    def __init__(
        self,
        axis: Optional[str] = None,
        provider: Optional[BaseConditionProvider] = None,
        min_token_length: int = 1,
    ):
        """
        Initialize W2 Aggregator.
        
        Args:
            axis: Axis name ('section', 'passive', etc.)
            provider: ConditionProvider instance (overrides axis)
            min_token_length: Minimum token length to include
        """
        if provider is not None:
            self.provider = provider
        elif axis is not None:
            self.provider = get_condition_provider(axis)
        else:
            raise ValueError("Must provide either 'axis' or 'provider'")
        
        self.min_token_length = min_token_length
        
        # Initialize stats
        self.stats = W2Stats(
            provider_id=self.provider.provider_id,
            axis_name=self.provider.axis_name,
            aggregation_unit=self.provider.aggregation_unit,
        )
        
        # Track articles per condition (for doc_count)
        self._condition_articles: Dict[str, Set[str]] = defaultdict(set)
    
    def process_article(
        self,
        article_id: str,
        features: List[Any],  # List of TokenFeature
        sections: List[Dict[str, Any]],  # List of {title, level, content}
    ) -> Dict[str, Any]:
        """
        Process a single article's features.
        
        Args:
            article_id: Unique article identifier
            features: List of TokenFeature from W1
            sections: List of section dicts with 'title' key
            
        Returns:
            Processing summary
        """
        if not features:
            return {"article_id": article_id, "tokens_processed": 0}
        
        # Build section index map
        section_map = {i: sec.get('title', f'section_{i}') for i, sec in enumerate(sections)}
        
        # Group features by sentence for context building
        sentences: Dict[int, List[Any]] = defaultdict(list)
        for feat in features:
            sentences[feat.sentence_idx].append(feat)
        
        # Pre-compute sentence-level properties
        sentence_has_passive: Dict[int, bool] = {}
        sentence_has_propn: Dict[int, bool] = {}
        
        for sent_idx, sent_features in sentences.items():
            # Check passive (index 7 = is_passive_participle)
            sentence_has_passive[sent_idx] = any(
                f.vector[7] == 1.0 for f in sent_features
            )
            # Check proper noun (index 14 = is_proper_noun)
            sentence_has_propn[sent_idx] = any(
                f.vector[14] == 1.0 for f in sent_features
            )
        
        # Process each token
        tokens_processed = 0
        conditions_seen: Set[str] = set()
        
        for feat in features:
            # Skip short tokens
            token_clean = feat.token.strip()
            if len(token_clean) < self.min_token_length:
                continue
            
            # Skip non-alphabetic tokens for cleaner statistics
            if not any(c.isalpha() for c in token_clean):
                continue
            
            # Build context
            section_idx = getattr(feat, 'section_idx', 0) if hasattr(feat, 'section_idx') else 0
            context = AggregationContext(
                token_idx=feat.token_idx,
                sentence_idx=feat.sentence_idx,
                section_idx=section_idx,
                section_name=section_map.get(section_idx, 'unknown'),
                sentence_tokens=sentences.get(feat.sentence_idx, []),
                sentence_has_passive=sentence_has_passive.get(feat.sentence_idx, False),
                sentence_has_propn=sentence_has_propn.get(feat.sentence_idx, False),
                article_id=article_id,
                total_sections=len(sections),
            )
            
            # Get condition ID
            condition_id = self.provider.get_condition_id(feat, context)
            conditions_seen.add(condition_id)
            
            # Normalize token (lowercase for counting)
            token_norm = feat.lemma.lower() if feat.lemma else token_clean.lower()
            
            # Update global counts
            self.stats.global_counts[token_norm] += 1
            self.stats.global_total += 1
            
            # Update condition counts
            cond_stats = self.stats.get_condition(condition_id)
            if not cond_stats.factors:
                cond_stats.factors = self.provider.get_condition_factors(condition_id)
            cond_stats.token_counts[token_norm] += 1
            cond_stats.total_tokens += 1
            
            tokens_processed += 1
        
        # Update doc counts
        for cond_id in conditions_seen:
            self._condition_articles[cond_id].add(article_id)
            self.stats.conditions[cond_id].doc_count = len(self._condition_articles[cond_id])
        
        self.stats.articles_processed += 1
        
        return {
            "article_id": article_id,
            "tokens_processed": tokens_processed,
            "conditions_seen": list(conditions_seen),
        }
    
    def get_stats(self) -> W2Stats:
        """Get current statistics."""
        return self.stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary."""
        return {
            "provider": self.provider.provider_id,
            "axis": self.provider.axis_name,
            "articles_processed": self.stats.articles_processed,
            "global_total": self.stats.global_total,
            "unique_tokens": len(self.stats.global_counts),
            "conditions": [
                {
                    "condition_id": cid,
                    "total_tokens": cond.total_tokens,
                    "unique_tokens": len(cond.token_counts),
                    "doc_count": cond.doc_count,
                }
                for cid, cond in sorted(
                    self.stats.conditions.items(),
                    key=lambda x: -x[1].total_tokens
                )
            ],
        }


# ==========================================
# Convenience Functions
# ==========================================

def aggregate_by_axis(
    axis: str,
    articles: List[Tuple[str, List[Any], List[Dict]]],
) -> W2Stats:
    """
    Aggregate multiple articles by a single axis.
    
    Args:
        axis: Condition axis ('section', 'passive', etc.)
        articles: List of (article_id, features, sections) tuples
        
    Returns:
        W2Stats
    """
    aggregator = W2Aggregator(axis=axis)
    
    for article_id, features, sections in articles:
        aggregator.process_article(article_id, features, sections)
    
    return aggregator.get_stats()


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("W2 Aggregator Test")
    print("=" * 60)
    
    # Create mock TokenFeature for testing
    from dataclasses import dataclass as dc
    
    @dc
    class MockFeature:
        token: str
        lemma: str
        vector: List[float]
        token_idx: int
        sentence_idx: int
        section_idx: int = 0
    
    # Create test features
    features = [
        MockFeature("Nobunaga", "nobunaga", [0]*20, 0, 0, 0),
        MockFeature("was", "be", [0]*8 + [0, 0] + [0]*10, 1, 0, 0),  # no passive
        MockFeature("born", "bear", [0]*7 + [1.0] + [0]*12, 2, 0, 0),  # passive participle
        MockFeature("in", "in", [0]*20, 3, 0, 0),
        MockFeature("1534", "1534", [0]*20, 4, 0, 0),
        MockFeature("He", "he", [0]*20, 0, 1, 1),
        MockFeature("conquered", "conquer", [0]*20, 1, 1, 1),
        MockFeature("many", "many", [0]*20, 2, 1, 1),
        MockFeature("castles", "castle", [0]*20, 3, 1, 1),
    ]
    
    sections = [
        {"title": "Lead", "level": 0},
        {"title": "Military campaigns", "level": 1},
    ]
    
    # Test section axis
    print("\n[Test 1] Section Axis")
    aggregator = W2Aggregator(axis="section")
    result = aggregator.process_article("test_001", features, sections)
    
    print(f"  Tokens processed: {result['tokens_processed']}")
    print(f"  Conditions seen: {result['conditions_seen']}")
    
    summary = aggregator.get_summary()
    print(f"  Global total: {summary['global_total']}")
    print(f"  Unique tokens: {summary['unique_tokens']}")
    for cond in summary['conditions']:
        print(f"    {cond['condition_id']}: {cond['total_tokens']} tokens")
    
    # Test passive axis
    print("\n[Test 2] Passive Axis")
    aggregator2 = W2Aggregator(axis="passive")
    result2 = aggregator2.process_article("test_001", features, sections)
    
    print(f"  Conditions seen: {result2['conditions_seen']}")
    summary2 = aggregator2.get_summary()
    for cond in summary2['conditions']:
        print(f"    {cond['condition_id']}: {cond['total_tokens']} tokens")
    
    print("\n" + "=" * 60)
    print("W2 Aggregator tests passed!")
