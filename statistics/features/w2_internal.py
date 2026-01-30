#!/usr/bin/env python3
"""
ESDE Phase 9: W2 Aggregator (Internal Conditions)
=================================================

Aggregates token statistics by internal condition (section/passive/etc).

This is NOT a replacement for the existing W2Aggregator.
It produces output compatible with existing W3Calculator.

Key Difference from Legacy W2:
  - Conditions from internal structure (ConditionProvider)
  - NOT from source_meta
  - Multiple conditions per article (expected!)

Usage:
    from statistics.features.w2_internal import W2InternalAggregator
    
    aggregator = W2InternalAggregator(axis="section")
    aggregator.process_features(article_id, features, sections)
    
    # Get W2GlobalStats compatible output
    w2_stats = aggregator.get_w2_stats()
    
    # Feed to existing W3Calculator
    calculator = W3Calculator(w1_stats, w2_stats)

Spec: Phase 9 W2 Internal v1.0
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set

from .condition_providers import (
    BaseConditionProvider,
    SentenceContext,
    get_condition_provider,
)
from .feature_extractor import TokenFeature


# ==========================================
# Constants (Legacy W2 Compatible)
# ==========================================

W2_VERSION = "v9.2.0-internal"
SOURCE_META_VERSION = "v1"


# ==========================================
# Helper: Build SentenceContext from Features
# ==========================================

def group_features_by_sentence(
    features: List[TokenFeature],
) -> Dict[int, List[TokenFeature]]:
    """Group TokenFeature list by sentence_idx."""
    grouped = defaultdict(list)
    for feat in features:
        grouped[feat.sentence_idx].append(feat)
    return dict(grouped)


def build_sentence_context(
    sentence_idx: int,
    sentence_features: List[TokenFeature],
    section_idx: int,
    section_name: str,
) -> SentenceContext:
    """
    Build SentenceContext from list of TokenFeatures in a sentence.
    
    Args:
        sentence_idx: Sentence index
        sentence_features: TokenFeatures in this sentence
        section_idx: Section index
        section_name: Section name
        
    Returns:
        SentenceContext with aggregated sentence-level features
    """
    # Aggregate sentence-level boolean flags
    has_passive = False
    has_paren = False
    has_quote = False
    has_proper_noun = False
    
    tokens = []
    
    for feat in sentence_features:
        # Extract token (normalized)
        token = feat.lemma.lower() if feat.lemma else feat.token.lower()
        if token and any(c.isalpha() for c in token):
            tokens.append(token)
        
        # Check vector flags
        if len(feat.vector) >= 20:
            # Index 7: is_passive_participle
            if feat.vector[7] == 1.0:
                has_passive = True
            # Index 8: inside_parentheses
            if feat.vector[8] == 1.0:
                has_paren = True
            # Index 9: is_in_quote
            if feat.vector[9] == 1.0:
                has_quote = True
            # Index 14: is_proper_noun
            if feat.vector[14] == 1.0:
                has_proper_noun = True
    
    return SentenceContext(
        sentence_idx=sentence_idx,
        section_idx=section_idx,
        section_name=section_name,
        token_count=len(tokens),
        has_passive=has_passive,
        has_paren=has_paren,
        has_quote=has_quote,
        has_proper_noun=has_proper_noun,
        tokens=tokens,
    )


# ==========================================
# W2 Statistics Structures (Legacy Compatible)
# ==========================================

@dataclass
class ConditionEntry:
    """Condition registry entry (legacy W2 compatible)."""
    signature: str
    factors: Dict[str, str]
    first_seen: str = ""
    total_token_count: int = 0
    total_doc_count: int = 0
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()


@dataclass
class W2Record:
    """W2 record for (token, condition) pair (legacy compatible)."""
    record_id: str
    token_norm: str
    condition_signature: str
    count: int = 0
    doc_freq: int = 0
    entropy: float = 0.0
    top_surface_form: str = ""
    normalizer_version: str = "v9.1.0"
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()


@dataclass
class W2GlobalStats:
    """Container for W2 statistics (legacy compatible)."""
    records: Dict[str, W2Record] = field(default_factory=dict)
    conditions: Dict[str, ConditionEntry] = field(default_factory=dict)
    total_records: int = 0
    total_conditions: int = 0
    w2_version: str = W2_VERSION
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ==========================================
# W2 Internal Aggregator
# ==========================================

class W2InternalAggregator:
    """
    W2 Aggregator using internal condition providers.
    
    Produces W2GlobalStats compatible with existing W3Calculator.
    
    Usage:
        aggregator = W2InternalAggregator(axis="section")
        aggregator.process_features("article_001", features, sections)
        w2_stats = aggregator.get_w2_stats()
    """
    
    def __init__(
        self,
        axis: str = "section",
        provider: Optional[BaseConditionProvider] = None,
        min_token_length: int = 1,
    ):
        """
        Initialize W2InternalAggregator.
        
        Args:
            axis: Condition axis ('section', 'passive', 'paren', 'quote', 'propn')
            provider: Custom provider (overrides axis)
            min_token_length: Minimum token length to count
        """
        if provider:
            self.provider = provider
        else:
            self.provider = get_condition_provider(axis)
        
        self.axis = self.provider.condition_axis
        self.min_token_length = min_token_length
        
        # Accumulators
        self._condition_tokens: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._condition_docs: Dict[str, Set[str]] = defaultdict(set)
        self._condition_total: Dict[str, int] = defaultdict(int)
        self._condition_factors: Dict[str, Dict[str, str]] = {}
        
        # Global accumulators (for W1 compatibility)
        self._global_tokens: Dict[str, int] = defaultdict(int)
        self._global_total: int = 0
        
        self._articles_processed: int = 0
    
    def process_features(
        self,
        article_id: str,
        features: List[TokenFeature],
        sections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process article features and aggregate statistics.
        
        Args:
            article_id: Unique article identifier
            features: List of TokenFeature from Phase 9 W1
            sections: Optional section info (for section axis)
            
        Returns:
            Processing summary
        """
        # Default sections if not provided
        if sections is None:
            sections = [{"title": "unknown", "level": 0}]
        
        # Build section map
        section_map = {}
        for i, sec in enumerate(sections):
            section_map[i] = sec.get("title", f"section_{i}")
        
        # Group features by sentence
        sentences = group_features_by_sentence(features)
        
        conditions_seen = set()
        tokens_processed = 0
        
        for sent_idx, sent_features in sentences.items():
            # Determine section for this sentence
            # Use section_idx from first feature, or default to 0
            section_idx = 0
            if sent_features:
                section_idx = getattr(sent_features[0], 'section_idx', 0)
            section_name = section_map.get(section_idx, "unknown")
            
            # Build sentence context
            ctx = build_sentence_context(
                sentence_idx=sent_idx,
                sentence_features=sent_features,
                section_idx=section_idx,
                section_name=section_name,
            )
            
            # Get condition_id
            condition_id = self.provider.get_condition_id(ctx)
            signature = self.provider.get_condition_signature(condition_id)
            conditions_seen.add(signature)
            
            # Store factors (for human readability)
            if signature not in self._condition_factors:
                self._condition_factors[signature] = self.provider.get_condition_factors(condition_id)
            
            # Count tokens
            for token in ctx.tokens:
                if len(token) < self.min_token_length:
                    continue
                
                # Condition-specific count
                self._condition_tokens[signature][token] += 1
                self._condition_total[signature] += 1
                
                # Global count
                self._global_tokens[token] += 1
                self._global_total += 1
                
                tokens_processed += 1
            
            # Track docs per condition
            self._condition_docs[signature].add(article_id)
        
        self._articles_processed += 1
        
        return {
            "article_id": article_id,
            "sentences_processed": len(sentences),
            "tokens_processed": tokens_processed,
            "conditions_seen": list(conditions_seen),
        }
    
    def get_w2_stats(self) -> W2GlobalStats:
        """
        Get W2GlobalStats compatible with existing W3Calculator.
        
        Returns:
            W2GlobalStats with conditions and records
        """
        stats = W2GlobalStats()
        
        # Build conditions
        for signature, factors in self._condition_factors.items():
            stats.conditions[signature] = ConditionEntry(
                signature=signature,
                factors=factors,
                total_token_count=self._condition_total[signature],
                total_doc_count=len(self._condition_docs[signature]),
            )
        
        # Build records
        for signature, token_counts in self._condition_tokens.items():
            for token, count in token_counts.items():
                record_id = self._compute_record_id(token, signature)
                
                stats.records[record_id] = W2Record(
                    record_id=record_id,
                    token_norm=token,
                    condition_signature=signature,
                    count=count,
                    doc_freq=1,  # Simplified
                    top_surface_form=token,
                )
        
        stats.total_records = len(stats.records)
        stats.total_conditions = len(stats.conditions)
        
        return stats
    
    def get_w1_stats(self) -> Dict[str, Any]:
        """
        Get W1-like global statistics.
        
        Returns:
            Dict with global token counts
        """
        return {
            "records": {
                token: {
                    "token_norm": token,
                    "total_count": count,
                }
                for token, count in self._global_tokens.items()
            },
            "total_tokens": self._global_total,
            "total_documents": self._articles_processed,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable summary."""
        return {
            "provider": self.provider.provider_id,
            "axis": self.axis,
            "articles_processed": self._articles_processed,
            "total_conditions": len(self._condition_factors),
            "total_tokens": self._global_total,
            "unique_tokens": len(self._global_tokens),
            "conditions": [
                {
                    "signature": sig[:16] + "...",
                    "factors": factors,
                    "total_tokens": self._condition_total[sig],
                    "docs": len(self._condition_docs[sig]),
                }
                for sig, factors in sorted(
                    self._condition_factors.items(),
                    key=lambda x: -self._condition_total[x[0]]
                )
            ],
        }
    
    def _compute_record_id(self, token: str, condition_signature: str) -> str:
        """Compute record_id for (token, condition) pair."""
        data = {"token_norm": token, "cond": condition_signature}
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


# ==========================================
# W1GlobalStats Adapter (for W3Calculator)
# ==========================================

@dataclass
class W1Record:
    """W1 record (legacy compatible)."""
    token_norm: str
    total_count: int = 0
    doc_freq: int = 0
    surface_forms: Dict[str, int] = field(default_factory=dict)


@dataclass
class W1GlobalStats:
    """W1 global statistics (legacy compatible)."""
    records: Dict[str, W1Record] = field(default_factory=dict)
    total_tokens: int = 0
    total_documents: int = 0


def build_w1_from_aggregator(aggregator: W2InternalAggregator) -> W1GlobalStats:
    """
    Build W1GlobalStats from W2InternalAggregator's global counts.
    
    This provides the "global distribution" needed by W3Calculator.
    """
    stats = W1GlobalStats()
    
    for token, count in aggregator._global_tokens.items():
        stats.records[token] = W1Record(
            token_norm=token,
            total_count=count,
            doc_freq=1,  # Simplified
            surface_forms={token: count},
        )
    
    stats.total_tokens = aggregator._global_total
    stats.total_documents = aggregator._articles_processed
    
    return stats


# ==========================================
# Export
# ==========================================

__all__ = [
    # Context Builder
    "group_features_by_sentence",
    "build_sentence_context",
    
    # Legacy Compatible Structures
    "ConditionEntry",
    "W2Record",
    "W2GlobalStats",
    "W1Record",
    "W1GlobalStats",
    
    # Main Aggregator
    "W2InternalAggregator",
    
    # Adapter
    "build_w1_from_aggregator",
]
