#!/usr/bin/env python3
"""
ESDE Phase 9: W2 Adapter
========================

Converts new W2Stats (from internal condition provider) to legacy W2GlobalStats/W1GlobalStats
format for use with existing W3-W6 pipeline.

This adapter enables:
  - New W1 (TokenFeature) → New W2 (ConditionProvider) pipeline
  - Output compatible with existing W3Calculator, W4Projector, W5Condensator, W6Analyzer

Usage:
    from statistics.pipeline.w2_adapter import convert_to_legacy_format
    
    # New pipeline
    aggregator = W2Aggregator(axis="section")
    aggregator.process_article(...)
    new_stats = aggregator.get_stats()
    
    # Convert for existing W3-W6
    w1_stats, w2_stats = convert_to_legacy_format(new_stats)
    
    # Use existing W3
    calculator = W3Calculator(w1_stats, w2_stats)

Spec: Phase 9 W2 Adapter v1.0
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Tuple, Any
from dataclasses import dataclass, field

# Import from new pipeline
from .w2_aggregator import W2Stats, ConditionStats

# We'll define legacy-compatible structures here to avoid import issues
# These mirror the existing schema_w2.py and schema.py structures


# ==========================================
# Legacy-Compatible Structures
# ==========================================

@dataclass
class LegacyW1Record:
    """Mirror of W1Record from statistics.schema."""
    token_norm: str
    total_count: int = 0
    doc_freq: int = 0
    surface_forms: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_norm": self.token_norm,
            "total_count": self.total_count,
            "doc_freq": self.doc_freq,
            "surface_forms": self.surface_forms,
        }


@dataclass
class LegacyW1GlobalStats:
    """Mirror of W1GlobalStats from statistics.schema."""
    records: Dict[str, 'LegacyW1Record'] = field(default_factory=dict)
    total_tokens: int = 0
    total_documents: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_documents": self.total_documents,
            "record_count": len(self.records),
        }


@dataclass
class LegacyConditionEntry:
    """Mirror of ConditionEntry from statistics.schema_w2."""
    signature: str
    factors: Dict[str, str]
    first_seen: str = ""
    total_token_count: int = 0
    total_doc_count: int = 0
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "factors": self.factors,
            "first_seen": self.first_seen,
            "total_token_count": self.total_token_count,
            "total_doc_count": self.total_doc_count,
        }


@dataclass
class LegacyW2Record:
    """Mirror of W2Record from statistics.schema_w2."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "token_norm": self.token_norm,
            "condition_signature": self.condition_signature,
            "count": self.count,
            "doc_freq": self.doc_freq,
            "entropy": round(self.entropy, 6),
            "top_surface_form": self.top_surface_form,
            "normalizer_version": self.normalizer_version,
            "updated_at": self.updated_at,
        }


@dataclass
class LegacyW2GlobalStats:
    """Mirror of W2GlobalStats from statistics.schema_w2."""
    records: Dict[str, 'LegacyW2Record'] = field(default_factory=dict)
    conditions: Dict[str, 'LegacyConditionEntry'] = field(default_factory=dict)
    total_records: int = 0
    total_conditions: int = 0
    w2_version: str = "v9.2.0"
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "total_conditions": self.total_conditions,
            "w2_version": self.w2_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ==========================================
# Conversion Functions
# ==========================================

def compute_condition_signature(factors: Dict[str, str]) -> str:
    """Compute SHA256 signature for condition factors."""
    canonical = json.dumps(factors, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def compute_record_id(token_norm: str, condition_signature: str) -> str:
    """Compute SHA256 record ID for (token, condition) pair."""
    data = {"token_norm": token_norm, "cond": condition_signature}
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def convert_to_legacy_format(
    new_stats: W2Stats,
) -> Tuple[LegacyW1GlobalStats, LegacyW2GlobalStats]:
    """
    Convert new W2Stats to legacy W1GlobalStats + W2GlobalStats format.
    
    This enables using the new ConditionProvider-based pipeline with
    existing W3-W6 components.
    
    Args:
        new_stats: W2Stats from new pipeline (ConditionProvider-based)
        
    Returns:
        (w1_stats, w2_stats) tuple compatible with W3Calculator
    """
    # Build W1 (global statistics)
    w1_stats = LegacyW1GlobalStats()
    
    for token_norm, count in new_stats.global_counts.items():
        w1_stats.records[token_norm] = LegacyW1Record(
            token_norm=token_norm,
            total_count=count,
            doc_freq=1,  # Simplified - would need tracking for accurate DF
            surface_forms={token_norm: count},
        )
    
    w1_stats.total_tokens = new_stats.global_total
    w1_stats.total_documents = new_stats.articles_processed
    
    # Build W2 (conditional statistics)
    w2_stats = LegacyW2GlobalStats()
    
    for cond_id, cond_stats in new_stats.conditions.items():
        # Create condition entry
        # Build factors from condition_id
        factors = {
            new_stats.axis_name: cond_id,
            "provider": new_stats.provider_id,
        }
        
        signature = compute_condition_signature(factors)
        
        w2_stats.conditions[signature] = LegacyConditionEntry(
            signature=signature,
            factors=factors,
            total_token_count=cond_stats.total_tokens,
            total_doc_count=cond_stats.doc_count,
        )
        
        # Create W2 records for each token under this condition
        for token_norm, count in cond_stats.token_counts.items():
            record_id = compute_record_id(token_norm, signature)
            
            w2_stats.records[record_id] = LegacyW2Record(
                record_id=record_id,
                token_norm=token_norm,
                condition_signature=signature,
                count=count,
                doc_freq=1,  # Simplified
                top_surface_form=token_norm,
            )
    
    w2_stats.total_records = len(w2_stats.records)
    w2_stats.total_conditions = len(w2_stats.conditions)
    
    return w1_stats, w2_stats


def get_conversion_summary(
    w1_stats: LegacyW1GlobalStats,
    w2_stats: LegacyW2GlobalStats,
) -> Dict[str, Any]:
    """Get summary of converted statistics."""
    return {
        "w1": {
            "total_tokens": w1_stats.total_tokens,
            "unique_tokens": len(w1_stats.records),
            "documents": w1_stats.total_documents,
        },
        "w2": {
            "total_records": w2_stats.total_records,
            "total_conditions": w2_stats.total_conditions,
            "conditions": [
                {
                    "signature": sig[:16] + "...",
                    "factors": entry.factors,
                    "total_tokens": entry.total_token_count,
                }
                for sig, entry in w2_stats.conditions.items()
            ],
        },
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("W2 Adapter Test")
    print("=" * 60)
    
    from .w2_aggregator import W2Stats, ConditionStats
    from collections import defaultdict
    
    # Create mock new W2Stats
    new_stats = W2Stats(
        provider_id="section_v1",
        axis_name="section",
        aggregation_unit="section",
    )
    
    # Global counts
    new_stats.global_counts = defaultdict(int, {
        "war": 50,
        "peace": 20,
        "battle": 30,
        "art": 15,
    })
    new_stats.global_total = sum(new_stats.global_counts.values())
    new_stats.articles_processed = 5
    
    # Condition: military
    military = ConditionStats(
        condition_id="military",
        factors={"section_name": "military"},
    )
    military.token_counts = defaultdict(int, {
        "war": 35,
        "battle": 25,
        "peace": 2,
    })
    military.total_tokens = sum(military.token_counts.values())
    military.doc_count = 3
    new_stats.conditions["military"] = military
    
    # Condition: culture
    culture = ConditionStats(
        condition_id="culture",
        factors={"section_name": "culture"},
    )
    culture.token_counts = defaultdict(int, {
        "art": 12,
        "peace": 10,
        "war": 2,
    })
    culture.total_tokens = sum(culture.token_counts.values())
    culture.doc_count = 2
    new_stats.conditions["culture"] = culture
    
    # Convert
    w1_stats, w2_stats = convert_to_legacy_format(new_stats)
    
    # Print summary
    summary = get_conversion_summary(w1_stats, w2_stats)
    
    print("\n[Conversion Result]")
    print(f"  W1 total_tokens: {summary['w1']['total_tokens']}")
    print(f"  W1 unique_tokens: {summary['w1']['unique_tokens']}")
    print(f"  W2 total_conditions: {summary['w2']['total_conditions']}")
    print(f"  W2 total_records: {summary['w2']['total_records']}")
    
    print("\n[Conditions]")
    for cond in summary['w2']['conditions']:
        print(f"  {cond['factors']}: {cond['total_tokens']} tokens")
    
    # Verify structure
    print("\n[Verification]")
    assert len(w2_stats.conditions) == 2, "Should have 2 conditions"
    assert len(w1_stats.records) == 4, "Should have 4 unique tokens"
    print("  ✅ Structure verified")
    
    print("\n" + "=" * 60)
    print("W2 Adapter tests passed!")
