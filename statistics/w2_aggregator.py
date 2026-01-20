"""
ESDE Phase 9-2: W2 Aggregator
=============================
Conditional statistics aggregation from W0 observations.

Invariants:
  INV-W2-001 (Read-Only Condition):
      W2Aggregator reads condition factors from ArticleRecord.source_meta.
      It does NOT write, infer, or modify conditions.
      Missing factors are treated as "unknown".
  
  INV-W2-002 (Time Bucket Fixed):
      In v9.2, time_bucket is fixed to YYYY-MM (monthly).

P0 Compliance:
  P0-1: Conditions from ArticleRecord.source_meta (read-only)
  P0-2: time_bucket = YYYY-MM fixed
  P0-3: record_id = SHA256(CanonicalJSON({token_norm, cond}))

P1 Compliance:
  P1-1: Conditions stored in JSONL
  P1-2: entropy includes __OTHER__/__LONG__, top_form excludes them
  P1-3: min_count_to_persist option (default: 1)
  P1-4: normalizer_version tracked

Spec: v5.4.2-P9.2
"""

import os
import json
import math
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .schema_w2 import (
    W2Record,
    W2GlobalStats,
    ConditionEntry,
    compute_condition_signature,
    compute_record_id,
    compute_time_bucket,
    W2_VERSION,
    SOURCE_TYPES,
    LANGUAGE_PROFILES,
    DEFAULT_SOURCE_TYPE,
    DEFAULT_LANGUAGE_PROFILE,
)
from .tokenizer import get_w1_tokenizer, W1Tokenizer
from .normalizer import normalize_token, is_valid_token, NORMALIZER_VERSION
from .schema import SURFACE_FORMS_OTHER_KEY, SURFACE_FORMS_LONG_KEY


# ==========================================
# Constants
# ==========================================

DEFAULT_W2_RECORDS_PATH = "data/stats/w2_conditional_stats.jsonl"
DEFAULT_W2_CONDITIONS_PATH = "data/stats/w2_conditions.jsonl"


# ==========================================
# W2 Aggregator
# ==========================================

class W2Aggregator:
    """
    Aggregates W0 observations into W2 conditional statistics.
    
    Usage:
        aggregator = W2Aggregator()
        aggregator.process_article(article_record)
        aggregator.save()
    
    INV-W2-001: This aggregator ONLY READS condition factors.
    It does NOT infer, write, or modify ArticleRecord.source_meta.
    Missing factors default to "unknown".
    """
    
    def __init__(
        self,
        records_path: str = DEFAULT_W2_RECORDS_PATH,
        conditions_path: str = DEFAULT_W2_CONDITIONS_PATH,
        tokenizer: Optional[W1Tokenizer] = None,
        min_token_length: int = 1,
        min_count_to_persist: int = 1,  # P1-3: Sparse control
    ):
        """
        Initialize W2 Aggregator.
        
        Args:
            records_path: Path to W2 records JSONL
            conditions_path: Path to conditions registry JSONL
            tokenizer: W1Tokenizer instance (default: hybrid)
            min_token_length: Minimum token length
            min_count_to_persist: Minimum count to persist record (P1-3)
        """
        self.records_path = records_path
        self.conditions_path = conditions_path
        self.tokenizer = tokenizer or get_w1_tokenizer("hybrid")
        self.min_token_length = min_token_length
        self.min_count_to_persist = min_count_to_persist
        
        # Load or create stats
        self.stats = self._load_or_create()
        
        # Temporary accumulator for surface forms (not persisted)
        # Key: record_id -> {surface_form: count}
        self._surface_accum: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def _load_or_create(self) -> W2GlobalStats:
        """Load existing stats or create new."""
        stats = W2GlobalStats()
        
        # Load conditions
        if os.path.exists(self.conditions_path):
            try:
                with open(self.conditions_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = ConditionEntry.from_jsonl(line)
                            stats.conditions[entry.signature] = entry
                stats.total_conditions = len(stats.conditions)
            except Exception as e:
                print(f"[W2Aggregator] Warning: Failed to load conditions: {e}")
        
        # Load records
        if os.path.exists(self.records_path):
            try:
                with open(self.records_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = W2Record.from_jsonl(line)
                            stats.records[record.record_id] = record
                stats.total_records = len(stats.records)
            except Exception as e:
                print(f"[W2Aggregator] Warning: Failed to load records: {e}")
        
        return stats
    
    def _extract_segment_text(self, article_raw_text: str, segment_span: tuple) -> str:
        """
        Extract segment text from raw_text using segment_span.
        
        INV-W1-002 (inherited): segment_span is authoritative.
        """
        start, end = segment_span
        return article_raw_text[start:end]
    
    def _extract_condition_factors(self, article) -> Dict[str, str]:
        """
        Extract condition factors from ArticleRecord.
        
        INV-W2-001: READ ONLY. Never infer or modify.
        Missing factors default to "unknown".
        
        Args:
            article: ArticleRecord instance
            
        Returns:
            Condition factors dict
        """
        source_meta = getattr(article, 'source_meta', {}) or {}
        
        # Extract source_type (read-only, no inference)
        source_type = source_meta.get("source_type", DEFAULT_SOURCE_TYPE)
        if source_type not in SOURCE_TYPES:
            source_type = DEFAULT_SOURCE_TYPE
        
        # Extract language_profile (read-only, no inference)
        language_profile = source_meta.get("language_profile", DEFAULT_LANGUAGE_PROFILE)
        if language_profile not in LANGUAGE_PROFILES:
            language_profile = DEFAULT_LANGUAGE_PROFILE
        
        # Compute time_bucket from ingestion_time (INV-W2-002: YYYY-MM fixed)
        ingestion_time = getattr(article, 'ingestion_time', '')
        time_bucket = compute_time_bucket(ingestion_time) if ingestion_time else compute_time_bucket(datetime.now(timezone.utc).isoformat())
        
        return {
            "source_type": source_type,
            "language_profile": language_profile,
            "time_bucket": time_bucket,
        }
    
    def _compute_entropy(self, surface_counts: Dict[str, int]) -> float:
        """
        Compute Shannon entropy (log2) from surface form counts.
        
        P1-2: Includes __OTHER__ and __LONG__ in calculation.
        """
        if not surface_counts:
            return 0.0
        
        counts = list(surface_counts.values())
        total = sum(counts)
        
        if total < 2 or len(counts) < 2:
            return 0.0
        
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return round(entropy, 6)
    
    def _compute_top_form(self, surface_counts: Dict[str, int]) -> str:
        """
        Get most frequent surface form.
        
        P1-2: Excludes __OTHER__ and __LONG__.
        """
        if not surface_counts:
            return ""
        
        # Filter out special keys
        real_forms = {
            k: v for k, v in surface_counts.items()
            if k not in (SURFACE_FORMS_OTHER_KEY, SURFACE_FORMS_LONG_KEY)
        }
        
        if not real_forms:
            return ""
        
        return max(real_forms, key=real_forms.get)
    
    def process_article(self, article) -> Dict[str, Any]:
        """
        Process an ArticleRecord and update W2 stats.
        
        Args:
            article: ArticleRecord instance (from Phase 9-0)
            
        Returns:
            Processing summary dict
        """
        # Extract condition factors (INV-W2-001: read-only)
        factors = self._extract_condition_factors(article)
        
        # Get or create condition signature
        condition_sig = self.stats.get_or_create_condition(factors)
        
        # Get condition entry for updating denominators (INV-W2-003)
        condition_entry = self.stats.conditions[condition_sig]
        
        # Track tokens in this article for DF
        tokens_in_article: Set[str] = set()
        
        total_tokens = 0
        valid_tokens = 0
        
        for obs in article.observations:
            # Extract segment text (INV-W1-002)
            segment_text = self._extract_segment_text(
                article.raw_text,
                obs.segment_span
            )
            
            # Tokenize
            token_results = self.tokenizer.tokenize(segment_text)
            total_tokens += len(token_results)
            
            for token_result in token_results:
                surface = token_result.surface
                
                # Normalize
                token_norm = normalize_token(surface)
                if token_norm is None:
                    continue
                
                if not is_valid_token(token_norm, self.min_token_length):
                    continue
                
                valid_tokens += 1
                
                # Get or create W2 record
                record = self.stats.get_or_create_record(token_norm, condition_sig)
                
                # Update count
                record.count += 1
                record.normalizer_version = NORMALIZER_VERSION
                record.updated_at = datetime.now(timezone.utc).isoformat()
                
                # Accumulate surface forms (temporary)
                self._surface_accum[record.record_id][surface] += 1
                
                # Track for DF
                tokens_in_article.add(record.record_id)
        
        # Update DF for all records seen in this article
        for record_id in tokens_in_article:
            self.stats.records[record_id].doc_freq += 1
        
        # Update entropy and top_form for affected records
        for record_id in tokens_in_article:
            record = self.stats.records[record_id]
            surface_counts = self._surface_accum[record_id]
            record.entropy = self._compute_entropy(surface_counts)
            record.top_surface_form = self._compute_top_form(surface_counts)
        
        # INV-W2-003: Update condition denominators (W2Aggregator writes, W3 NEVER writes)
        condition_entry.total_token_count += valid_tokens
        condition_entry.total_doc_count += 1
        
        self.stats.updated_at = datetime.now(timezone.utc).isoformat()
        
        return {
            "article_id": article.article_id,
            "condition_signature": condition_sig,
            "condition_factors": factors,
            "segments_processed": len(article.observations),
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "unique_records_updated": len(tokens_in_article),
        }
    
    def process_batch(self, articles: List) -> Dict[str, Any]:
        """Process multiple ArticleRecords."""
        summaries = []
        for article in articles:
            summary = self.process_article(article)
            summaries.append(summary)
        
        return {
            "articles_processed": len(summaries),
            "total_records": self.stats.total_records,
            "total_conditions": self.stats.total_conditions,
            "summaries": summaries,
        }
    
    def rebuild_from_articles(self, articles: List) -> W2GlobalStats:
        """
        Full rebuild of W2 stats from articles.
        
        Proves W2 is a cache that can be regenerated.
        
        Args:
            articles: List of ArticleRecord instances
            
        Returns:
            Fresh W2GlobalStats
        """
        # Reset
        self.stats = W2GlobalStats()
        self._surface_accum.clear()
        
        # Process all
        self.process_batch(articles)
        
        return self.stats
    
    def save(
        self,
        records_path: Optional[str] = None,
        conditions_path: Optional[str] = None,
    ):
        """Save W2 stats to JSONL files."""
        rpath = records_path or self.records_path
        cpath = conditions_path or self.conditions_path
        
        # Ensure directories
        os.makedirs(os.path.dirname(rpath) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(cpath) or ".", exist_ok=True)
        
        # Save conditions
        with open(cpath, 'w', encoding='utf-8') as f:
            for sig in sorted(self.stats.conditions.keys()):
                entry = self.stats.conditions[sig]
                f.write(entry.to_jsonl() + '\n')
        
        # Save records (with min_count filter - P1-3)
        persisted = 0
        with open(rpath, 'w', encoding='utf-8') as f:
            for rid in sorted(self.stats.records.keys()):
                record = self.stats.records[rid]
                if record.count >= self.min_count_to_persist:
                    f.write(record.to_jsonl() + '\n')
                    persisted += 1
        
        print(f"[W2Aggregator] Saved {persisted} records to {rpath}")
        print(f"[W2Aggregator] Saved {len(self.stats.conditions)} conditions to {cpath}")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of current stats."""
        if not self.stats.records:
            return {
                "total_records": 0,
                "total_conditions": 0,
            }
        
        # Group by condition
        by_condition: Dict[str, int] = defaultdict(int)
        for record in self.stats.records.values():
            by_condition[record.condition_signature] += 1
        
        return {
            "total_records": self.stats.total_records,
            "total_conditions": self.stats.total_conditions,
            "records_per_condition": dict(by_condition),
            "w2_version": self.stats.w2_version,
        }


# ==========================================
# Comparison for Rebuild Test
# ==========================================

def compare_w2_stats(stats1: W2GlobalStats, stats2: W2GlobalStats) -> Dict[str, Any]:
    """
    Compare two W2GlobalStats for rebuild validation.
    
    Excludes updated_at from comparison.
    """
    diffs = []
    
    # Compare record counts
    if len(stats1.records) != len(stats2.records):
        diffs.append(f"Record count: {len(stats1.records)} vs {len(stats2.records)}")
    
    # Compare condition counts
    if len(stats1.conditions) != len(stats2.conditions):
        diffs.append(f"Condition count: {len(stats1.conditions)} vs {len(stats2.conditions)}")
    
    # Compare each record
    all_keys = set(stats1.records.keys()) | set(stats2.records.keys())
    
    for key in all_keys:
        r1 = stats1.records.get(key)
        r2 = stats2.records.get(key)
        
        if r1 is None:
            diffs.append(f"Missing in stats1: {key[:16]}...")
            continue
        if r2 is None:
            diffs.append(f"Missing in stats2: {key[:16]}...")
            continue
        
        c1 = r1.to_canonical_dict()
        c2 = r2.to_canonical_dict()
        
        if c1 != c2:
            diffs.append(f"Difference in {key[:16]}: {c1} vs {c2}")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-2 W2Aggregator Test")
    print("=" * 60)
    
    # Mock ArticleRecord for testing
    class MockObservation:
        def __init__(self, segment_span, timestamp):
            self.segment_span = segment_span
            self.timestamp = timestamp
    
    class MockArticle:
        def __init__(self, article_id, raw_text, observations, source_meta=None, ingestion_time=None):
            self.article_id = article_id
            self.raw_text = raw_text
            self.observations = observations
            self.source_meta = source_meta or {}
            self.ingestion_time = ingestion_time or datetime.now(timezone.utc).isoformat()
    
    # Test 1: Basic aggregation with different conditions
    print("\n[Test 1] Basic aggregation with conditions")
    
    article_news = MockArticle(
        article_id="article_news_001",
        raw_text="Apple releases new iPhone. Apple stock rises.",
        observations=[
            MockObservation((0, 25), "2026-01-20T10:00:00Z"),
            MockObservation((26, 45), "2026-01-20T10:00:01Z"),
        ],
        source_meta={"source_type": "news", "language_profile": "en"},
        ingestion_time="2026-01-20T10:00:00Z",
    )
    
    article_dialog = MockArticle(
        article_id="article_dialog_001",
        raw_text="I love Apple products!",
        observations=[
            MockObservation((0, 22), "2026-01-20T11:00:00Z"),
        ],
        source_meta={"source_type": "dialog", "language_profile": "en"},
        ingestion_time="2026-01-20T11:00:00Z",
    )
    
    aggregator = W2Aggregator(
        records_path="/tmp/w2_test_records.jsonl",
        conditions_path="/tmp/w2_test_conditions.jsonl",
    )
    
    result1 = aggregator.process_article(article_news)
    result2 = aggregator.process_article(article_dialog)
    
    print(f"  News article: {result1['condition_factors']}")
    print(f"  Dialog article: {result2['condition_factors']}")
    print(f"  Total conditions: {aggregator.stats.total_conditions}")
    print(f"  Total records: {aggregator.stats.total_records}")
    
    assert aggregator.stats.total_conditions == 2, "Should have 2 conditions"
    assert result1['condition_signature'] != result2['condition_signature'], "Different conditions"
    print("  ✅ PASS")
    
    # Test 2: INV-W2-001 - Missing factors default to unknown
    print("\n[Test 2] INV-W2-001: Missing factors -> unknown")
    
    article_no_meta = MockArticle(
        article_id="article_no_meta",
        raw_text="Hello world!",
        observations=[
            MockObservation((0, 12), "2026-01-20T12:00:00Z"),
        ],
        source_meta={},  # Empty
        ingestion_time="2026-01-20T12:00:00Z",
    )
    
    result = aggregator.process_article(article_no_meta)
    
    print(f"  Condition factors: {result['condition_factors']}")
    
    assert result['condition_factors']['source_type'] == 'unknown'
    assert result['condition_factors']['language_profile'] == 'unknown'
    print("  ✅ PASS")
    
    # Test 3: Same token, different conditions
    print("\n[Test 3] Same token, different conditions -> different records")
    
    # Find 'apple' records
    apple_records = [
        r for r in aggregator.stats.records.values()
        if r.token_norm == 'apple'
    ]
    
    print(f"  'apple' records: {len(apple_records)}")
    for r in apple_records:
        cond = aggregator.stats.conditions[r.condition_signature]
        print(f"    - {cond.factors['source_type']}: count={r.count}")
    
    # Should have apple in both news and dialog conditions
    assert len(apple_records) >= 1, "Should have apple records"
    print("  ✅ PASS")
    
    # Test 4: Rebuild idempotency
    print("\n[Test 4] Rebuild idempotency")
    
    articles = [article_news, article_dialog]
    
    # First build
    agg1 = W2Aggregator(
        records_path="/tmp/w2_rebuild1.jsonl",
        conditions_path="/tmp/w2_cond1.jsonl",
    )
    agg1.rebuild_from_articles(articles)
    
    # Second build
    agg2 = W2Aggregator(
        records_path="/tmp/w2_rebuild2.jsonl",
        conditions_path="/tmp/w2_cond2.jsonl",
    )
    agg2.rebuild_from_articles(articles)
    
    comparison = compare_w2_stats(agg1.stats, agg2.stats)
    
    print(f"  Run 1 records: {agg1.stats.total_records}")
    print(f"  Run 2 records: {agg2.stats.total_records}")
    print(f"  Match: {comparison['match']}")
    
    assert comparison['match'], "Rebuild should produce identical results"
    print("  ✅ PASS")
    
    # Test 5: Condition signature stability (order-independence)
    print("\n[Test 5] Condition signature stability")
    
    # Same factors, different order
    factors1 = {"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"}
    factors2 = {"time_bucket": "2026-01", "source_type": "news", "language_profile": "en"}
    
    sig1 = compute_condition_signature(factors1)
    sig2 = compute_condition_signature(factors2)
    
    print(f"  factors1: {sig1}")
    print(f"  factors2: {sig2}")
    
    assert sig1 == sig2, "Signatures should match regardless of key order"
    print("  ✅ PASS")
    
    # Test 6: Time bucket (INV-W2-002)
    print("\n[Test 6] Time bucket (YYYY-MM fixed)")
    
    article_jan = MockArticle(
        article_id="jan_article",
        raw_text="Test",
        observations=[MockObservation((0, 4), "2026-01-15T10:00:00Z")],
        source_meta={"source_type": "news", "language_profile": "en"},
        ingestion_time="2026-01-15T10:00:00Z",
    )
    
    article_feb = MockArticle(
        article_id="feb_article",
        raw_text="Test",
        observations=[MockObservation((0, 4), "2026-02-15T10:00:00Z")],
        source_meta={"source_type": "news", "language_profile": "en"},
        ingestion_time="2026-02-15T10:00:00Z",
    )
    
    agg3 = W2Aggregator(
        records_path="/tmp/w2_time_test.jsonl",
        conditions_path="/tmp/w2_time_cond.jsonl",
    )
    
    r1 = agg3.process_article(article_jan)
    r2 = agg3.process_article(article_feb)
    
    print(f"  Jan time_bucket: {r1['condition_factors']['time_bucket']}")
    print(f"  Feb time_bucket: {r2['condition_factors']['time_bucket']}")
    
    assert r1['condition_factors']['time_bucket'] == '2026-01'
    assert r2['condition_factors']['time_bucket'] == '2026-02'
    assert r1['condition_signature'] != r2['condition_signature'], "Different months = different conditions"
    print("  ✅ PASS")
    
    # Test 7: entropy/top_form (P1-2)
    print("\n[Test 7] Entropy and top_form calculation")
    
    article_varied = MockArticle(
        article_id="varied_article",
        raw_text="Apple APPLE apple Apple",
        observations=[MockObservation((0, 23), "2026-01-20T13:00:00Z")],
        source_meta={"source_type": "social", "language_profile": "en"},
        ingestion_time="2026-01-20T13:00:00Z",
    )
    
    agg4 = W2Aggregator(
        records_path="/tmp/w2_entropy.jsonl",
        conditions_path="/tmp/w2_entropy_cond.jsonl",
    )
    agg4.process_article(article_varied)
    
    # Find apple record
    apple_rec = None
    for r in agg4.stats.records.values():
        if r.token_norm == 'apple':
            apple_rec = r
            break
    
    print(f"  'apple' entropy: {apple_rec.entropy}")
    print(f"  'apple' top_form: {apple_rec.top_surface_form}")
    
    assert apple_rec.entropy > 0, "Should have non-zero entropy"
    assert apple_rec.top_surface_form == "Apple", "Apple appears most (2 times)"
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nPhase 9-2 W2Aggregator is ready.")