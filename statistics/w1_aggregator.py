"""
ESDE Phase 9-1: W1 Aggregator
=============================
Cross-sectional statistics aggregation from W0 observations.

Invariants:
  INV-W1-001 (Recalculability):
      W1 can always be regenerated from W0. Loss of W1 is not system corruption.
  
  INV-W1-002 (Input Authority):
      W1 input is ArticleRecord.raw_text sliced by ObservationEvent.segment_span.
      ObservationEvent.segment_text is cache only, not authoritative.

P0 Compliance:
  P0-1: Input from raw_text + segment_span (not segment_text)
  P0-2: Explicit tokenizer (English word / CJK bigram)
  P0-3: DF counted once per source_id
  P0-4: surface_forms limited to 50, overflow to __OTHER__, long to __LONG__

Spec: v5.4.1-P9.1
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field

# Import from integration (Phase 9-0)
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .schema import W1Record, W1GlobalStats, AGGREGATION_VERSION
from .tokenizer import get_w1_tokenizer, W1Tokenizer, TokenResult
from .normalizer import normalize_token, is_valid_token


# ==========================================
# Constants
# ==========================================

DEFAULT_STORAGE_PATH = "data/stats/w1_global_stats.json"


# ==========================================
# Source ID Prefix (P1-4)
# ==========================================

def format_source_id(source_id: str, source_type: str = "A") -> str:
    """
    Format source_id with type prefix (P1-4).
    
    Args:
        source_id: Raw source ID (article_id or dialog_id)
        source_type: "A" for Article, "D" for Dialog
        
    Returns:
        Prefixed source_id (e.g., "A:uuid...")
    """
    if source_id.startswith(("A:", "D:")):
        return source_id
    return f"{source_type}:{source_id}"


# ==========================================
# W1 Aggregator
# ==========================================

class W1Aggregator:
    """
    Aggregates W0 observations into W1 cross-sectional statistics.
    
    Usage:
        aggregator = W1Aggregator()
        aggregator.process_article(article_record)
        aggregator.save()
    
    Constraints (must not violate):
      - No semantic inference (just counting)
      - No modification of source ArticleRecord or ObservationEvent
      - No Axis generation
    """
    
    def __init__(
        self,
        storage_path: str = DEFAULT_STORAGE_PATH,
        tokenizer: Optional[W1Tokenizer] = None,
        min_token_length: int = 1,
    ):
        """
        Initialize W1 Aggregator.
        
        Args:
            storage_path: Path to W1 stats JSON file
            tokenizer: W1Tokenizer instance (default: hybrid)
            min_token_length: Minimum token length to track
        """
        self.storage_path = storage_path
        self.tokenizer = tokenizer or get_w1_tokenizer("hybrid")
        self.min_token_length = min_token_length
        
        # Load or create stats
        self.stats = self._load_or_create()
        
        # Track processed source_ids for DF calculation (P0-3)
        self._processed_sources: Set[str] = set()
    
    def _load_or_create(self) -> W1GlobalStats:
        """Load existing stats or create new."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    return W1GlobalStats.from_json(f.read())
            except Exception as e:
                print(f"[W1Aggregator] Warning: Failed to load stats: {e}")
        
        return W1GlobalStats()
    
    def _extract_segment_text(self, article_raw_text: str, segment_span: tuple) -> str:
        """
        Extract segment text from article raw_text using segment_span.
        
        INV-W1-002: This is the authoritative way to get segment text.
        ObservationEvent.segment_text is cache only.
        
        Args:
            article_raw_text: ArticleRecord.raw_text
            segment_span: (start, end) tuple
            
        Returns:
            Extracted segment text
        """
        start, end = segment_span
        return article_raw_text[start:end]
    
    def process_article(self, article) -> Dict[str, Any]:
        """
        Process an ArticleRecord and update W1 stats.
        
        This is the main entry point for stream processing.
        
        Args:
            article: ArticleRecord instance (from Phase 9-0)
            
        Returns:
            Processing summary dict
        """
        source_id = format_source_id(article.article_id, "A")
        
        # Track tokens seen in this article (for DF calculation, P0-3)
        tokens_in_article: Set[str] = set()
        
        total_tokens = 0
        valid_tokens = 0
        
        for obs in article.observations:
            # P0-1: Extract from raw_text + segment_span (INV-W1-002)
            segment_text = self._extract_segment_text(
                article.raw_text,
                obs.segment_span
            )
            
            # Tokenize
            token_results = self.tokenizer.tokenize(segment_text)
            total_tokens += len(token_results)
            
            # Process each token
            for token_result in token_results:
                surface = token_result.surface
                
                # Normalize
                token_norm = normalize_token(surface)
                if token_norm is None:
                    continue
                
                if not is_valid_token(token_norm, self.min_token_length):
                    continue
                
                valid_tokens += 1
                
                # Get or create record
                record = self.stats.get_or_create(token_norm)
                
                # Update count
                record.total_count += 1
                
                # Update surface forms (P0-4: limits handled in W1Record)
                record.add_surface_form(surface)
                
                # Track for DF (P0-3)
                tokens_in_article.add(token_norm)
                
                # Update timestamps (P1-3: ObservationEvent.timestamp is authoritative)
                obs_timestamp = obs.timestamp
                if record.first_seen_at is None or obs_timestamp < record.first_seen_at:
                    record.first_seen_at = obs_timestamp
                if record.last_seen_at is None or obs_timestamp > record.last_seen_at:
                    record.last_seen_at = obs_timestamp
                    record.last_seen_source = source_id
        
        # Update DF for all tokens seen in this article (P0-3: once per source_id)
        for token_norm in tokens_in_article:
            self.stats.records[token_norm].document_frequency += 1
        
        # Update entropies
        for token_norm in tokens_in_article:
            self.stats.records[token_norm].update_entropy()
        
        # Update global stats
        self.stats.total_tokens_processed += valid_tokens
        self.stats.total_articles_processed += 1
        self.stats.updated_at = datetime.now(timezone.utc).isoformat()
        
        self._processed_sources.add(source_id)
        
        return {
            "source_id": source_id,
            "segments_processed": len(article.observations),
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "unique_tokens": len(tokens_in_article),
        }
    
    def process_batch(self, articles: List) -> Dict[str, Any]:
        """
        Process multiple ArticleRecords.
        
        Args:
            articles: List of ArticleRecord instances
            
        Returns:
            Batch processing summary
        """
        summaries = []
        for article in articles:
            summary = self.process_article(article)
            summaries.append(summary)
        
        return {
            "articles_processed": len(summaries),
            "total_tokens": sum(s["valid_tokens"] for s in summaries),
            "unique_tokens_global": len(self.stats.records),
            "summaries": summaries,
        }
    
    def rebuild_from_articles(self, articles: List) -> W1GlobalStats:
        """
        Full rebuild of W1 stats from W0 observations (INV-W1-001).
        
        This proves W1 is a cache that can be regenerated from W0.
        
        Args:
            articles: List of ArticleRecord instances
            
        Returns:
            Fresh W1GlobalStats
        """
        # Create fresh stats
        self.stats = W1GlobalStats()
        self._processed_sources.clear()
        
        # Process all articles
        self.process_batch(articles)
        
        return self.stats
    
    def save(self, path: Optional[str] = None):
        """Save W1 stats to JSON file."""
        save_path = path or self.storage_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(self.stats.to_json())
        
        print(f"[W1Aggregator] Saved {len(self.stats.records)} records to {save_path}")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of current stats."""
        if not self.stats.records:
            return {
                "total_records": 0,
                "total_tokens_processed": 0,
                "total_articles_processed": 0,
            }
        
        records = list(self.stats.records.values())
        
        return {
            "total_records": len(records),
            "total_tokens_processed": self.stats.total_tokens_processed,
            "total_articles_processed": self.stats.total_articles_processed,
            "top_by_count": sorted(
                [(r.token_norm, r.total_count) for r in records],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "top_by_entropy": sorted(
                [(r.token_norm, r.entropy) for r in records if r.entropy > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }


# ==========================================
# Rebuild Comparison (P1-5)
# ==========================================

def compare_w1_stats(stats1: W1GlobalStats, stats2: W1GlobalStats) -> Dict[str, Any]:
    """
    Compare two W1GlobalStats for rebuild validation (P1-5).
    
    Rules:
      - Exclude updated_at from comparison
      - Use canonical dict (sorted keys, fixed precision)
      
    Returns:
        Comparison result dict
    """
    diffs = []
    
    # Compare record counts
    if len(stats1.records) != len(stats2.records):
        diffs.append(f"Record count: {len(stats1.records)} vs {len(stats2.records)}")
    
    # Compare each record
    all_keys = set(stats1.records.keys()) | set(stats2.records.keys())
    
    for key in all_keys:
        r1 = stats1.records.get(key)
        r2 = stats2.records.get(key)
        
        if r1 is None:
            diffs.append(f"Missing in stats1: {key}")
            continue
        if r2 is None:
            diffs.append(f"Missing in stats2: {key}")
            continue
        
        # Compare canonical dicts (excludes updated_at)
        c1 = r1.to_canonical_dict()
        c2 = r2.to_canonical_dict()
        
        if c1 != c2:
            diffs.append(f"Difference in '{key}': {c1} vs {c2}")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],  # Limit to first 10
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-1 W1Aggregator Test")
    print("=" * 60)
    
    # Create mock ArticleRecord-like objects for testing
    # (Simulating Phase 9-0 output)
    
    class MockObservation:
        def __init__(self, segment_span, timestamp):
            self.segment_span = segment_span
            self.timestamp = timestamp
    
    class MockArticle:
        def __init__(self, article_id, raw_text, observations):
            self.article_id = article_id
            self.raw_text = raw_text
            self.observations = observations
    
    # Test 1: Basic aggregation
    print("\n[Test 1] Basic aggregation")
    
    article1 = MockArticle(
        article_id="article_001",
        raw_text="I love Apple. Apple is great!",
        observations=[
            MockObservation((0, 13), "2026-01-20T10:00:00Z"),   # "I love Apple."
            MockObservation((14, 29), "2026-01-20T10:00:01Z"),  # "Apple is great!"
        ]
    )
    
    aggregator = W1Aggregator(storage_path="/tmp/w1_test.json")
    result = aggregator.process_article(article1)
    
    print(f"  Processed: {result}")
    
    # Check "apple" record (appears twice with different cases)
    apple_record = aggregator.stats.records.get("apple")
    print(f"  'apple' total_count: {apple_record.total_count if apple_record else 'N/A'}")
    print(f"  'apple' surface_forms: {apple_record.surface_forms if apple_record else 'N/A'}")
    print(f"  'apple' DF: {apple_record.document_frequency if apple_record else 'N/A'}")
    
    assert apple_record is not None, "Should have 'apple' record"
    assert apple_record.total_count == 2, f"Expected 2, got {apple_record.total_count}"
    assert "Apple" in apple_record.surface_forms, "Should have 'Apple' surface form"
    assert apple_record.document_frequency == 1, "DF should be 1 (same article)"
    print("  ✅ PASS")
    
    # Test 2: DF calculation across articles (P0-3)
    print("\n[Test 2] Document Frequency (P0-3: once per source_id)")
    
    article2 = MockArticle(
        article_id="article_002",
        raw_text="Apple makes iPhones.",
        observations=[
            MockObservation((0, 20), "2026-01-20T11:00:00Z"),
        ]
    )
    
    aggregator.process_article(article2)
    
    apple_record = aggregator.stats.records["apple"]
    print(f"  'apple' total_count: {apple_record.total_count}")
    print(f"  'apple' DF: {apple_record.document_frequency}")
    
    assert apple_record.total_count == 3, f"Expected 3, got {apple_record.total_count}"
    assert apple_record.document_frequency == 2, f"Expected DF=2, got {apple_record.document_frequency}"
    print("  ✅ PASS")
    
    # Test 3: Surface variance tracking
    print("\n[Test 3] Surface variance (case sensitivity)")
    
    article3 = MockArticle(
        article_id="article_003",
        raw_text="APPLE apple Apple",
        observations=[
            MockObservation((0, 17), "2026-01-20T12:00:00Z"),
        ]
    )
    
    aggregator.process_article(article3)
    
    apple_record = aggregator.stats.records["apple"]
    print(f"  'apple' surface_forms: {apple_record.surface_forms}")
    print(f"  'apple' entropy: {apple_record.entropy:.4f}")
    
    assert "APPLE" in apple_record.surface_forms
    assert "apple" in apple_record.surface_forms
    assert "Apple" in apple_record.surface_forms
    assert apple_record.entropy > 0, "Entropy should be > 0 with multiple forms"
    print("  ✅ PASS")
    
    # Test 4: Rebuild test (INV-W1-001, P1-5)
    print("\n[Test 4] Rebuild from W0 (INV-W1-001)")
    
    articles = [article1, article2, article3]
    
    # Rebuild fresh
    fresh_stats = aggregator.rebuild_from_articles(articles)
    
    # Compare
    print(f"  Rebuilt {len(fresh_stats.records)} records")
    
    # Should be identical (except updated_at)
    apple_fresh = fresh_stats.records["apple"]
    print(f"  Rebuilt 'apple' total_count: {apple_fresh.total_count}")
    print(f"  Rebuilt 'apple' DF: {apple_fresh.document_frequency}")
    
    assert apple_fresh.total_count == 6, f"Expected 6, got {apple_fresh.total_count}"
    assert apple_fresh.document_frequency == 3, f"Expected 3, got {apple_fresh.document_frequency}"
    print("  ✅ PASS")
    
    # Test 5: Japanese text (CJK bigram)
    print("\n[Test 5] Japanese text (CJK bigram)")
    
    aggregator2 = W1Aggregator(storage_path="/tmp/w1_test_jp.json")
    
    article_jp = MockArticle(
        article_id="article_jp_001",
        raw_text="日本語テスト",
        observations=[
            MockObservation((0, 6), "2026-01-20T13:00:00Z"),
        ]
    )
    
    result = aggregator2.process_article(article_jp)
    print(f"  Processed: {result}")
    print(f"  Records: {list(aggregator2.stats.records.keys())}")
    
    # Should have bigrams
    assert "日本" in aggregator2.stats.records, "Should have '日本' bigram"
    assert "本語" in aggregator2.stats.records, "Should have '本語' bigram"
    print("  ✅ PASS")
    
    # Test 6: Mixed English/Japanese
    print("\n[Test 6] Mixed English/Japanese")
    
    article_mixed = MockArticle(
        article_id="article_mixed_001",
        raw_text="I love 日本!",
        observations=[
            MockObservation((0, 10), "2026-01-20T14:00:00Z"),
        ]
    )
    
    result = aggregator2.process_article(article_mixed)
    print(f"  Records: {list(aggregator2.stats.records.keys())}")
    
    assert "love" in aggregator2.stats.records or "i" in aggregator2.stats.records
    print("  ✅ PASS")
    
    # Test 7: Source ID prefix (P1-4)
    print("\n[Test 7] Source ID prefix (P1-4)")
    
    apple_record = aggregator.stats.records["apple"]
    print(f"  last_seen_source: {apple_record.last_seen_source}")
    
    assert apple_record.last_seen_source.startswith("A:"), "Should have 'A:' prefix"
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nPhase 9-1 W1Aggregator is ready.")
