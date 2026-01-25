"""
ESDE Phase 9-2: W2 Aggregator
=============================
Conditional statistics aggregation from W0 observations.

Migration Phase 2 Update:
  - Policy-based condition signature generation from Substrate
  - Legacy fallback when substrate_ref is not available
  - Canonical JSON unified with Substrate spec

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

Migration Phase 2 P0 Compliance:
  P0-MIG-1: Policy優先、Legacy fallback
  P0-MIG-2: substrate_refが無い or registry.get()がNone → Legacy
  P0-MIG-3: Legacy hashもSubstrate準拠（separators無空白）

Spec: v5.4.2-P9.2 + Migration Phase 2 v0.2.1
"""

import os
import json
import math
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Any, Tuple, TYPE_CHECKING
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

# Migration Phase 2: Policy imports (conditional to avoid circular imports)
if TYPE_CHECKING:
    from .policies.base import BaseConditionPolicy
    from substrate import SubstrateRegistry


# ==========================================
# Constants
# ==========================================

DEFAULT_W2_RECORDS_PATH = "data/stats/w2_conditional_stats.jsonl"
DEFAULT_W2_CONDITIONS_PATH = "data/stats/w2_conditions.jsonl"

# Migration Phase 2: Canonical JSON settings (Substrate unified)
CANONICAL_SORT_KEYS = True
CANONICAL_ENSURE_ASCII = False
CANONICAL_SEPARATORS = (',', ':')  # No spaces - matches Substrate spec


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
    
    Migration Phase 2 Usage:
        from substrate import SubstrateRegistry
        from statistics.policies import StandardConditionPolicy
        
        registry = SubstrateRegistry()
        policy = StandardConditionPolicy(
            policy_id="legacy_migration_v1",
            target_keys=["legacy:source_type", "legacy:language_profile"],
        )
        
        aggregator = W2Aggregator(registry=registry, policy=policy)
        aggregator.process_article(article_record)  # Uses Policy if substrate_ref exists
    
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
        # === Migration Phase 2: Policy-based signature ===
        registry: Optional["SubstrateRegistry"] = None,
        policy: Optional["BaseConditionPolicy"] = None,
    ):
        """
        Initialize W2 Aggregator.
        
        Args:
            records_path: Path to W2 records JSONL
            conditions_path: Path to conditions registry JSONL
            tokenizer: W1Tokenizer instance (default: hybrid)
            min_token_length: Minimum token length
            min_count_to_persist: Minimum count to persist record (P1-3)
            registry: SubstrateRegistry instance (Migration Phase 2)
            policy: BaseConditionPolicy instance (Migration Phase 2)
        """
        self.records_path = records_path
        self.conditions_path = conditions_path
        self.tokenizer = tokenizer or get_w1_tokenizer("hybrid")
        self.min_token_length = min_token_length
        self.min_count_to_persist = min_count_to_persist
        
        # Migration Phase 2: Policy-based signature
        self.registry = registry
        self.policy = policy
        
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
    
    # ==========================================
    # Migration Phase 2: Policy-based Signature
    # ==========================================
    
    def _get_condition_signature(self, article) -> str:
        """
        Get condition signature using Policy (if available) or Legacy fallback.
        
        Migration Phase 2 P0 Compliance:
          P0-MIG-1: Policy優先、Legacy fallback
          P0-MIG-2: substrate_refが無い or registry.get()がNone → 必ずLegacy
        
        Args:
            article: ArticleRecord instance
            
        Returns:
            Condition signature (64 hex chars for Policy, 32 hex for Legacy)
        """
        # Case A: Substrate Ref exists and Policy is configured
        substrate_ref = getattr(article, 'substrate_ref', None)
        
        if substrate_ref and self.registry and self.policy:
            # Try to get ContextRecord from registry
            try:
                record = self.registry.get(substrate_ref)
                if record:
                    # Use Policy for signature generation
                    return self.policy.compute_signature(record)
            except Exception as e:
                # Log but don't fail - fall through to legacy
                print(f"[W2Aggregator] Policy signature failed, using legacy: {e}")
        
        # Case B: Legacy fallback
        factors = self._extract_legacy_factors(article)
        return self._compute_legacy_hash(factors)
    
    def _extract_legacy_factors(self, article) -> Dict[str, str]:
        """
        Extract condition factors from ArticleRecord (Legacy method).
        
        This is the original _extract_condition_factors logic.
        
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
    
    def _compute_legacy_hash(self, factors: Dict[str, str]) -> str:
        """
        Compute condition signature using Legacy method.
        
        Migration Phase 2 P0-MIG-3: Uses Substrate-unified Canonical JSON.
        
        NOTE: This differs from the original compute_condition_signature()
        which did not specify separators. For Phase 2 migration, we unify
        all hash computation to use the same canonical JSON spec.
        
        Args:
            factors: Condition factors dict
            
        Returns:
            Full SHA256 hex (64 chars)
        """
        # Use Substrate-unified canonical JSON
        canonical = json.dumps(
            factors,
            sort_keys=CANONICAL_SORT_KEYS,
            ensure_ascii=CANONICAL_ENSURE_ASCII,
            separators=CANONICAL_SEPARATORS,
        )
        # Return full hash (64 chars) for consistency with Policy
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def _extract_condition_factors(self, article) -> Dict[str, str]:
        """
        Extract condition factors from ArticleRecord.
        
        For W6 output/debugging. Uses Policy if available, otherwise Legacy.
        
        NOTE: This is NOT used for signature generation in Phase 2.
        Signature generation uses _get_condition_signature() which delegates
        to Policy.compute_signature() or _compute_legacy_hash().
        
        Args:
            article: ArticleRecord instance
            
        Returns:
            Condition factors dict
        """
        # Try Policy-based extraction
        substrate_ref = getattr(article, 'substrate_ref', None)
        
        if substrate_ref and self.registry and self.policy:
            try:
                record = self.registry.get(substrate_ref)
                if record:
                    return self.policy.extract_factors(record)
            except Exception:
                pass
        
        # Legacy fallback
        return self._extract_legacy_factors(article)
    
    # ==========================================
    # Statistics Computation
    # ==========================================
    
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
    
    # ==========================================
    # Article Processing
    # ==========================================
    
    def process_article(self, article) -> Dict[str, Any]:
        """
        Process an ArticleRecord and update W2 stats.
        
        Migration Phase 2: Uses _get_condition_signature() for Policy-based
        or Legacy signature generation.
        
        Args:
            article: ArticleRecord instance (from Phase 9-0)
            
        Returns:
            Processing summary dict
        """
        # Migration Phase 2: Use unified signature method
        condition_sig = self._get_condition_signature(article)
        
        # Extract factors for logging/debugging (not for signature)
        factors = self._extract_condition_factors(article)
        
        # Get or create condition entry
        # NOTE: For Phase 2, we create a new condition entry if signature doesn't exist
        if condition_sig not in self.stats.conditions:
            self.stats.conditions[condition_sig] = ConditionEntry(
                signature=condition_sig,
                factors=factors,
            )
            self.stats.total_conditions = len(self.stats.conditions)
        
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
            # Migration Phase 2: Indicate which path was used
            "signature_source": "policy" if (
                getattr(article, 'substrate_ref', None) and 
                self.registry and 
                self.policy
            ) else "legacy",
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
            # Migration Phase 2: Indicate if Policy is configured
            "policy_enabled": self.policy is not None,
            "registry_enabled": self.registry is not None,
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
    print("Phase 9-2 W2Aggregator Test (Migration Phase 2)")
    print("=" * 60)
    
    # Mock ArticleRecord for testing
    class MockObservation:
        def __init__(self, segment_span, timestamp):
            self.segment_span = segment_span
            self.timestamp = timestamp
    
    class MockArticle:
        def __init__(self, article_id, raw_text, observations, source_meta=None, 
                     ingestion_time=None, substrate_ref=None):
            self.article_id = article_id
            self.raw_text = raw_text
            self.observations = observations
            self.source_meta = source_meta or {}
            self.ingestion_time = ingestion_time or datetime.now(timezone.utc).isoformat()
            self.substrate_ref = substrate_ref  # Migration Phase 2
    
    # Test 1: Legacy mode (no registry/policy)
    print("\n[Test 1] Legacy mode (no registry/policy)")
    
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
    
    aggregator = W2Aggregator(
        records_path="/tmp/w2_test_records.jsonl",
        conditions_path="/tmp/w2_test_conditions.jsonl",
    )
    
    result = aggregator.process_article(article_news)
    
    print(f"  signature_source: {result['signature_source']}")
    print(f"  condition_signature: {result['condition_signature'][:32]}...")
    print(f"  signature length: {len(result['condition_signature'])}")
    
    assert result['signature_source'] == 'legacy', "Should use legacy path"
    assert len(result['condition_signature']) == 64, "Legacy hash should be 64 chars now"
    print("  ✅ Legacy mode works")
    
    # Test 2: Backward compatibility (no substrate_ref)
    print("\n[Test 2] Backward compatibility (no substrate_ref)")
    
    article_no_ref = MockArticle(
        article_id="article_no_ref",
        raw_text="Hello world!",
        observations=[MockObservation((0, 12), "2026-01-20T12:00:00Z")],
        source_meta={"source_type": "dialog"},
        substrate_ref=None,  # Explicitly None
    )
    
    result2 = aggregator.process_article(article_no_ref)
    
    print(f"  signature_source: {result2['signature_source']}")
    assert result2['signature_source'] == 'legacy'
    print("  ✅ Backward compatible")
    
    # Test 3: Canonical JSON verification (no spaces)
    print("\n[Test 3] Canonical JSON verification")
    
    factors = {"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"}
    
    # Use the aggregator's method
    canonical = json.dumps(
        factors,
        sort_keys=CANONICAL_SORT_KEYS,
        ensure_ascii=CANONICAL_ENSURE_ASCII,
        separators=CANONICAL_SEPARATORS,
    )
    
    print(f"  Canonical JSON: {canonical}")
    assert ': ' not in canonical, "Should not have space after colon"
    assert ', ' not in canonical, "Should not have space after comma"
    print("  ✅ No spaces in canonical JSON")
    
    # Test 4: Full hash (64 chars)
    print("\n[Test 4] Full hash (64 chars)")
    
    legacy_hash = aggregator._compute_legacy_hash(factors)
    
    print(f"  Legacy hash: {legacy_hash}")
    print(f"  Length: {len(legacy_hash)}")
    
    assert len(legacy_hash) == 64, f"Expected 64, got {len(legacy_hash)}"
    assert all(c in '0123456789abcdef' for c in legacy_hash), "Should be hex"
    print("  ✅ Full SHA256 (64 chars)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nMigration Phase 2 W2Aggregator ready.")
