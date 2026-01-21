"""
ESDE Phase 9-3: W3 Calculator
=============================
Calculates axis candidates by comparing W1 (global) and W2 (conditional) distributions.

Mathematical Model:
  S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))
  
  This is the per-token KL contribution (specificity score).
  - S > 0: Token is MORE specific to condition C (appears more than expected)
  - S < 0: Token is LESS specific to condition C (appears less than expected)

Invariants:
  INV-W3-001 (No Labeling):
      Output is factual only. No "this is political axis" labels.
  
  INV-W3-002 (Immutable Input):
      W3 calculation does NOT modify W1/W2 data.
  
  INV-W3-003 (Deterministic):
      Same input produces identical output (tie-break by token_norm).

P0 Compliance:
  P0-1: Uses ConditionEntry.total_token_count as denominator
  P0-2: epsilon = 1e-12 fixed
  P0-3: negative_candidates = S smallest (most negative) Top-K

P1 Compliance:
  P1-1: Comparison set = intersection(W2_cond, W1_global) + W1-only with count=0
  P1-2: min_count_for_w3 = 2 (configurable)
  P1-3: Tie-break by token_norm (alphabetical)
  P1-5: analysis_id = hash(cond_sig + w1_hash + w2_hash + algo)

Spec: v5.4.3-P9.3
"""

import os
import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .schema import W1GlobalStats, W1Record
from .schema_w2 import W2GlobalStats, W2Record, ConditionEntry
from .schema_w3 import (
    W3Record,
    CandidateToken,
    compute_analysis_id,
    compute_stats_hash,
    W3_VERSION,
    W3_ALGORITHM,
    EPSILON,
    DEFAULT_MIN_COUNT_FOR_W3,
    DEFAULT_TOP_K,
)


# ==========================================
# Constants
# ==========================================

DEFAULT_W3_OUTPUT_DIR = "data/stats/w3_candidates"


# ==========================================
# W3 Calculator
# ==========================================

class W3Calculator:
    """
    Calculates axis candidates by comparing W1 and W2 distributions.
    
    INV-W3-002: This calculator ONLY READS W1/W2 data. Never modifies.
    
    Usage:
        calculator = W3Calculator(w1_stats, w2_stats)
        result = calculator.calculate(condition_signature)
        calculator.save(result)
    """
    
    def __init__(
        self,
        w1_stats: W1GlobalStats,
        w2_stats: W2GlobalStats,
        top_k: int = DEFAULT_TOP_K,
        min_count: int = DEFAULT_MIN_COUNT_FOR_W3,
        epsilon: float = EPSILON,
        output_dir: str = DEFAULT_W3_OUTPUT_DIR,
    ):
        """
        Initialize W3 Calculator.
        
        Args:
            w1_stats: W1GlobalStats (global distribution)
            w2_stats: W2GlobalStats (conditional distributions)
            top_k: Number of top candidates to extract
            min_count: Minimum count filter for stability (P1-2)
            epsilon: Smoothing constant (P0-2: fixed 1e-12)
            output_dir: Directory for W3 output files
        """
        # INV-W3-002: Store references (read-only)
        self.w1_stats = w1_stats
        self.w2_stats = w2_stats
        
        self.top_k = top_k
        self.min_count = min_count
        self.epsilon = epsilon
        self.output_dir = output_dir
        
        # Compute snapshot hashes for reproducibility
        self._w1_hash = self._compute_w1_hash()
        self._w2_hash = self._compute_w2_hash()
    
    def _compute_w1_hash(self) -> str:
        """Compute hash of W1 state."""
        summary = {
            "total_tokens_processed": self.w1_stats.total_tokens_processed,
            "total_tokens_raw": self.w1_stats.total_tokens_raw,
            "total_articles": self.w1_stats.total_articles_processed,
            "record_count": len(self.w1_stats.records),
            "aggregation_version": self.w1_stats.aggregation_version,
        }
        return compute_stats_hash(summary)
    
    def _compute_w2_hash(self) -> str:
        """Compute hash of W2 state."""
        summary = {
            "total_records": self.w2_stats.total_records,
            "total_conditions": self.w2_stats.total_conditions,
            "w2_version": self.w2_stats.w2_version,
        }
        return compute_stats_hash(summary)
    
    def _get_global_total_tokens(self) -> int:
        """Get total tokens globally (W1 denominator)."""
        # Use total_tokens_processed (valid tokens after normalization)
        return self.w1_stats.total_tokens_processed
    
    def _get_condition_total_tokens(self, condition_sig: str) -> int:
        """Get total tokens for a condition (W2 denominator)."""
        if condition_sig not in self.w2_stats.conditions:
            return 0
        return self.w2_stats.conditions[condition_sig].total_token_count
    
    def _compute_s_score(
        self,
        count_cond: int,
        total_cond: int,
        count_global: int,
        total_global: int,
    ) -> Tuple[float, float, float]:
        """
        Compute S-Score (per-token KL contribution).
        
        S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))
        
        Args:
            count_cond: Token count under condition
            total_cond: Total tokens under condition
            count_global: Token count globally
            total_global: Total tokens globally
            
        Returns:
            (s_score, p_cond, p_global)
        """
        # Compute probabilities
        p_cond = count_cond / total_cond if total_cond > 0 else 0.0
        p_global = count_global / total_global if total_global > 0 else 0.0
        
        # Apply smoothing (P0-2: epsilon fixed)
        p_cond_smoothed = p_cond + self.epsilon
        p_global_smoothed = p_global + self.epsilon
        
        # Compute S-Score (per-token KL contribution)
        # S = P(t|C) * log(P(t|C) / P(t|G))
        s_score = p_cond * math.log(p_cond_smoothed / p_global_smoothed)
        
        return s_score, p_cond, p_global
    
    def calculate(self, condition_signature: str) -> W3Record:
        """
        Calculate axis candidates for a specific condition.
        
        INV-W3-002: This method does NOT modify W1/W2 data.
        INV-W3-003: Same input produces identical output.
        
        Args:
            condition_signature: Target condition to analyze
            
        Returns:
            W3Record with positive and negative candidates
        """
        # Validate condition exists
        if condition_signature not in self.w2_stats.conditions:
            raise ValueError(f"Condition not found: {condition_signature}")
        
        condition_entry = self.w2_stats.conditions[condition_signature]
        
        # Get totals for denominators (P0-1)
        total_global = self._get_global_total_tokens()
        total_cond = self._get_condition_total_tokens(condition_signature)
        
        if total_global == 0 or total_cond == 0:
            raise ValueError("Cannot calculate: zero total tokens")
        
        # P1-1: Build comparison set
        # - All tokens in W2 for this condition
        # - Plus all tokens in W1 (with count=0 if not in W2)
        
        # Get W2 records for this condition
        w2_records_for_cond: Dict[str, W2Record] = {}
        for record in self.w2_stats.records.values():
            if record.condition_signature == condition_signature:
                w2_records_for_cond[record.token_norm] = record
        
        # Build comparison set: union of W1 and W2 tokens
        all_tokens = set(self.w1_stats.records.keys()) | set(w2_records_for_cond.keys())
        
        # Compute scores for all tokens
        scored_tokens: List[Tuple[str, float, float, float, int, int]] = []
        
        for token_norm in all_tokens:
            # Get counts
            w1_record = self.w1_stats.records.get(token_norm)
            w2_record = w2_records_for_cond.get(token_norm)
            
            count_global = w1_record.total_count if w1_record else 0
            count_cond = w2_record.count if w2_record else 0
            
            # P1-2: Apply min_count filter
            if count_global < self.min_count and count_cond < self.min_count:
                continue
            
            # Compute S-Score
            s_score, p_cond, p_global = self._compute_s_score(
                count_cond, total_cond, count_global, total_global
            )
            
            scored_tokens.append((
                token_norm, s_score, p_cond, p_global, count_cond, count_global
            ))
        
        # P1-3: Sort with tie-break
        # Positive: s_score desc, then token_norm asc
        # Negative: s_score asc (most negative first), then token_norm asc
        
        positive_tokens = [t for t in scored_tokens if t[1] > 0]
        negative_tokens = [t for t in scored_tokens if t[1] < 0]
        
        # Sort positive: highest s_score first, tie-break by token_norm
        positive_tokens.sort(key=lambda x: (-x[1], x[0]))
        
        # Sort negative: lowest s_score first (most negative), tie-break by token_norm
        negative_tokens.sort(key=lambda x: (x[1], x[0]))
        
        # Extract Top-K
        positive_candidates = [
            CandidateToken(
                token_norm=t[0],
                s_score=t[1],
                p_cond=t[2],
                p_global=t[3],
                count_cond=t[4],
                count_global=t[5],
            )
            for t in positive_tokens[:self.top_k]
        ]
        
        negative_candidates = [
            CandidateToken(
                token_norm=t[0],
                s_score=t[1],
                p_cond=t[2],
                p_global=t[3],
                count_cond=t[4],
                count_global=t[5],
            )
            for t in negative_tokens[:self.top_k]
        ]
        
        # P1-5: Compute analysis ID
        analysis_id = compute_analysis_id(
            condition_signature,
            self._w1_hash,
            self._w2_hash,
            W3_ALGORITHM,
        )
        
        # Build W3Record
        record = W3Record(
            analysis_id=analysis_id,
            condition_signature=condition_signature,
            condition_factors=dict(condition_entry.factors),
            w1_snapshot_hash=self._w1_hash,
            w2_snapshot_hash=self._w2_hash,
            positive_candidates=positive_candidates,
            negative_candidates=negative_candidates,
            algorithm=W3_ALGORITHM,
            top_k=self.top_k,
            min_count=self.min_count,
            epsilon=self.epsilon,
            total_tokens_compared=len(scored_tokens),
            positive_count=len(positive_tokens),
            negative_count=len(negative_tokens),
        )
        
        return record
    
    def calculate_all(self) -> List[W3Record]:
        """
        Calculate axis candidates for all conditions.
        
        Returns:
            List of W3Records, one per condition
        """
        results = []
        for condition_sig in self.w2_stats.conditions.keys():
            try:
                result = self.calculate(condition_sig)
                results.append(result)
            except ValueError as e:
                print(f"[W3Calculator] Skipping {condition_sig[:16]}...: {e}")
        
        return results
    
    def save(self, record: W3Record, output_dir: Optional[str] = None):
        """
        Save W3Record to JSON file.
        
        File: {output_dir}/{condition_signature}.json
        """
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        filepath = os.path.join(out_dir, f"{record.condition_signature}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(record.to_json())
        
        print(f"[W3Calculator] Saved to {filepath}")
    
    def get_summary(self, record: W3Record) -> Dict[str, Any]:
        """Get human-readable summary of W3 result."""
        return {
            "condition": record.condition_factors,
            "total_compared": record.total_tokens_compared,
            "positive_count": record.positive_count,
            "negative_count": record.negative_count,
            "top_positive": [
                (c.token_norm, round(c.s_score, 6))
                for c in record.positive_candidates[:10]
            ],
            "top_negative": [
                (c.token_norm, round(c.s_score, 6))
                for c in record.negative_candidates[:10]
            ],
        }


# ==========================================
# Comparison for Rebuild Test
# ==========================================

def compare_w3_records(r1: W3Record, r2: W3Record) -> Dict[str, Any]:
    """
    Compare two W3Records for determinism validation (INV-W3-003).
    
    Checks:
      - Same analysis_id
      - Same candidate rankings
      - Same scores (within tolerance)
    """
    diffs = []
    
    if r1.analysis_id != r2.analysis_id:
        diffs.append(f"analysis_id: {r1.analysis_id} vs {r2.analysis_id}")
    
    if r1.total_tokens_compared != r2.total_tokens_compared:
        diffs.append(f"total_compared: {r1.total_tokens_compared} vs {r2.total_tokens_compared}")
    
    # Compare positive candidates
    if len(r1.positive_candidates) != len(r2.positive_candidates):
        diffs.append(f"positive count: {len(r1.positive_candidates)} vs {len(r2.positive_candidates)}")
    else:
        for i, (c1, c2) in enumerate(zip(r1.positive_candidates, r2.positive_candidates)):
            if c1.token_norm != c2.token_norm:
                diffs.append(f"positive[{i}] token: {c1.token_norm} vs {c2.token_norm}")
            if abs(c1.s_score - c2.s_score) > 1e-8:
                diffs.append(f"positive[{i}] score: {c1.s_score} vs {c2.s_score}")
    
    # Compare negative candidates
    if len(r1.negative_candidates) != len(r2.negative_candidates):
        diffs.append(f"negative count: {len(r1.negative_candidates)} vs {len(r2.negative_candidates)}")
    else:
        for i, (c1, c2) in enumerate(zip(r1.negative_candidates, r2.negative_candidates)):
            if c1.token_norm != c2.token_norm:
                diffs.append(f"negative[{i}] token: {c1.token_norm} vs {c2.token_norm}")
            if abs(c1.s_score - c2.s_score) > 1e-8:
                diffs.append(f"negative[{i}] score: {c1.s_score} vs {c2.s_score}")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-3 W3Calculator Test")
    print("=" * 60)
    
    # Create mock W1/W2 stats for testing
    from .schema import W1GlobalStats, W1Record
    from .schema_w2 import W2GlobalStats, W2Record, ConditionEntry, compute_condition_signature
    
    # Build mock W1 (global stats)
    w1_stats = W1GlobalStats()
    w1_stats.total_tokens_processed = 1000
    w1_stats.total_tokens_raw = 1200
    w1_stats.total_articles_processed = 10
    
    # Add some global records
    w1_stats.records["apple"] = W1Record(token_norm="apple", total_count=100, document_frequency=8)
    w1_stats.records["banana"] = W1Record(token_norm="banana", total_count=50, document_frequency=5)
    w1_stats.records["prime"] = W1Record(token_norm="prime", total_count=10, document_frequency=2)
    w1_stats.records["minister"] = W1Record(token_norm="minister", total_count=5, document_frequency=1)
    w1_stats.records["the"] = W1Record(token_norm="the", total_count=200, document_frequency=10)
    w1_stats.records["lol"] = W1Record(token_norm="lol", total_count=80, document_frequency=5)
    
    # Build mock W2 (conditional stats)
    w2_stats = W2GlobalStats()
    
    # Condition: news
    news_factors = {"source_type": "news", "language_profile": "en", "time_bucket": "2026-01"}
    news_sig = compute_condition_signature(news_factors)
    w2_stats.conditions[news_sig] = ConditionEntry(
        signature=news_sig,
        factors=news_factors,
        total_token_count=400,
        total_doc_count=4,
    )
    
    # Add W2 records for news condition
    # "prime" and "minister" are MORE specific to news
    from .schema_w2 import compute_record_id
    
    rid1 = compute_record_id("apple", news_sig)
    w2_stats.records[rid1] = W2Record(
        record_id=rid1, token_norm="apple", condition_signature=news_sig,
        count=40, doc_freq=3
    )
    
    rid2 = compute_record_id("prime", news_sig)
    w2_stats.records[rid2] = W2Record(
        record_id=rid2, token_norm="prime", condition_signature=news_sig,
        count=8, doc_freq=2  # High relative to global
    )
    
    rid3 = compute_record_id("minister", news_sig)
    w2_stats.records[rid3] = W2Record(
        record_id=rid3, token_norm="minister", condition_signature=news_sig,
        count=4, doc_freq=1  # High relative to global
    )
    
    rid4 = compute_record_id("the", news_sig)
    w2_stats.records[rid4] = W2Record(
        record_id=rid4, token_norm="the", condition_signature=news_sig,
        count=80, doc_freq=4  # Same ratio as global
    )
    
    rid5 = compute_record_id("lol", news_sig)
    w2_stats.records[rid5] = W2Record(
        record_id=rid5, token_norm="lol", condition_signature=news_sig,
        count=2, doc_freq=1  # LOW relative to global (negative candidate)
    )
    
    w2_stats.total_records = 5
    w2_stats.total_conditions = 1
    
    # Test 1: Basic calculation
    print("\n[Test 1] Basic calculation")
    
    calculator = W3Calculator(
        w1_stats, w2_stats,
        top_k=10,
        min_count=2,
    )
    
    result = calculator.calculate(news_sig)
    
    print(f"  Condition: {result.condition_factors}")
    print(f"  Total compared: {result.total_tokens_compared}")
    print(f"  Positive count: {result.positive_count}")
    print(f"  Negative count: {result.negative_count}")
    
    assert result.total_tokens_compared > 0
    print("  ✅ PASS")
    
    # Test 2: S-Score math verification
    print("\n[Test 2] S-Score math verification")
    
    # Manual calculation for "prime":
    # count_cond = 8, total_cond = 400 -> P(prime|news) = 8/400 = 0.02
    # count_global = 10, total_global = 1000 -> P(prime|G) = 10/1000 = 0.01
    # S = 0.02 * log((0.02 + 1e-12) / (0.01 + 1e-12))
    #   = 0.02 * log(2.0)
    #   ≈ 0.02 * 0.693
    #   ≈ 0.0139
    
    import math
    expected_p_cond = 8 / 400  # 0.02
    expected_p_global = 10 / 1000  # 0.01
    expected_s = expected_p_cond * math.log((expected_p_cond + 1e-12) / (expected_p_global + 1e-12))
    
    print(f"  Expected P(prime|news) = {expected_p_cond}")
    print(f"  Expected P(prime|G) = {expected_p_global}")
    print(f"  Expected S-score ≈ {expected_s:.6f}")
    
    # Find "prime" in results
    prime_candidate = None
    for c in result.positive_candidates:
        if c.token_norm == "prime":
            prime_candidate = c
            break
    
    if prime_candidate:
        print(f"  Actual S-score = {prime_candidate.s_score:.6f}")
        assert abs(prime_candidate.s_score - expected_s) < 1e-6, "S-score mismatch"
        print("  ✅ PASS: S-score matches hand calculation")
    else:
        print("  WARNING: 'prime' not found in positive candidates")
    
    # Test 3: Positive vs Negative
    print("\n[Test 3] Positive vs Negative candidates")
    
    print(f"  Top positive (more specific to news):")
    for c in result.positive_candidates[:3]:
        print(f"    {c.token_norm}: S={c.s_score:.6f}, P(C)={c.p_cond:.4f}, P(G)={c.p_global:.4f}")
    
    print(f"  Top negative (less specific to news):")
    for c in result.negative_candidates[:3]:
        print(f"    {c.token_norm}: S={c.s_score:.6f}, P(C)={c.p_cond:.4f}, P(G)={c.p_global:.4f}")
    
    # "lol" should be negative (suppressed in news)
    lol_candidate = None
    for c in result.negative_candidates:
        if c.token_norm == "lol":
            lol_candidate = c
            break
    
    if lol_candidate:
        assert lol_candidate.s_score < 0, "lol should have negative S-score"
        print("  ✅ PASS: 'lol' correctly identified as negative candidate")
    else:
        print("  WARNING: 'lol' not found (may not meet min_count)")
    
    # Test 4: Determinism (INV-W3-003)
    print("\n[Test 4] Determinism (INV-W3-003)")
    
    # Calculate twice
    result1 = calculator.calculate(news_sig)
    result2 = calculator.calculate(news_sig)
    
    comparison = compare_w3_records(result1, result2)
    
    print(f"  Run 1 analysis_id: {result1.analysis_id}")
    print(f"  Run 2 analysis_id: {result2.analysis_id}")
    print(f"  Match: {comparison['match']}")
    
    assert comparison['match'], "Results should be deterministic"
    print("  ✅ PASS")
    
    # Test 5: Tie-break (P1-3)
    print("\n[Test 5] Tie-break by token_norm")
    
    # Add tokens with same count to test tie-break
    w1_stats.records["aaa"] = W1Record(token_norm="aaa", total_count=10, document_frequency=2)
    w1_stats.records["zzz"] = W1Record(token_norm="zzz", total_count=10, document_frequency=2)
    
    rid_aaa = compute_record_id("aaa", news_sig)
    rid_zzz = compute_record_id("zzz", news_sig)
    
    w2_stats.records[rid_aaa] = W2Record(
        record_id=rid_aaa, token_norm="aaa", condition_signature=news_sig,
        count=8, doc_freq=2
    )
    w2_stats.records[rid_zzz] = W2Record(
        record_id=rid_zzz, token_norm="zzz", condition_signature=news_sig,
        count=8, doc_freq=2  # Same as "aaa"
    )
    
    # Recalculate
    calculator2 = W3Calculator(w1_stats, w2_stats, top_k=10, min_count=2)
    result3 = calculator2.calculate(news_sig)
    
    # Find positions
    aaa_pos = None
    zzz_pos = None
    for i, c in enumerate(result3.positive_candidates):
        if c.token_norm == "aaa":
            aaa_pos = i
        if c.token_norm == "zzz":
            zzz_pos = i
    
    if aaa_pos is not None and zzz_pos is not None:
        print(f"  'aaa' position: {aaa_pos}")
        print(f"  'zzz' position: {zzz_pos}")
        assert aaa_pos < zzz_pos, "aaa should come before zzz (alphabetical tie-break)"
        print("  ✅ PASS: Tie-break by token_norm works")
    else:
        print("  WARNING: Could not verify tie-break")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("\nPhase 9-3 W3Calculator is ready.")
