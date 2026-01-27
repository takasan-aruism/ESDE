"""
ESDE Phase 9-3: W3 Calculator
=============================
Calculates axis candidates by comparing W1 (global) and W2 (conditional) distributions.

Mathematical Model:
  S(t, C) = P(t|C) * log((P(t|C) + ε) / (P(t|G) + ε))
  
  This is the per-token KL contribution (specificity score).
  - S > 0: Token is MORE specific to condition C (appears more than expected)
  - S < 0: Token is LESS specific to condition C (appears less than expected)

Migration Phase 3 Update:
  - analysis_scope_id for statistics isolation
  - Path resolution via utils.resolve_w3_output_dir()
  - P0-1: unknown_scope fallback prohibited in Policy mode

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

Migration Phase 3 P0 Compliance:
  P0-MIG3-1: Policy mode で analysis_scope_id が必須
  P0-MIG3-2: パストラバーサル対策（ID検証）

Spec: v5.4.3-P9.3 + Migration Phase 3 v0.3.1
"""

import os
import math
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING

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

# Migration Phase 3: Policy imports (conditional to avoid circular imports)
if TYPE_CHECKING:
    from .policies.base import BaseConditionPolicy


# ==========================================
# Constants
# ==========================================

DEFAULT_W3_OUTPUT_DIR = "data/stats/w3_candidates"
DEFAULT_W3_BASE_DIR = "data/stats"


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
    
    Migration Phase 3 Usage:
        # Policy mode with scope isolation
        calculator = W3Calculator(
            w1_stats, w2_stats,
            policy=policy,
            analysis_scope_id="run_20260125",  # Required in Policy mode
        )
        # Files saved to: data/stats/policies/{policy_id}/{version}/{scope}/w3_candidates/
    """
    
    def __init__(
        self,
        w1_stats: W1GlobalStats,
        w2_stats: W2GlobalStats,
        top_k: int = DEFAULT_TOP_K,
        min_count: int = DEFAULT_MIN_COUNT_FOR_W3,
        epsilon: float = EPSILON,
        output_dir: Optional[str] = None,
        # === Migration Phase 3: Statistics isolation ===
        policy: Optional["BaseConditionPolicy"] = None,
        analysis_scope_id: Optional[str] = None,
        base_dir: str = DEFAULT_W3_BASE_DIR,
    ):
        """
        Initialize W3 Calculator.
        
        Args:
            w1_stats: W1GlobalStats (global distribution)
            w2_stats: W2GlobalStats (conditional distributions)
            top_k: Number of top candidates to extract
            min_count: Minimum count filter for stability (P1-2)
            epsilon: Smoothing constant (P0-2: fixed 1e-12)
            output_dir: Directory for W3 output files (overrides path resolution)
            policy: BaseConditionPolicy instance (Migration Phase 3)
            analysis_scope_id: Scope identifier for statistics isolation (Migration Phase 3)
            base_dir: Base directory for statistics data (Migration Phase 3)
        
        Raises:
            ScopeValidationError: If policy is set but analysis_scope_id is not provided
        """
        # INV-W3-002: Store references (read-only)
        self.w1_stats = w1_stats
        self.w2_stats = w2_stats
        
        self.top_k = top_k
        self.min_count = min_count
        self.epsilon = epsilon
        
        # Migration Phase 3: Statistics isolation
        self.policy = policy
        self.analysis_scope_id = analysis_scope_id
        self.base_dir = base_dir
        
        # Resolve output directory
        self.output_dir = self._resolve_output_dir(output_dir)
        
        # Compute snapshot hashes for reproducibility
        self._w1_hash = self._compute_w1_hash()
        self._w2_hash = self._compute_w2_hash()
    
    def _resolve_output_dir(self, explicit_output_dir: Optional[str]) -> str:
        """
        Resolve output directory for W3 data.
        
        Migration Phase 3: Uses utils.resolve_w3_output_dir() for Policy mode.
        
        Priority:
          1. Explicit output_dir (if provided)
          2. Path resolution (Policy mode → isolated directory)
          3. Default path (Legacy mode)
        
        Args:
            explicit_output_dir: Explicit output directory (or None)
            
        Returns:
            Resolved output directory path
            
        Raises:
            ScopeValidationError: If Policy mode but no scope_id
        """
        # If output_dir is explicitly provided, use it directly
        if explicit_output_dir:
            return explicit_output_dir
        
        # Import here to avoid circular imports
        from .utils import resolve_w3_output_dir, ScopeValidationError
        
        # Use path resolver (validates scope in Policy mode)
        try:
            return resolve_w3_output_dir(
                self.base_dir,
                self.policy,
                self.analysis_scope_id,
            )
        except ScopeValidationError:
            # Re-raise with more context
            raise ScopeValidationError(
                f"W3Calculator requires analysis_scope_id when policy is set. "
                f"Policy: {self.policy.policy_id if self.policy else 'None'}, "
                f"Scope: {self.analysis_scope_id}"
            )
    
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
            
            # P1-2: Filter by min_count (stability)
            if count_cond < self.min_count and count_global < self.min_count:
                continue
            
            # Compute S-Score
            s_score, p_cond, p_global = self._compute_s_score(
                count_cond, total_cond,
                count_global, total_global,
            )
            
            scored_tokens.append((
                token_norm,
                s_score,
                p_cond,
                p_global,
                count_cond,
                count_global,
            ))
        
        # Sort: Primary by s_score (desc), Secondary by token_norm (P1-3 tie-break)
        # For positive: highest first
        # For negative: lowest first (most negative)
        
        # Extract positive (S > 0)
        positive_tokens = [t for t in scored_tokens if t[1] > 0]
        positive_tokens.sort(key=lambda x: (-x[1], x[0]))  # Desc by score, asc by name
        
        # Extract negative (S < 0) - P0-3
        negative_tokens = [t for t in scored_tokens if t[1] < 0]
        negative_tokens.sort(key=lambda x: (x[1], x[0]))  # Asc by score (most negative first)
        
        # Build CandidateToken lists
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
    
    def save_all(self, records: List[W3Record], output_dir: Optional[str] = None):
        """Save multiple W3Records."""
        for record in records:
            self.save(record, output_dir)
    
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
            # Migration Phase 3: Include scope info
            "analysis_scope_id": self.analysis_scope_id,
            "output_dir": self.output_dir,
        }
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of calculator configuration."""
        return {
            "w1_hash": self._w1_hash,
            "w2_hash": self._w2_hash,
            "top_k": self.top_k,
            "min_count": self.min_count,
            "epsilon": self.epsilon,
            "conditions_available": len(self.w2_stats.conditions),
            # Migration Phase 3: Policy info
            "policy_enabled": self.policy is not None,
            "analysis_scope_id": self.analysis_scope_id,
            "output_dir": self.output_dir,
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
