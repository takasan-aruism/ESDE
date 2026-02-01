"""
ESDE Phase 9: W4 Vector Projector
====================================

Vector-mode W4: Computes pairwise similarity between condition mean vectors.

Unlike token-mode W4 (resonance vectors from S-Score projections), this:
  - Uses μ_c (condition mean vectors) directly
  - Computes cosine similarity between all pairs
  - Feeds into W5 clustering (same interface as token-mode W4)

Design source:
  - GPT audit: "W4 similarity = μ_doc cosine (not Δ_doc cosine)"
  - GPT audit: "Δ is for explanation (W3), μ is for distance (W4)"

Input:  W3VectorResult (condition mean vectors)
Output: Pairwise similarity matrix + per-condition vectors (W5 compatible)

Spec: Phase 9 W4 Vector v1.0
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple

from .w3_vector import W3VectorResult


# ==========================================
# Constants
# ==========================================

W4_VECTOR_VERSION = "v1.1"  # v1.0→v1.1: μ_c cosine → z-score cosine


# ==========================================
# Data Structures (W5-compatible)
# ==========================================

@dataclass
class W4VectorRecord:
    """
    W4 vector record for a single condition.
    
    Compatible with W5 clustering interface:
      - article_id → condition_id
      - resonance_vector → mean_vector
    """
    condition_id: str
    mean_vector: List[float]
    token_count: int


@dataclass 
class W4VectorResult:
    """Complete W4 vector analysis result."""
    records: Dict[str, W4VectorRecord]
    similarities: List[Dict[str, Any]]  # [{a, b, similarity}, ...]
    version: str = W4_VECTOR_VERSION


# ==========================================
# Math
# ==========================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value in [-1, 1]. Returns 0.0 if either vector is zero/near-zero.
    Handles NaN/Inf gracefully.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    
    result = dot / (norm_a * norm_b)
    
    # Clamp and sanitize
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return max(-1.0, min(1.0, result))


# ==========================================
# W4 Vector Projector
# ==========================================

class W4VectorProjector:
    """
    Computes pairwise cosine similarities from condition z-score vectors.
    
    W4 role in vector mode: "How similar are conditions' deviation profiles?"
    
    Design evolution:
      v1.0: Used μ_c (raw mean vectors) → all similarities ~1.0000
            because baseline (English Wikipedia) dominates.
      v1.1: Uses z_c (z-score vectors from W3) → compares "direction of 
            deviation from global mean", which is where the actual signal is.
    
    The z-score normalizes dimension scales, preventing high-magnitude
    dimensions (e.g., word_length ~4.5) from drowning out low-magnitude
    ones (e.g., valence ~0.01).
    """
    
    def project(self, w3_result: W3VectorResult) -> W4VectorResult:
        """
        Compute pairwise similarities using z-score vectors.
        
        Args:
            w3_result: W3 vector profiles with z-score vectors
            
        Returns:
            W4VectorResult with records and similarity pairs
        """
        # Build records using z-score vectors (not raw means)
        records = {}
        for cid, profile in w3_result.profiles.items():
            records[cid] = W4VectorRecord(
                condition_id=cid,
                mean_vector=profile.z_score_vector,  # z-score, not μ
                token_count=profile.token_count,
            )
        
        # Compute all pairs
        cids = sorted(records.keys())
        similarities = []
        
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a_id, b_id = cids[i], cids[j]
                sim = cosine_similarity(
                    records[a_id].mean_vector,
                    records[b_id].mean_vector,
                )
                similarities.append({
                    "a": a_id,
                    "b": b_id,
                    "similarity": round(sim, 6),
                })
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: -x["similarity"])
        
        return W4VectorResult(
            records=records,
            similarities=similarities,
        )


# ==========================================
# W5 Adapter (convert to W5-compatible format)
# ==========================================

def convert_to_w5_input(
    w4_result: W4VectorResult,
) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    """
    Convert W4 vector results to W5-compatible format.
    
    W5 expects:
      - vectors: {article_id: resonance_vector}
      - similarities: [{a, b, similarity}]
    
    Returns:
        (vectors_dict, similarities_list)
    """
    vectors = {
        cid: rec.mean_vector
        for cid, rec in w4_result.records.items()
    }
    
    return vectors, w4_result.similarities


# ==========================================
# Export Helpers
# ==========================================

def export_vector_similarities(result: W4VectorResult) -> Dict[str, Any]:
    """Export W4 vector result to JSON-serializable dict."""
    return {
        "version": result.version,
        "condition_count": len(result.records),
        "similarity_pairs": len(result.similarities),
        "records": {
            cid: {
                "mean_vector": rec.mean_vector,
                "token_count": rec.token_count,
            }
            for cid, rec in result.records.items()
        },
        "top_similarities": result.similarities[:20],
        "bottom_similarities": result.similarities[-10:] if len(result.similarities) > 10 else [],
    }
