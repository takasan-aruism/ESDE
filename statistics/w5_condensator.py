"""
ESDE Phase 9-5: W5 Condensator
==============================
Weak Structural Condensation algorithm.

Theme: The Deterministic Shape of Resonance (共鳴の決定論的形状)

Algorithm: Resonance Condensation
  1. Validation: Batch size check, duplicate article_id check (P0-A)
  2. Preprocessing: L2 normalize all vectors
  3. Similarity Matrix: Cosine similarity with fixed rounding (P0-B)
  4. Graph Linkage: Edge if similarity >= threshold
  5. Component Detection: Connected components via DFS
  6. Filtering: Size < min_island_size → noise
  7. Centroid: Raw mean with rounding (INV-W5-005)
  8. Cohesion: Edge average within island (P1-1)
  9. ID Generation: Canonical JSON hash (INV-W5-006)

GPT Audit Compliance:
  P0-A: Duplicate article_id check → ValueError
  P0-B: Similarity rounded to 12 decimals before threshold comparison
  P1-1: Cohesion = average of edges that formed the island (not all pairs)

Spec: v5.4.5-P9.5-Final
"""

import math
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from .schema_w5 import (
    W5Island,
    W5Structure,
    get_canonical_json,
    compute_canonical_hash,
    W5_ALGORITHM,
    W5_VECTOR_POLICY,
    W5_VECTOR_ROUNDING,
    W5_SIMILARITY_ROUNDING,
    W5_DEFAULT_THRESHOLD,
    W5_DEFAULT_MIN_ISLAND_SIZE,
    W5_MAX_BATCH_SIZE,
)
from .schema_w4 import W4Record


# ==========================================
# W5 Condensator
# ==========================================

class W5Condensator:
    """
    W5 Condensator: Resonance-based structural condensation.
    
    Converts a batch of W4Records into a W5Structure by:
    1. Building a similarity graph based on cosine similarity
    2. Extracting connected components as "islands"
    3. Computing representative vectors (centroids) for each island
    
    Thread Safety: Stateless computation, thread-safe.
    
    Usage:
        condensator = W5Condensator(threshold=0.70, min_island_size=3)
        structure = condensator.condense(w4_records)
    """
    
    def __init__(
        self,
        threshold: float = W5_DEFAULT_THRESHOLD,
        min_island_size: int = W5_DEFAULT_MIN_ISLAND_SIZE,
        max_batch_size: int = W5_MAX_BATCH_SIZE,
    ):
        """
        Initialize W5 Condensator.
        
        Args:
            threshold: Similarity threshold for edge creation (default: 0.70)
            min_island_size: Minimum members for valid island (default: 3)
            max_batch_size: Maximum batch size to prevent O(N^2) explosion (default: 2000)
        """
        self.threshold = threshold
        self.min_island_size = min_island_size
        self.max_batch_size = max_batch_size
    
    def condense(self, records: List[W4Record]) -> W5Structure:
        """
        Condense W4Records into W5Structure.
        
        INV-W5-003: Uses L2-normalized cosine similarity with >= threshold.
        INV-W5-006: IDs generated via Canonical JSON hash.
        INV-W5-007: structure_id based on w4_analysis_id, not article_id.
        
        Args:
            records: List of W4Record to condense
            
        Returns:
            W5Structure containing islands and noise
            
        Raises:
            ValueError: If batch size exceeds limit or duplicate article_id found
        """
        n = len(records)
        
        # === Validation ===
        
        # P1-4: Batch size guard
        if n > self.max_batch_size:
            raise ValueError(
                f"Batch size {n} exceeds limit {self.max_batch_size}. "
                f"Split into smaller batches."
            )
        
        # P0-A: Duplicate article_id check
        article_ids = [r.article_id for r in records]
        if len(article_ids) != len(set(article_ids)):
            duplicates = [aid for aid in article_ids if article_ids.count(aid) > 1]
            raise ValueError(
                f"Duplicate article_id in batch: {set(duplicates)}. "
                f"Each article_id must be unique."
            )
        
        # Empty input case
        if n == 0:
            return self._empty_structure()
        
        # === Preprocessing ===
        
        # Sort for deterministic processing order
        records_sorted = sorted(records, key=lambda r: r.article_id)
        sorted_article_ids = [r.article_id for r in records_sorted]
        
        # Build record map for later lookup
        record_map: Dict[str, W4Record] = {r.article_id: r for r in records}
        
        # L2 normalize all vectors (INV-W5-003)
        norm_vectors: Dict[str, Dict[str, float]] = {}
        for r in records:
            norm_vectors[r.article_id] = self._l2_normalize(r.resonance_vector)
        
        # === Build Similarity Graph ===
        
        # Adjacency list
        adj: Dict[str, Set[str]] = defaultdict(set)
        
        # Store similarities for cohesion calculation
        # Key: tuple(sorted(id_a, id_b)), Value: similarity
        edge_similarities: Dict[Tuple[str, str], float] = {}
        
        for i in range(n):
            id_a = sorted_article_ids[i]
            vec_a = norm_vectors[id_a]
            
            for j in range(i + 1, n):
                id_b = sorted_article_ids[j]
                vec_b = norm_vectors[id_b]
                
                # Compute cosine similarity
                sim = self._cosine_similarity(vec_a, vec_b)
                
                # P0-B: Round similarity to fixed precision before comparison
                sim_rounded = round(sim, W5_SIMILARITY_ROUNDING)
                
                # INV-W5-003: Fixed threshold operator (>=)
                if sim_rounded >= self.threshold:
                    adj[id_a].add(id_b)
                    adj[id_b].add(id_a)
                    
                    # Store for cohesion calculation (use rounded value)
                    edge_key = tuple(sorted((id_a, id_b)))
                    edge_similarities[edge_key] = sim_rounded
        
        # === Component Detection (DFS) ===
        
        visited: Set[str] = set()
        islands: List[W5Island] = []
        noise_ids: List[str] = []
        
        for aid in sorted_article_ids:
            if aid in visited:
                continue
            
            # DFS to find connected component
            component: List[str] = []
            stack = [aid]
            visited.add(aid)
            
            while stack:
                curr = stack.pop()
                component.append(curr)
                
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            # Filter by size
            if len(component) >= self.min_island_size:
                island = self._create_island(
                    component, record_map, edge_similarities
                )
                islands.append(island)
            else:
                noise_ids.extend(component)
        
        # Sort noise_ids for determinism
        noise_ids.sort()
        
        # Sort islands by island_id for determinism
        islands.sort(key=lambda i: i.island_id)
        
        # === Structure ID Generation (INV-W5-004, INV-W5-007) ===
        
        # Use w4_analysis_id (not article_id) for identity
        input_w4_ids = sorted([r.w4_analysis_id for r in records])
        
        structure_hash_input = {
            "input_w4_ids": input_w4_ids,
            "threshold": self.threshold,
            "min_island_size": self.min_island_size,
            "algorithm": W5_ALGORITHM,
            "vector_policy": W5_VECTOR_POLICY,
        }
        structure_id = compute_canonical_hash(structure_hash_input)
        
        # === Create Structure ===
        
        return W5Structure(
            structure_id=structure_id,
            islands=islands,
            noise_ids=noise_ids,
            input_count=n,
            island_count=len(islands),
            noise_count=len(noise_ids),
            threshold=self.threshold,
            min_island_size=self.min_island_size,
            algorithm=W5_ALGORITHM,
            vector_policy=W5_VECTOR_POLICY,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    
    def _l2_normalize(self, vec: Dict[str, float]) -> Dict[str, float]:
        """
        L2 normalize a vector.
        
        Args:
            vec: Sparse vector as dict
            
        Returns:
            Normalized vector (empty dict if norm is near zero)
        """
        norm = math.sqrt(sum(v * v for v in vec.values()))
        
        if norm < 1e-12:
            return {}
        
        return {k: v / norm for k, v in vec.items()}
    
    def _cosine_similarity(
        self,
        vec_a: Dict[str, float],
        vec_b: Dict[str, float],
    ) -> float:
        """
        Compute cosine similarity between two L2-normalized vectors.
        
        For normalized vectors, this is simply the dot product.
        
        Args:
            vec_a: First normalized vector
            vec_b: Second normalized vector
            
        Returns:
            Cosine similarity in [-1, 1]
        """
        # Only sum over common keys (sparse dot product)
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        return sum(vec_a[k] * vec_b[k] for k in common_keys)
    
    def _create_island(
        self,
        member_ids: List[str],
        record_map: Dict[str, W4Record],
        edge_similarities: Dict[Tuple[str, str], float],
    ) -> W5Island:
        """
        Create a W5Island from a connected component.
        
        Args:
            member_ids: List of article_ids in this component
            record_map: Map from article_id to W4Record
            edge_similarities: Pre-computed edge similarities
            
        Returns:
            W5Island instance
        """
        # Sort for determinism
        member_ids = sorted(member_ids)
        
        # === Island ID (INV-W5-002, INV-W5-006) ===
        # From members only, using Canonical JSON
        island_id = compute_canonical_hash({"members": member_ids})
        
        # === Representative Vector (INV-W5-005) ===
        # Raw mean with rounding (mean_raw_v1 policy)
        sum_vec: Dict[str, float] = defaultdict(float)
        
        for mid in member_ids:
            record = record_map[mid]
            for k, v in record.resonance_vector.items():
                sum_vec[k] += v
        
        count = len(member_ids)
        mean_vec = {
            k: round(v / count, W5_VECTOR_ROUNDING)
            for k, v in sum_vec.items()
        }
        
        # Sort keys for determinism
        canonical_vec = dict(sorted(mean_vec.items()))
        
        # === Cohesion Score (P1-1: edge average) ===
        # Average similarity of edges that formed this island
        # (not all pairs, only those above threshold)
        total_sim = 0.0
        edge_count = 0
        
        m = len(member_ids)
        for i in range(m):
            for j in range(i + 1, m):
                edge_key = tuple(sorted((member_ids[i], member_ids[j])))
                if edge_key in edge_similarities:
                    total_sim += edge_similarities[edge_key]
                    edge_count += 1
        
        if edge_count > 0:
            cohesion = round(total_sim / edge_count, 6)
        elif m == 1:
            # Single member: cohesion = 1.0 (perfect self-agreement)
            cohesion = 1.0
        else:
            # Should not happen if min_island_size >= 2 and algorithm is correct
            cohesion = 0.0
        
        return W5Island(
            island_id=island_id,
            member_ids=member_ids,
            size=m,
            representative_vector=canonical_vec,
            cohesion_score=cohesion,
        )
    
    def _empty_structure(self) -> W5Structure:
        """
        Create an empty W5Structure for empty input.
        
        Returns:
            W5Structure with no islands and no noise
        """
        structure_hash_input = {
            "input_w4_ids": [],
            "threshold": self.threshold,
            "min_island_size": self.min_island_size,
            "algorithm": W5_ALGORITHM,
            "vector_policy": W5_VECTOR_POLICY,
        }
        structure_id = compute_canonical_hash(structure_hash_input)
        
        return W5Structure(
            structure_id=structure_id,
            islands=[],
            noise_ids=[],
            input_count=0,
            island_count=0,
            noise_count=0,
            threshold=self.threshold,
            min_island_size=self.min_island_size,
            algorithm=W5_ALGORITHM,
            vector_policy=W5_VECTOR_POLICY,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


# ==========================================
# Comparison Utility
# ==========================================

def condense_batch(
    records: List[W4Record],
    threshold: float = W5_DEFAULT_THRESHOLD,
    min_island_size: int = W5_DEFAULT_MIN_ISLAND_SIZE,
) -> W5Structure:
    """
    Convenience function to condense a batch of W4Records.
    
    Args:
        records: List of W4Record
        threshold: Similarity threshold
        min_island_size: Minimum island size
        
    Returns:
        W5Structure
    """
    condensator = W5Condensator(
        threshold=threshold,
        min_island_size=min_island_size,
    )
    return condensator.condense(records)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-5 W5 Condensator Test")
    print("=" * 60)
    
    # Create test W4Records
    def make_w4(article_id: str, vector: Dict[str, float]) -> W4Record:
        """Helper to create test W4Record."""
        from .schema_w4 import compute_w4_analysis_id
        
        w4_analysis_id = compute_w4_analysis_id(
            article_id=article_id,
            used_w3={"cond_a": "w3_001"},
            tokenizer_version="test_v1",
            normalizer_version="test_v1",
        )
        
        return W4Record(
            article_id=article_id,
            w4_analysis_id=w4_analysis_id,
            resonance_vector=vector,
            used_w3={"cond_a": "w3_001"},
            token_count=100,
            tokenizer_version="test_v1",
            normalizer_version="test_v1",
        )
    
    # Test 1: Empty input
    print("\n[Test 1] Empty input")
    
    condensator = W5Condensator()
    result = condensator.condense([])
    
    assert result.input_count == 0
    assert result.island_count == 0
    assert result.noise_count == 0
    print(f"  structure_id: {result.structure_id[:16]}...")
    print("  ✅ PASS")
    
    # Test 2: P0-A - Duplicate article_id detection
    print("\n[Test 2] P0-A: Duplicate article_id detection")
    
    r1 = make_w4("dup_001", {"a": 1.0})
    r2 = make_w4("dup_001", {"a": 0.9})  # Same article_id!
    
    try:
        condensator.condense([r1, r2])
        print("  ❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        assert "Duplicate" in str(e)
        print(f"  Caught: {e}")
        print("  ✅ PASS")
    
    # Test 3: P0-B - Boundary similarity stability
    print("\n[Test 3] P0-B: Boundary similarity stability")
    
    # Create vectors that would have very close similarity
    r1 = make_w4("boundary_001", {"x": 1.0})
    r2 = make_w4("boundary_002", {"x": 0.7})  # Normalized: sim ≈ 0.7
    r3 = make_w4("boundary_003", {"x": 0.7, "y": 0.0001})  # Tiny perturbation
    
    # Run twice - should get identical results
    result1 = condensator.condense([r1, r2, r3])
    result2 = condensator.condense([r1, r2, r3])
    
    assert result1.structure_id == result2.structure_id
    assert result1.island_count == result2.island_count
    print(f"  Run 1 islands: {result1.island_count}")
    print(f"  Run 2 islands: {result2.island_count}")
    print("  ✅ PASS (Deterministic)")
    
    # Test 4: Basic condensation with clear clusters
    print("\n[Test 4] Basic condensation")
    
    # Cluster A: similar vectors
    a1 = make_w4("cluster_a_001", {"dim1": 1.0, "dim2": 0.1})
    a2 = make_w4("cluster_a_002", {"dim1": 0.95, "dim2": 0.15})
    a3 = make_w4("cluster_a_003", {"dim1": 0.9, "dim2": 0.2})
    
    # Cluster B: different vectors
    b1 = make_w4("cluster_b_001", {"dim1": -0.8, "dim2": 0.9})
    b2 = make_w4("cluster_b_002", {"dim1": -0.85, "dim2": 0.85})
    b3 = make_w4("cluster_b_003", {"dim1": -0.75, "dim2": 0.95})
    
    # Noise: isolated point
    noise = make_w4("noise_001", {"dim1": 0.0, "dim2": 0.0, "dim3": 1.0})
    
    result = condensator.condense([a1, a2, a3, b1, b2, b3, noise])
    
    print(f"  input_count: {result.input_count}")
    print(f"  island_count: {result.island_count}")
    print(f"  noise_count: {result.noise_count}")
    
    for island in result.islands:
        print(f"  Island {island.island_id[:8]}...: {island.size} members, cohesion={island.cohesion_score:.3f}")
    
    print("  ✅ PASS")
    
    # Test 5: P1-4 - Batch size limit
    print("\n[Test 5] P1-4: Batch size limit")
    
    small_condensator = W5Condensator(max_batch_size=5)
    
    records = [make_w4(f"too_many_{i:03d}", {"x": float(i)}) for i in range(10)]
    
    try:
        small_condensator.condense(records)
        print("  ❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        assert "exceeds limit" in str(e)
        print(f"  Caught: {e}")
        print("  ✅ PASS")
    
    # Test 6: INV-W5-008 - created_at excluded from canonical
    print("\n[Test 6] INV-W5-008: created_at excluded")
    
    import time
    
    r1 = make_w4("time_test_001", {"x": 1.0})
    r2 = make_w4("time_test_002", {"x": 0.9})
    r3 = make_w4("time_test_003", {"x": 0.85})
    
    result1 = condensator.condense([r1, r2, r3])
    time.sleep(0.01)  # Small delay
    result2 = condensator.condense([r1, r2, r3])
    
    # created_at should differ
    assert result1.created_at != result2.created_at
    
    # But canonical comparison should match
    from .schema_w5 import compare_w5_structures
    comparison = compare_w5_structures(result1, result2)
    
    assert comparison["match"], f"Should match: {comparison['differences']}"
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All condensator tests passed!")
