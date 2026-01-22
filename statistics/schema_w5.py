"""
ESDE Phase 9-5: W5 Schema
=========================
W5 (Weak Structural Condensation) data structures.

Theme: The Deterministic Shape of Resonance (共鳴の決定論的形状)

Design Principles:
  - W5 = Structural condensation of W4 resonance vectors
  - Complete bit-level determinism guaranteed
  - Collision-safe ID generation via Canonical JSON

Invariants:
  INV-W5-001 (No Naming):
      Output must not contain any natural language labels.
      
  INV-W5-002 (Topological Identity):
      island_id is generated from member IDs only, no floating-point.
      
  INV-W5-003 (Fixed Metric):
      Similarity = L2-normalized Cosine, threshold operator is >= (fixed).
      
  INV-W5-004 (Parameter Traceability):
      structure_id includes input w4_analysis_ids and observation parameters.
      
  INV-W5-005 (Canonical Vector):
      Vector values must be rounded before storage.
      
  INV-W5-006 (ID Collision Safety):
      ID generation uses Canonical JSON serialization. String join forbidden.
      
  INV-W5-007 (Structure Identity):
      structure_id uses w4_analysis_id (not article_id) as input identity.
      
  INV-W5-008 (Canonical Output):
      created_at and other non-deterministic fields excluded from identity check.

Spec: v5.4.5-P9.5-Final
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# ==========================================
# Constants
# ==========================================

W5_VERSION = "v9.5.0"
W5_ALGORITHM = "ResonanceCondensation-v1"
W5_VECTOR_POLICY = "mean_raw_v1"
W5_VECTOR_ROUNDING = 9           # Decimal places for vector values
W5_SIMILARITY_ROUNDING = 12      # Decimal places for similarity (P0-B)

# Default parameters
W5_DEFAULT_THRESHOLD = 0.70
W5_DEFAULT_MIN_ISLAND_SIZE = 3
W5_MAX_BATCH_SIZE = 2000         # P1-4: Prevent O(N^2) explosion

# Output directory
DEFAULT_W5_OUTPUT_DIR = "data/stats/w5_structures"


# ==========================================
# Canonical JSON Utility (INV-W5-006)
# ==========================================

def get_canonical_json(data: Any) -> bytes:
    """
    Safe canonical JSON serialization for hashing.
    
    INV-W5-006: All ID generation must use this function.
    String concatenation (join) is strictly forbidden.
    
    Args:
        data: Any JSON-serializable data
        
    Returns:
        UTF-8 encoded canonical JSON bytes
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True
    ).encode('utf-8')


def compute_canonical_hash(data: Any, length: int = 64) -> str:
    """
    Compute SHA256 hash of canonical JSON.
    
    Args:
        data: Data to hash
        length: Hash string length (default: full 64 chars)
        
    Returns:
        Hex digest string
    """
    return hashlib.sha256(get_canonical_json(data)).hexdigest()[:length]


# ==========================================
# W5 Island
# ==========================================

@dataclass
class W5Island:
    """
    W5 Island: A condensation unit (connected component).
    
    Represents a cluster of W4Records with high resonance similarity.
    
    INV-W5-002: island_id is SHA256 of {"members": sorted_member_ids}
    """
    
    # Identity (INV-W5-002: from members only)
    island_id: str
    
    # Members (sorted for determinism)
    member_ids: List[str]         # Sorted article_ids
    size: int
    
    # Representative vector (INV-W5-005: rounded)
    representative_vector: Dict[str, float]  # Raw mean, rounded
    
    # Cohesion (P1-1: edge average within island)
    cohesion_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "island_id": self.island_id,
            "member_ids": self.member_ids,
            "size": self.size,
            "representative_vector": dict(sorted(self.representative_vector.items())),
            "cohesion_score": round(self.cohesion_score, 6),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W5Island':
        """Create from dict."""
        return cls(
            island_id=data["island_id"],
            member_ids=data["member_ids"],
            size=data["size"],
            representative_vector=data["representative_vector"],
            cohesion_score=data["cohesion_score"],
        )


# ==========================================
# W5 Structure
# ==========================================

@dataclass
class W5Structure:
    """
    W5 Structure: A snapshot of structural condensation.
    
    Contains all islands and noise from a batch of W4Records.
    
    INV-W5-004: structure_id includes input w4_analysis_ids + parameters
    INV-W5-007: Uses w4_analysis_id, not article_id
    INV-W5-008: created_at excluded from canonical comparison
    """
    
    # Identity (INV-W5-004, INV-W5-007)
    structure_id: str
    
    # Results
    islands: List[W5Island]
    noise_ids: List[str]          # article_ids that didn't form islands
    
    # Metadata
    input_count: int
    island_count: int
    noise_count: int
    
    # Observation Parameters (Traceability)
    threshold: float
    min_island_size: int
    algorithm: str
    vector_policy: str
    
    # Non-deterministic field (INV-W5-008: excluded from canonical check)
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "structure_id": self.structure_id,
            "islands": [i.to_dict() for i in self.islands],
            "noise_ids": self.noise_ids,
            "input_count": self.input_count,
            "island_count": self.island_count,
            "noise_count": self.noise_count,
            "threshold": self.threshold,
            "min_island_size": self.min_island_size,
            "algorithm": self.algorithm,
            "vector_policy": self.vector_policy,
            "created_at": self.created_at,
        }
    
    def get_canonical_dict(self) -> Dict[str, Any]:
        """
        Returns dict for determinism comparison.
        
        INV-W5-008: Excludes created_at.
        """
        d = self.to_dict()
        d.pop('created_at', None)
        return d
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'W5Structure':
        """Create from dict."""
        return cls(
            structure_id=data["structure_id"],
            islands=[W5Island.from_dict(i) for i in data.get("islands", [])],
            noise_ids=data.get("noise_ids", []),
            input_count=data.get("input_count", 0),
            island_count=data.get("island_count", 0),
            noise_count=data.get("noise_count", 0),
            threshold=data.get("threshold", W5_DEFAULT_THRESHOLD),
            min_island_size=data.get("min_island_size", W5_DEFAULT_MIN_ISLAND_SIZE),
            algorithm=data.get("algorithm", W5_ALGORITHM),
            vector_policy=data.get("vector_policy", W5_VECTOR_POLICY),
            created_at=data.get("created_at", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'W5Structure':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# ==========================================
# Comparison Utility
# ==========================================

def compare_w5_structures(
    struct1: W5Structure,
    struct2: W5Structure,
) -> Dict[str, Any]:
    """
    Compare two W5Structures for determinism verification.
    
    INV-W5-008: Uses canonical dict (excludes created_at).
    
    Args:
        struct1: First W5Structure
        struct2: Second W5Structure
        
    Returns:
        Comparison result dict
    """
    diffs = []
    
    c1 = struct1.get_canonical_dict()
    c2 = struct2.get_canonical_dict()
    
    # Compare structure_id
    if c1["structure_id"] != c2["structure_id"]:
        diffs.append(f"structure_id mismatch")
    
    # Compare counts
    for key in ["input_count", "island_count", "noise_count"]:
        if c1.get(key) != c2.get(key):
            diffs.append(f"{key}: {c1.get(key)} vs {c2.get(key)}")
    
    # Compare island_ids
    ids1 = [i["island_id"] for i in c1.get("islands", [])]
    ids2 = [i["island_id"] for i in c2.get("islands", [])]
    if ids1 != ids2:
        diffs.append(f"island_ids mismatch: {len(ids1)} vs {len(ids2)}")
    
    # Compare noise_ids
    if c1.get("noise_ids") != c2.get("noise_ids"):
        diffs.append("noise_ids mismatch")
    
    return {
        "match": len(diffs) == 0,
        "differences": diffs[:10],
        "total_differences": len(diffs),
    }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-5 W5 Schema Test")
    print("=" * 60)
    
    # Test 1: Canonical JSON
    print("\n[Test 1] Canonical JSON (INV-W5-006)")
    
    data1 = {"b": 2, "a": 1}
    data2 = {"a": 1, "b": 2}
    
    json1 = get_canonical_json(data1)
    json2 = get_canonical_json(data2)
    
    print(f"  data1: {data1} -> {json1}")
    print(f"  data2: {data2} -> {json2}")
    assert json1 == json2, "Canonical JSON should be order-independent"
    print("  ✅ PASS")
    
    # Test 2: W5Island creation
    print("\n[Test 2] W5Island creation")
    
    island_id = compute_canonical_hash({"members": ["art_001", "art_002", "art_003"]})
    
    island = W5Island(
        island_id=island_id,
        member_ids=["art_001", "art_002", "art_003"],
        size=3,
        representative_vector={"cond_a": 0.123456789, "cond_b": -0.987654321},
        cohesion_score=0.85,
    )
    
    print(f"  island_id: {island.island_id[:16]}...")
    print(f"  size: {island.size}")
    print("  ✅ PASS")
    
    # Test 3: W5Structure creation
    print("\n[Test 3] W5Structure creation")
    
    structure = W5Structure(
        structure_id="test_structure_id",
        islands=[island],
        noise_ids=["art_noise_001"],
        input_count=4,
        island_count=1,
        noise_count=1,
        threshold=0.70,
        min_island_size=3,
        algorithm=W5_ALGORITHM,
        vector_policy=W5_VECTOR_POLICY,
        created_at="2026-01-22T00:00:00Z",
    )
    
    print(f"  input_count: {structure.input_count}")
    print(f"  island_count: {structure.island_count}")
    print(f"  noise_count: {structure.noise_count}")
    print("  ✅ PASS")
    
    # Test 4: Canonical dict excludes created_at
    print("\n[Test 4] Canonical dict (INV-W5-008)")
    
    canonical = structure.get_canonical_dict()
    
    assert "created_at" not in canonical, "created_at should be excluded"
    assert "structure_id" in canonical, "structure_id should be included"
    print("  ✅ PASS")
    
    # Test 5: JSON round-trip
    print("\n[Test 5] JSON serialization")
    
    json_str = structure.to_json()
    restored = W5Structure.from_json(json_str)
    
    assert structure.structure_id == restored.structure_id
    assert structure.island_count == restored.island_count
    assert len(structure.islands) == len(restored.islands)
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All schema tests passed!")
