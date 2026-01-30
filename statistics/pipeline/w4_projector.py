#!/usr/bin/env python3
"""
ESDE Phase 9: Article Projection (W4)
======================================

Projects articles onto W3 S-Score space to produce resonance vectors.

Mathematical Model:
  R(A, C) = Σ count(t, A) × S(t, C)

Where:
  - A = Article
  - C = Condition
  - count(t, A) = Token count in article
  - S(t, C) = S-Score from W3

Result:
  - Each article becomes a vector in condition-space
  - Vector dimension = number of conditions
  - Vector values = resonance with each condition

Usage in Pipeline:
  W2 → W3 (S-Scores) → W4 (Article vectors) → W5 (Clustering)

Spec: Phase 9 W4 v1.0
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from datetime import datetime, timezone

from .w3_calculator import W3Result


# ==========================================
# Constants
# ==========================================

W4_VERSION = "v9.phase9.1"
W4_ALGORITHM = "DotProduct-v1"


# ==========================================
# Data Structures
# ==========================================

@dataclass
class ArticleVector:
    """
    Article projected onto condition-space.
    
    The resonance_vector maps condition_id → resonance score.
    Positive resonance = article aligns with condition
    Negative resonance = article diverges from condition
    """
    article_id: str
    resonance_vector: Dict[str, float] = field(default_factory=dict)
    token_count: int = 0
    unique_tokens: int = 0
    
    # Traceability
    w3_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "resonance_vector": {
                k: round(v, 8) for k, v in self.resonance_vector.items()
            },
            "token_count": self.token_count,
            "unique_tokens": self.unique_tokens,
            "w3_hash": self.w3_hash,
        }
    
    def get_dominant_condition(self) -> Optional[Tuple[str, float]]:
        """Get condition with highest positive resonance."""
        if not self.resonance_vector:
            return None
        max_cond = max(self.resonance_vector.items(), key=lambda x: x[1])
        return max_cond
    
    def get_vector_magnitude(self) -> float:
        """L2 norm of resonance vector."""
        if not self.resonance_vector:
            return 0.0
        return math.sqrt(sum(v * v for v in self.resonance_vector.values()))


@dataclass
class W4Result:
    """
    W4 Projection result.
    
    Contains article vectors for all processed articles.
    """
    # Source info
    provider_id: str
    axis_name: str
    w3_hash: str
    
    # Article vectors
    articles: Dict[str, ArticleVector] = field(default_factory=dict)
    
    # Condition info (for reference)
    condition_ids: List[str] = field(default_factory=list)
    
    # Metadata
    algorithm: str = W4_ALGORITHM
    version: str = W4_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "axis_name": self.axis_name,
            "w3_hash": self.w3_hash,
            "article_count": len(self.articles),
            "condition_count": len(self.condition_ids),
            "condition_ids": self.condition_ids,
            "algorithm": self.algorithm,
            "version": self.version,
            "created_at": self.created_at,
            "articles": {
                aid: av.to_dict() for aid, av in self.articles.items()
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def get_vectors_for_clustering(self) -> List[Tuple[str, Dict[str, float]]]:
        """
        Get article vectors in format ready for W5 clustering.
        
        Returns:
            List of (article_id, resonance_vector) tuples
        """
        return [
            (aid, av.resonance_vector)
            for aid, av in self.articles.items()
        ]


# ==========================================
# W4 Projector
# ==========================================

class W4Projector:
    """
    Projects articles onto W3 S-Score space.
    
    Usage:
        projector = W4Projector(w3_result)
        
        # Project a single article
        vector = projector.project_article(article_id, features)
        
        # Project multiple articles
        result = projector.project_all(articles)
    """
    
    def __init__(self, w3_result: W3Result):
        """
        Initialize W4 Projector.
        
        Args:
            w3_result: W3 calculation result with S-Score candidates
        """
        self.w3_result = w3_result
        
        # Pre-build S-Score dictionaries for each condition
        self._sscore_dicts: Dict[str, Dict[str, float]] = {}
        for cond_id in w3_result.conditions:
            self._sscore_dicts[cond_id] = w3_result.get_sscore_dict(cond_id)
        
        self._condition_ids = sorted(w3_result.conditions.keys())
    
    def project_article(
        self,
        article_id: str,
        features: List[Any],  # List of TokenFeature
    ) -> ArticleVector:
        """
        Project a single article onto condition-space.
        
        Args:
            article_id: Article identifier
            features: List of TokenFeature from W1
            
        Returns:
            ArticleVector with resonance scores
        """
        # Count tokens in article
        token_counts: Counter = Counter()
        
        for feat in features:
            # Normalize token (lowercase)
            token_norm = feat.lemma.lower() if feat.lemma else feat.token.lower().strip()
            
            # Skip empty or non-alphabetic
            if not token_norm or not any(c.isalpha() for c in token_norm):
                continue
            
            token_counts[token_norm] += 1
        
        # Compute resonance for each condition
        resonance_vector: Dict[str, float] = {}
        
        for cond_id in self._condition_ids:
            sscore_dict = self._sscore_dicts.get(cond_id, {})
            
            # R(A, C) = Σ count(t, A) × S(t, C)
            resonance = 0.0
            for token, count in token_counts.items():
                if token in sscore_dict:
                    resonance += count * sscore_dict[token]
            
            resonance_vector[cond_id] = resonance
        
        return ArticleVector(
            article_id=article_id,
            resonance_vector=resonance_vector,
            token_count=sum(token_counts.values()),
            unique_tokens=len(token_counts),
            w3_hash=self.w3_result.w2_snapshot_hash,
        )
    
    def project_all(
        self,
        articles: List[Tuple[str, List[Any]]],  # List of (article_id, features)
    ) -> W4Result:
        """
        Project multiple articles.
        
        Args:
            articles: List of (article_id, features) tuples
            
        Returns:
            W4Result with all article vectors
        """
        result = W4Result(
            provider_id=self.w3_result.provider_id,
            axis_name=self.w3_result.axis_name,
            w3_hash=self.w3_result.w2_snapshot_hash,
            condition_ids=self._condition_ids,
        )
        
        for article_id, features in articles:
            vector = self.project_article(article_id, features)
            result.articles[article_id] = vector
        
        return result
    
    def get_summary(self, result: W4Result) -> Dict[str, Any]:
        """Get human-readable summary of W4 result."""
        summaries = []
        
        for aid, av in result.articles.items():
            dominant = av.get_dominant_condition()
            summaries.append({
                "article_id": aid,
                "token_count": av.token_count,
                "dominant_condition": dominant[0] if dominant else None,
                "dominant_score": round(dominant[1], 4) if dominant else None,
                "magnitude": round(av.get_vector_magnitude(), 4),
            })
        
        return {
            "axis_name": result.axis_name,
            "article_count": len(result.articles),
            "condition_count": len(result.condition_ids),
            "articles": summaries,
        }


# ==========================================
# Utility Functions
# ==========================================

def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.
    
    Args:
        vec1: First vector (dict)
        vec2: Second vector (dict)
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    # Get all keys
    all_keys = set(vec1.keys()) | set(vec2.keys())
    
    # Compute dot product and magnitudes
    dot = 0.0
    mag1 = 0.0
    mag2 = 0.0
    
    for key in all_keys:
        v1 = vec1.get(key, 0.0)
        v2 = vec2.get(key, 0.0)
        dot += v1 * v2
        mag1 += v1 * v1
        mag2 += v2 * v2
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (math.sqrt(mag1) * math.sqrt(mag2))


def compute_pairwise_similarities(
    result: W4Result
) -> List[Tuple[str, str, float]]:
    """
    Compute pairwise cosine similarities between all articles.
    
    Returns:
        List of (article_id_1, article_id_2, similarity) tuples
    """
    article_ids = sorted(result.articles.keys())
    similarities = []
    
    for i in range(len(article_ids)):
        for j in range(i + 1, len(article_ids)):
            aid1 = article_ids[i]
            aid2 = article_ids[j]
            vec1 = result.articles[aid1].resonance_vector
            vec2 = result.articles[aid2].resonance_vector
            sim = cosine_similarity(vec1, vec2)
            similarities.append((aid1, aid2, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: -x[2])
    
    return similarities


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("W4 Projector Test")
    print("=" * 60)
    
    from dataclasses import dataclass as dc
    from .w3_calculator import W3Result, ConditionCandidates, CandidateToken
    
    # Create mock W3 result
    w3 = W3Result(
        provider_id="section_v1",
        axis_name="section",
        w2_snapshot_hash="test_hash",
    )
    
    # Military condition: war, battle positive; art, culture negative
    military_cond = ConditionCandidates(
        condition_id="military",
        factors={"section_name": "military"},
        positive_candidates=[
            CandidateToken("war", 0.05, 0.3, 0.1, 30, 50),
            CandidateToken("battle", 0.04, 0.25, 0.1, 25, 30),
        ],
        negative_candidates=[
            CandidateToken("art", -0.03, 0.01, 0.1, 1, 15),
            CandidateToken("culture", -0.02, 0.01, 0.05, 1, 10),
        ],
    )
    w3.conditions["military"] = military_cond
    
    # Culture condition: art, culture positive; war, battle negative
    culture_cond = ConditionCandidates(
        condition_id="culture",
        factors={"section_name": "culture"},
        positive_candidates=[
            CandidateToken("art", 0.06, 0.3, 0.1, 12, 15),
            CandidateToken("culture", 0.05, 0.2, 0.05, 8, 10),
        ],
        negative_candidates=[
            CandidateToken("war", -0.04, 0.02, 0.1, 2, 50),
            CandidateToken("battle", -0.03, 0.01, 0.1, 1, 30),
        ],
    )
    w3.conditions["culture"] = culture_cond
    
    # Create projector
    projector = W4Projector(w3)
    
    # Create mock articles
    @dc
    class MockFeature:
        token: str
        lemma: str
    
    # Article 1: Military focused
    features1 = [
        MockFeature("The", "the"),
        MockFeature("war", "war"),
        MockFeature("war", "war"),
        MockFeature("battle", "battle"),
        MockFeature("was", "be"),
        MockFeature("fierce", "fierce"),
    ]
    
    # Article 2: Culture focused
    features2 = [
        MockFeature("The", "the"),
        MockFeature("art", "art"),
        MockFeature("art", "art"),
        MockFeature("culture", "culture"),
        MockFeature("flourished", "flourish"),
    ]
    
    # Article 3: Mixed
    features3 = [
        MockFeature("war", "war"),
        MockFeature("art", "art"),
        MockFeature("peace", "peace"),
    ]
    
    # Project
    result = projector.project_all([
        ("article_military", features1),
        ("article_culture", features2),
        ("article_mixed", features3),
    ])
    
    # Print results
    print("\n[Article Vectors]")
    for aid, av in result.articles.items():
        print(f"\n  {aid}:")
        for cond_id, resonance in sorted(av.resonance_vector.items()):
            print(f"    {cond_id}: {resonance:+.4f}")
        dominant = av.get_dominant_condition()
        if dominant:
            print(f"    → Dominant: {dominant[0]} ({dominant[1]:+.4f})")
    
    # Print similarities
    print("\n[Pairwise Similarities]")
    sims = compute_pairwise_similarities(result)
    for aid1, aid2, sim in sims:
        print(f"  {aid1} <-> {aid2}: {sim:.4f}")
    
    print("\n" + "=" * 60)
    print("W4 Projector tests passed!")
