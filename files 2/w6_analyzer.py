"""
ESDE Phase 9-6: W6 Analyzer
===========================
Observation analysis for W5 structures.

Theme: The Observatory - From Structure to Evidence (構造から証拠へ)

Migration Phase 3 Update:
  - ExecutionContext support for policy metadata propagation
  - Policy metadata included in W6Observatory output

Algorithm:
  1. Scope Validation (INV-W6-009): Verify W5 members match input W4/Articles
  2. Evidence Extraction (P0-X1): mean_s_score_v1 formula
  3. Topology Calculation (P0-1): W5 representative_vector distance only
  4. Representative Sampling: Top articles by resonance magnitude
  5. ID Generation (P0-4): Canonical JSON hash excluding floats

GPT Audit Compliance:
  P0-X1: Evidence formula fixed (mean_s_score_v1)
  P0-X2: Scope closure enforced (INV-W6-009)
  P0-1: Topology uses W5 vectors only
  P0-4: observation_id excludes floating-point values

Spec: v5.4.6-P9.6-Final + Migration Phase 3 v0.3.1
"""

import math
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any, TYPE_CHECKING

from .schema_w6 import (
    W6Observatory,
    W6IslandDetail,
    W6TopologyPair,
    W6EvidenceToken,
    W6RepresentativeArticle,
    get_canonical_json,
    compute_canonical_hash,
    W6_VERSION,
    W6_CODE_VERSION,
    W6_EVIDENCE_POLICY,
    W6_SNIPPET_POLICY,
    W6_METRIC_POLICY,
    W6_DIGEST_POLICY,
    W6_SNIPPET_LENGTH,
    W6_DIGEST_K,
    W6_EVIDENCE_K,
    W6_REP_ARTICLE_K,
    W6_TOPOLOGY_K,
    W6_DISTANCE_ROUNDING,
    W6_EVIDENCE_ROUNDING,
)

if TYPE_CHECKING:
    from statistics.utils import ExecutionContext


# ==========================================
# W6 Analyzer
# ==========================================

class W6Analyzer:
    """
    W6 Analyzer: Extracts observations from W5 structures.
    
    Converts W5Structure + W4Records + W3Records + ArticleRecords
    into a W6Observatory containing:
    - Evidence tokens for each island
    - Topology (inter-island distances)
    - Representative article snippets
    
    Migration Phase 3 Update:
    - Accepts ExecutionContext for policy metadata propagation
    - Policy metadata automatically included in W6Observatory
    
    INV-W6-003: Read-only (never modifies input data)
    INV-W6-004: No new statistics, only extraction/transformation
    INV-W6-009: Validates scope closure before processing
    """
    
    # Fixed Policies (hardcoded for determinism)
    POLICIES = {
        "evidence_policy": W6_EVIDENCE_POLICY,
        "snippet_policy": W6_SNIPPET_POLICY,
        "metric_policy": W6_METRIC_POLICY,
        "digest_policy": W6_DIGEST_POLICY,
        "snippet_length": W6_SNIPPET_LENGTH,
        "digest_k": W6_DIGEST_K,
        "evidence_k": W6_EVIDENCE_K,
        "rep_article_k": W6_REP_ARTICLE_K,
        "topology_k": W6_TOPOLOGY_K,
    }
    
    def __init__(
        self,
        scope_id: str = "",
        tokenizer_version: str = "",
        normalizer_version: str = "",
        execution_context: Optional["ExecutionContext"] = None,
    ):
        """
        Initialize W6 Analyzer.
        
        Args:
            scope_id: Analysis scope identifier (e.g., "news_202501")
            tokenizer_version: Tokenizer version for tracking
            normalizer_version: Normalizer version for tracking
            execution_context: Optional ExecutionContext from StatisticsPipelineRunner
                              (Migration Phase 3 - for policy metadata propagation)
        """
        self.scope_id = scope_id
        self.tokenizer_version = tokenizer_version
        self.normalizer_version = normalizer_version
        self.execution_context = execution_context
        
        # Extract policy metadata from ExecutionContext if available
        self._policy_metadata = self._extract_policy_metadata()
    
    def _extract_policy_metadata(self) -> Dict[str, Any]:
        """
        Extract policy metadata from ExecutionContext.
        
        Migration Phase 3: Returns policy metadata dict or legacy defaults.
        """
        if self.execution_context is not None:
            return self.execution_context.to_metadata_dict()
        
        # Legacy defaults
        return {
            "analysis_scope_id": self.scope_id,
            "policy_id": None,
            "policy_version": None,
            "policy_signature_hash": None,
            "signature_source": "legacy",
        }
    
    def analyze(
        self,
        w5_structure,  # W5Structure
        w4_records: List,  # List[W4Record]
        w3_records: List,  # List[W3Record]
        article_records: List,  # List[ArticleRecord]
    ) -> W6Observatory:
        """
        Analyze W5 structure and produce W6 observation.
        
        INV-W6-009: Validates that W5 members match input sets.
        INV-W6-003: Does not modify any input.
        
        Args:
            w5_structure: W5Structure from condensation
            w4_records: W4Records for scope (must match W5 members)
            w3_records: W3Records used in W4 projection
            article_records: ArticleRecords for scope (must match W5 members)
            
        Returns:
            W6Observatory with evidence and topology
            
        Raises:
            ValueError: If scope closure violated (INV-W6-009)
        """
        # === INV-W6-009: Scope Closure Validation ===
        self._validate_scope_closure(w5_structure, w4_records, article_records)
        
        # === Build Index Maps ===
        w4_map = {r.article_id: r for r in w4_records}
        article_map = {a.article_id: a for a in article_records}
        w3_map = {r.condition_signature: r for r in w3_records}
        
        # Build S-Score lookup: {cond_sig: {token_norm: s_score}}
        sscore_lookup = self._build_sscore_lookup(w3_records)
        
        # === Process Islands ===
        island_details: List[W6IslandDetail] = []
        
        for island in w5_structure.islands:
            detail = self._process_island(
                island=island,
                w4_map=w4_map,
                article_map=article_map,
                w3_map=w3_map,
                sscore_lookup=sscore_lookup,
            )
            island_details.append(detail)
        
        # Sort islands by size (desc), then by island_id (INV-W6-006)
        island_details.sort(key=lambda x: (-x.size, x.island_id))
        
        # === Calculate Topology ===
        topology_pairs = self._calculate_topology(w5_structure.islands)
        
        # === Compute W3 Versions Digest ===
        w3_versions_digest = self._compute_w3_digest(w3_records)
        
        # === Generate Observation ID (P0-4: excludes floats) ===
        observation_id = self._generate_observation_id(
            w5_structure.structure_id,
            [d.island_id for d in island_details],
        )
        
        # === Build W6Observatory with Policy Metadata ===
        policy_meta = self._policy_metadata
        
        return W6Observatory(
            observation_id=observation_id,
            input_structure_id=w5_structure.structure_id,
            analysis_scope_id=policy_meta.get("analysis_scope_id", self.scope_id),
            islands=island_details,
            topology_pairs=topology_pairs,
            tokenizer_version=self.tokenizer_version,
            normalizer_version=self.normalizer_version,
            w3_versions_digest=w3_versions_digest,
            code_version=W6_CODE_VERSION,
            # Migration Phase 3: Policy metadata
            policy_id=policy_meta.get("policy_id"),
            policy_version=policy_meta.get("policy_version"),
            policy_signature_hash=policy_meta.get("policy_signature_hash"),
            signature_source=policy_meta.get("signature_source", "legacy"),
            params=dict(self.POLICIES),
            noise_count=w5_structure.noise_count,
            noise_ids=sorted(w5_structure.noise_ids),
        )
    
    def _validate_scope_closure(
        self,
        w5_structure,
        w4_records: List,
        article_records: List,
    ) -> None:
        """
        Validate INV-W6-009: Scope Closure.
        
        All W5 members must exist in W4Records and ArticleRecords.
        """
        # Collect all member IDs from W5
        w5_member_ids: Set[str] = set()
        for island in w5_structure.islands:
            w5_member_ids.update(island.member_ids)
        w5_member_ids.update(w5_structure.noise_ids)
        
        # Collect IDs from inputs
        w4_ids = {r.article_id for r in w4_records}
        article_ids = {a.article_id for a in article_records}
        
        # Check closure
        missing_w4 = w5_member_ids - w4_ids
        missing_article = w5_member_ids - article_ids
        
        if missing_w4:
            raise ValueError(
                f"INV-W6-009 Violation: {len(missing_w4)} W5 members not in W4Records. "
                f"First few: {list(missing_w4)[:3]}"
            )
        
        if missing_article:
            raise ValueError(
                f"INV-W6-009 Violation: {len(missing_article)} W5 members not in ArticleRecords. "
                f"First few: {list(missing_article)[:3]}"
            )
    
    def _build_sscore_lookup(
        self,
        w3_records: List,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build S-Score lookup from W3Records.
        
        Returns:
            {condition_signature: {token_norm: s_score}}
        """
        lookup: Dict[str, Dict[str, float]] = {}
        
        for w3 in w3_records:
            cond_sig = w3.condition_signature
            lookup[cond_sig] = {}
            
            # Add positive candidates
            for cand in w3.positive_candidates:
                lookup[cond_sig][cand.token_norm] = cand.s_score
            
            # Add negative candidates
            for cand in w3.negative_candidates:
                lookup[cond_sig][cand.token_norm] = cand.s_score
        
        return lookup
    
    def _process_island(
        self,
        island,
        w4_map: Dict,
        article_map: Dict,
        w3_map: Dict,
        sscore_lookup: Dict[str, Dict[str, float]],
    ) -> W6IslandDetail:
        """
        Process a single island to extract evidence and representatives.
        """
        # === Representative Vector Digest ===
        # Sort by |value| desc, then by key (INV-W6-006)
        rep_vec = island.representative_vector
        digest_items = sorted(
            rep_vec.items(),
            key=lambda x: (-abs(x[1]), x[0])
        )[:W6_DIGEST_K]
        
        # === Evidence Tokens ===
        evidence_tokens = self._extract_evidence_tokens(
            island.member_ids,
            w4_map,
            sscore_lookup,
        )
        
        # === Used W3 Digest ===
        used_w3_ids = set()
        for article_id in island.member_ids:
            if article_id in w4_map:
                w4 = w4_map[article_id]
                used_w3_ids.update(w4.used_w3.values())
        
        used_w3_digest = compute_canonical_hash(sorted(used_w3_ids), length=32)
        
        # === Representative Articles ===
        rep_articles = self._select_representative_articles(
            island.member_ids,
            w4_map,
            article_map,
        )
        
        return W6IslandDetail(
            island_id=island.island_id,
            size=island.size,
            cohesion_score=island.cohesion_score,
            representative_vec_digest=digest_items,
            evidence_tokens=evidence_tokens,
            used_w3_digest=used_w3_digest,
            representative_articles=rep_articles,
            member_ids=sorted(island.member_ids),
        )
    
    def _extract_evidence_tokens(
        self,
        member_ids: List[str],
        w4_map: Dict,
        sscore_lookup: Dict[str, Dict[str, float]],
    ) -> List[W6EvidenceToken]:
        """
        Extract evidence tokens using mean_s_score_v1 formula.
        
        P0-X1: evidence_score = mean of S-Scores across island members
        """
        # Aggregate: {token_norm: {"total_score": float, "count": int, "w3_ids": set}}
        token_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_score": 0.0, "count": 0, "w3_ids": set()}
        )
        
        for article_id in member_ids:
            if article_id not in w4_map:
                continue
            
            w4 = w4_map[article_id]
            
            # For each condition in resonance_vector
            for cond_sig, resonance in w4.resonance_vector.items():
                if cond_sig not in sscore_lookup:
                    continue
                
                # Get W3 analysis ID for this condition
                w3_id = w4.used_w3.get(cond_sig, "")
                
                # For each token in this condition's S-Score dict
                for token_norm, s_score in sscore_lookup[cond_sig].items():
                    token_stats[token_norm]["total_score"] += s_score
                    token_stats[token_norm]["count"] += 1
                    if w3_id:
                        token_stats[token_norm]["w3_ids"].add(w3_id)
        
        # Build evidence tokens
        evidence_list: List[W6EvidenceToken] = []
        
        for token_norm, stats in token_stats.items():
            if stats["count"] == 0:
                continue
            
            mean_score = stats["total_score"] / stats["count"]
            
            evidence_list.append(W6EvidenceToken(
                token_norm=token_norm,
                evidence_score=mean_score,
                article_presence_count=stats["count"],
                source_w3_ids=sorted(stats["w3_ids"]),
            ))
        
        # Sort by |evidence_score| desc, then by token_norm (INV-W6-006)
        evidence_list.sort(key=lambda x: (-abs(x.evidence_score), x.token_norm))
        
        return evidence_list[:W6_EVIDENCE_K]
    
    def _select_representative_articles(
        self,
        member_ids: List[str],
        w4_map: Dict,
        article_map: Dict,
    ) -> List[W6RepresentativeArticle]:
        """
        Select representative articles by resonance magnitude.
        
        P1-2: snippet_policy = head_chars_v1 (first 200 chars)
        """
        candidates: List[Tuple[float, str, str, str]] = []  # (mag, article_id, w4_id, snippet)
        
        for article_id in member_ids:
            if article_id not in w4_map or article_id not in article_map:
                continue
            
            w4 = w4_map[article_id]
            article = article_map[article_id]
            
            # Compute resonance magnitude
            magnitude = math.sqrt(sum(v * v for v in w4.resonance_vector.values()))
            
            # Extract snippet (head_chars_v1)
            raw_text = article.raw_text if hasattr(article, 'raw_text') else ""
            snippet = raw_text[:W6_SNIPPET_LENGTH]
            
            candidates.append((magnitude, article_id, w4.w4_analysis_id, snippet))
        
        # Sort by magnitude desc, then by article_id (INV-W6-006)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Take top K
        result: List[W6RepresentativeArticle] = []
        for mag, art_id, w4_id, snippet in candidates[:W6_REP_ARTICLE_K]:
            result.append(W6RepresentativeArticle(
                article_id=art_id,
                w4_analysis_id=w4_id,
                resonance_magnitude=mag,
                snippet=snippet,
            ))
        
        return result
    
    def _calculate_topology(
        self,
        islands: List,
    ) -> List[W6TopologyPair]:
        """
        Calculate inter-island topology (distances).
        
        P1-3: metric_policy = cosine_dist_v1
        INV-W6-004: Uses W5 representative_vector ONLY
        """
        pairs: List[W6TopologyPair] = []
        
        n = len(islands)
        for i in range(n):
            for j in range(i + 1, n):
                island_a = islands[i]
                island_b = islands[j]
                
                # Compute cosine distance
                distance = self._cosine_distance(
                    island_a.representative_vector,
                    island_b.representative_vector,
                )
                
                # Generate pair_id
                pair_id = compute_canonical_hash({
                    "island_a": island_a.island_id,
                    "island_b": island_b.island_id,
                    "metric": W6_METRIC_POLICY,
                }, length=32)
                
                pairs.append(W6TopologyPair(
                    pair_id=pair_id,
                    island_a_id=island_a.island_id,
                    island_b_id=island_b.island_id,
                    distance=distance,
                    metric=W6_METRIC_POLICY,
                ))
        
        # Sort by distance asc, then by pair_id (INV-W6-006)
        pairs.sort(key=lambda x: (x.distance, x.pair_id))
        
        return pairs[:W6_TOPOLOGY_K]
    
    def _cosine_distance(
        self,
        vec_a: Dict[str, float],
        vec_b: Dict[str, float],
    ) -> float:
        """
        Compute cosine distance between two sparse vectors.
        
        P1-3 (cosine_dist_v1):
          distance = 1.0 - round(cos_sim, 12)
          
        Where cos_sim = dot(a, b) / (|a| * |b|)
        """
        # Compute dot product (sparse)
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
        
        # Compute norms
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        
        if norm_a < 1e-12 or norm_b < 1e-12:
            # Undefined similarity, treat as maximum distance
            return 2.0
        
        cos_sim = dot / (norm_a * norm_b)
        
        # P1-3: Round similarity before computing distance
        cos_sim_rounded = round(cos_sim, W6_DISTANCE_ROUNDING)
        
        # Distance = 1 - similarity
        distance = 1.0 - cos_sim_rounded
        
        # Round final distance for consistency
        return round(distance, W6_DISTANCE_ROUNDING)
    
    def _compute_w3_digest(self, w3_records: List) -> str:
        """Compute hash of W3 versions/analysis IDs."""
        w3_ids = sorted([w3.analysis_id for w3 in w3_records])
        return compute_canonical_hash(w3_ids, length=32)
    
    def _generate_observation_id(
        self,
        structure_id: str,
        island_ids: List[str],
    ) -> str:
        """
        Generate observation ID.
        
        P0-4: Excludes floating-point values for determinism.
        """
        # Use only string/int values
        id_data = {
            "structure_id": structure_id,
            "island_ids": sorted(island_ids),
            "scope_id": self.scope_id,
            "tokenizer_version": self.tokenizer_version,
            "normalizer_version": self.normalizer_version,
            "code_version": W6_CODE_VERSION,
        }
        
        return compute_canonical_hash(id_data, length=64)
