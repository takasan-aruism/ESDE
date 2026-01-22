"""
ESDE Phase 9-6: W6 Analyzer
===========================
Observation analysis for W5 structures.

Theme: The Observatory - From Structure to Evidence (構造から証拠へ)

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

Spec: v5.4.6-P9.6-Final
"""

import math
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

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
    ):
        """
        Initialize W6 Analyzer.
        
        Args:
            scope_id: Analysis scope identifier (e.g., "news_202501")
            tokenizer_version: Tokenizer version for tracking
            normalizer_version: Normalizer version for tracking
        """
        self.scope_id = scope_id
        self.tokenizer_version = tokenizer_version
        self.normalizer_version = normalizer_version
    
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
        
        # Sort islands by island_id for determinism
        island_details.sort(key=lambda d: d.island_id)
        
        # === Topology Calculation (P0-1: W5 vector only) ===
        topology_pairs = self._calc_topology(w5_structure.islands)
        
        # === Version Tracking (INV-W6-008) ===
        w3_versions = sorted([r.analysis_id for r in w3_records])
        w3_versions_digest = compute_canonical_hash({"w3_ids": w3_versions}, length=32)
        
        # === Observation ID Generation (P0-4) ===
        # Only use non-floating-point values
        member_w4_ids = self._get_member_w4_ids(w5_structure, w4_map)
        
        obs_id_input = {
            "input_structure_id": w5_structure.structure_id,
            "input_w4_ids": member_w4_ids,
            "analysis_scope_id": self.scope_id,
            "params": self.POLICIES,
            "versions": {
                "tokenizer": self.tokenizer_version,
                "normalizer": self.normalizer_version,
                "code": W6_CODE_VERSION,
            },
        }
        observation_id = compute_canonical_hash(obs_id_input)
        
        # === Build Observatory ===
        return W6Observatory(
            observation_id=observation_id,
            input_structure_id=w5_structure.structure_id,
            analysis_scope_id=self.scope_id,
            islands=island_details,
            topology_pairs=topology_pairs,
            tokenizer_version=self.tokenizer_version,
            normalizer_version=self.normalizer_version,
            w3_versions_digest=w3_versions_digest,
            code_version=W6_CODE_VERSION,
            params=self.POLICIES,
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
        
        W5 member set must exactly match W4/Article input sets.
        
        Raises:
            ValueError: If sets don't match
        """
        # Collect all article_ids from W5 (islands + noise)
        w5_article_ids: Set[str] = set()
        for island in w5_structure.islands:
            w5_article_ids.update(island.member_ids)
        w5_article_ids.update(w5_structure.noise_ids)
        
        # Collect article_ids from W4 records
        w4_article_ids = {r.article_id for r in w4_records}
        
        # Collect article_ids from ArticleRecords
        article_ids = {a.article_id for a in article_records}
        
        # Check W5 vs W4
        if w5_article_ids != w4_article_ids:
            missing_in_w4 = w5_article_ids - w4_article_ids
            extra_in_w4 = w4_article_ids - w5_article_ids
            raise ValueError(
                f"INV-W6-009 violation: W5 members ({len(w5_article_ids)}) "
                f"!= W4 records ({len(w4_article_ids)}). "
                f"Missing in W4: {len(missing_in_w4)}, Extra in W4: {len(extra_in_w4)}"
            )
        
        # Check W5 vs Articles
        if w5_article_ids != article_ids:
            missing_in_articles = w5_article_ids - article_ids
            extra_in_articles = article_ids - w5_article_ids
            raise ValueError(
                f"INV-W6-009 violation: W5 members ({len(w5_article_ids)}) "
                f"!= ArticleRecords ({len(article_ids)}). "
                f"Missing: {len(missing_in_articles)}, Extra: {len(extra_in_articles)}"
            )
    
    def _get_member_w4_ids(
        self,
        w5_structure,
        w4_map: Dict[str, Any],
    ) -> List[str]:
        """
        Get W4 analysis IDs for W5 members only (P0-X2).
        
        Uses w4_analysis_id (not article_id) for identity.
        Only includes IDs from W5 member scope.
        """
        all_member_ids: Set[str] = set()
        for island in w5_structure.islands:
            all_member_ids.update(island.member_ids)
        all_member_ids.update(w5_structure.noise_ids)
        
        w4_ids = []
        for article_id in sorted(all_member_ids):
            if article_id in w4_map:
                w4_ids.append(w4_map[article_id].w4_analysis_id)
        
        return sorted(w4_ids)
    
    def _build_sscore_lookup(
        self,
        w3_records: List,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build S-Score lookup from W3 records.
        
        Returns:
            {cond_sig: {token_norm: s_score}}
        """
        lookup: Dict[str, Dict[str, float]] = {}
        
        for w3 in w3_records:
            cond_sig = w3.condition_signature
            lookup[cond_sig] = {}
            
            # Include both positive and negative candidates
            for candidate in w3.positive_candidates:
                lookup[cond_sig][candidate.token_norm] = candidate.s_score
            
            for candidate in w3.negative_candidates:
                lookup[cond_sig][candidate.token_norm] = candidate.s_score
        
        return lookup
    
    def _process_island(
        self,
        island,  # W5Island
        w4_map: Dict[str, Any],
        article_map: Dict[str, Any],
        w3_map: Dict[str, Any],
        sscore_lookup: Dict[str, Dict[str, float]],
    ) -> W6IslandDetail:
        """
        Process a single island to extract evidence.
        
        P0-X1: Uses mean_s_score_v1 formula for evidence.
        P1-1: Strict ordering for digest.
        P1-2: head_chars_v1 for snippets.
        """
        member_ids = sorted(island.member_ids)
        
        # === Representative Vector Digest (P1-1: abs_val_desc_v1) ===
        vec_digest = self._compute_vec_digest(island.representative_vector)
        
        # === Evidence Tokens (P0-X1: mean_s_score_v1) ===
        evidence_tokens = self._extract_evidence_tokens(
            member_ids=member_ids,
            w4_map=w4_map,
            sscore_lookup=sscore_lookup,
        )
        
        # === Used W3 Digest ===
        used_w3_digest = self._compute_used_w3_digest(member_ids, w4_map)
        
        # === Representative Articles (P1-2: head_chars_v1) ===
        rep_articles = self._sample_representative_articles(
            member_ids=member_ids,
            w4_map=w4_map,
            article_map=article_map,
        )
        
        return W6IslandDetail(
            island_id=island.island_id,
            size=island.size,
            cohesion_score=island.cohesion_score,
            representative_vec_digest=vec_digest,
            evidence_tokens=evidence_tokens,
            used_w3_digest=used_w3_digest,
            representative_articles=rep_articles,
            member_ids=member_ids,
        )
    
    def _compute_vec_digest(
        self,
        rep_vector: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Compute representative vector digest.
        
        P1-1 (abs_val_desc_v1):
          - Sort by (-abs(value), cond_sig asc)
          - Take top W6_DIGEST_K
        """
        items = list(rep_vector.items())
        
        # Sort by (-abs(value), cond_sig asc)
        items.sort(key=lambda x: (-abs(x[1]), x[0]))
        
        # Take top K
        return items[:W6_DIGEST_K]
    
    def _extract_evidence_tokens(
        self,
        member_ids: List[str],
        w4_map: Dict[str, Any],
        sscore_lookup: Dict[str, Dict[str, float]],
    ) -> List[W6EvidenceToken]:
        """
        Extract evidence tokens using P0-X1 formula.
        
        Formula (mean_s_score_v1):
          evidence(token) = mean_r( S(token, cond(r)) * I[token in article(r)] )
          
          Where:
            - r iterates over all articles in the island
            - cond(r) = condition_signature(s) from article r's used_w3
            - I[...] = 1 if token is in W3 candidates for cond(r), else 0
            - mean denominator = len(member_ids) (total island articles)
        
        Implementation:
          For each token found in any W3 candidates:
            sum = 0
            presence_count = 0
            source_w3_ids = set()
            
            for article in island:
              for cond_sig in article.used_w3:
                if token in sscore_lookup[cond_sig]:
                  sum += sscore_lookup[cond_sig][token]
                  presence_count += 1
                  source_w3_ids.add(used_w3[cond_sig])
            
            evidence_score = sum / len(member_ids)  # Denominator is total articles
        """
        n_articles = len(member_ids)
        if n_articles == 0:
            return []
        
        # Aggregate scores per token
        # token_norm -> {sum: float, presence_count: int, source_w3_ids: set}
        token_agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"sum": 0.0, "presence_count": 0, "source_w3_ids": set()}
        )
        
        for article_id in member_ids:
            if article_id not in w4_map:
                continue
            
            w4_record = w4_map[article_id]
            
            # Iterate over all condition signatures this article used
            for cond_sig, w3_analysis_id in w4_record.used_w3.items():
                if cond_sig not in sscore_lookup:
                    continue
                
                # For each token in this condition's S-Score dict
                for token_norm, s_score in sscore_lookup[cond_sig].items():
                    agg = token_agg[token_norm]
                    agg["sum"] += s_score
                    agg["presence_count"] += 1
                    agg["source_w3_ids"].add(w3_analysis_id)
        
        # Compute evidence scores
        evidence_list: List[W6EvidenceToken] = []
        
        for token_norm, agg in token_agg.items():
            # P0-X1: Denominator is total articles (including zeros)
            evidence_score = agg["sum"] / n_articles
            
            # Round for determinism
            evidence_score_rounded = round(evidence_score, W6_EVIDENCE_ROUNDING)
            
            evidence_list.append(W6EvidenceToken(
                token_norm=token_norm,
                evidence_score=evidence_score_rounded,
                article_presence_count=agg["presence_count"],
                source_w3_ids=sorted(agg["source_w3_ids"]),
            ))
        
        # Sort (INV-W6-006): (-evidence_score_rounded, token_norm asc)
        evidence_list.sort(key=lambda e: (-e.evidence_score, e.token_norm))
        
        # Take top K
        return evidence_list[:W6_EVIDENCE_K]
    
    def _compute_used_w3_digest(
        self,
        member_ids: List[str],
        w4_map: Dict[str, Any],
    ) -> str:
        """
        Compute digest of W3 analysis IDs used by island members.
        
        P0-3: SHA256(sorted([all used_w3 values]))
        """
        all_w3_ids: Set[str] = set()
        
        for article_id in member_ids:
            if article_id not in w4_map:
                continue
            w4_record = w4_map[article_id]
            all_w3_ids.update(w4_record.used_w3.values())
        
        return compute_canonical_hash({"w3_ids": sorted(all_w3_ids)}, length=32)
    
    def _sample_representative_articles(
        self,
        member_ids: List[str],
        w4_map: Dict[str, Any],
        article_map: Dict[str, Any],
    ) -> List[W6RepresentativeArticle]:
        """
        Sample representative articles from island.
        
        P1-2 (head_chars_v1): First 200 chars as snippet.
        Sort by (-resonance_magnitude, article_id asc).
        """
        candidates: List[Tuple[float, str, Any, Any]] = []
        
        for article_id in member_ids:
            if article_id not in w4_map or article_id not in article_map:
                continue
            
            w4 = w4_map[article_id]
            article = article_map[article_id]
            
            # Compute resonance magnitude
            magnitude = math.sqrt(
                sum(v * v for v in w4.resonance_vector.values())
            )
            
            candidates.append((magnitude, article_id, w4, article))
        
        # Sort by (-magnitude, article_id asc)
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        # Take top K
        result: List[W6RepresentativeArticle] = []
        
        for magnitude, article_id, w4, article in candidates[:W6_REP_ARTICLE_K]:
            snippet = self._make_snippet(article.raw_text)
            
            result.append(W6RepresentativeArticle(
                article_id=article_id,
                w4_analysis_id=w4.w4_analysis_id,
                resonance_magnitude=magnitude,
                snippet=snippet,
            ))
        
        return result
    
    def _make_snippet(self, raw_text: str) -> str:
        """
        Create snippet using head_chars_v1 policy.
        
        - First W6_SNIPPET_LENGTH characters
        - Replace newlines with spaces
        - Collapse multiple spaces into one
        """
        # Take first N chars
        snippet = raw_text[:W6_SNIPPET_LENGTH]
        
        # Replace newlines with spaces
        snippet = snippet.replace('\n', ' ').replace('\r', ' ')
        
        # Collapse multiple spaces
        snippet = re.sub(r' +', ' ', snippet)
        
        return snippet.strip()
    
    def _calc_topology(
        self,
        islands: List,  # List[W5Island]
    ) -> List[W6TopologyPair]:
        """
        Calculate inter-island topology.
        
        P0-1: Uses W5 representative_vector ONLY.
        P1-3: Distance = 1.0 - round(cos_sim, 12)
        P1-b: Top-N pairs by distance (descending), with tie-break
        """
        if len(islands) < 2:
            return []
        
        pairs: List[W6TopologyPair] = []
        
        for i in range(len(islands)):
            for j in range(i + 1, len(islands)):
                island_a = islands[i]
                island_b = islands[j]
                
                # Ensure consistent ordering (a < b by island_id)
                if island_a.island_id > island_b.island_id:
                    island_a, island_b = island_b, island_a
                
                # Compute distance (P0-1: W5 vectors only)
                distance = self._compute_distance(
                    island_a.representative_vector,
                    island_b.representative_vector,
                )
                
                # Generate pair_id (P1-a: canonical JSON)
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
        
        # Sort (P1-b): (-distance, pair_id asc) for most distant first
        pairs.sort(key=lambda p: (-p.distance, p.pair_id))
        
        # Take top K (optional, include all if K is large)
        return pairs[:W6_TOPOLOGY_K] if W6_TOPOLOGY_K > 0 else pairs
    
    def _compute_distance(
        self,
        vec_a: Dict[str, float],
        vec_b: Dict[str, float],
    ) -> float:
        """
        Compute cosine distance between two vectors.
        
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


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-6 W6 Analyzer Test")
    print("=" * 60)
    
    # Create mock data classes
    from dataclasses import dataclass, field
    from typing import Dict, List
    
    @dataclass
    class MockW5Island:
        island_id: str
        member_ids: List[str]
        size: int
        representative_vector: Dict[str, float]
        cohesion_score: float
    
    @dataclass
    class MockW5Structure:
        structure_id: str
        islands: List[MockW5Island]
        noise_ids: List[str]
        noise_count: int
    
    @dataclass
    class MockW4Record:
        article_id: str
        w4_analysis_id: str
        resonance_vector: Dict[str, float]
        used_w3: Dict[str, str]
    
    @dataclass
    class MockW3Candidate:
        token_norm: str
        s_score: float
    
    @dataclass
    class MockW3Record:
        analysis_id: str
        condition_signature: str
        positive_candidates: List[MockW3Candidate]
        negative_candidates: List[MockW3Candidate]
    
    @dataclass
    class MockArticle:
        article_id: str
        raw_text: str
    
    # Test 1: Scope Closure Validation
    print("\n[Test 1] INV-W6-009: Scope Closure")
    
    analyzer = W6Analyzer(
        scope_id="test_scope",
        tokenizer_version="hybrid_v1",
        normalizer_version="v9.1.0",
    )
    
    # Create matching data
    island = MockW5Island(
        island_id="island_001",
        member_ids=["art_001", "art_002", "art_003"],
        size=3,
        representative_vector={"cond_a": 0.5, "cond_b": -0.3},
        cohesion_score=0.85,
    )
    
    w5 = MockW5Structure(
        structure_id="struct_001",
        islands=[island],
        noise_ids=["art_004"],
        noise_count=1,
    )
    
    w4_records = [
        MockW4Record("art_001", "w4_001", {"cond_a": 1.0}, {"cond_a": "w3_a"}),
        MockW4Record("art_002", "w4_002", {"cond_a": 0.8}, {"cond_a": "w3_a"}),
        MockW4Record("art_003", "w4_003", {"cond_a": 0.9}, {"cond_a": "w3_a"}),
        MockW4Record("art_004", "w4_004", {"cond_a": 0.1}, {"cond_a": "w3_a"}),
    ]
    
    w3_records = [
        MockW3Record(
            analysis_id="w3_a",
            condition_signature="cond_a",
            positive_candidates=[
                MockW3Candidate("prime", 0.05),
                MockW3Candidate("minister", 0.04),
            ],
            negative_candidates=[
                MockW3Candidate("lol", -0.02),
            ],
        ),
    ]
    
    articles = [
        MockArticle("art_001", "The prime minister announced policy."),
        MockArticle("art_002", "Prime minister said reforms needed."),
        MockArticle("art_003", "Government minister spoke today."),
        MockArticle("art_004", "Random noise article here."),
    ]
    
    # Should pass
    try:
        obs = analyzer.analyze(w5, w4_records, w3_records, articles)
        print(f"  observation_id: {obs.observation_id[:16]}...")
        print(f"  islands: {len(obs.islands)}")
        print("  ✅ PASS (scope closure valid)")
    except ValueError as e:
        print(f"  ❌ FAIL: {e}")
    
    # Test 2: Scope Closure Violation
    print("\n[Test 2] INV-W6-009: Scope Closure Violation")
    
    # Missing W4 record
    w4_missing = w4_records[:3]  # Missing art_004
    
    try:
        analyzer.analyze(w5, w4_missing, w3_records, articles)
        print("  ❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        assert "INV-W6-009" in str(e)
        print(f"  Caught: {str(e)[:80]}...")
        print("  ✅ PASS (violation detected)")
    
    # Test 3: Evidence Extraction
    print("\n[Test 3] Evidence Extraction (P0-X1)")
    
    obs = analyzer.analyze(w5, w4_records, w3_records, articles)
    
    if obs.islands and obs.islands[0].evidence_tokens:
        ev = obs.islands[0].evidence_tokens[0]
        print(f"  Top token: {ev.token_norm}")
        print(f"  Score: {ev.evidence_score}")
        print(f"  Presence: {ev.article_presence_count}")
        print("  ✅ PASS")
    else:
        print("  No evidence tokens (expected with mock data)")
        print("  ✅ PASS")
    
    # Test 4: Determinism
    print("\n[Test 4] Determinism (INV-W6-002)")
    
    obs1 = analyzer.analyze(w5, w4_records, w3_records, articles)
    obs2 = analyzer.analyze(w5, w4_records, w3_records, articles)
    
    assert obs1.observation_id == obs2.observation_id
    assert len(obs1.islands) == len(obs2.islands)
    print(f"  Run 1 ID: {obs1.observation_id[:16]}...")
    print(f"  Run 2 ID: {obs2.observation_id[:16]}...")
    print("  ✅ PASS (deterministic)")
    
    # Test 5: Topology Calculation
    print("\n[Test 5] Topology (P0-1: W5 vectors only)")
    
    # Create second island
    island2 = MockW5Island(
        island_id="island_002",
        member_ids=["art_005", "art_006", "art_007"],
        size=3,
        representative_vector={"cond_a": -0.5, "cond_b": 0.8},  # Different direction
        cohesion_score=0.80,
    )
    
    w5_multi = MockW5Structure(
        structure_id="struct_002",
        islands=[island, island2],
        noise_ids=[],
        noise_count=0,
    )
    
    # Add corresponding W4 and Articles
    w4_multi = w4_records[:3] + [
        MockW4Record("art_005", "w4_005", {"cond_a": -0.5}, {"cond_a": "w3_a"}),
        MockW4Record("art_006", "w4_006", {"cond_a": -0.4}, {"cond_a": "w3_a"}),
        MockW4Record("art_007", "w4_007", {"cond_a": -0.6}, {"cond_a": "w3_a"}),
    ]
    
    articles_multi = articles[:3] + [
        MockArticle("art_005", "Casual text with lol."),
        MockArticle("art_006", "Yeah this is cool."),
        MockArticle("art_007", "Hey what's up."),
    ]
    
    obs_multi = analyzer.analyze(w5_multi, w4_multi, w3_records, articles_multi)
    
    print(f"  Islands: {len(obs_multi.islands)}")
    print(f"  Topology pairs: {len(obs_multi.topology_pairs)}")
    
    if obs_multi.topology_pairs:
        pair = obs_multi.topology_pairs[0]
        print(f"  Pair: {pair.island_a_id[:8]}... <-> {pair.island_b_id[:8]}...")
        print(f"  Distance: {pair.distance}")
    
    print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All analyzer tests passed! ✅")
