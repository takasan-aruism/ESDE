#!/usr/bin/env python3
"""
ESDE Phase 9: W5/W6 Adapter
============================

Adapts new pipeline output (W4Result) to existing W5Condensator and W6 components.

This enables:
  - New W4 ArticleVector → Legacy W4Record format
  - Use existing W5Condensator for clustering
  - Use existing W6Analyzer/Exporter for output

Spec: Phase 9 W5/W6 Adapter v1.0
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone

from .w4_projector import W4Result, ArticleVector


# ==========================================
# W4Record-Compatible Structure
# ==========================================

@dataclass
class W4RecordCompat:
    """
    W4Record compatible structure for W5Condensator.
    
    Mirrors the schema expected by existing W5.
    """
    article_id: str
    w4_analysis_id: str
    resonance_vector: Dict[str, float]
    used_w3: Dict[str, str]
    token_count: int
    tokenizer_version: str = "pipeline_v1"
    normalizer_version: str = "v9.1.0"
    projection_norm: str = "raw"
    algorithm: str = "DotProduct-v1"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ==========================================
# Adapter Functions
# ==========================================

def compute_w4_analysis_id(
    article_id: str,
    w3_hash: str,
) -> str:
    """Compute deterministic W4 analysis ID."""
    data = {
        "article_id": article_id,
        "w3_hash": w3_hash,
    }
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def convert_to_w4_records(w4_result: W4Result) -> List[W4RecordCompat]:
    """
    Convert W4Result to list of W4RecordCompat for W5.
    
    Args:
        w4_result: W4Result from new pipeline
        
    Returns:
        List of W4RecordCompat compatible with W5Condensator
    """
    records = []
    
    for article_id, av in w4_result.articles.items():
        analysis_id = compute_w4_analysis_id(article_id, w4_result.w3_hash)
        
        # Build used_w3 mapping (condition_id → w3_hash)
        used_w3 = {cid: w4_result.w3_hash for cid in w4_result.condition_ids}
        
        record = W4RecordCompat(
            article_id=article_id,
            w4_analysis_id=analysis_id,
            resonance_vector=av.resonance_vector,
            used_w3=used_w3,
            token_count=av.token_count,
        )
        records.append(record)
    
    return records


# ==========================================
# Simple W5 Condensator (Standalone)
# ==========================================

@dataclass
class SimpleIsland:
    """Simple island structure."""
    island_id: str
    member_ids: List[str]
    size: int
    representative_vector: Dict[str, float]
    cohesion_score: float


@dataclass
class SimpleStructure:
    """Simple W5 structure."""
    structure_id: str
    islands: List[SimpleIsland]
    noise_ids: List[str]
    input_count: int
    island_count: int
    noise_count: int
    threshold: float
    min_island_size: int


def l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """L2 normalize a vector."""
    magnitude = math.sqrt(sum(v * v for v in vec.values()))
    if magnitude == 0:
        return vec
    return {k: v / magnitude for k, v in vec.items()}


def cosine_sim(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """Compute cosine similarity."""
    keys = set(v1.keys()) | set(v2.keys())
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in keys)
    m1 = math.sqrt(sum(v1.get(k, 0) ** 2 for k in keys))
    m2 = math.sqrt(sum(v2.get(k, 0) ** 2 for k in keys))
    if m1 == 0 or m2 == 0:
        return 0.0
    return dot / (m1 * m2)


def compute_structure_hash(data: Dict) -> str:
    """Compute canonical hash."""
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class SimpleCondensator:
    """
    Simple W5-style condensator for the new pipeline.
    
    Uses cosine similarity threshold to form islands.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        min_island_size: int = 2,
    ):
        self.threshold = threshold
        self.min_island_size = min_island_size
    
    def condense(self, records: List[W4RecordCompat]) -> SimpleStructure:
        """
        Condense W4 records into islands.
        
        Args:
            records: List of W4RecordCompat
            
        Returns:
            SimpleStructure with islands
        """
        if not records:
            return SimpleStructure(
                structure_id=compute_structure_hash({"empty": True}),
                islands=[],
                noise_ids=[],
                input_count=0,
                island_count=0,
                noise_count=0,
                threshold=self.threshold,
                min_island_size=self.min_island_size,
            )
        
        n = len(records)
        
        # Build adjacency from similarity
        adj: Dict[str, set] = {r.article_id: set() for r in records}
        edge_sims: Dict[tuple, float] = {}
        
        for i in range(n):
            for j in range(i + 1, n):
                r1, r2 = records[i], records[j]
                v1 = l2_normalize(r1.resonance_vector)
                v2 = l2_normalize(r2.resonance_vector)
                sim = round(cosine_sim(v1, v2), 12)
                
                if sim >= self.threshold:
                    adj[r1.article_id].add(r2.article_id)
                    adj[r2.article_id].add(r1.article_id)
                    edge_sims[(r1.article_id, r2.article_id)] = sim
        
        # Find connected components (DFS)
        visited = set()
        components = []
        
        for r in records:
            if r.article_id in visited:
                continue
            
            component = []
            stack = [r.article_id]
            visited.add(r.article_id)
            
            while stack:
                curr = stack.pop()
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            components.append(sorted(component))
        
        # Build islands
        islands = []
        noise_ids = []
        
        record_map = {r.article_id: r for r in records}
        
        for comp in components:
            if len(comp) < self.min_island_size:
                noise_ids.extend(comp)
                continue
            
            # Compute centroid
            dim_sums: Dict[str, float] = {}
            for aid in comp:
                vec = record_map[aid].resonance_vector
                for k, v in vec.items():
                    dim_sums[k] = dim_sums.get(k, 0) + v
            
            centroid = {k: v / len(comp) for k, v in dim_sums.items()}
            
            # Compute cohesion (average edge similarity)
            edges = []
            for i, a1 in enumerate(comp):
                for a2 in comp[i+1:]:
                    key = tuple(sorted((a1, a2)))
                    if key in edge_sims:
                        edges.append(edge_sims[key])
            
            cohesion = sum(edges) / len(edges) if edges else 1.0
            
            # Island ID
            island_id = compute_structure_hash({"members": comp})
            
            islands.append(SimpleIsland(
                island_id=island_id,
                member_ids=comp,
                size=len(comp),
                representative_vector=centroid,
                cohesion_score=cohesion,
            ))
        
        # Structure ID
        structure_id = compute_structure_hash({
            "input_ids": sorted([r.article_id for r in records]),
            "threshold": self.threshold,
            "min_island_size": self.min_island_size,
        })
        
        return SimpleStructure(
            structure_id=structure_id,
            islands=sorted(islands, key=lambda x: x.island_id),
            noise_ids=sorted(noise_ids),
            input_count=n,
            island_count=len(islands),
            noise_count=len(noise_ids),
            threshold=self.threshold,
            min_island_size=self.min_island_size,
        )
    
    def condense_from_edges(
        self,
        node_ids: List[str],
        node_vectors: Dict[str, Dict[str, float]],
        edges: List[tuple],
    ) -> 'SimpleStructure':
        """
        Build structure from pre-filtered edges.
        
        Unlike condense(), this method does NOT recompute similarities.
        It takes edges that have already been filtered (e.g., by Mutual-kNN
        EdgeSelector) and builds connected components from them.
        
        Args:
            node_ids: All node identifiers.
            node_vectors: {node_id: {dim: value, ...}} for centroid computation.
            edges: [(a, b, similarity), ...] — pre-filtered, all above threshold.
            
        Returns:
            SimpleStructure with islands.
        """
        n = len(node_ids)
        
        if n == 0:
            return SimpleStructure(
                structure_id=compute_structure_hash({"empty": True, "from_edges": True}),
                islands=[],
                noise_ids=[],
                input_count=0,
                island_count=0,
                noise_count=0,
                threshold=self.threshold,
                min_island_size=self.min_island_size,
            )
        
        # Build adjacency from provided edges
        adj: Dict[str, set] = {nid: set() for nid in node_ids}
        edge_sims: Dict[tuple, float] = {}
        
        for a, b, sim in edges:
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
                edge_sims[tuple(sorted((a, b)))] = sim
        
        # Find connected components (DFS) — same as condense()
        visited = set()
        components = []
        
        for nid in sorted(node_ids):
            if nid in visited:
                continue
            
            component = []
            stack = [nid]
            visited.add(nid)
            
            while stack:
                curr = stack.pop()
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            
            components.append(sorted(component))
        
        # Build islands — same logic as condense()
        islands = []
        noise_ids_list = []
        
        for comp in components:
            if len(comp) < self.min_island_size:
                noise_ids_list.extend(comp)
                continue
            
            # Compute centroid
            dim_sums: Dict[str, float] = {}
            for aid in comp:
                vec = node_vectors.get(aid, {})
                for k, v in vec.items():
                    dim_sums[k] = dim_sums.get(k, 0) + v
            
            centroid = {k: v / len(comp) for k, v in dim_sums.items()}
            
            # Compute cohesion (average edge similarity within component)
            edge_list = []
            for i, a1 in enumerate(comp):
                for a2 in comp[i+1:]:
                    key = tuple(sorted((a1, a2)))
                    if key in edge_sims:
                        edge_list.append(edge_sims[key])
            
            cohesion = sum(edge_list) / len(edge_list) if edge_list else 1.0
            
            island_id = compute_structure_hash({"members": comp})
            
            islands.append(SimpleIsland(
                island_id=island_id,
                member_ids=comp,
                size=len(comp),
                representative_vector=centroid,
                cohesion_score=cohesion,
            ))
        
        structure_id = compute_structure_hash({
            "input_ids": sorted(node_ids),
            "threshold": self.threshold,
            "min_island_size": self.min_island_size,
            "method": "from_edges",
        })
        
        return SimpleStructure(
            structure_id=structure_id,
            islands=sorted(islands, key=lambda x: x.island_id),
            noise_ids=sorted(noise_ids_list),
            input_count=n,
            island_count=len(islands),
            noise_count=len(noise_ids_list),
            threshold=self.threshold,
            min_island_size=self.min_island_size,
        )


# ==========================================
# Simple W6 Exporter
# ==========================================

def export_structure_markdown(
    structure: SimpleStructure,
    w3_result: Any,
    output_path: str,
) -> str:
    """
    Export W5 structure to Markdown.
    
    Args:
        structure: SimpleStructure from condensator
        w3_result: W3Result for evidence tokens
        output_path: Output file path
        
    Returns:
        Markdown content
    """
    lines = []
    lines.append("# ESDE Phase 9: Structural Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Input Articles**: {structure.input_count}")
    lines.append(f"- **Islands Found**: {structure.island_count}")
    lines.append(f"- **Noise Articles**: {structure.noise_count}")
    lines.append(f"- **Threshold**: {structure.threshold}")
    lines.append(f"- **Min Island Size**: {structure.min_island_size}")
    lines.append("")
    
    # Islands
    lines.append("## Islands")
    lines.append("")
    
    for i, island in enumerate(structure.islands, 1):
        lines.append(f"### Island {i}: {island.size} members")
        lines.append("")
        lines.append(f"- **ID**: `{island.island_id[:16]}...`")
        lines.append(f"- **Cohesion**: {island.cohesion_score:.4f}")
        lines.append(f"- **Members**: {', '.join(island.member_ids)}")
        lines.append("")
        
        # Representative vector
        lines.append("**Representative Vector:**")
        lines.append("")
        lines.append("| Condition | Score |")
        lines.append("|-----------|-------|")
        for cid, score in sorted(island.representative_vector.items(), key=lambda x: -x[1]):
            lines.append(f"| {cid} | {score:+.4f} |")
        lines.append("")
    
    # Noise
    if structure.noise_ids:
        lines.append("## Noise (Unclustered)")
        lines.append("")
        lines.append(f"Articles: {', '.join(structure.noise_ids)}")
        lines.append("")
    
    # Conditions (from W3)
    if w3_result:
        lines.append("## Condition S-Scores")
        lines.append("")
        
        for cid, cond in w3_result.conditions.items():
            lines.append(f"### {cid}")
            lines.append("")
            
            if cond.positive_candidates:
                lines.append("**Top Positive:**")
                for c in cond.positive_candidates[:5]:
                    lines.append(f"- {c.token}: S={c.s_score:+.6f}")
                lines.append("")
            
            if cond.negative_candidates:
                lines.append("**Top Negative:**")
                for c in cond.negative_candidates[:3]:
                    lines.append(f"- {c.token}: S={c.s_score:+.6f}")
                lines.append("")
    
    content = "\n".join(lines)
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content


def export_structure_json(
    structure: SimpleStructure,
    w3_result: Any,
    w4_result: W4Result,
    output_path: str,
) -> Dict:
    """
    Export full analysis to JSON.
    """
    data = {
        "version": "phase9_pipeline_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "structure": {
            "structure_id": structure.structure_id,
            "input_count": structure.input_count,
            "island_count": structure.island_count,
            "noise_count": structure.noise_count,
            "threshold": structure.threshold,
            "min_island_size": structure.min_island_size,
            "islands": [
                {
                    "island_id": isl.island_id,
                    "size": isl.size,
                    "members": isl.member_ids,
                    "cohesion": isl.cohesion_score,
                    "representative_vector": {
                        k: round(v, 6) for k, v in isl.representative_vector.items()
                    },
                }
                for isl in structure.islands
            ],
            "noise_ids": structure.noise_ids,
        },
        "articles": {
            aid: {
                "token_count": av.token_count,
                "resonance_vector": {k: round(v, 6) for k, v in av.resonance_vector.items()},
            }
            for aid, av in w4_result.articles.items()
        },
        "conditions": {
            cid: {
                "positive": [
                    {"token": c.token, "s_score": round(c.s_score, 6)}
                    for c in cond.positive_candidates[:10]
                ],
                "negative": [
                    {"token": c.token, "s_score": round(c.s_score, 6)}
                    for c in cond.negative_candidates[:10]
                ],
            }
            for cid, cond in w3_result.conditions.items()
        },
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("W5/W6 Adapter Test")
    print("=" * 60)
    
    # Create mock W4 data
    from .w4_projector import W4Result, ArticleVector
    
    w4_result = W4Result(
        provider_id="section_v1",
        axis_name="section",
        w3_hash="test_hash",
        condition_ids=["lead", "military", "death"],
    )
    
    w4_result.articles["nobunaga"] = ArticleVector(
        article_id="nobunaga",
        resonance_vector={"lead": 0.5, "military": 0.3, "death": 0.2},
        token_count=50,
    )
    w4_result.articles["hideyoshi"] = ArticleVector(
        article_id="hideyoshi",
        resonance_vector={"lead": 0.4, "military": 0.35, "death": 0.25},
        token_count=45,
    )
    w4_result.articles["ieyasu"] = ArticleVector(
        article_id="ieyasu",
        resonance_vector={"lead": 0.45, "military": 0.25, "death": 0.3},
        token_count=48,
    )
    
    # Convert to W4Records
    print("\n[1] Converting to W4RecordCompat...")
    records = convert_to_w4_records(w4_result)
    print(f"  Converted {len(records)} records")
    
    # Condense
    print("\n[2] Condensing...")
    condensator = SimpleCondensator(threshold=0.9, min_island_size=2)
    structure = condensator.condense(records)
    
    print(f"  Islands: {structure.island_count}")
    print(f"  Noise: {structure.noise_count}")
    
    for isl in structure.islands:
        print(f"  Island: {isl.member_ids} (cohesion: {isl.cohesion_score:.4f})")
    
    print("\n✅ W5/W6 Adapter working!")
