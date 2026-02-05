"""
ESDE Integration - Relation Logger
====================================
Grounds SVO triples onto ESDE Atoms via Synapse lookup,
then writes relation edges to JSONL.

Pipeline:
  ParserAdapter (SVO triples)
      ↓
  RelationLogger (this module)
      ↓
  relations_edges.jsonl  (raw per-sentence edges)

Grounding strategy:
  - verb_lemma → WordNet synsets → Synapse edges → Atom candidates
  - subject/object → Named Entity or raw text (no Atom grounding for entities)
  - No winner selection. All candidates preserved (describe, don't decide)
  - Operator is always ▷ (ACT) for this prototype

Spec: Phase 8 Integration Design Brief, Step 2
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


def _ensure_wordnet():
    """Download WordNet data if not present."""
    if not WORDNET_AVAILABLE:
        return False
    try:
        wn.synsets("test")
        return True
    except LookupError:
        import nltk
        print("[SynapseGrounder] Downloading WordNet data...")
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        return True

from .parser_adapter import SVOTriple, ExtractionResult


VERSION = "0.1.0"

# Default Synapse file location (relative to project root esde/)
DEFAULT_SYNAPSE_PATH = "esde_synapses_v3.json"

# WordNet POS tags to search for verbs
VERB_POS = ["v"]

# Max synsets per verb lemma
MAX_SYNSETS_PER_VERB = 10


# ==========================================
# Synapse Grounding
# ==========================================

class SynapseGrounder:
    """
    Grounds verb lemmas onto ESDE Atoms via WordNet → Synapse lookup.
    
    Uses the same Synapse data as Sensor V2, but only for verb predicates.
    Does NOT select a winner. Returns all candidates with scores.
    
    Can operate with or without Synapse data:
    - With Synapse: verb_lemma → synsets → Synapse edges → Atom candidates
    - Without Synapse: verb_lemma stored as raw text, atom_candidates = []
    """
    
    def __init__(self, synapse_data: Optional[Dict[str, List[Dict]]] = None):
        """
        Args:
            synapse_data: The "synapses" dict from synapse_v3.json.
                          If None, operates in raw mode (no grounding).
        """
        self.synapses = synapse_data or {}
        self._cache: Dict[str, List[Dict]] = {}
    
    @classmethod
    def from_file(cls, filepath: str) -> "SynapseGrounder":
        """Load from synapse JSON file."""
        path = Path(filepath)
        if not path.exists():
            print(f"[SynapseGrounder] File not found: {filepath}, running in raw mode")
            return cls(synapse_data=None)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        synapses = data.get("synapses", {})
        print(f"[SynapseGrounder] Loaded {len(synapses)} synsets from Synapse")
        return cls(synapse_data=synapses)
    
    def ground_verb(self, verb_lemma: str) -> List[Dict[str, Any]]:
        """
        Ground a verb lemma to ESDE Atom candidates.
        
        Returns list of candidates, each:
          {"concept_id": "ACT.attack", "axis": "...", "raw_score": 0.85, "synset": "..."}
        
        Returns empty list if no grounding found (raw mode or no match).
        """
        if verb_lemma in self._cache:
            return self._cache[verb_lemma]
        
        if not self.synapses or not WORDNET_AVAILABLE:
            self._cache[verb_lemma] = []
            return []
        
        # Ensure WordNet data is available (auto-download on first call)
        if not hasattr(self, '_wordnet_ready'):
            self._wordnet_ready = _ensure_wordnet()
            if not self._wordnet_ready:
                self._cache[verb_lemma] = []
                return []
        
        # Step 1: verb_lemma → WordNet synsets
        synsets = wn.synsets(verb_lemma, pos=wn.VERB)[:MAX_SYNSETS_PER_VERB]
        
        if not synsets:
            self._cache[verb_lemma] = []
            return []
        
        # Step 2: synsets → Synapse edges → Atom candidates
        candidates_map: Dict[str, Dict] = {}  # concept_id → best candidate
        
        for syn in synsets:
            synset_id = syn.name()
            edges = self.synapses.get(synset_id, [])
            
            for edge in edges:
                cid = edge.get("concept_id", "")
                raw_score = edge.get("raw_score", 0.0)
                
                if cid not in candidates_map or raw_score > candidates_map[cid]["raw_score"]:
                    candidates_map[cid] = {
                        "concept_id": cid,
                        "axis": edge.get("axis", ""),
                        "level": edge.get("level", ""),
                        "raw_score": raw_score,
                        "synset": synset_id,
                    }
        
        # Sort by score descending, then concept_id for determinism
        candidates = sorted(
            candidates_map.values(),
            key=lambda c: (-c["raw_score"], c["concept_id"])
        )
        
        self._cache[verb_lemma] = candidates
        return candidates


# ==========================================
# Relation Edge
# ==========================================

def _build_edge(
    triple: SVOTriple,
    atom_candidates: List[Dict],
    section_name: str,
    article_id: str,
) -> Dict[str, Any]:
    """
    Build a single relation edge from an SVO triple + grounding result.
    
    Output format matches the design brief:
    {
        "source": "Nobunaga",
        "target": "Azai",
        "atom": "ACT.attack",          # top candidate or "UNGROUNDED"
        "atom_candidates": [...],       # all candidates with scores
        "operator": "ACT",
        "negated": false,
        "passive": false,
        "verb_lemma": "attack",
        "verb_raw": "attacked",
        "section": "battle_of_anegawa__course",
        "article": "oda_nobunaga",
        "sentence_idx": 3,
        "text_ref": "Nobunaga attacked the Azai clan"
    }
    """
    # Top atom candidate (or UNGROUNDED)
    top_atom = "UNGROUNDED"
    if atom_candidates:
        top_atom = atom_candidates[0]["concept_id"]
    
    return {
        "source": triple.subject,
        "target": triple.object,
        "atom": top_atom,
        "atom_candidates": atom_candidates[:5],  # Keep top 5 for audit
        "operator": "ACT",
        "negated": triple.negated,
        "passive": triple.passive,
        "verb_lemma": triple.verb_lemma,
        "verb_raw": triple.verb,
        "section": section_name,
        "article": article_id,
        "sentence_idx": triple.sentence_idx,
        "text_ref": triple.sentence_text.strip(),
    }


# ==========================================
# Relation Logger (Public API)
# ==========================================

class RelationLogger:
    """
    Converts SVO triples to ESDE relation edges with Synapse grounding.
    
    Usage:
        grounder = SynapseGrounder.from_file("esde_synapses_v3.json")
        logger = RelationLogger(grounder)
        
        # Process one section
        edges = logger.process_section(extraction_result, "battle_section", "nobunaga")
        
        # Write all edges to JSONL
        logger.write_jsonl("output/relations_edges.jsonl")
    """
    
    def __init__(self, grounder: Optional[SynapseGrounder] = None):
        self.grounder = grounder or SynapseGrounder()
        self._all_edges: List[Dict[str, Any]] = []
        self._stats = {
            "sections_processed": 0,
            "triples_processed": 0,
            "grounded": 0,
            "ungrounded": 0,
        }
    
    def process_section(
        self,
        result: ExtractionResult,
        section_name: str,
        article_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Process SVO extraction result for one section.
        
        Returns list of relation edges.
        """
        self._stats["sections_processed"] += 1
        edges = []
        
        for triple in result.triples:
            self._stats["triples_processed"] += 1
            
            # Ground verb
            candidates = self.grounder.ground_verb(triple.verb_lemma)
            
            if candidates:
                self._stats["grounded"] += 1
            else:
                self._stats["ungrounded"] += 1
            
            edge = _build_edge(triple, candidates, section_name, article_id)
            edges.append(edge)
        
        self._all_edges.extend(edges)
        return edges
    
    def process_article(
        self,
        sections: Dict[str, ExtractionResult],
        article_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Process all sections for an article.
        
        Args:
            sections: {section_name: ExtractionResult}
            article_id: Article identifier
        
        Returns:
            All edges for the article.
        """
        all_edges = []
        for section_name, result in sections.items():
            edges = self.process_section(result, section_name, article_id)
            all_edges.extend(edges)
        return all_edges
    
    def write_jsonl(self, filepath: str) -> int:
        """
        Write all accumulated edges to JSONL file.
        
        Returns number of edges written.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            for edge in self._all_edges:
                f.write(json.dumps(edge, ensure_ascii=False) + "\n")
        
        return len(self._all_edges)
    
    def get_edges(self) -> List[Dict[str, Any]]:
        """Return all accumulated edges."""
        return self._all_edges
    
    def get_stats(self) -> Dict[str, Any]:
        """Return processing statistics."""
        total = self._stats["grounded"] + self._stats["ungrounded"]
        grounding_rate = (
            self._stats["grounded"] / total if total > 0 else 0.0
        )
        return {
            "version": VERSION,
            **self._stats,
            "grounding_rate": round(grounding_rate, 4),
            "total_edges": len(self._all_edges),
        }
    
    def clear(self):
        """Clear accumulated edges."""
        self._all_edges = []


# ==========================================
# Aggregation: Entity Graph + Section Profile
# ==========================================

def aggregate_entity_graph(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate edges into an entity graph (UI-facing).
    
    Output: entity_graph.json format from design brief.
    """
    nodes: Dict[str, Dict] = defaultdict(lambda: {
        "degree": 0,
        "atoms": defaultdict(int),
        "as_source": 0,
        "as_target": 0,
    })
    
    edge_agg: Dict[str, Dict] = {}  # (source, target, atom) → aggregated edge
    
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        atom = e["atom"]
        
        # Update nodes
        nodes[src]["degree"] += 1
        nodes[src]["as_source"] += 1
        nodes[src]["atoms"][atom] += 1
        
        nodes[tgt]["degree"] += 1
        nodes[tgt]["as_target"] += 1
        nodes[tgt]["atoms"][atom] += 1
        
        # Aggregate edges
        key = f"{src}||{tgt}||{atom}"
        if key not in edge_agg:
            edge_agg[key] = {
                "source": src,
                "target": tgt,
                "atom": atom,
                "count": 0,
                "text_refs": [],
                "sections": set(),
            }
        edge_agg[key]["count"] += 1
        if e.get("text_ref") and len(edge_agg[key]["text_refs"]) < 3:
            edge_agg[key]["text_refs"].append(e["text_ref"][:100])
        edge_agg[key]["sections"].add(e.get("section", ""))
    
    # Format output
    formatted_nodes = {}
    for name, data in nodes.items():
        top_atoms = sorted(data["atoms"].items(), key=lambda x: -x[1])[:5]
        formatted_nodes[name] = {
            "degree": data["degree"],
            "as_source": data["as_source"],
            "as_target": data["as_target"],
            "top_atoms": [{"atom": a, "count": c} for a, c in top_atoms],
        }
    
    formatted_edges = []
    for data in edge_agg.values():
        formatted_edges.append({
            "source": data["source"],
            "target": data["target"],
            "atom": data["atom"],
            "count": data["count"],
            "text_refs": data["text_refs"],
            "sections": sorted(data["sections"]),
        })
    
    # Sort edges by count descending
    formatted_edges.sort(key=lambda e: -e["count"])
    
    return {
        "nodes": formatted_nodes,
        "edges": formatted_edges,
        "meta": {
            "version": VERSION,
            "total_nodes": len(formatted_nodes),
            "total_edges": len(formatted_edges),
            "total_raw_edges": len(edges),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def aggregate_section_profile(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate edges into section-level profiles (Phase 9 Lens input).
    
    Output: section_relation_profile.json format from design brief.
    Each section gets a vector of predicate atom counts + structural stats.
    """
    sections: Dict[str, Dict] = defaultdict(lambda: {
        "predicate_atoms": defaultdict(int),
        "entity_count": set(),
        "edge_count": 0,
        "source_count": 0,
        "target_count": 0,
        "negated_count": 0,
        "passive_count": 0,
    })
    
    for e in edges:
        sec = e.get("section", "unknown")
        atom = e["atom"]
        
        sections[sec]["predicate_atoms"][atom] += 1
        sections[sec]["entity_count"].add(e["source"])
        sections[sec]["entity_count"].add(e["target"])
        sections[sec]["edge_count"] += 1
        sections[sec]["source_count"] += 1
        if e.get("negated"):
            sections[sec]["negated_count"] += 1
        if e.get("passive"):
            sections[sec]["passive_count"] += 1
    
    # Format output
    result = {}
    for sec_name, data in sections.items():
        total = data["edge_count"]
        # Directionality: ratio of unique sources to total entities
        entity_count = len(data["entity_count"])
        directionality = (
            data["source_count"] / total if total > 0 else 0.0
        )
        
        result[sec_name] = {
            "predicate_atoms": dict(data["predicate_atoms"]),
            "entity_count": entity_count,
            "edge_count": total,
            "negated_ratio": round(data["negated_count"] / total, 4) if total else 0.0,
            "passive_ratio": round(data["passive_count"] / total, 4) if total else 0.0,
            "directionality": round(directionality, 4),
        }
    
    return result


# ==========================================
# Full Pipeline: Extract → Ground → Aggregate → Write
# ==========================================

def run_relation_pipeline(
    sections: List[Dict[str, str]],
    article_id: str,
    synapse_path: Optional[str] = None,
    output_dir: str = "output",
) -> Dict[str, Any]:
    """
    Run the full relation extraction pipeline for an article.
    
    Args:
        sections: List of {"title": "...", "content": "..."} dicts
        article_id: Article identifier
        synapse_path: Path to esde_synapses_v3.json (optional, auto-detects if None)
        output_dir: Output directory
    
    Returns:
        Summary dict with file paths and statistics.
    """
    from .parser_adapter import ParserAdapter
    
    # Initialize
    adapter = ParserAdapter()
    grounder = (
        SynapseGrounder.from_file(synapse_path)
        if synapse_path
        else SynapseGrounder()
    )
    logger = RelationLogger(grounder)
    
    # Extract + Ground
    print(f"[RelationPipeline] Processing {len(sections)} sections for '{article_id}'...")
    
    for sec in sections:
        title = sec.get("title", "untitled")
        content = sec.get("content", "")
        
        extraction = adapter.extract(content, section_name=title)
        logger.process_section(extraction, title, article_id)
    
    edges = logger.get_edges()
    stats = logger.get_stats()
    
    print(f"  Edges: {len(edges)}")
    print(f"  Grounding rate: {stats['grounding_rate']:.1%}")
    
    # Write raw edges
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = out / "relations_edges.jsonl"
    logger.write_jsonl(str(jsonl_path))
    
    # Aggregate: entity graph
    entity_graph = aggregate_entity_graph(edges)
    graph_path = out / "entity_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(entity_graph, f, indent=2, ensure_ascii=False)
    
    # Aggregate: section profile
    section_profile = aggregate_section_profile(edges)
    profile_path = out / "section_relation_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(section_profile, f, indent=2, ensure_ascii=False)
    
    # Also write relations.json for app.py (combined format)
    relations_json = {
        "nodes": entity_graph["nodes"],
        "edges": entity_graph["edges"],
        "meta": entity_graph["meta"],
    }
    relations_path = out / "relations.json"
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(relations_json, f, indent=2, ensure_ascii=False)
    
    return {
        "article_id": article_id,
        "files": {
            "edges_jsonl": str(jsonl_path),
            "entity_graph": str(graph_path),
            "section_profile": str(profile_path),
            "relations_json": str(relations_path),
        },
        "stats": stats,
        "parser_stats": adapter.get_stats(),
    }