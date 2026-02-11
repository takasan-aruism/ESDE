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

Grounding strategy (v0.3.2 — Conditional Guard + Penalized Fallback):
  - verb_lemma → WordNet synsets (POS=VERB) → Synapse edges → Atom candidates
  - Conditional Primary-Lemma Guard (v0.3.1):
      If primary-lemma synsets have Synapse edges → block secondary paths
      If no primary-lemma synset has edges → allow secondary with penalty (v0.3.2)
  - Secondary Fallback Penalty (v0.3.2):
      Secondary path candidates receive score *= SECONDARY_PENALTY (0.7)
      This preserves coverage while demoting uncertain groundings
  - POS Guard: Atom candidates with noun-category (NAT/MAT/PRP/SPA) are filtered
  - Light Verb Stoplist: Functional verbs bypass grounding → UNGROUNDED_LIGHTVERB
  - Minimum Score Threshold: Candidates below threshold → UNGROUNDED
  - subject/object → Named Entity or raw text (no Atom grounding for entities)
  - No winner selection. All candidates preserved (describe, don't decide)
  - Operator is always ▷ (ACT) for this prototype

Spec: Phase 8 Integration Design Brief, Step 2 + Diagnostic Prescription C-1
3AI Approval: Gemini (design) → GPT (audit) → Claude (implementation)

Changelog:
  v0.2.0 — Hardened filters: POS Guard, Light Verb Stoplist, Score Threshold
  v0.3.0 — Primary-Lemma Guard (hard): filter ALL secondary lemma paths
            Too aggressive — grounding rate dropped from 62% to 40%
  v0.3.1 — Conditional Primary-Lemma Guard:
            Block secondary ONLY when primary has edges; else allow fallback.
            Recovered to 49.4% but win→EMO.pride still leaked via fallback.
  v0.3.2 — Penalized Fallback:
            Secondary fallback candidates get score *= 0.7 penalty.
            Preserves coverage while demoting uncertain secondary groundings.
            "If A is undefined, accept B as provisional footing — but mark it weaker"
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

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


VERSION = "0.3.2"

# Default Synapse file location (relative to project root esde/)
DEFAULT_SYNAPSE_PATH = "esde_synapses_v3.json"

# WordNet POS tags to search for verbs
VERB_POS = ["v"]

# Max synsets per verb lemma
MAX_SYNSETS_PER_VERB = 10


# ==========================================
# Grounding Filters (v0.2.0 + v0.3.2)
# ==========================================

# --- Filter 1: Light Verb Stoplist ---
LIGHT_VERB_STOPLIST = frozenset({
    "have", "make", "do", "get", "take", "give", "go", "come",
    "be", "become", "include", "feature", "provide",
})

# --- Filter 2: POS Guard (Atom Category Filter) ---
POS_GUARD_BLOCKED_CATEGORIES = frozenset({
    "NAT",  # Nature (e.g., NAT.water) — not a verb concept
    "MAT",  # Material (e.g., MAT.metal) — not a verb concept
    "PRP",  # Property (e.g., PRP.young, PRP.dirty) — adjective, not verb
    "SPA",  # Space (e.g., SPA.inside) — spatial, not action
})

# --- Filter 3: Minimum Score Threshold ---
DEFAULT_MIN_SCORE = 0.45

# --- Filter 4: Secondary Fallback Penalty (v0.3.2) ---
# When no primary-lemma synset has Synapse edges, secondary paths are accepted
# but their raw_score is multiplied by this factor to demote them.
# GPT audit recommendation: start at 0.7, adjust based on diagnostics.
SECONDARY_PENALTY = 0.9


# ==========================================
# Synapse Grounding
# ==========================================

class SynapseGrounder:
    """
    Grounds verb lemmas onto ESDE Atoms via WordNet → Synapse lookup.
    
    v0.3.2 Filters (3AI approved):
      0. Conditional Primary-Lemma Guard:
         - If any primary-lemma synset has Synapse edges → block secondary paths
         - If no primary-lemma synset has edges → allow secondary with penalty
      1. Secondary Fallback Penalty: score *= 0.7 for secondary path candidates
      2. POS Guard: Block noun-category Atoms (NAT/MAT/PRP/SPA)
      3. Score Threshold: Drop candidates below min_score
    
    Does NOT select a winner. Returns all surviving candidates with scores.
    """
    
    def __init__(
        self,
        synapse_data: Optional[Dict[str, List[Dict]]] = None,
        min_score: float = DEFAULT_MIN_SCORE,
        secondary_penalty: float = SECONDARY_PENALTY,
    ):
        self.synapses = synapse_data or {}
        self.min_score = min_score
        self.secondary_penalty = secondary_penalty
        self._cache: Dict[str, List[Dict]] = {}
        self._filter_log = {
            "pos_guard_dropped": 0,
            "threshold_dropped": 0,
            "threshold_drop_details": [],
            "secondary_lemma_filtered": 0,
            "secondary_lemma_accepted": 0,
            "secondary_lemma_penalty_kills": 0,
            "secondary_lemma_details": [],
        }
    
    @classmethod
    def from_file(
        cls,
        filepath: str,
        min_score: float = DEFAULT_MIN_SCORE,
        secondary_penalty: float = SECONDARY_PENALTY,
    ) -> "SynapseGrounder":
        """Load from synapse JSON file."""
        path = Path(filepath)
        if not path.exists():
            print(f"[SynapseGrounder] File not found: {filepath}, running in raw mode")
            return cls(synapse_data=None, min_score=min_score,
                       secondary_penalty=secondary_penalty)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        synapses = data.get("synapses", {})
        print(f"[SynapseGrounder] Loaded {len(synapses)} synsets from Synapse")
        print(f"[SynapseGrounder] min_score={min_score}, "
              f"secondary_penalty={secondary_penalty}, "
              f"POS Guard={sorted(POS_GUARD_BLOCKED_CATEGORIES)}")
        return cls(synapse_data=synapses, min_score=min_score,
                   secondary_penalty=secondary_penalty)
    
    def ground_verb(self, verb_lemma: str) -> List[Dict[str, Any]]:
        """
        Ground a verb lemma to ESDE Atom candidates.
        
        Pipeline (v0.3.2):
          1. verb_lemma → WordNet synsets (POS=VERB only)
          2. Conditional Primary-Lemma Guard:
             a. Classify synsets as primary or secondary
             b. If ANY primary synset has Synapse edges → block secondary
             c. If NO primary synset has edges → accept secondary with penalty
          3. Accepted synsets → Synapse edges → raw candidates
             (secondary candidates: raw_score *= secondary_penalty)
          4. POS Guard: remove candidates with blocked categories
          5. Score Threshold: remove candidates below min_score
          6. Sort by score descending
        
        Returns list of surviving candidates, each:
          {"concept_id": str, "axis": str, "raw_score": float,
           "synset": str, "path_kind": "primary"|"secondary"}
        """
        if verb_lemma in self._cache:
            return self._cache[verb_lemma]
        
        if not self.synapses or not WORDNET_AVAILABLE:
            self._cache[verb_lemma] = []
            return []
        
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
        
        # ── Step 2: Conditional Primary-Lemma Guard ──
        primary_synsets = []
        secondary_synsets = []
        
        for syn in synsets:
            primary_lemma = syn.lemmas()[0].name().lower().replace("_", " ")
            if primary_lemma == verb_lemma:
                primary_synsets.append(syn)
            else:
                secondary_synsets.append(syn)
        
        primary_has_edges = False
        for syn in primary_synsets:
            if self.synapses.get(syn.name(), []):
                primary_has_edges = True
                break
        
        block_secondary = primary_has_edges
        
        # ── Step 3: Collect candidates ──
        candidates_map: Dict[str, Dict] = {}
        
        # Always process primary synsets (no penalty)
        for syn in primary_synsets:
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
                        "path_kind": "primary",
                    }
        
        # Process secondary synsets (conditionally, with penalty)
        for syn in secondary_synsets:
            synset_id = syn.name()
            primary_lemma = syn.lemmas()[0].name().lower().replace("_", " ")
            edges = self.synapses.get(synset_id, [])
            
            if not edges:
                continue
            
            top_edge = edges[0]
            top_atom = top_edge.get("concept_id", "?")
            top_score = round(top_edge.get("raw_score", 0.0), 4)
            
            if block_secondary:
                self._filter_log["secondary_lemma_filtered"] += 1
                if len(self._filter_log["secondary_lemma_details"]) < 200:
                    self._filter_log["secondary_lemma_details"].append(
                        (verb_lemma, synset_id, primary_lemma,
                         top_atom, top_score, "blocked")
                    )
                continue
            else:
                # No primary edges → accept secondary with penalty (v0.3.2)
                self._filter_log["secondary_lemma_accepted"] += 1
                if len(self._filter_log["secondary_lemma_details"]) < 200:
                    self._filter_log["secondary_lemma_details"].append(
                        (verb_lemma, synset_id, primary_lemma,
                         top_atom, top_score, "accepted_penalized")
                    )
                
                for edge in edges:
                    cid = edge.get("concept_id", "")
                    raw_score = edge.get("raw_score", 0.0)
                    
                    # v0.3.2: Apply penalty to secondary path scores
                    penalized_score = round(raw_score * self.secondary_penalty, 4)
                    
                    if cid not in candidates_map or penalized_score > candidates_map[cid]["raw_score"]:
                        candidates_map[cid] = {
                            "concept_id": cid,
                            "axis": edge.get("axis", ""),
                            "level": edge.get("level", ""),
                            "raw_score": penalized_score,
                            "synset": synset_id,
                            "path_kind": "secondary",
                        }
        
        # Step 4: POS Guard
        pos_filtered = {}
        for cid, cand in candidates_map.items():
            cat = cid.split(".")[0] if "." in cid else ""
            if cat in POS_GUARD_BLOCKED_CATEGORIES:
                self._filter_log["pos_guard_dropped"] += 1
            else:
                pos_filtered[cid] = cand
        
        # Step 5: Score Threshold
        score_filtered = {}
        for cid, cand in pos_filtered.items():
            if cand["raw_score"] < self.min_score:
                self._filter_log["threshold_dropped"] += 1
                if len(self._filter_log["threshold_drop_details"]) < 200:
                    self._filter_log["threshold_drop_details"].append(
                        (verb_lemma, cid, round(cand["raw_score"], 4))
                    )
                # Track penalty kills specifically
                if cand.get("path_kind") == "secondary":
                    self._filter_log["secondary_lemma_penalty_kills"] += 1
            else:
                score_filtered[cid] = cand
        
        # Step 6: Sort by score descending, then concept_id for determinism
        candidates = sorted(
            score_filtered.values(),
            key=lambda c: (-c["raw_score"], c["concept_id"])
        )
        
        self._cache[verb_lemma] = candidates
        return candidates
    
    def get_filter_log(self) -> Dict[str, Any]:
        """Return diagnostic log of filtered candidates."""
        return {
            "pos_guard_dropped": self._filter_log["pos_guard_dropped"],
            "threshold_dropped": self._filter_log["threshold_dropped"],
            "secondary_lemma_filtered": self._filter_log["secondary_lemma_filtered"],
            "secondary_lemma_accepted": self._filter_log["secondary_lemma_accepted"],
            "secondary_lemma_penalty_kills": self._filter_log["secondary_lemma_penalty_kills"],
            "secondary_penalty": self.secondary_penalty,
            "min_score": self.min_score,
            "threshold_drop_top": self._filter_log["threshold_drop_details"][:30],
            "secondary_lemma_top": self._filter_log["secondary_lemma_details"][:30],
        }
    
    def is_light_verb(self, verb_lemma: str) -> bool:
        """Check if verb is in the light verb stoplist."""
        return verb_lemma in LIGHT_VERB_STOPLIST


# ==========================================
# Relation Edge
# ==========================================

def _build_edge(
    triple: SVOTriple,
    atom_candidates: List[Dict],
    section_name: str,
    article_id: str,
    grounding_status: str = "GROUNDED",
) -> Dict[str, Any]:
    """Build a single relation edge from an SVO triple + grounding result."""
    if grounding_status == "UNGROUNDED_LIGHTVERB":
        top_atom = "UNGROUNDED_LIGHTVERB"
    elif atom_candidates:
        top_atom = atom_candidates[0]["concept_id"]
    else:
        top_atom = "UNGROUNDED"
    
    return {
        "source": triple.subject,
        "target": triple.object,
        "atom": top_atom,
        "atom_candidates": atom_candidates[:5],
        "operator": "ACT",
        "negated": triple.negated,
        "passive": triple.passive,
        "verb_lemma": triple.verb_lemma,
        "verb_raw": triple.verb,
        "section": section_name,
        "article": article_id,
        "sentence_idx": triple.sentence_idx,
        "text_ref": triple.sentence_text.strip(),
        "grounding_status": grounding_status,
    }


# ==========================================
# Relation Logger (Public API)
# ==========================================

class RelationLogger:
    """
    Converts SVO triples to ESDE relation edges with Synapse grounding.
    
    v0.3.2 Filter pipeline:
      1. Light Verb check → UNGROUNDED_LIGHTVERB (edge preserved, atom suppressed)
      2. Synapse grounding (Conditional Guard + Penalized Fallback + POS Guard + Threshold)
      3. Surviving candidates → GROUNDED; none → UNGROUNDED
    """
    
    def __init__(self, grounder: Optional[SynapseGrounder] = None):
        self.grounder = grounder or SynapseGrounder()
        self._all_edges: List[Dict[str, Any]] = []
        self._stats = {
            "sections_processed": 0,
            "triples_processed": 0,
            "grounded": 0,
            "ungrounded": 0,
            "lightverb": 0,
        }
    
    def process_section(
        self,
        result: ExtractionResult,
        section_name: str,
        article_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Process SVO extraction result for one section."""
        self._stats["sections_processed"] += 1
        edges = []
        
        for triple in result.triples:
            self._stats["triples_processed"] += 1
            
            if self.grounder.is_light_verb(triple.verb_lemma):
                self._stats["lightverb"] += 1
                edge = _build_edge(
                    triple, [], section_name, article_id,
                    grounding_status="UNGROUNDED_LIGHTVERB",
                )
                edges.append(edge)
                continue
            
            candidates = self.grounder.ground_verb(triple.verb_lemma)
            
            if candidates:
                self._stats["grounded"] += 1
                grounding_status = "GROUNDED"
            else:
                self._stats["ungrounded"] += 1
                grounding_status = "UNGROUNDED"
            
            edge = _build_edge(
                triple, candidates, section_name, article_id,
                grounding_status=grounding_status,
            )
            edges.append(edge)
        
        self._all_edges.extend(edges)
        return edges
    
    def process_article(
        self,
        sections: Dict[str, ExtractionResult],
        article_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Process all sections for an article."""
        all_edges = []
        for section_name, result in sections.items():
            edges = self.process_section(result, section_name, article_id)
            all_edges.extend(edges)
        return all_edges
    
    def write_jsonl(self, filepath: str) -> int:
        """Write all accumulated edges to JSONL file."""
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
        grounded = self._stats["grounded"]
        ungrounded = self._stats["ungrounded"]
        lightverb = self._stats["lightverb"]
        
        groundable = grounded + ungrounded
        grounding_rate = (
            grounded / groundable if groundable > 0 else 0.0
        )
        
        stats = {
            "version": VERSION,
            **self._stats,
            "grounding_rate": round(grounding_rate, 4),
            "total_edges": len(self._all_edges),
        }
        
        if hasattr(self.grounder, 'get_filter_log'):
            stats["filter_log"] = self.grounder.get_filter_log()
        
        return stats
    
    def clear(self):
        """Clear accumulated edges."""
        self._all_edges = []


# ==========================================
# Aggregation: Entity Graph + Section Profile
# ==========================================

def aggregate_entity_graph(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate edges into an entity graph (UI-facing)."""
    nodes: Dict[str, Dict] = defaultdict(lambda: {
        "degree": 0,
        "atoms": defaultdict(int),
        "as_source": 0,
        "as_target": 0,
    })
    
    edge_agg: Dict[str, Dict] = {}
    
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        atom = e["atom"]
        
        nodes[src]["degree"] += 1
        nodes[src]["as_source"] += 1
        nodes[src]["atoms"][atom] += 1
        
        nodes[tgt]["degree"] += 1
        nodes[tgt]["as_target"] += 1
        nodes[tgt]["atoms"][atom] += 1
        
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
    """Aggregate edges into section-level profiles (Phase 9 Lens input)."""
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
    
    result = {}
    for sec_name, data in sections.items():
        total = data["edge_count"]
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
    """Run the full relation extraction pipeline for an article."""
    from .parser_adapter import ParserAdapter
    
    adapter = ParserAdapter()
    grounder = (
        SynapseGrounder.from_file(synapse_path)
        if synapse_path
        else SynapseGrounder()
    )
    logger = RelationLogger(grounder)
    
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
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = out / "relations_edges.jsonl"
    logger.write_jsonl(str(jsonl_path))
    
    entity_graph = aggregate_entity_graph(edges)
    graph_path = out / "entity_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(entity_graph, f, indent=2, ensure_ascii=False)
    
    section_profile = aggregate_section_profile(edges)
    profile_path = out / "section_relation_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(section_profile, f, indent=2, ensure_ascii=False)
    
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