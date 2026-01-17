"""
ESDE Engine v5.3.2 - Activation Collectors
"""
from collections import defaultdict
from typing import Dict, List, Any, Optional

from nltk.corpus import wordnet as wn

from .config import (
    DEDUP_PER_TOKEN_CONCEPT, ENABLE_PROXIMITY_EXPANSION, MAX_PROXY_DEPTH,
    DEBUG_PROXIMITY, DECAY_SIMILAR, DECAY_HYPERNYM, DECAY_DERIVATIONAL,
    DECAY_INSTANCE_HYPERNYM, DECAY_HOLONYM, VARIANCE_MARGIN_THRESHOLD,
    VARIANCE_ENTROPY_THRESHOLD, VARIANCE_DOWNWEIGHT, MIN_SCORE_THRESHOLD,
    DYNAMIC_FLOOR_RATIO, TOP_CONCEPTS, AXIS_TOP_LEVELS
)
from .utils import compute_variance_metrics, is_proper_noun_candidate
from .loaders import SynapseLoader


class ProximityExplorer:
    """Explore related synsets to find proxy edges."""
    
    def __init__(self, synapse_loader: SynapseLoader,
                 max_depth: int = MAX_PROXY_DEPTH,
                 debug: bool = DEBUG_PROXIMITY):
        self.synapse_loader = synapse_loader
        self.max_depth = max_depth
        self.debug = debug
    
    def find_proxy(self, synset_id: str, token: str = "") -> Optional[Dict]:
        """Find a proxy synset that has edges in the synapse map."""
        try:
            synset = wn.synset(synset_id)
        except Exception:
            return None
        
        candidates = []
        
        # Similar synsets
        for s in synset.similar_tos()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_SIMILAR,
                    "proxy_synset_id": s.name(), "relation": "similar", "depth": 1
                })
        
        # Also-see synsets
        for s in synset.also_sees()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_SIMILAR,
                    "proxy_synset_id": s.name(), "relation": "also_see", "depth": 1
                })
        
        # Hypernyms
        for s in synset.hypernyms()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_HYPERNYM,
                    "proxy_synset_id": s.name(), "relation": "hypernym", "depth": 1
                })
        
        # Instance hypernyms
        for s in synset.instance_hypernyms()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_INSTANCE_HYPERNYM,
                    "proxy_synset_id": s.name(), "relation": "instance_hypernym", "depth": 1
                })
        
        # Member holonyms
        for s in synset.member_holonyms()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_HOLONYM,
                    "proxy_synset_id": s.name(), "relation": "member_holonym", "depth": 1
                })
        
        # Part holonyms
        for s in synset.part_holonyms()[:10]:
            edges = self.synapse_loader.get_edges(s.name())
            if edges:
                candidates.append({
                    "edges": edges, "decay": DECAY_HOLONYM,
                    "proxy_synset_id": s.name(), "relation": "part_holonym", "depth": 1
                })
        
        # Derivationally related forms
        deriv_count = 0
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                deriv_count += 1
                if deriv_count > 10:
                    break
                related_synset = related.synset()
                edges = self.synapse_loader.get_edges(related_synset.name())
                if edges:
                    candidates.append({
                        "edges": edges, "decay": DECAY_DERIVATIONAL,
                        "proxy_synset_id": related_synset.name(),
                        "relation": "derivational", "depth": 1
                    })
        
        if not candidates:
            return None
        return self._select_top_1(candidates)
    
    def _select_top_1(self, candidates: List[Dict]) -> Dict:
        """Select the best proxy candidate."""
        def get_max_raw_score(c):
            edges = c.get("edges", [])
            return max((e.get("raw_score", 0.0) for e in edges), default=0.0)
        
        def sort_key(c):
            return (-get_max_raw_score(c), -c.get("decay", 0.0), c.get("proxy_synset_id", ""))
        
        candidates.sort(key=sort_key)
        return candidates[0]


class ActivationCollector:
    """Collect concept activations from synsets."""
    
    def __init__(self, synapse_loader: SynapseLoader,
                 dedup_per_token: bool = DEDUP_PER_TOKEN_CONCEPT,
                 enable_proximity: bool = ENABLE_PROXIMITY_EXPANSION,
                 max_proxy_depth: int = MAX_PROXY_DEPTH,
                 debug_proximity: bool = DEBUG_PROXIMITY):
        self.synapse_loader = synapse_loader
        self.dedup = dedup_per_token
        self.enable_proximity = enable_proximity
        self.proximity_explorer = (
            ProximityExplorer(synapse_loader, max_proxy_depth, debug_proximity)
            if enable_proximity else None
        )
    
    def collect(self, token_synsets: Dict[str, List[str]],
                routing_queue: Dict[str, List[Dict]],
                all_tokens: List[str]) -> Dict[str, Any]:
        """Collect activations from token-synset mappings."""
        concept_scores = defaultdict(float)
        level_scores = defaultdict(float)
        evidence = []
        unknown_records = []
        
        abstain_summary = {
            "no_synapse_edges": 0,
            "high_variance": 0,
            "below_dynamic_threshold": 0,
            "proxy_failed": 0
        }
        
        # Build token to index mapping
        token_to_indices = defaultdict(list)
        for idx, t in enumerate(all_tokens):
            token_to_indices[t].append(idx)
        
        for token, synset_ids in token_synsets.items():
            seen_concepts_this_token = set()
            token_had_any_activation = False
            token_abstain_synsets = []
            proxy_traces = []
            
            # Get first index for this token
            token_idx = token_to_indices[token][0] if token_to_indices[token] else -1
            
            for synset_id in synset_ids:
                edges = self.synapse_loader.get_edges(synset_id)
                decay = 1.0
                mode = "direct"
                proxy_info = None
                
                if not edges and self.enable_proximity and self.proximity_explorer:
                    proxy_result = self.proximity_explorer.find_proxy(synset_id, token)
                    if proxy_result:
                        edges = proxy_result["edges"]
                        decay = proxy_result["decay"]
                        mode = "proxy"
                        proxy_info = {
                            "src": synset_id, "via": proxy_result["proxy_synset_id"],
                            "relation": proxy_result["relation"], "depth": proxy_result["depth"]
                        }
                        proxy_traces.append(proxy_info)
                    else:
                        abstain_summary["proxy_failed"] += 1
                
                if not edges:
                    token_abstain_synsets.append(synset_id)
                    abstain_summary["no_synapse_edges"] += 1
                    continue
                
                weights = [e.get("weight", 0.0) for e in edges]
                variance = compute_variance_metrics(
                    weights, VARIANCE_MARGIN_THRESHOLD, VARIANCE_ENTROPY_THRESHOLD
                )
                variance_multiplier = 1.0
                decision = "normal"
                if variance["high_variance"]:
                    variance_multiplier = VARIANCE_DOWNWEIGHT
                    decision = "downweight"
                    abstain_summary["high_variance"] += 1
                
                top_weight = max(weights) if weights else 0.0
                dynamic_floor = max(MIN_SCORE_THRESHOLD, top_weight * DYNAMIC_FLOOR_RATIO)
                
                selected = []
                for edge in edges:
                    concept_id = edge.get("concept_id", "")
                    axis = edge.get("axis", "")
                    level = edge.get("level", "")
                    weight = edge.get("weight", 0.0)
                    
                    if weight < dynamic_floor:
                        abstain_summary["below_dynamic_threshold"] += 1
                        continue
                    if self.dedup and concept_id in seen_concepts_this_token:
                        continue
                    seen_concepts_this_token.add(concept_id)
                    effective_weight = weight * decay * variance_multiplier
                    concept_scores[concept_id] += effective_weight
                    level_scores[(concept_id, axis, level)] += effective_weight
                    selected.append({
                        "concept": concept_id, "axis": axis,
                        "level": level, "weight": round(effective_weight, 4)
                    })
                    token_had_any_activation = True
                
                evidence_entry = {
                    "token": token, "mode": mode, "synset": synset_id,
                    "edge_count": len(edges), "selected": selected
                }
                if mode == "proxy" and proxy_info:
                    evidence_entry["proxy"] = proxy_info
                if variance["high_variance"]:
                    evidence_entry["variance"] = variance
                    evidence_entry["decision"] = decision
                evidence.append(evidence_entry)
            
            # Record unknown tokens
            if not token_had_any_activation and token_abstain_synsets:
                # Determine POS from first synset
                try:
                    pos = wn.synset(token_abstain_synsets[0]).pos()
                except:
                    pos = "?"
                
                # Check if proper noun
                if is_proper_noun_candidate(token_abstain_synsets):
                    routing_queue["proper_noun_candidates"].append({
                        "token": token,
                        "token_index": token_idx,
                        "synsets": token_abstain_synsets,
                        "reason": "proper_noun_pattern"
                    })
                else:
                    routing_queue["molecule_candidates"].append({
                        "token": token,
                        "token_index": token_idx,
                        "synsets": token_abstain_synsets,
                        "reason": "no_synapse_edges"
                    })
                
                # Create unknown record for queue
                unknown_records.append({
                    "token": token,
                    "token_index": token_idx,
                    "pos": pos,
                    "reason": "no_synapse_edges",
                    "wordnet_synsets": token_abstain_synsets,
                    "direct_edges_found": 0,
                    "proxy_edges_found": 0,
                    "proxy_trace": proxy_traces
                })
        
        return {
            "concept_scores": dict(concept_scores),
            "level_scores": {f"{k[0]}:{k[1]}:{k[2]}": v for k, v in level_scores.items()},
            "evidence": evidence,
            "routing_queue": routing_queue,
            "abstain_summary": abstain_summary,
            "unknown_records": unknown_records
        }


class OutputGate:
    """Gate and filter output concepts."""
    
    def __init__(self, top_concepts: int = TOP_CONCEPTS,
                 axis_top_levels: int = AXIS_TOP_LEVELS):
        self.top_concepts = top_concepts
        self.axis_top_levels = axis_top_levels
    
    def apply(self, activations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply gating to activation results."""
        concept_scores = activations.get("concept_scores", {})
        level_scores_raw = activations.get("level_scores", {})
        
        level_scores = {}
        for key, score in level_scores_raw.items():
            parts = key.split(":")
            if len(parts) == 3:
                level_scores[(parts[0], parts[1], parts[2])] = score
        
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        top_concept_ids = [c[0] for c in sorted_concepts[:self.top_concepts]]
        top_concepts = [
            {"concept_id": cid, "score": round(concept_scores[cid], 4)}
            for cid in top_concept_ids
        ]
        
        top_levels = []
        for concept_id in top_concept_ids:
            axis_levels = defaultdict(list)
            for (cid, axis, level), score in level_scores.items():
                if cid == concept_id:
                    axis_levels[axis].append((level, score))
            for axis, levels in axis_levels.items():
                levels.sort(key=lambda x: x[1], reverse=True)
                for level, score in levels[:self.axis_top_levels]:
                    top_levels.append({
                        "concept_id": concept_id, "axis": axis,
                        "level": level, "score": round(score, 4)
                    })
        
        return {"top_concepts": top_concepts, "top_levels": top_levels, "gated": True}
