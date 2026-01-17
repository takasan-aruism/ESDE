"""
ESDE Sensor - Candidate Ranker
Aggregates scores and ranks candidates with deterministic output.

GPT Audit:
  - Deterministic sort: score DESC, concept_id ASC
  - Hash includes config_snapshot
  - Returns both full and short hash
"""
import json
import hashlib
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from .loader_synapse import SynapseLoader


VERSION = "2.0.0"
DEFAULT_TOP_K = 5


class CandidateRanker:
    """
    Aggregate scores and rank candidates with deterministic output.
    """
    
    def __init__(self, top_k: int = None):
        self.top_k = top_k or DEFAULT_TOP_K
    
    def aggregate_scores(self, 
                         token_synsets: Dict[str, List[str]],
                         synapse_loader: SynapseLoader,
                         tokens_with_index: List[Tuple[int, str]]) -> Tuple[Dict[str, Dict], Dict[str, int]]:
        """
        Aggregate scores across all tokens.
        
        GPT Audit: Returns counters for fallback debugging.
        
        Args:
            token_synsets: {token: [synset_ids]}
            synapse_loader: SynapseLoader instance
            tokens_with_index: [(index, token), ...]
        
        Returns:
            Tuple of:
              - {concept_id: {score, axis, level, evidence}}
              - counters: {synsets_checked, synsets_with_edges, total_edges}
        """
        concept_scores: Dict[str, Dict] = defaultdict(lambda: {
            "score": 0.0,
            "axis": None,
            "level": None,
            "evidence": [],
            "max_raw_score": 0.0
        })
        
        # GPT Audit: Counters for debugging
        counters = {
            "synsets_checked": 0,
            "synsets_with_edges": 0,
            "total_edges": 0
        }
        
        # Build token to index map
        token_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, token in tokens_with_index:
            token_to_indices[token].append(idx)
        
        for token, synset_ids in token_synsets.items():
            for synset_id in synset_ids:
                counters["synsets_checked"] += 1
                edges = synapse_loader.get_edges(synset_id)
                
                if edges:
                    counters["synsets_with_edges"] += 1
                    counters["total_edges"] += len(edges)
                
                for edge in edges:
                    concept_id = edge.get("concept_id")
                    if not concept_id:
                        continue
                    
                    raw_score = edge.get("raw_score", 0.0)
                    weight = edge.get("weight", 1.0)
                    
                    # Aggregate score
                    score_contribution = round(raw_score * weight, 8)
                    concept_scores[concept_id]["score"] += score_contribution
                    
                    # Track max raw_score for axis/level selection
                    if raw_score > concept_scores[concept_id]["max_raw_score"]:
                        concept_scores[concept_id]["max_raw_score"] = raw_score
                        concept_scores[concept_id]["axis"] = edge.get("axis")
                        concept_scores[concept_id]["level"] = edge.get("level")
                    
                    # Add evidence
                    for token_idx in token_to_indices.get(token, []):
                        concept_scores[concept_id]["evidence"].append({
                            "token": token,
                            "token_index": token_idx,
                            "synset": synset_id,
                            "raw_score": raw_score,
                            "weight": weight
                        })
        
        return dict(concept_scores), counters
    
    def rank(self, concept_scores: Dict[str, Dict]) -> List[Dict]:
        """
        Rank candidates with deterministic ordering.
        
        Sort by:
          1. score (DESC)
          2. concept_id (ASC) - for tie-breaking
        
        Returns:
            List of ranked candidates
        """
        candidates = []
        
        for concept_id, data in concept_scores.items():
            candidates.append({
                "concept_id": concept_id,
                "axis": data["axis"],
                "level": data["level"],
                "score": round(data["score"], 8),
                "evidence": data["evidence"]
            })
        
        # Deterministic sort: score DESC, concept_id ASC
        candidates.sort(key=lambda x: (-x["score"], x["concept_id"]))
        
        # Apply Top-K
        return candidates[:self.top_k]
    
    def compute_determinism_hash(self, 
                                  candidates: List[Dict],
                                  config_snapshot: Dict,
                                  synapse_hash: str) -> Tuple[str, str]:
        """
        Compute deterministic hash for audit trail.
        
        GPT Audit: Returns FULL hash, display can be truncated.
        
        Returns:
            Tuple of (full_hash, short_hash)
        """
        # Build input string
        hash_input = (
            synapse_hash +
            VERSION +
            str(config_snapshot.get("top_k", "")) +
            str(config_snapshot.get("max_synsets", "")) +
            str(config_snapshot.get("strict_mode", "")) +
            json.dumps(candidates, sort_keys=True, ensure_ascii=False)
        )
        
        full_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        short_hash = full_hash[:16]
        
        return (f"sha256:{full_hash}", f"sha256:{short_hash}")
