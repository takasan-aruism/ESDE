"""
ESDE Sensor - Audit Tracer
Handles counters, hash computation, and evidence formatting for audit trail.

GPT Audit Requirements:
  - Full hash preservation (determinism_hash_full)
  - Fallback counters for debugging
  - Top evidence per candidate
"""
from typing import Dict, List, Any
from datetime import datetime, timezone


class AuditTracer:
    """
    Manages audit-related data: counters, evidence, timestamps.
    """
    
    def __init__(self):
        self.run_start = datetime.now(timezone.utc)
    
    def create_timestamp(self) -> str:
        """Create ISO format timestamp."""
        return datetime.now(timezone.utc).isoformat()
    
    def build_counters(self,
                       input_tokens: int,
                       tokens_with_synsets: int,
                       synapse_counters: Dict[str, int],
                       concepts_aggregated: int,
                       candidates_returned: int,
                       fallback_candidates: int = 0) -> Dict[str, int]:
        """
        Build counter dict for debugging.
        
        GPT Audit: Essential for understanding "why 0 candidates"
        """
        return {
            "input_tokens": input_tokens,
            "tokens_with_synsets": tokens_with_synsets,
            "synsets_checked": synapse_counters.get("synsets_checked", 0),
            "synsets_with_edges": synapse_counters.get("synsets_with_edges", 0),
            "total_edges_found": synapse_counters.get("total_edges", 0),
            "concepts_aggregated": concepts_aggregated,
            "candidates_returned": candidates_returned,
            "fallback_candidates": fallback_candidates
        }
    
    def add_top_evidence(self, candidates: List[Dict]) -> List[Dict]:
        """
        Add top_evidence (highest raw_score) to each candidate.
        
        GPT Audit: "最低限、各候補について最大寄与evidenceを1件だけ"
        """
        for cand in candidates:
            evidence_list = cand.get("evidence", [])
            if evidence_list:
                top_ev = max(evidence_list, key=lambda e: e.get("raw_score", 0))
                cand["top_evidence"] = top_ev
            else:
                cand["top_evidence"] = None
        return candidates
    
    def build_config_snapshot(self,
                              top_k: int,
                              max_synsets: int,
                              strict_mode: bool,
                              allowed_pos: set) -> Dict[str, Any]:
        """Build config snapshot for audit."""
        return {
            "top_k": top_k,
            "max_synsets": max_synsets,
            "strict_mode": strict_mode,
            "allowed_pos": sorted(list(allowed_pos))
        }
    
    def build_meta(self,
                   engine_type: str,
                   version: str,
                   timestamp: str,
                   hash_short: str,
                   hash_full: str,
                   config_snapshot: Dict,
                   counters: Dict) -> Dict[str, Any]:
        """Build complete meta object."""
        return {
            "engine": engine_type,
            "version": version,
            "timestamp": timestamp,
            "determinism_hash": hash_short,
            "determinism_hash_full": hash_full,
            "config_snapshot": config_snapshot,
            "counters": counters
        }
    
    def build_empty_meta(self,
                         version: str,
                         timestamp: str,
                         config_snapshot: Dict,
                         reason: str) -> Dict[str, Any]:
        """Build meta for empty result."""
        return {
            "engine": "v2_synapse",
            "version": version,
            "timestamp": timestamp,
            "determinism_hash": "empty",
            "determinism_hash_full": "empty",
            "config_snapshot": config_snapshot,
            "counters": {
                "input_tokens": 0,
                "tokens_with_synsets": 0,
                "synsets_checked": 0,
                "synsets_with_edges": 0,
                "total_edges_found": 0,
                "concepts_aggregated": 0,
                "candidates_returned": 0,
                "fallback_candidates": 0
            },
            "empty_reason": reason
        }
