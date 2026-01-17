"""
ESDE Engine v5.3.2 - Unknown Queue Writer
"""
import json
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    VERSION, QUEUE_FILE_PATH, QUEUE_INCLUDE_NOISE, QUEUE_BUFFER_SIZE, STOPWORDS
)
from .utils import (
    generate_run_id, compute_dedup_key, find_all_typo_candidates, guess_category
)


class UnknownQueueWriter:
    """
    Enhanced Unknown Queue Writer with full observation points.
    
    Responsibilities:
    - Generate and track run_id
    - Create properly formatted records with all observation points
    - Append to JSONL file with dedup within run
    - Buffer and flush for performance
    - Track typo_candidates even for Route B tokens
    
    Dedup: Same token only queued once per run (sha1-based key)
    """
    
    def __init__(self,
                 queue_path: str = QUEUE_FILE_PATH,
                 include_noise: bool = QUEUE_INCLUDE_NOISE,
                 buffer_size: int = QUEUE_BUFFER_SIZE):
        self.queue_path = queue_path
        self.include_noise = include_noise
        self.buffer_size = buffer_size
        self.buffer: List[Dict] = []
        self.run_id: Optional[str] = None
        self.run_start: Optional[datetime] = None
        self.record_count = 0
        self.dedup_count = 0
        self.config_meta: Dict[str, Any] = {}
        self._seen_keys: Set[str] = set()
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create queue directory if it doesn't exist."""
        path = Path(self.queue_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_run(self, config_meta: Optional[Dict[str, Any]] = None) -> str:
        """Start a new processing run."""
        self.run_id = generate_run_id()
        self.run_start = datetime.now(timezone.utc)
        self.record_count = 0
        self.dedup_count = 0
        self._seen_keys.clear()
        self.config_meta = config_meta or {}
        return self.run_id
    
    def create_record(self,
                      input_text: str,
                      token: str,
                      token_raw: str,
                      token_index: int,
                      reason: str,
                      route: Optional[str],
                      context_window: List[str],
                      pos_guess: str = "u",
                      is_active: bool = False,
                      wn_synsets: Optional[List[Dict[str, str]]] = None,
                      wn_selected: Optional[str] = None,
                      has_direct_edges: bool = False,
                      has_proxy_edges: bool = False,
                      direct_edge_count: int = 0,
                      proxy_edge_count: int = 0,
                      proxy_evidence: Optional[List[Dict]] = None,
                      uncertainty: Optional[Dict[str, Any]] = None,
                      abstain_reasons: Optional[List[str]] = None,
                      typo_candidates: Optional[List[Dict]] = None,
                      routing_report: Optional[Dict[str, Any]] = None,
                      notes: str = "") -> Dict[str, Any]:
        """
        Create enhanced queue record with routing_report.
        
        Schema Categories:
          A. Tracking: engine_version, timestamp_utc, run_id, input_text, token_index
          B. Token: token_raw, token_norm, pos_guess, is_stopword, is_active
          C. WordNet: wn_synsets, wn_selected
          D. Synapse: has_direct_edges, has_proxy_edges, edge counts
          E. Uncertainty: entropy, margin, is_volatile
          F. Routing: route, category_guess, action_hint, confidence (provisional)
          G. Audit: abstain_reasons, proxy_evidence, typo_candidates
          H. RoutingReport: Full multi-hypothesis evaluation
        """
        now = datetime.now(timezone.utc)
        token_norm = token.lower().strip()
        
        # Get top synset for dedup key
        top_synset = None
        if wn_synsets and len(wn_synsets) > 0:
            top_synset = wn_synsets[0].get("id") if isinstance(wn_synsets[0], dict) else wn_synsets[0]
        
        # Compute dedup key
        dedup_key = compute_dedup_key(token_norm, pos_guess, top_synset)
        
        # Find typo candidates if not provided
        if typo_candidates is None:
            typo_candidates = find_all_typo_candidates(token_norm)
        
        # Extract from routing_report if available
        if routing_report:
            decision = routing_report.get("decision", {})
            category_guess = decision.get("action", "unknown")
            action_hint = decision.get("winner") or "abstain"
            confidence = decision.get("margin")
            # Update route from decision if available
            if decision.get("winner"):
                route = decision.get("winner")
        else:
            # Guess category (provisional)
            synset_ids = [s.get("id") if isinstance(s, dict) else s for s in (wn_synsets or [])]
            has_edges = has_direct_edges or has_proxy_edges
            category_guess, action_hint, confidence = guess_category(
                token_norm, synset_ids, has_edges, typo_candidates
            )
        
        # Build uncertainty dict
        unc = uncertainty or {}
        uncertainty_record = {
            "entropy": unc.get("entropy"),
            "margin": unc.get("margin"),
            "is_volatile": unc.get("high_variance", unc.get("is_volatile"))
        }
        
        record = {
            # A. Tracking & Reproducibility
            "engine_version": VERSION,
            "timestamp_utc": now.isoformat(),
            "run_id": self.run_id or "no_run",
            "input_text": input_text,
            "token_index": token_index,
            
            # B. Token Observation
            "token_raw": token_raw,
            "token_norm": token_norm,
            "pos_guess": pos_guess,
            "is_stopword": token_norm in STOPWORDS,
            "is_active": is_active,
            "is_alpha": token_norm.isalpha(),
            "len": len(token_norm),
            
            # C. WordNet Observation
            "wn_synsets": wn_synsets or [],
            "wn_selected": wn_selected,
            
            # D. Synapse Observation
            "has_direct_edges": has_direct_edges,
            "has_proxy_edges": has_proxy_edges,
            "direct_edge_count": direct_edge_count,
            "proxy_edge_count": proxy_edge_count,
            
            # E. Uncertainty
            "uncertainty": uncertainty_record,
            
            # F. Routing (provisional - not final classification)
            "route": route,
            "category_guess": category_guess,
            "action_hint": action_hint,
            "confidence": confidence,
            
            # G. Audit Evidence
            "reason": reason,
            "abstain_reasons": abstain_reasons or [reason],
            "proxy_evidence": proxy_evidence or [],
            "typo_candidates": typo_candidates,
            
            # H. RoutingReport - Full multi-hypothesis evaluation
            "routing_report": routing_report,
            
            # Context
            "context_window": context_window,
            "notes": notes,
            
            # Dedup tracking
            "dedup_key": dedup_key,
            "seen_count_in_run": 1,
            
            # Artifacts (config metadata)
            "artifacts": self.config_meta.get("artifacts", {})
        }
        
        return record
    
    def enqueue(self, record: Dict[str, Any]) -> bool:
        """
        Add record to queue with dedup and noise filtering.
        
        Returns True if record was queued, False if filtered/deduped.
        """
        # Filter noise if configured
        if not self.include_noise and record.get("reason") == "stopword_or_noise":
            return False
        
        # Dedup check
        dedup_key = record.get("dedup_key", "")
        if dedup_key in self._seen_keys:
            self.dedup_count += 1
            return False
        
        self._seen_keys.add(dedup_key)
        self.buffer.append(record)
        self.record_count += 1
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        
        return True
    
    def flush(self):
        """Write buffered records to file."""
        if not self.buffer:
            return
        
        try:
            with open(self.queue_path, 'a', encoding='utf-8') as f:
                for record in self.buffer:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.buffer.clear()
        except Exception as e:
            print(f"[UnknownQueueWriter] Flush error: {e}")
    
    def end_run(self) -> Dict[str, Any]:
        """End the current run and return summary."""
        self.flush()
        
        summary = {
            "run_id": self.run_id,
            "queued_records": self.record_count,
            "deduped_records": self.dedup_count,
            "path": self.queue_path,
            "started": self.run_start.isoformat() if self.run_start else None,
            "ended": datetime.now(timezone.utc).isoformat()
        }
        
        self.run_id = None
        self.run_start = None
        self._seen_keys.clear()
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        return {
            "run_id": self.run_id,
            "queued_records": self.record_count,
            "deduped_records": self.dedup_count,
            "buffered": len(self.buffer),
            "path": self.queue_path,
            "include_noise": self.include_noise
        }
