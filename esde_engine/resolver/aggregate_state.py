"""
ESDE Engine v5.3.4 - Phase 7B+: Aggregate State Manager

Extends QueueStateManager for 7B+ aggregate-level state management.

Key Changes from v5.3.3:
    - aggregate_key = hash(token_norm, pos, sorted(route_set))
    - Two-stage processing: seen vs finalized
    - Observation model stored in state
    - Re-evaluation conditions controlled by state
"""
import json
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Constants
# =============================================================================

MIN_OBSERVE_COUNT = 3  # Minimum observations before candidate can skip re-eval
COOLDOWN_HOURS = 24    # Hours before defer/quarantine can be re-evaluated


# =============================================================================
# Aggregate State Manager (v5.3.4)
# =============================================================================

class AggregateStateManager:
    """
    Manages state at aggregate_key level for Phase 7B+.
    
    State format per aggregate:
    {
        "aggregate_key": str,         # Primary key
        "token_norm": str,
        "pos": str,
        "route_set": List[str],       # Sorted list of routes
        
        # Processing flags (v5.3.4)
        "seen": bool,                 # Has been evaluated by 7B+
        "finalized": bool,            # Human review complete (skip target)
        
        # Observation model (v5.3.4)
        "count_total": int,           # Total occurrences across all runs
        "first_seen_ts": str,         # ISO timestamp
        "last_seen_ts": str,          # ISO timestamp
        "last_run_id": str,
        
        # Evaluation results (v5.3.4)
        "status_last": str,           # candidate/defer/quarantine
        "volatility_last": float,
        "volatility_max": float,
        "volatility_history": List[float],  # Last N volatility values
        "scores_avg": Dict[str, float],     # A/B/C/D average scores
        "competing_routes_last": List[str],
        
        # Re-evaluation control (v5.3.4)
        "needs_observation": bool,    # True = stay in re-eval loop
        "observe_count": int,         # How many times observed
        "cooldown_until": str,        # ISO timestamp (defer/quarantine cooldown)
    }
    """
    
    STATE_FILE = "./data/unknown_queue_state_7bplus.json"
    MAX_VOLATILITY_HISTORY = 10
    
    def __init__(self, state_file: str = None):
        self.state_file = state_file or self.STATE_FILE
        self.state: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._dirty = False
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create state directory if needed."""
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def compute_aggregate_key(token_norm: str, pos: str, route_set: Set[str]) -> str:
        """
        Compute aggregate_key per v5.3.4 spec.
        
        aggregate_key = hash(token_norm, pos, sorted(route_set))
        """
        pos_str = pos if pos else "UNK"
        route_str = ",".join(sorted(route_set))
        key_tuple = f"{token_norm}|{pos_str}|{route_str}"
        return hashlib.sha256(key_tuple.encode('utf-8')).hexdigest()[:16]
    
    def load(self) -> bool:
        """Load state from file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.state = data.get("aggregates", {})
                self._meta = data.get("_meta", {})
                return True
            return False
        except Exception as e:
            print(f"[AggregateStateManager] Load error: {e}")
            return False
    
    def save(self) -> bool:
        """Save state to file."""
        try:
            self._meta["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._meta["aggregate_count"] = len(self.state)
            self._meta["spec_version"] = "5.3.4"
            
            data = {
                "_meta": self._meta,
                "aggregates": self.state
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._dirty = False
            return True
        except Exception as e:
            print(f"[AggregateStateManager] Save error: {e}")
            return False
    
    def get_entry(self, aggregate_key: str) -> Optional[Dict[str, Any]]:
        """Get state entry by aggregate_key."""
        return self.state.get(aggregate_key)
    
    def should_process(self, aggregate_key: str) -> Tuple[bool, str]:
        """
        Determine if aggregate should be processed.
        
        Returns: (should_process, reason)
        
        v5.3.4 Rules:
            - finalized=True → skip
            - needs_observation=True → process
            - status_last in {defer, quarantine} and cooldown_expired → process
            - status_last == candidate and observe_count < MIN_OBSERVE → process
            - otherwise → skip
        """
        entry = self.state.get(aggregate_key)
        
        if entry is None:
            return True, "new_aggregate"
        
        # Rule 1: finalized = always skip
        if entry.get("finalized", False):
            return False, "finalized"
        
        # Rule 2: needs_observation = always process
        if entry.get("needs_observation", False):
            return True, "needs_observation"
        
        # Rule 3: Not seen yet = process
        if not entry.get("seen", False):
            return True, "not_seen"
        
        status = entry.get("status_last", "")
        observe_count = entry.get("observe_count", 0)
        
        # Rule 4: defer/quarantine with cooldown expired
        if status in ("defer", "quarantine"):
            cooldown_until = entry.get("cooldown_until")
            if cooldown_until:
                try:
                    cooldown_dt = datetime.fromisoformat(cooldown_until.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) >= cooldown_dt:
                        return True, "cooldown_expired"
                except:
                    pass
            return False, f"in_cooldown_{status}"
        
        # Rule 5: candidate with insufficient observations
        if status == "candidate" and observe_count < MIN_OBSERVE_COUNT:
            return True, f"observe_count_{observe_count}_lt_{MIN_OBSERVE_COUNT}"
        
        # Default: skip (already sufficiently observed candidate)
        return False, "already_observed"
    
    def upsert_observation(self, 
                           aggregate_key: str,
                           token_norm: str,
                           pos: str,
                           route_set: Set[str],
                           evaluation_result: Dict[str, Any],
                           run_id: str) -> Dict[str, Any]:
        """
        Insert or update an aggregate observation.
        
        Args:
            aggregate_key: Pre-computed aggregate key
            token_norm: Normalized token
            pos: Part of speech
            route_set: Set of routes
            evaluation_result: Result from evaluate_all_hypotheses containing:
                - status: candidate/defer/quarantine
                - global_volatility: float
                - scores: Dict[str, float] for A/B/C/D
                - competing_routes: List[str]
            run_id: Current run identifier
        
        Returns:
            Updated state entry
        """
        now = datetime.now(timezone.utc).isoformat()
        
        if aggregate_key in self.state:
            entry = self.state[aggregate_key]
            entry["count_total"] = entry.get("count_total", 0) + 1
            entry["last_seen_ts"] = now
            entry["last_run_id"] = run_id
            entry["observe_count"] = entry.get("observe_count", 0) + 1
        else:
            entry = {
                "aggregate_key": aggregate_key,
                "token_norm": token_norm,
                "pos": pos or "UNK",
                "route_set": sorted(route_set),
                
                # Processing flags
                "seen": False,
                "finalized": False,
                
                # Observation tracking
                "count_total": 1,
                "first_seen_ts": now,
                "last_seen_ts": now,
                "last_run_id": run_id,
                "observe_count": 1,
                
                # Evaluation results (will be filled)
                "status_last": None,
                "volatility_last": None,
                "volatility_max": 0.0,
                "volatility_history": [],
                "scores_avg": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0},
                "competing_routes_last": [],
                
                # Re-evaluation control
                "needs_observation": True,
                "cooldown_until": None
            }
            self.state[aggregate_key] = entry
        
        # Update evaluation results
        status = evaluation_result.get("status", "defer")
        volatility = evaluation_result.get("global_volatility", 0.0)
        scores = evaluation_result.get("scores", {})
        competing = evaluation_result.get("competing_routes", [])
        
        entry["seen"] = True
        entry["status_last"] = status
        entry["volatility_last"] = volatility
        entry["volatility_max"] = max(entry.get("volatility_max", 0.0), volatility)
        entry["competing_routes_last"] = competing
        
        # Update volatility history
        history = entry.get("volatility_history", [])
        history.append(volatility)
        if len(history) > self.MAX_VOLATILITY_HISTORY:
            history = history[-self.MAX_VOLATILITY_HISTORY:]
        entry["volatility_history"] = history
        
        # Update scores (running average)
        old_count = entry.get("observe_count", 1) - 1
        new_count = entry["observe_count"]
        for route in ["A", "B", "C", "D"]:
            old_avg = entry["scores_avg"].get(route, 0.0)
            new_score = scores.get(route, 0.0)
            # Incremental average
            if old_count > 0:
                entry["scores_avg"][route] = round(
                    (old_avg * old_count + new_score) / new_count, 3
                )
            else:
                entry["scores_avg"][route] = round(new_score, 3)
        
        # Determine needs_observation
        entry["needs_observation"] = self._compute_needs_observation(entry)
        
        # Set cooldown for defer/quarantine
        if status in ("defer", "quarantine"):
            cooldown_dt = datetime.now(timezone.utc)
            from datetime import timedelta
            cooldown_dt += timedelta(hours=COOLDOWN_HOURS)
            entry["cooldown_until"] = cooldown_dt.isoformat()
        else:
            entry["cooldown_until"] = None
        
        self._dirty = True
        return entry
    
    def _compute_needs_observation(self, entry: Dict[str, Any]) -> bool:
        """
        Determine if aggregate needs continued observation.
        
        Rules:
            - quarantine → always True (high volatility needs monitoring)
            - defer → True if volatility trending up
            - candidate with observe_count < MIN_OBSERVE → True
            - candidate with stable low volatility → False
        """
        status = entry.get("status_last", "defer")
        observe_count = entry.get("observe_count", 0)
        volatility_history = entry.get("volatility_history", [])
        
        # Quarantine always needs observation
        if status == "quarantine":
            return True
        
        # Defer needs observation if volatility is trending up
        if status == "defer":
            if len(volatility_history) >= 2:
                recent = volatility_history[-2:]
                if recent[-1] > recent[-2]:
                    return True  # Volatility increasing
            return True  # Default to observe for defer
        
        # Candidate: check if sufficiently observed
        if status == "candidate":
            if observe_count < MIN_OBSERVE_COUNT:
                return True
            
            # Check volatility stability
            if len(volatility_history) >= MIN_OBSERVE_COUNT:
                recent = volatility_history[-MIN_OBSERVE_COUNT:]
                volatility_range = max(recent) - min(recent)
                if volatility_range < 0.1:  # Stable
                    return False
        
        return True  # Default to observe
    
    def mark_finalized(self, aggregate_key: str, reason: str = None) -> bool:
        """Mark aggregate as finalized (human-reviewed)."""
        if aggregate_key not in self.state:
            return False
        
        entry = self.state[aggregate_key]
        entry["finalized"] = True
        entry["needs_observation"] = False
        entry["finalized_at"] = datetime.now(timezone.utc).isoformat()
        entry["finalized_reason"] = reason
        self._dirty = True
        return True
    
    def reset_for_reprocess(self, include_finalized: bool = False) -> int:
        """
        Reset states for reprocessing.
        
        Args:
            include_finalized: If True, also reset finalized entries
        
        Returns:
            Number of entries reset
        """
        count = 0
        for entry in self.state.values():
            if entry.get("finalized") and not include_finalized:
                continue
            
            entry["seen"] = False
            entry["needs_observation"] = True
            entry["cooldown_until"] = None
            count += 1
        
        self._dirty = True
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregate state."""
        stats = {
            "total": len(self.state),
            "by_status": {},
            "seen_count": 0,
            "finalized_count": 0,
            "needs_observation_count": 0,
            "avg_observe_count": 0.0
        }
        
        total_observe = 0
        for entry in self.state.values():
            status = entry.get("status_last", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            
            if entry.get("seen"):
                stats["seen_count"] += 1
            if entry.get("finalized"):
                stats["finalized_count"] += 1
            if entry.get("needs_observation"):
                stats["needs_observation_count"] += 1
            
            total_observe += entry.get("observe_count", 0)
        
        if stats["total"] > 0:
            stats["avg_observe_count"] = round(total_observe / stats["total"], 2)
        
        return stats
    
    def get_pending_aggregates(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get aggregates that should be processed."""
        pending = []
        
        for agg_key, entry in self.state.items():
            should_process, reason = self.should_process(agg_key)
            if should_process:
                entry_copy = entry.copy()
                entry_copy["_process_reason"] = reason
                pending.append(entry_copy)
        
        # Sort by: needs_observation first, then by observe_count ascending
        pending.sort(key=lambda x: (
            not x.get("needs_observation", False),
            x.get("observe_count", 0)
        ))
        
        return pending[:limit]
