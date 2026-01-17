"""
ESDE Engine v5.3.2 - Phase 7B: Queue State Management

Manages deduplication, frequency tracking, and status of unknown tokens.
"""
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path


class QueueStateManager:
    """
    Manages unknown_queue_state.json for dedup and status tracking.
    
    State format per token:
    {
        "hash": str,           # Dedup key
        "token_norm": str,     # Normalized token
        "route": str,          # Primary route (A/B/C/D)
        "status": str,         # pending/resolved/quarantine/ignored
        "first_seen": str,     # ISO timestamp
        "last_seen": str,      # ISO timestamp
        "count": int,          # Occurrence count
        "last_action": str,    # Last action taken
        "priority": float,     # Computed priority score
    }
    """
    
    STATE_FILE = "./data/unknown_queue_state.json"
    
    STATUS_PENDING = "pending"
    STATUS_RESOLVED = "resolved"
    STATUS_QUARANTINE = "quarantine"  # High volatility
    STATUS_IGNORED = "ignored"
    STATUS_DRAFT = "draft"  # Needs human review
    
    def __init__(self, state_file: str = None):
        self.state_file = state_file or self.STATE_FILE
        self.state: Dict[str, Dict[str, Any]] = {}
        self._meta: Dict[str, Any] = {}
        self._dirty = False
        self._ensure_directory()
    
    def _ensure_directory(self):
        """Create state directory if needed."""
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, token_norm: str, route: str) -> str:
        """Compute dedup hash for token+route."""
        key = f"{token_norm}|{route}"
        return hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]
    
    def load(self) -> bool:
        """Load state from file."""
        try:
            if Path(self.state_file).exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.state = data.get("tokens", {})
                self._meta = data.get("_meta", {})
                return True
            return False
        except Exception as e:
            print(f"[QueueStateManager] Load error: {e}")
            return False
    
    def save(self) -> bool:
        """Save state to file."""
        try:
            self._meta["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._meta["token_count"] = len(self.state)
            
            data = {
                "_meta": self._meta,
                "tokens": self.state
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self._dirty = False
            return True
        except Exception as e:
            print(f"[QueueStateManager] Save error: {e}")
            return False
    
    def upsert(self, token_norm: str, route: str, 
               synsets: List[str] = None,
               context: str = None) -> Dict[str, Any]:
        """
        Insert or update a token in state.
        Returns the updated state entry.
        """
        now = datetime.now(timezone.utc).isoformat()
        hash_key = self._compute_hash(token_norm, route)
        
        if hash_key in self.state:
            # Update existing
            entry = self.state[hash_key]
            entry["count"] += 1
            entry["last_seen"] = now
        else:
            # Insert new
            entry = {
                "hash": hash_key,
                "token_norm": token_norm,
                "route": route,
                "status": self.STATUS_PENDING,
                "first_seen": now,
                "last_seen": now,
                "count": 1,
                "last_action": None,
                "priority": 0.0,
                "synsets": synsets or [],
                "sample_context": context
            }
            self.state[hash_key] = entry
        
        self._dirty = True
        return entry
    
    def get(self, token_norm: str, route: str) -> Optional[Dict[str, Any]]:
        """Get state entry for token+route."""
        hash_key = self._compute_hash(token_norm, route)
        return self.state.get(hash_key)
    
    def get_by_hash(self, hash_key: str) -> Optional[Dict[str, Any]]:
        """Get state entry by hash key."""
        return self.state.get(hash_key)
    
    def update_status(self, hash_key: str, status: str, 
                      action: str = None) -> bool:
        """Update status of a token."""
        if hash_key not in self.state:
            return False
        
        self.state[hash_key]["status"] = status
        self.state[hash_key]["last_action"] = action
        self.state[hash_key]["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._dirty = True
        return True
    
    def compute_priorities(self) -> None:
        """
        Compute priority scores for all pending tokens.
        Priority = frequency * recency * (1 if has_synsets else 0.5)
        """
        now = datetime.now(timezone.utc)
        
        for hash_key, entry in self.state.items():
            if entry["status"] != self.STATUS_PENDING:
                continue
            
            # Frequency factor (log scale)
            count = entry.get("count", 1)
            freq_factor = min(1.0, (count / 100) ** 0.5)
            
            # Recency factor (decay over 30 days)
            try:
                last_seen = datetime.fromisoformat(entry["last_seen"].replace('Z', '+00:00'))
                days_ago = (now - last_seen).days
                recency_factor = max(0.1, 1.0 - (days_ago / 30))
            except:
                recency_factor = 0.5
            
            # Has synsets factor
            synset_factor = 1.0 if entry.get("synsets") else 0.5
            
            # Combine
            priority = freq_factor * recency_factor * synset_factor
            entry["priority"] = round(priority, 4)
        
        self._dirty = True
    
    def get_pending_by_priority(self, limit: int = 100, 
                                 route_filter: str = None) -> List[Dict[str, Any]]:
        """Get pending tokens sorted by priority."""
        pending = []
        
        for entry in self.state.values():
            if entry["status"] != self.STATUS_PENDING:
                continue
            if route_filter and entry["route"] != route_filter:
                continue
            pending.append(entry)
        
        # Sort by priority descending
        pending.sort(key=lambda x: x.get("priority", 0), reverse=True)
        
        return pending[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about queue state."""
        stats = {
            "total": len(self.state),
            "by_status": {},
            "by_route": {}
        }
        
        for entry in self.state.values():
            status = entry.get("status", "unknown")
            route = entry.get("route", "unknown")
            
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["by_route"][route] = stats["by_route"].get(route, 0) + 1
        
        return stats
