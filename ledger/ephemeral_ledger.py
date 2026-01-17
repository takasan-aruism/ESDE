"""
ESDE Phase 8-5: Ephemeral Ledger
================================
In-memory semantic memory with decay and reinforcement.

Spec: v8.5 (Semantic Memory Preconditions)

Constraints:
- NO Persistence: All data is in-memory only.
- NO Hash Chain: No tamper-proofing (Phase 8-6+).
- NO Semantic Logic: Operators are not semantically unified.
- NO Emergence Classification: No creative/destructive tagging.

Conflict Handling:
- Contradictory meanings (e.g., Love vs Hate) are stored as separate entries.
- No cancellation or subtraction.
- Natural selection via weight competition.
"""

import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from copy import deepcopy

from .memory_math import (
    decay,
    reinforce,
    should_purge,
    get_tau_from_molecule,
    generate_fingerprint,
    compute_dt,
    now_utc,
    ALPHA,
    EPSILON,
    DEFAULT_TAU,
)


# ==========================================
# Data Structures
# ==========================================

@dataclass
class MemoryEntry:
    """Single memory entry in the ledger."""
    
    # Identity
    entry_id: str
    fingerprint: str
    
    # Content
    molecule: Dict
    source_text_hash: str
    
    # Temporal
    created_at: datetime
    last_updated: datetime
    tau: int
    
    # Weight
    weight: float
    observation_count: int
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        d['last_updated'] = self.last_updated.isoformat()
        return d


@dataclass
class LedgerStats:
    """Statistics for the ledger."""
    total_entries: int = 0
    total_observations: int = 0
    purged_count: int = 0
    avg_weight: float = 0.0
    max_weight: float = 0.0
    min_weight: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ==========================================
# Ephemeral Ledger
# ==========================================

class EphemeralLedger:
    """
    In-memory semantic memory ledger.
    
    Features:
    - Observation recording with fingerprint-based identity
    - Exponential decay over time
    - Reinforcement on re-observation
    - Automatic purge below threshold
    - Conflict co-existence (no cancellation)
    """
    
    def __init__(self, 
                 alpha: float = ALPHA,
                 epsilon: float = EPSILON,
                 default_tau: int = DEFAULT_TAU):
        """
        Initialize ephemeral ledger.
        
        Args:
            alpha: Reinforcement learning rate
            epsilon: Oblivion threshold
            default_tau: Default time constant (seconds)
        """
        self._entries: Dict[str, MemoryEntry] = {}
        self._alpha = alpha
        self._epsilon = epsilon
        self._default_tau = default_tau
        
        # Statistics
        self._total_observations = 0
        self._purged_count = 0
        
        # Audit log (in-memory only)
        self._audit_log: List[Dict] = []
    
    # ==========================================
    # Core Operations
    # ==========================================
    
    def upsert(self, 
               molecule: Dict, 
               source_text: str,
               timestamp: Optional[datetime] = None) -> MemoryEntry:
        """
        Record observation and reinforce if exists.
        
        - If fingerprint exists: reinforce weight
        - If new: create entry with weight=1.0
        
        Args:
            molecule: Molecule dict from generator
            source_text: Original input text
            timestamp: Observation time (default: now)
        
        Returns:
            Updated or created MemoryEntry
        """
        if timestamp is None:
            timestamp = now_utc()
        
        # Generate fingerprint
        fingerprint = generate_fingerprint(molecule)
        source_hash = hashlib.sha256(source_text.encode('utf-8')).hexdigest()[:16]
        
        self._total_observations += 1
        
        if fingerprint in self._entries:
            # Existing entry - reinforce
            entry = self._entries[fingerprint]
            old_weight = entry.weight
            
            # Apply decay first (time since last update)
            dt = compute_dt(entry.last_updated, timestamp)
            decayed_weight = decay(entry.weight, dt, entry.tau)
            
            # Then reinforce
            new_weight = reinforce(decayed_weight, self._alpha)
            
            entry.weight = new_weight
            entry.last_updated = timestamp
            entry.observation_count += 1
            
            self._log_audit('reinforce', fingerprint, {
                'old_weight': old_weight,
                'decayed_weight': decayed_weight,
                'new_weight': new_weight,
                'dt': dt,
            })
            
        else:
            # New entry
            tau = get_tau_from_molecule(molecule)
            
            entry = MemoryEntry(
                entry_id=str(uuid.uuid4())[:8],
                fingerprint=fingerprint,
                molecule=deepcopy(molecule),
                source_text_hash=source_hash,
                created_at=timestamp,
                last_updated=timestamp,
                tau=tau,
                weight=1.0,
                observation_count=1,
            )
            
            self._entries[fingerprint] = entry
            
            self._log_audit('create', fingerprint, {
                'weight': 1.0,
                'tau': tau,
            })
        
        return entry
    
    def decay_all(self, current_timestamp: Optional[datetime] = None) -> int:
        """
        Apply decay to all entries and purge if below threshold.
        
        Args:
            current_timestamp: Current time (default: now)
        
        Returns:
            Number of entries purged
        """
        if current_timestamp is None:
            current_timestamp = now_utc()
        
        purged = 0
        to_purge = []
        
        for fingerprint, entry in self._entries.items():
            dt = compute_dt(entry.last_updated, current_timestamp)
            
            if dt > 0:
                old_weight = entry.weight
                new_weight = decay(entry.weight, dt, entry.tau)
                entry.weight = new_weight
                entry.last_updated = current_timestamp
                
                if should_purge(new_weight, self._epsilon):
                    to_purge.append(fingerprint)
                    self._log_audit('purge', fingerprint, {
                        'final_weight': new_weight,
                        'reason': 'below_epsilon',
                    })
        
        # Purge entries
        for fingerprint in to_purge:
            del self._entries[fingerprint]
            purged += 1
        
        self._purged_count += purged
        return purged
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get current state of all memories.
        
        Returns:
            Dict with entries and statistics
        """
        entries_list = [e.to_dict() for e in self._entries.values()]
        
        # Sort by weight descending
        entries_list.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'timestamp': now_utc().isoformat(),
            'stats': self.get_stats().to_dict(),
            'entries': entries_list,
        }
    
    # ==========================================
    # Query Operations
    # ==========================================
    
    def get_entry(self, fingerprint: str) -> Optional[MemoryEntry]:
        """Get entry by fingerprint."""
        return self._entries.get(fingerprint)
    
    def get_active(self, min_weight: float = 0.1) -> List[MemoryEntry]:
        """
        Get entries with weight above threshold.
        
        Args:
            min_weight: Minimum weight threshold
        
        Returns:
            List of active entries (sorted by weight desc)
        """
        active = [e for e in self._entries.values() if e.weight >= min_weight]
        active.sort(key=lambda e: e.weight, reverse=True)
        return active
    
    def get_by_atom(self, atom: str) -> List[MemoryEntry]:
        """
        Get entries containing specific atom.
        
        Args:
            atom: Atom ID (e.g., "EMO.love")
        
        Returns:
            List of matching entries
        """
        results = []
        for entry in self._entries.values():
            atoms_in_entry = [
                aa.get('atom') 
                for aa in entry.molecule.get('active_atoms', [])
            ]
            if atom in atoms_in_entry:
                results.append(entry)
        return results
    
    # ==========================================
    # Statistics
    # ==========================================
    
    def get_stats(self) -> LedgerStats:
        """Get current ledger statistics."""
        if not self._entries:
            return LedgerStats(
                total_entries=0,
                total_observations=self._total_observations,
                purged_count=self._purged_count,
            )
        
        weights = [e.weight for e in self._entries.values()]
        
        return LedgerStats(
            total_entries=len(self._entries),
            total_observations=self._total_observations,
            purged_count=self._purged_count,
            avg_weight=sum(weights) / len(weights),
            max_weight=max(weights),
            min_weight=min(weights),
        )
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._entries)
    
    # ==========================================
    # Audit Log (Internal)
    # ==========================================
    
    def _log_audit(self, action: str, fingerprint: str, details: Dict):
        """Log audit event (in-memory only)."""
        self._audit_log.append({
            'timestamp': now_utc().isoformat(),
            'action': action,
            'fingerprint': fingerprint[:8],
            'details': details,
        })
        
        # Keep only last 1000 events
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
    
    def get_audit_log(self, last_n: int = 100) -> List[Dict]:
        """Get recent audit log entries."""
        return self._audit_log[-last_n:]
    
    # ==========================================
    # Debug / Testing
    # ==========================================
    
    def clear(self):
        """Clear all entries (for testing)."""
        self._entries.clear()
        self._audit_log.clear()
        self._total_observations = 0
        self._purged_count = 0
    
    def simulate_time_passage(self, seconds: float) -> int:
        """
        Simulate time passage for testing.
        
        Args:
            seconds: Seconds to advance
        
        Returns:
            Number of entries purged
        """
        future = datetime.now(timezone.utc)
        from datetime import timedelta
        future = future + timedelta(seconds=seconds)
        return self.decay_all(future)
