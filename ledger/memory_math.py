"""
ESDE Phase 8-5: Memory Mathematics
==================================
Decay, Reinforcement, Tau Policy, and Fingerprint generation.

Spec: v8.5 (Semantic Memory Preconditions)

Mathematical Model:
  Decay:         w = w_prev * exp(-dt / tau)
  Reinforcement: w = w + alpha * (1 - w)
  Oblivion:      if w < epsilon: purge
"""

import math
import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set


# ==========================================
# Constants (Spec v8.5)
# ==========================================

# Learning rate for reinforcement
ALPHA = 0.2

# Oblivion threshold
EPSILON = 0.01

# Valid operators (v0.3 complete list)
VALID_OPERATORS: Set[str] = {
    '×',   # 結合 (Connection)
    '▷',   # 作用 (Action)
    '→',   # 遷移 (Transition)
    '⊕',   # 並置 (Juxtaposition)
    '|',   # 条件 (Condition)
    '◯',   # 対象 (Target)
    '↺',   # 再帰 (Recursion)
    '〈',   # 階層開始 (Hierarchy Open)
    '〉',   # 階層終了 (Hierarchy Close)
    '≡',   # 等価 (Equivalence)
    '≃',   # 実用等価 (Practical Equivalence)
    '¬',   # 否定 (Negation)
    '⇒',   # 創発 (Emergence)
    '⇒+',  # 創造的創発 (Creative Emergence)
    '-|>', # 破壊的創発 (Destructive Emergence)
}

# Temporal Tau Policy (seconds)
TAU_POLICY: Dict[str, int] = {
    'permanence': 86400,      # 24h - 真理・恒久
    'continuation': 3600,     # 1h - 継続的状態
    'establishment': 3600,    # 1h - 確立された事実
    'transformation': 1800,   # 30m - 変容中
    'indication': 300,        # 5m - 兆し
    'emergence': 60,          # 1m - 瞬間の創発
}

DEFAULT_TAU = 300  # 5 minutes


# ==========================================
# Time Handling
# ==========================================

def compute_dt(last_updated: datetime, now: datetime) -> float:
    """
    Compute time delta in seconds.
    
    - All timestamps must be UTC timezone-aware.
    - dt is clamped to >= 0 (handles clock skew).
    
    Returns:
        dt in seconds (float, >= 0)
    """
    dt = (now - last_updated).total_seconds()
    if dt < 0:
        # Clock skew detected - clamp to 0
        return 0.0
    return dt


def now_utc() -> datetime:
    """Get current UTC timestamp (timezone-aware)."""
    return datetime.now(timezone.utc)


# ==========================================
# Memory Math Functions
# ==========================================

def decay(w_prev: float, dt: float, tau: float) -> float:
    """
    Apply exponential decay to weight.
    
    Formula: w = w_prev * exp(-dt / tau)
    
    Args:
        w_prev: Previous weight [0, 1]
        dt: Time delta in seconds (>= 0)
        tau: Time constant in seconds (> 0)
    
    Returns:
        New weight after decay [0, 1]
    """
    if tau <= 0:
        tau = DEFAULT_TAU
    if dt < 0:
        dt = 0
    
    w = w_prev * math.exp(-dt / tau)
    return max(0.0, min(1.0, w))


def reinforce(w_prev: float, alpha: float = ALPHA) -> float:
    """
    Apply reinforcement (asymptotic approach to 1.0).
    
    Formula: w = w + alpha * (1 - w)
    
    Args:
        w_prev: Previous weight [0, 1]
        alpha: Learning rate (default: 0.2)
    
    Returns:
        New weight after reinforcement [0, 1]
    """
    w = w_prev + alpha * (1.0 - w_prev)
    return max(0.0, min(1.0, w))


def should_purge(weight: float, epsilon: float = EPSILON) -> bool:
    """
    Check if memory should be purged (forgotten).
    
    Args:
        weight: Current weight
        epsilon: Oblivion threshold (default: 0.01)
    
    Returns:
        True if weight < epsilon (should purge)
    """
    return weight < epsilon


# ==========================================
# Tau Policy
# ==========================================

def get_tau_from_molecule(molecule: Dict) -> int:
    """
    Determine tau from molecule's temporal axis levels.
    
    Policy:
    - Scan active_atoms for temporal axis/level.
    - Select the longest tau (most permanent).
    - Unknown temporal levels are ignored.
    - If no recognized level, use Default.
    
    Args:
        molecule: Molecule dict with 'active_atoms' list
    
    Returns:
        tau in seconds
    """
    active_atoms = molecule.get('active_atoms', [])
    
    found_taus = []
    for aa in active_atoms:
        axis = aa.get('axis')
        level = aa.get('level')
        
        # Check if this atom has temporal axis
        if axis == 'temporal' and level:
            if level in TAU_POLICY:
                found_taus.append(TAU_POLICY[level])
    
    if found_taus:
        # Return the longest tau (most permanent)
        return max(found_taus)
    
    return DEFAULT_TAU


# ==========================================
# Fingerprint Generation
# ==========================================

def extract_operator_type(formula: Optional[str]) -> str:
    """
    Extract operator type string from formula.
    
    - Parse formula string.
    - Extract operator symbols from allowed set.
    - Return sorted unique operators joined by '|'.
    - If no operators, return 'ATOM_ONLY'.
    
    Args:
        formula: Formula string (e.g., "aa_1 ▷ aa_2")
    
    Returns:
        OperatorType string (e.g., "▷" or "ATOM_ONLY")
    """
    if not formula:
        return "ATOM_ONLY"
    
    found_ops = set()
    
    # Check multi-char operators first
    for op in ['⇒+', '-|>']:
        if op in formula:
            found_ops.add(op)
    
    # Check single-char operators
    for char in formula:
        if char in VALID_OPERATORS and char not in ['⇒', '-', '|', '>']:
            # Avoid partial matches with multi-char ops
            found_ops.add(char)
    
    # Special handling for ⇒ (not part of ⇒+)
    if '⇒' in formula and '⇒+' not in formula:
        found_ops.add('⇒')
    
    if not found_ops:
        return "ATOM_ONLY"
    
    return "|".join(sorted(found_ops))


def generate_fingerprint(molecule: Dict) -> str:
    """
    Generate canonical fingerprint for molecule identity.
    
    Key = SHA256( "|".join(sorted(atom_ids)) + "::" + OperatorType )
    
    Args:
        molecule: Molecule dict with 'active_atoms' and 'formula'
    
    Returns:
        Fingerprint string (32 hex chars)
    """
    active_atoms = molecule.get('active_atoms', [])
    formula = molecule.get('formula', '')
    
    # Extract atom IDs
    atom_ids = []
    for aa in active_atoms:
        atom = aa.get('atom')
        if atom:
            atom_ids.append(atom)
    
    # Sort for canonical order
    atom_ids_str = "|".join(sorted(atom_ids))
    
    # Extract operator type
    op_type = extract_operator_type(formula)
    
    # Combine
    key_material = f"{atom_ids_str}::{op_type}"
    
    # Hash
    fingerprint = hashlib.sha256(key_material.encode('utf-8')).hexdigest()[:32]
    
    return fingerprint


# ==========================================
# Utility Functions
# ==========================================

def format_tau(tau_seconds: int) -> str:
    """Format tau as human-readable string."""
    if tau_seconds >= 86400:
        return f"{tau_seconds // 86400}d"
    elif tau_seconds >= 3600:
        return f"{tau_seconds // 3600}h"
    elif tau_seconds >= 60:
        return f"{tau_seconds // 60}m"
    else:
        return f"{tau_seconds}s"
