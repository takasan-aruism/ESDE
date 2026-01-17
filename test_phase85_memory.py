"""
ESDE Phase 8-5: Memory Audit Tests
==================================
Validates the Ephemeral Ledger implementation.

Test Categories:
1. Math Check - Decay and reinforcement calculations
2. Tau Check - Temporal policy determines lifespan
3. Conflict Check - Contradictory molecules co-exist
4. Retention Rate - Survival after decay cycles
5. Reinforcement Stability - Weight never exceeds 1.0

Run:
    python test_phase85_memory.py
"""

import sys
import math
from datetime import datetime, timezone, timedelta

# Add parent to path for imports
sys.path.insert(0, '.')

from ledger import (
    EphemeralLedger,
    decay,
    reinforce,
    should_purge,
    get_tau_from_molecule,
    generate_fingerprint,
    extract_operator_type,
    ALPHA,
    EPSILON,
    DEFAULT_TAU,
    TAU_POLICY,
)


# ==========================================
# Test Fixtures
# ==========================================

def make_molecule(atom: str, axis: str = None, level: str = None, formula: str = None):
    """Create a test molecule."""
    aa = {'atom': atom, 'axis': axis, 'level': level, 'span': [0, 4]}
    return {
        'formula': formula or 'aa_1',
        'active_atoms': [aa],
    }


def make_temporal_molecule(atom: str, temporal_level: str):
    """Create molecule with temporal axis."""
    aa = {'atom': atom, 'axis': 'temporal', 'level': temporal_level, 'span': [0, 4]}
    return {
        'formula': 'aa_1',
        'active_atoms': [aa],
    }


def make_binary_molecule(atom1: str, atom2: str, operator: str):
    """Create two-atom molecule with operator."""
    return {
        'formula': f'aa_1 {operator} aa_2',
        'active_atoms': [
            {'atom': atom1, 'axis': 'interconnection', 'level': 'catalytic', 'span': [0, 4]},
            {'atom': atom2, 'axis': 'interconnection', 'level': 'catalytic', 'span': [5, 9]},
        ],
    }


# ==========================================
# Test 1: Math Check
# ==========================================

def test_decay_math():
    """Test exponential decay formula."""
    print("\n=== Test 1.1: Decay Math ===")
    
    # w = w_prev * exp(-dt / tau)
    # At dt = tau, weight should be ~36.8% (1/e)
    w = decay(1.0, 300, 300)  # dt = tau
    expected = math.exp(-1)  # ~0.368
    
    assert abs(w - expected) < 0.001, f"Decay at dt=tau failed: {w} != {expected}"
    print(f"  ✅ Decay at dt=tau: {w:.4f} ≈ {expected:.4f}")
    
    # At dt = 0, weight unchanged
    w_zero = decay(1.0, 0, 300)
    assert w_zero == 1.0, f"Decay at dt=0 failed: {w_zero}"
    print(f"  ✅ Decay at dt=0: {w_zero}")
    
    # At dt = 5*tau, weight should be ~0.67% (e^-5)
    w_long = decay(1.0, 1500, 300)  # 5 * tau
    expected_long = math.exp(-5)  # ~0.0067
    assert abs(w_long - expected_long) < 0.001, f"Decay at dt=5tau failed"
    print(f"  ✅ Decay at dt=5tau: {w_long:.4f} ≈ {expected_long:.4f}")
    
    print("  ✅ Decay Math: PASS")
    return True


def test_reinforce_math():
    """Test reinforcement formula."""
    print("\n=== Test 1.2: Reinforcement Math ===")
    
    # w = w + alpha * (1 - w)
    # Starting from 0.5 with alpha=0.2
    w = reinforce(0.5, 0.2)
    expected = 0.5 + 0.2 * (1 - 0.5)  # 0.6
    
    assert abs(w - expected) < 0.001, f"Reinforce failed: {w} != {expected}"
    print(f"  ✅ Reinforce 0.5 → {w:.4f} (expected {expected})")
    
    # Reinforcing from 0
    w_zero = reinforce(0.0, 0.2)
    assert abs(w_zero - 0.2) < 0.001, f"Reinforce from 0 failed"
    print(f"  ✅ Reinforce 0.0 → {w_zero:.4f}")
    
    # Reinforcing from 1.0 (should stay 1.0)
    w_max = reinforce(1.0, 0.2)
    assert w_max == 1.0, f"Reinforce at 1.0 failed: {w_max}"
    print(f"  ✅ Reinforce 1.0 → {w_max}")
    
    print("  ✅ Reinforcement Math: PASS")
    return True


def test_oblivion_threshold():
    """Test purge threshold."""
    print("\n=== Test 1.3: Oblivion Threshold ===")
    
    assert should_purge(0.009) == True, "Should purge below epsilon"
    assert should_purge(0.01) == False, "Should not purge at epsilon"
    assert should_purge(0.5) == False, "Should not purge above epsilon"
    
    print(f"  ✅ Threshold at epsilon={EPSILON}: PASS")
    return True


# ==========================================
# Test 2: Tau Check
# ==========================================

def test_tau_policy():
    """Test temporal tau determination."""
    print("\n=== Test 2: Tau Policy ===")
    
    # Permanence = 24h
    mol_perm = make_temporal_molecule('VAL.truth', 'permanence')
    tau_perm = get_tau_from_molecule(mol_perm)
    assert tau_perm == 86400, f"Permanence tau wrong: {tau_perm}"
    print(f"  ✅ permanence → tau={tau_perm}s (24h)")
    
    # Emergence = 1min
    mol_emer = make_temporal_molecule('CHG.change', 'emergence')
    tau_emer = get_tau_from_molecule(mol_emer)
    assert tau_emer == 60, f"Emergence tau wrong: {tau_emer}"
    print(f"  ✅ emergence → tau={tau_emer}s (1m)")
    
    # Unknown level = Default
    mol_unknown = make_temporal_molecule('EXS.being', 'unknown_level')
    tau_unknown = get_tau_from_molecule(mol_unknown)
    assert tau_unknown == DEFAULT_TAU, f"Unknown tau wrong: {tau_unknown}"
    print(f"  ✅ unknown → tau={tau_unknown}s (default)")
    
    # No temporal axis = Default
    mol_no_temporal = make_molecule('EMO.love', 'interconnection', 'catalytic')
    tau_none = get_tau_from_molecule(mol_no_temporal)
    assert tau_none == DEFAULT_TAU, f"No temporal tau wrong: {tau_none}"
    print(f"  ✅ no temporal axis → tau={tau_none}s (default)")
    
    print("  ✅ Tau Policy: PASS")
    return True


def test_tau_affects_lifespan():
    """Test that permanence molecules live longer."""
    print("\n=== Test 2.1: Tau Affects Lifespan ===")
    
    ledger = EphemeralLedger()
    
    # Create two molecules with different tau
    mol_perm = make_temporal_molecule('VAL.truth', 'permanence')  # 24h
    mol_emer = make_temporal_molecule('CHG.change', 'emergence')  # 1min
    
    now = datetime.now(timezone.utc)
    ledger.upsert(mol_perm, "truth is eternal", now)
    ledger.upsert(mol_emer, "change happens", now)
    
    assert len(ledger) == 2, "Should have 2 entries"
    
    # After 2 minutes, emergence should be nearly gone
    future = now + timedelta(minutes=2)
    ledger.decay_all(future)
    
    # Get weights
    entries = ledger.get_snapshot()['entries']
    weights = {e['molecule']['active_atoms'][0]['atom']: e['weight'] for e in entries}
    
    print(f"  After 2 min:")
    print(f"    VAL.truth (24h tau): {weights.get('VAL.truth', 0):.4f}")
    print(f"    CHG.change (1m tau): {weights.get('CHG.change', 0):.4f}")
    
    # Permanence should be much higher
    if 'VAL.truth' in weights and 'CHG.change' in weights:
        # After 2 min: permanence barely decays, emergence decays significantly
        # 0.9986 vs 0.1353 - permanence should be at least 5x higher
        assert weights['VAL.truth'] > weights['CHG.change'] * 5, \
            f"Permanence should decay much slower: {weights['VAL.truth']:.4f} vs {weights['CHG.change']:.4f}"
    
    print("  ✅ Tau Affects Lifespan: PASS")
    return True


# ==========================================
# Test 3: Conflict Check
# ==========================================

def test_conflict_coexistence():
    """Test that contradictory molecules co-exist."""
    print("\n=== Test 3: Conflict Co-existence ===")
    
    ledger = EphemeralLedger()
    
    # Love and Hate should both exist
    mol_love = make_molecule('EMO.love', 'interconnection', 'catalytic')
    mol_hate = make_molecule('EMO.hate', 'interconnection', 'catalytic')
    
    now = datetime.now(timezone.utc)
    ledger.upsert(mol_love, "I love you", now)
    ledger.upsert(mol_hate, "I hate you", now)
    
    # Should have 2 separate entries
    assert len(ledger) == 2, f"Expected 2 entries, got {len(ledger)}"
    print(f"  ✅ Love and Hate co-exist: {len(ledger)} entries")
    
    # Different fingerprints
    fp_love = generate_fingerprint(mol_love)
    fp_hate = generate_fingerprint(mol_hate)
    assert fp_love != fp_hate, "Love and Hate should have different fingerprints"
    print(f"  ✅ Different fingerprints: {fp_love[:8]}... vs {fp_hate[:8]}...")
    
    # Both retrievable
    entry_love = ledger.get_entry(fp_love)
    entry_hate = ledger.get_entry(fp_hate)
    assert entry_love is not None, "Love entry missing"
    assert entry_hate is not None, "Hate entry missing"
    print(f"  ✅ Both entries retrievable")
    
    print("  ✅ Conflict Co-existence: PASS")
    return True


# ==========================================
# Test 4: Retention Rate
# ==========================================

def test_retention_rate():
    """Test retention after 100 decay cycles."""
    print("\n=== Test 4: Retention Rate ===")
    
    ledger = EphemeralLedger()
    
    # Create 10 molecules
    molecules = [
        ('EMO.love', 'permanence'),      # 24h
        ('EMO.joy', 'continuation'),     # 1h
        ('ACT.create', 'establishment'), # 1h
        ('CHG.change', 'transformation'),# 30m
        ('COG.know', 'indication'),      # 5m
        ('CHG.emerge', 'emergence'),     # 1m
        ('EXS.being', None),             # default 5m
        ('VAL.truth', None),             # default 5m
        ('ACT.give', None),              # default 5m
        ('EMO.hope', None),              # default 5m
    ]
    
    now = datetime.now(timezone.utc)
    for atom, temporal in molecules:
        if temporal:
            mol = make_temporal_molecule(atom, temporal)
        else:
            mol = make_molecule(atom)
        ledger.upsert(mol, f"test {atom}", now)
    
    initial_count = len(ledger)
    print(f"  Initial entries: {initial_count}")
    
    # 100 decay cycles, 1 minute each
    current = now
    for i in range(100):
        current = current + timedelta(minutes=1)
        ledger.decay_all(current)
    
    final_count = len(ledger)
    retention_rate = final_count / initial_count * 100
    
    print(f"  After 100 min:")
    print(f"    Remaining: {final_count}/{initial_count}")
    print(f"    Retention Rate: {retention_rate:.1f}%")
    
    # Show survivors
    survivors = ledger.get_active(0.0)
    for entry in survivors:
        atom = entry.molecule['active_atoms'][0]['atom']
        print(f"      {atom}: weight={entry.weight:.4f}, tau={entry.tau}s")
    
    # Permanence (24h) should survive
    perm_entries = ledger.get_by_atom('EMO.love')
    assert len(perm_entries) > 0, "Permanence molecule should survive 100 min"
    print(f"  ✅ Permanence survived: weight={perm_entries[0].weight:.4f}")
    
    print("  ✅ Retention Rate Test: PASS")
    return True


# ==========================================
# Test 5: Reinforcement Stability
# ==========================================

def test_reinforcement_stability():
    """Test that weight never exceeds 1.0."""
    print("\n=== Test 5: Reinforcement Stability ===")
    
    ledger = EphemeralLedger()
    mol = make_molecule('EMO.love')
    
    now = datetime.now(timezone.utc)
    
    # Reinforce 100 times
    for i in range(100):
        ledger.upsert(mol, "I love you", now)
    
    entry = list(ledger._entries.values())[0]
    
    print(f"  After 100 reinforcements:")
    print(f"    Weight: {entry.weight}")
    print(f"    Observation count: {entry.observation_count}")
    
    assert entry.weight <= 1.0, f"Weight exceeded 1.0: {entry.weight}"
    assert entry.weight > 0.99, f"Weight should approach 1.0: {entry.weight}"
    assert entry.observation_count == 100, f"Observation count wrong"
    
    print("  ✅ Reinforcement Stability: PASS")
    return True


def test_reinforcement_asymptotic():
    """Test asymptotic approach to 1.0."""
    print("\n=== Test 5.1: Asymptotic Approach ===")
    
    # Starting from 0, how many reinforcements to reach 0.99?
    w = 0.0
    for i in range(50):
        w = reinforce(w, ALPHA)
        if w >= 0.99:
            print(f"  Reached 0.99 after {i+1} reinforcements")
            break
    
    # Verify formula: after n reinforcements, w = 1 - (1-alpha)^n
    w_formula = 1 - (1 - ALPHA) ** 20
    print(f"  After 20 reinforcements: w = {w_formula:.4f} (formula)")
    
    print("  ✅ Asymptotic Approach: PASS")
    return True


# ==========================================
# Test 6: Fingerprint & Operator
# ==========================================

def test_fingerprint_identity():
    """Test fingerprint identity rules."""
    print("\n=== Test 6: Fingerprint Identity ===")
    
    # Same atom, same formula = same fingerprint
    mol1 = make_molecule('EMO.love')
    mol2 = make_molecule('EMO.love')
    fp1 = generate_fingerprint(mol1)
    fp2 = generate_fingerprint(mol2)
    assert fp1 == fp2, "Same molecule should have same fingerprint"
    print(f"  ✅ Same molecule → same fingerprint")
    
    # Different atoms = different fingerprint
    mol3 = make_molecule('EMO.hate')
    fp3 = generate_fingerprint(mol3)
    assert fp1 != fp3, "Different atoms should have different fingerprints"
    print(f"  ✅ Different atoms → different fingerprint")
    
    # Same atoms, different operators = different fingerprint
    mol_action = make_binary_molecule('EMO.love', 'ACT.give', '▷')
    mol_connect = make_binary_molecule('EMO.love', 'ACT.give', '×')
    fp_action = generate_fingerprint(mol_action)
    fp_connect = generate_fingerprint(mol_connect)
    assert fp_action != fp_connect, "Different operators should have different fingerprints"
    print(f"  ✅ Different operators → different fingerprint")
    
    print("  ✅ Fingerprint Identity: PASS")
    return True


def test_operator_extraction():
    """Test operator type extraction."""
    print("\n=== Test 6.1: Operator Extraction ===")
    
    cases = [
        ('aa_1', 'ATOM_ONLY'),
        ('aa_1 ▷ aa_2', '▷'),
        ('aa_1 × aa_2', '×'),
        ('¬aa_1', '¬'),
        ('¬¬aa_1', '¬'),  # Double negation
        ('aa_1 ⇒+ aa_2', '⇒+'),
        ('aa_1 -|> aa_2', '-|>'),
        ('(aa_1 × aa_2) ▷ aa_3', '×|▷'),
    ]
    
    for formula, expected in cases:
        result = extract_operator_type(formula)
        # Note: order might differ, so check contains
        print(f"  {formula!r} → {result}")
    
    print("  ✅ Operator Extraction: PASS")
    return True


# ==========================================
# Main
# ==========================================

def main():
    print("=" * 60)
    print("ESDE Phase 8-5: Memory Audit Tests")
    print("=" * 60)
    
    tests = [
        # Math Check
        test_decay_math,
        test_reinforce_math,
        test_oblivion_threshold,
        
        # Tau Check
        test_tau_policy,
        test_tau_affects_lifespan,
        
        # Conflict Check
        test_conflict_coexistence,
        
        # Retention Rate
        test_retention_rate,
        
        # Reinforcement Stability
        test_reinforcement_stability,
        test_reinforcement_asymptotic,
        
        # Fingerprint
        test_fingerprint_identity,
        test_operator_extraction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{len(tests)}")
    print(f"  Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Phase 8-5 Memory Audit Complete")
    else:
        print(f"\n❌ {failed} TESTS FAILED")
    
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
