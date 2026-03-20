#!/usr/bin/env python3
"""
ESDE — SimpleSynapseLoader Regression Test
============================================
GPT Audit Point E: Verify patch application correctness.

Usage:
  python3 tests/test_projection_loader.py --synapse esde_synapses_v3.json

Author: Claude (per GPT Audit v3.4 §6)
Date: 2026-03-04
"""

import json
import sys
import tempfile
import os
from pathlib import Path

# Add parent dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_projection_experiment import SimpleSynapseLoader


def _make_temp_patch(patches_list: list) -> str:
    """Write a temporary patch JSON file."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({"patches": patches_list}, tmp)
    tmp.close()
    return tmp.name


# ============================================================
# Tests
# ============================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, condition: bool, name: str, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
            print(msg)
            self.errors.append(msg)
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for e in self.errors:
                print(e)
        return self.failed == 0


def test_add_edge(synapse_path: str, results: TestResults):
    """Test that add_edge creates correct entries."""
    print("\n--- Test: add_edge ---")
    
    patch_path = _make_temp_patch([
        {
            "op": "add_edge",
            "edge_key": "capital.n.03::SOC.official",
            "synset_id": "capital.n.03",
            "atom": "SOC.official",
            "score": 0.85,
            "reason": "test_add_edge",
            "metadata": {}
        },
        {
            "op": "add_edge",
            "edge_key": "capital.n.03::SOC.city",
            "synset_id": "capital.n.03",
            "atom": "SOC.city",
            "score": 0.80,
            "reason": "test_add_edge",
            "metadata": {}
        },
        {
            "op": "add_edge",
            "edge_key": "capital.n.03::SPC.place",
            "synset_id": "capital.n.03",
            "atom": "SPC.place",
            "score": 0.65,
            "reason": "test_add_edge",
            "metadata": {}
        },
    ])
    
    try:
        loader = SimpleSynapseLoader(synapse_path, patches=[patch_path])
        
        edges = loader.get_edges("capital.n.03")
        results.check(len(edges) == 3,
                       "capital.n.03 has 3 edges",
                       f"got {len(edges)}")
        
        atoms = {e.get("concept_id") for e in edges}
        results.check(atoms == {"SOC.official", "SOC.city", "SPC.place"},
                       "capital.n.03 atoms correct",
                       f"got {atoms}")
        
        # Check field names are translated
        for e in edges:
            results.check("concept_id" in e,
                           f"Edge has concept_id field ({e.get('concept_id','')})")
            results.check("raw_score" in e,
                           f"Edge has raw_score field ({e.get('raw_score','')})")
            break  # check one is enough
        
        # Check score values
        scores = {e["concept_id"]: e["raw_score"] for e in edges}
        results.check(abs(scores.get("SOC.official", 0) - 0.85) < 0.001,
                       "SOC.official score = 0.85",
                       f"got {scores.get('SOC.official')}")
    finally:
        os.unlink(patch_path)


def test_disable_edge(synapse_path: str, results: TestResults):
    """Test that disable_edge removes specific atom, not entire synset."""
    print("\n--- Test: disable_edge ---")
    
    patch_path = _make_temp_patch([
        {
            "op": "disable_edge",
            "edge_key": "capital.n.01::STA.wealth",
            "synset_id": "capital.n.01",
            "atom": "STA.wealth",
            "score": 0.0,
            "reason": "test_disable",
            "metadata": {}
        }
    ])
    
    try:
        loader = SimpleSynapseLoader(synapse_path, patches=[patch_path])
        
        edges_n01 = loader.get_edges("capital.n.01")
        atoms_n01 = {e.get("concept_id") for e in edges_n01}
        results.check("STA.wealth" not in atoms_n01,
                       "STA.wealth removed from capital.n.01",
                       f"atoms still contain: {atoms_n01}")
        
        # capital.n.02 should be unaffected
        edges_n02 = loader.get_edges("capital.n.02")
        atoms_n02 = {e.get("concept_id") for e in edges_n02}
        results.check("STA.wealth" in atoms_n02,
                       "capital.n.02 unaffected (STA.wealth preserved)",
                       f"atoms: {atoms_n02}")
    finally:
        os.unlink(patch_path)


def test_existing_edges_preserved(synapse_path: str, results: TestResults):
    """Test that patching doesn't corrupt existing entries."""
    print("\n--- Test: existing edges preserved ---")
    
    loader_base = SimpleSynapseLoader(synapse_path)
    
    patch_path = _make_temp_patch([
        {
            "op": "add_edge",
            "edge_key": "capital.n.03::SOC.official",
            "synset_id": "capital.n.03",
            "atom": "SOC.official",
            "score": 0.85,
            "reason": "test",
            "metadata": {}
        }
    ])
    
    try:
        loader_patched = SimpleSynapseLoader(synapse_path, patches=[patch_path])
        
        # capital.n.01 should still have STA.wealth in both
        base_n01 = loader_base.get_edges("capital.n.01")
        patch_n01 = loader_patched.get_edges("capital.n.01")
        
        base_atoms = {e.get("concept_id") for e in base_n01}
        patch_atoms = {e.get("concept_id") for e in patch_n01}
        
        results.check(base_atoms == patch_atoms,
                       "capital.n.01 edges identical before/after patch",
                       f"base={base_atoms}, patched={patch_atoms}")
        
        # Total synsets should increase (new synset added)
        base_count = len(loader_base.synapses)
        patched_count = len(loader_patched.synapses)
        results.check(patched_count >= base_count,
                       f"Synset count: {base_count} → {patched_count} (not decreased)",
                       f"base={base_count}, patched={patched_count}")
        
        # Specifically, capital.n.03 is new
        results.check("capital.n.03" not in loader_base.synapses,
                       "capital.n.03 absent in base",
                       "was already present")
        results.check("capital.n.03" in loader_patched.synapses,
                       "capital.n.03 present after patch")
    finally:
        os.unlink(patch_path)


def test_v34_integration(synapse_path: str, results: TestResults):
    """Integration test with actual v3.4 patch file."""
    print("\n--- Test: v3.4 patch integration ---")
    
    v34_path = Path(synapse_path).parent / "patches" / "synapse_v3.4.json"
    if not v34_path.exists():
        # Try relative to CWD
        v34_path = Path("patches/synapse_v3.4.json")
    
    if not v34_path.exists():
        print("  ⚠️  patches/synapse_v3.4.json not found — skipping integration test")
        return
    
    loader = SimpleSynapseLoader(synapse_path, patches=[str(v34_path)])
    
    # capital.n.03 should have 3 edges
    edges_n03 = loader.get_edges("capital.n.03")
    results.check(len(edges_n03) == 3,
                   "v3.4: capital.n.03 has 3 edges",
                   f"got {len(edges_n03)}")
    
    # capital.n.05 should have 2 edges
    edges_n05 = loader.get_edges("capital.n.05")
    results.check(len(edges_n05) == 2,
                   "v3.4: capital.n.05 has 2 edges",
                   f"got {len(edges_n05)}")
    
    # capital.n.06 should have 1 edge
    edges_n06 = loader.get_edges("capital.n.06")
    results.check(len(edges_n06) == 1,
                   "v3.4: capital.n.06 has 1 edge",
                   f"got {len(edges_n06)}")
    
    # Verify SOC.official is reachable for capital
    all_capital_atoms = set()
    from nltk.corpus import wordnet as wn
    for ss in wn.synsets("capital", pos="n"):
        for e in loader.get_edges(ss.name()):
            all_capital_atoms.add(e.get("concept_id"))
    
    results.check("SOC.official" in all_capital_atoms,
                   "SOC.official reachable via capital synsets",
                   f"reachable atoms: {all_capital_atoms}")
    results.check("SOC.city" in all_capital_atoms,
                   "SOC.city reachable via capital synsets")
    results.check("STA.wealth" in all_capital_atoms,
                   "STA.wealth still reachable (n.01/n.02 preserved)")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SimpleSynapseLoader Regression Test")
    parser.add_argument("--synapse", default="esde_synapses_v3.json",
                        help="Base Synapse JSON path")
    args = parser.parse_args()
    
    if not os.path.exists(args.synapse):
        print(f"ERROR: {args.synapse} not found")
        sys.exit(1)
    
    print(f"SimpleSynapseLoader Regression Test")
    print(f"Base Synapse: {args.synapse}")
    
    results = TestResults()
    
    test_add_edge(args.synapse, results)
    test_disable_edge(args.synapse, results)
    test_existing_edges_preserved(args.synapse, results)
    test_v34_integration(args.synapse, results)
    
    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
