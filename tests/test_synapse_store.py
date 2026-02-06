#!/usr/bin/env python3
"""
ESDE Synapse Store - Test Suite
================================

Covers GPT Audit Checklist (Phase 1 gate):
  1. SynapseStore.load(base, patches) returns deterministic results
  2. Tombstone: disabled edge_key never appears in get_edges()
  3. [OVERLAY_CONFLICT] logged on conflict
  4. Patch-free behavior = exact current behavior (zero regression)
  5. Relation Pipeline and Phase 8 use same SynapseStore (GO condition)

Usage:
    python -m pytest tests/test_synapse_store.py -v
    python tests/test_synapse_store.py
"""

import json
import sys
import os
import tempfile
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synapse.store import SynapseStore
from synapse.schema import SynapsePatchEntry


# ==========================================
# Test Fixtures
# ==========================================

def _make_base_synapse(tmp: str) -> str:
    """Create a minimal base synapse JSON for testing."""
    base = {
        "_meta": {
            "version": "3.0-test",
            "config": {"global_top_k": 3},
        },
        "synapses": {
            "love.v.01": [
                {"concept_id": "EMO.love", "raw_score": 0.92},
                {"concept_id": "REL.bond", "raw_score": 0.45},
            ],
            "kill.v.01": [
                {"concept_id": "ACT.destroy", "raw_score": 0.38},
            ],
            "run.v.01": [
                {"concept_id": "ACT.move", "raw_score": 0.70},
            ],
        },
    }
    path = os.path.join(tmp, "base_synapses.json")
    with open(path, "w") as f:
        json.dump(base, f)
    return path


def _make_patch_add(tmp: str, filename: str = "patch_v3.1.json") -> str:
    """Create a patch that adds a new edge."""
    patch = {
        "patches": [
            {
                "op": "add_edge",
                "edge_key": "kill.v.01::EXS.death",
                "synset_id": "kill.v.01",
                "atom": "EXS.death",
                "score": 0.85,
                "reason": "auto_proposal_v2.0",
                "metadata": {"source": "patch_v3.1"},
            },
        ]
    }
    path = os.path.join(tmp, filename)
    with open(path, "w") as f:
        json.dump(patch, f)
    return path


def _make_patch_disable(tmp: str, filename: str = "patch_disable.json") -> str:
    """Create a patch that disables an existing edge."""
    patch = {
        "patches": [
            {
                "op": "disable_edge",
                "edge_key": "love.v.01::REL.bond",
                "synset_id": "love.v.01",
                "atom": "REL.bond",
                "score": 0.0,
                "reason": "manual_review_rejected",
                "metadata": {},
            },
        ]
    }
    path = os.path.join(tmp, filename)
    with open(path, "w") as f:
        json.dump(patch, f)
    return path


def _make_patch_overwrite(tmp: str, filename: str = "patch_overwrite.json") -> str:
    """Create a patch that overwrites an existing edge's score."""
    patch = {
        "patches": [
            {
                "op": "add_edge",
                "edge_key": "kill.v.01::ACT.destroy",
                "synset_id": "kill.v.01",
                "atom": "ACT.destroy",
                "score": 0.72,
                "reason": "score_update_v3.1",
                "metadata": {"source": "patch_v3.1"},
            },
        ]
    }
    path = os.path.join(tmp, filename)
    with open(path, "w") as f:
        json.dump(patch, f)
    return path


def _make_patch_add_then_disable(tmp: str) -> tuple:
    """Create two patches: first adds, second disables the same edge."""
    p1 = {
        "patches": [
            {
                "op": "add_edge",
                "edge_key": "defeat.v.01::ACT.destroy",
                "synset_id": "defeat.v.01",
                "atom": "ACT.destroy",
                "score": 0.65,
                "reason": "auto_proposal",
                "metadata": {},
            },
        ]
    }
    p2 = {
        "patches": [
            {
                "op": "disable_edge",
                "edge_key": "defeat.v.01::ACT.destroy",
                "synset_id": "defeat.v.01",
                "atom": "ACT.destroy",
                "score": 0.0,
                "reason": "regression_detected",
                "metadata": {},
            },
        ]
    }
    path1 = os.path.join(tmp, "patch_add.json")
    path2 = os.path.join(tmp, "patch_disable_after.json")
    with open(path1, "w") as f:
        json.dump(p1, f)
    with open(path2, "w") as f:
        json.dump(p2, f)
    return path1, path2


def _make_patch_jsonl(tmp: str, filename: str = "patch.jsonl") -> str:
    """Create a JSONL format patch file."""
    entries = [
        {
            "op": "add_edge",
            "edge_key": "write.v.01::ACT.create",
            "synset_id": "write.v.01",
            "atom": "ACT.create",
            "score": 0.60,
            "reason": "auto_proposal",
            "metadata": {},
        },
        {
            "op": "add_edge",
            "edge_key": "host.v.01::SOC.official",
            "synset_id": "host.v.01",
            "atom": "SOC.official",
            "score": 0.55,
            "reason": "auto_proposal",
            "metadata": {},
        },
    ]
    path = os.path.join(tmp, filename)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


# ==========================================
# Audit Check 1: Determinism
# ==========================================

def test_determinism():
    """
    Audit Check 1: load(base, patches) returns deterministic results.
    
    Run load() twice with identical inputs → identical output.
    """
    print("\n[Test 1] Determinism...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        patch = _make_patch_add(tmp)
        
        store1 = SynapseStore()
        store1.load(base, patches=[patch])
        
        store2 = SynapseStore()
        store2.load(base, patches=[patch])
        
        # Compare resolved synapses
        assert store1.synapses == store2.synapses, "Synapses differ between runs"
        
        # Compare tombstones
        assert store1._tombstones == store2._tombstones, "Tombstones differ"
        
        # Compare specific edge
        edges1 = store1.get_edges("kill.v.01")
        edges2 = store2.get_edges("kill.v.01")
        assert edges1 == edges2, "get_edges() differs between runs"
        
        # Check the patch actually applied
        kill_atoms = [e["concept_id"] for e in edges1]
        assert "EXS.death" in kill_atoms, "Patched edge not found"
        
        print("  ✅ Determinism verified")
        return True


# ==========================================
# Audit Check 2: Tombstone
# ==========================================

def test_tombstone_basic():
    """
    Audit Check 2: disabled edge_key never appears in get_edges().
    """
    print("\n[Test 2a] Tombstone — basic disable...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        patch = _make_patch_disable(tmp)
        
        store = SynapseStore()
        store.load(base, patches=[patch])
        
        # love.v.01 should still exist but without REL.bond
        edges = store.get_edges("love.v.01")
        atoms = [e["concept_id"] for e in edges]
        assert "EMO.love" in atoms, "EMO.love should survive"
        assert "REL.bond" not in atoms, "REL.bond should be tombstoned"
        
        # Tombstone state check
        assert store.is_tombstoned("love.v.01::REL.bond"), "Should be tombstoned"
        assert not store.is_tombstoned("love.v.01::EMO.love"), "Should NOT be tombstoned"
        
        print("  ✅ Basic tombstone works")
        return True


def test_tombstone_add_then_disable():
    """
    Audit Check 2 (extended): add_edge then disable_edge → disabled wins.
    """
    print("\n[Test 2b] Tombstone — add then disable...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        p_add, p_disable = _make_patch_add_then_disable(tmp)
        
        store = SynapseStore()
        store.load(base, patches=[p_add, p_disable])
        
        # defeat.v.01 was added then disabled → should not exist
        edges = store.get_edges("defeat.v.01")
        assert len(edges) == 0, f"defeat.v.01 should be empty, got {edges}"
        assert not store.has_synset("defeat.v.01"), "defeat.v.01 should not appear"
        assert store.is_tombstoned("defeat.v.01::ACT.destroy"), "Should be tombstoned"
        
        print("  ✅ Add-then-disable: tombstone wins")
        return True


def test_tombstone_blocks_readd():
    """
    Audit Check 2 (critical): disable then add → disabled STILL wins.
    
    Design Spec v2.1 §2: "disable_edge が適用された edge_key は、
    最終的に「削除」状態となる"
    """
    print("\n[Test 2c] Tombstone — disable then re-add attempt...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        
        # Patch 1: disable love.v.01::REL.bond
        p_disable = _make_patch_disable(tmp, "p1_disable.json")
        
        # Patch 2: try to re-add the same edge
        p_readd_data = {
            "patches": [
                {
                    "op": "add_edge",
                    "edge_key": "love.v.01::REL.bond",
                    "synset_id": "love.v.01",
                    "atom": "REL.bond",
                    "score": 0.99,
                    "reason": "attempt_readd",
                    "metadata": {},
                },
            ]
        }
        p_readd = os.path.join(tmp, "p2_readd.json")
        with open(p_readd, "w") as f:
            json.dump(p_readd_data, f)
        
        store = SynapseStore()
        store.load(base, patches=[p_disable, p_readd])
        
        # REL.bond must still be absent — tombstone is permanent
        edges = store.get_edges("love.v.01")
        atoms = [e["concept_id"] for e in edges]
        assert "REL.bond" not in atoms, "Tombstoned edge must NOT be re-added"
        assert store.is_tombstoned("love.v.01::REL.bond"), "Tombstone must persist"
        
        print("  ✅ Tombstone blocks re-add (permanent)")
        return True


# ==========================================
# Audit Check 3: Conflict Logging
# ==========================================

def test_overlay_conflict_logging_v2():
    """
    Audit Check 3: [OVERLAY_CONFLICT] logged + overwrite takes effect.
    """
    print("\n[Test 3] Overlay conflict logging...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        patch = _make_patch_overwrite(tmp)
        
        # Capture log output
        synapse_logger = logging.getLogger("synapse.store")
        captured = []
        
        class CaptureHandler(logging.Handler):
            def emit(self, record):
                captured.append(record.getMessage())
        
        handler = CaptureHandler()
        handler.setLevel(logging.DEBUG)
        synapse_logger.addHandler(handler)
        old_level = synapse_logger.level
        synapse_logger.setLevel(logging.DEBUG)
        
        try:
            store = SynapseStore()
            store.load(base, patches=[patch])
            
            # Check conflict log
            conflict_msgs = [m for m in captured if "[OVERLAY_CONFLICT]" in m]
            assert len(conflict_msgs) > 0, (
                f"Expected [OVERLAY_CONFLICT] log. Captured: {captured}"
            )
            
            # Verify last-one-wins
            edges = store.get_edges("kill.v.01")
            destroy_edges = [e for e in edges if e["concept_id"] == "ACT.destroy"]
            assert len(destroy_edges) == 1
            assert destroy_edges[0]["raw_score"] == 0.72
            
            # Verify patch_source metadata is present
            assert destroy_edges[0].get("patch_source") == "score_update_v3.1"
            
            print(f"  Captured {len(conflict_msgs)} conflict message(s)")
            print("  ✅ [OVERLAY_CONFLICT] logged + overwrite applied")
        finally:
            synapse_logger.removeHandler(handler)
            synapse_logger.setLevel(old_level)
        
        return True


# ==========================================
# Audit Check 4: Zero Regression (no patches)
# ==========================================

def test_no_patch_backward_compat():
    """
    Audit Check 4: Patch-free behavior = exact current behavior.
    
    Load with no patches → identical to raw JSON parse.
    """
    print("\n[Test 4] Zero regression (no patches)...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        
        # Load via SynapseStore (no patches)
        store = SynapseStore()
        store.load(base)
        
        # Load raw JSON directly
        with open(base, "r") as f:
            raw = json.load(f)
        raw_synapses = raw["synapses"]
        
        # Compare
        assert store.synapses == raw_synapses, "No-patch load must equal raw JSON"
        assert len(store._tombstones) == 0, "No tombstones without patches"
        assert len(store._applied_patches) == 0, "No patches applied"
        
        # Check individual synsets
        for synset_id, expected_edges in raw_synapses.items():
            actual = store.get_edges(synset_id)
            assert actual == expected_edges, (
                f"Mismatch for {synset_id}: {actual} != {expected_edges}"
            )
        
        # Meta preserved
        assert store.get_meta_top_k() == 3, "Meta top_k should be 3"
        
        print("  ✅ No-patch behavior = raw JSON (zero regression)")
        return True


# ==========================================
# Audit Check 5: GO Condition (shared store)
# ==========================================

def test_go_condition_shared_store():
    """
    Audit Check 5 (GO Condition):
    Phase 8 Sensor and Obs C can use the SAME SynapseStore.
    
    Simulates both consumers reading from the same store instance
    and verifies they see identical data including patches.
    """
    print("\n[Test 5] GO Condition — shared store for Phase 8 + Obs C...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        patch = _make_patch_add(tmp)
        
        # Single store instance
        store = SynapseStore()
        store.load(base, patches=[patch])
        
        # --- Simulate Phase 8 Sensor access ---
        # SynapseLoader compatibility: get_edges(), has_synset(), get_file_hash()
        sensor_edges = store.get_edges("kill.v.01")
        sensor_has = store.has_synset("kill.v.01")
        sensor_hash = store.get_file_hash()
        
        assert sensor_has is True
        sensor_atoms = [e["concept_id"] for e in sensor_edges]
        assert "EXS.death" in sensor_atoms, "Sensor must see patched edge"
        assert "ACT.destroy" in sensor_atoms, "Sensor must see base edge"
        
        # --- Simulate Obs C (SynapseGrounder) access ---
        # SynapseGrounder uses: self.synapses dict directly
        grounder_synapses = store.get_synapse_dict()
        grounder_edges = grounder_synapses.get("kill.v.01", [])
        
        grounder_atoms = [e["concept_id"] for e in grounder_edges]
        assert "EXS.death" in grounder_atoms, "Grounder must see patched edge"
        assert "ACT.destroy" in grounder_atoms, "Grounder must see base edge"
        
        # --- Verify IDENTICAL view ---
        assert sensor_edges == grounder_edges, (
            "Phase 8 Sensor and Obs C MUST see identical edges"
        )
        
        # --- Singleton pattern ---
        SynapseStore.reset_instance()
        shared = SynapseStore.get_instance(base, patches=[patch])
        shared2 = SynapseStore.get_instance()  # Should return same instance
        assert shared is shared2, "Singleton must return same instance"
        assert shared.get_edges("kill.v.01") == sensor_edges, (
            "Singleton must have same data"
        )
        SynapseStore.reset_instance()
        
        print("  ✅ GO Condition verified: Phase 8 + Obs C share identical view")
        return True


# ==========================================
# Additional: JSONL format support
# ==========================================

def test_jsonl_patch_format():
    """Test JSONL patch file format."""
    print("\n[Test 6] JSONL patch format...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        patch = _make_patch_jsonl(tmp)
        
        store = SynapseStore()
        store.load(base, patches=[patch])
        
        # Check new edges
        assert store.has_synset("write.v.01"), "write.v.01 should be added"
        assert store.has_synset("host.v.01"), "host.v.01 should be added"
        
        write_edges = store.get_edges("write.v.01")
        assert len(write_edges) == 1
        assert write_edges[0]["concept_id"] == "ACT.create"
        assert write_edges[0]["raw_score"] == 0.60
        
        print("  ✅ JSONL format works")
        return True


# ==========================================
# Additional: Schema validation
# ==========================================

def test_schema_validation():
    """Test SynapsePatchEntry validation."""
    print("\n[Test 7] Schema validation...")
    
    # Valid entry
    entry = SynapsePatchEntry(
        op="add_edge",
        edge_key="kill.v.01::EXS.death",
        synset_id="kill.v.01",
        atom="EXS.death",
        score=0.85,
    )
    assert entry.edge_key == "kill.v.01::EXS.death"
    
    # Invalid op
    try:
        SynapsePatchEntry(
            op="delete",
            edge_key="x::y",
            synset_id="x",
            atom="y",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Mismatched edge_key
    try:
        SynapsePatchEntry(
            op="add_edge",
            edge_key="wrong_key",
            synset_id="kill.v.01",
            atom="EXS.death",
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # make_key
    assert SynapsePatchEntry.make_key("kill.v.01", "EXS.death") == "kill.v.01::EXS.death"
    
    # Round-trip serialization
    d = entry.to_dict()
    entry2 = SynapsePatchEntry.from_dict(d)
    assert entry2.op == entry.op
    assert entry2.edge_key == entry.edge_key
    assert entry2.score == entry.score
    
    # to_synapse_edge
    edge = entry.to_synapse_edge()
    assert edge["concept_id"] == "EXS.death"
    assert edge["raw_score"] == 0.85
    
    print("  ✅ Schema validation correct")
    return True


# ==========================================
# Additional: Audit info
# ==========================================

def test_audit_info():
    """Test audit trail completeness."""
    print("\n[Test 8] Audit trail...")
    
    with tempfile.TemporaryDirectory() as tmp:
        base = _make_base_synapse(tmp)
        p1 = _make_patch_add(tmp, "p1.json")
        p2 = _make_patch_disable(tmp, "p2.json")
        
        store = SynapseStore()
        store.load(base, patches=[p1, p2])
        
        info = store.get_audit_info()
        assert info["base_path"] is not None
        assert info["base_hash"].startswith("sha256:")
        assert len(info["patches_applied"]) == 2
        assert len(info["patch_hashes"]) == 2
        assert info["tombstone_count"] == 1
        assert info["patch_stats"]["edges_added"] == 1
        assert info["patch_stats"]["edges_disabled"] == 1
        
        print(f"  Audit: {info['synset_count']} synsets, "
              f"{info['total_edges']} edges, "
              f"{info['tombstone_count']} tombstones")
        print("  ✅ Audit trail complete")
        return True


# ==========================================
# Runner
# ==========================================

def run_all():
    """Run all tests."""
    print("=" * 60)
    print("ESDE SynapseStore Test Suite")
    print("Design Spec v2.1 — Phase 1 Audit Gate")
    print("=" * 60)
    
    results = []
    
    # Audit checklist
    results.append(("1. Determinism", test_determinism()))
    results.append(("2a. Tombstone basic", test_tombstone_basic()))
    results.append(("2b. Tombstone add→disable", test_tombstone_add_then_disable()))
    results.append(("2c. Tombstone blocks re-add", test_tombstone_blocks_readd()))
    results.append(("3. Conflict logging", test_overlay_conflict_logging_v2()))
    results.append(("4. Zero regression", test_no_patch_backward_compat()))
    results.append(("5. GO condition", test_go_condition_shared_store()))
    
    # Additional
    results.append(("6. JSONL format", test_jsonl_patch_format()))
    results.append(("7. Schema validation", test_schema_validation()))
    results.append(("8. Audit trail", test_audit_info()))
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    if failed > 0:
        print("\n  ⚠️  AUDIT GATE: FAIL — do not proceed to Phase 2")
        return False
    else:
        print("\n  ✅ AUDIT GATE: PASS — Phase 1 complete")
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    success = run_all()
    sys.exit(0 if success else 1)
