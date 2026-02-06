"""
ESDE Synapse - Phase 3 CLI Tests
==================================
Tests the propose-synapse → evaluate-synapse-patch cycle.

Mock Strategy:
  - Pipeline runner: returns synthetic diagnostic reports
  - Proposer: injected with mock deps (from Phase 2 tests)

Test Coverage:
  1-3:  DiagnosticResult accessors and diff logic
  4-6:  Audit gate: PASS / WARN / FAIL conditions
  7-9:  propose-synapse command
  10-12: evaluate-synapse-patch command
  13:   Run ID collision resistance
  14:   GPT §4: no writes outside run-dir

3AI: Gemini (design) → GPT (audit) → Claude (implementation)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.diagnostic import DiagnosticResult
from synapse.cli import (
    cmd_propose_synapse,
    cmd_evaluate_synapse_patch,
    generate_run_id,
    render_diff_report,
)
from synapse.schema import SynapsePatchEntry
from synapse.store import SynapseStore


# ═══════════════════════════════════════════════════════════════════════
#  Mock Fixtures
# ═══════════════════════════════════════════════════════════════════════

def make_diagnostic_report(
    grounding_rate: float = 0.55,
    grounded: int = 100,
    ungrounded: int = 80,
    coverage_gaps: List[Dict] = None,
    consistent_misgrounds: List[Dict] = None,
    category_mismatches: List[Dict] = None,
) -> Dict[str, Any]:
    """Create a synthetic diagnostic report."""
    symptoms = []

    if coverage_gaps is not None:
        symptoms.extend(coverage_gaps)
    else:
        # Default gaps
        symptoms.extend([
            {
                "type": "SYNAPSE_COVERAGE_GAP",
                "severity": "HIGH",
                "description": "Verb 'kill' ungrounded 24x",
                "verb": "kill",
                "count": 24,
            },
            {
                "type": "SYNAPSE_COVERAGE_GAP",
                "severity": "HIGH",
                "description": "Verb 'defeat' ungrounded 10x",
                "verb": "defeat",
                "count": 10,
            },
        ])

    if consistent_misgrounds:
        symptoms.extend(consistent_misgrounds)

    if category_mismatches:
        symptoms.extend(category_mismatches)

    return {
        "meta": {
            "dataset": "test",
            "articles": 5,
            "generated_at": "2026-02-06T12:00:00Z",
            "version": "0.2.0",
        },
        "summary": {
            "total_triples": grounded + ungrounded,
            "grounded": grounded,
            "ungrounded": ungrounded,
            "lightverb": 10,
            "grounding_rate": round(grounding_rate, 4),
        },
        "symptoms": symptoms,
        "top_verb_atom_mappings": [],
        "top_ungrounded_verbs": [("kill", 24), ("defeat", 10)],
        "suspicious_groundings_sample": [],
        "per_article": [],
    }


def make_mock_pipeline_runner(report_factory):
    """
    Create a mock pipeline runner that returns synthetic reports.

    report_factory: callable(dataset, store, min_score, output_dir) -> raw dict
    """
    def runner(dataset, store, min_score, output_dir):
        return report_factory(dataset, store, min_score, output_dir)
    return runner


# Minimal mock for SynapseEdgeProposer (uses Phase 2 mock pattern)
def make_mock_proposer(tmp_dir: str):
    """Create a proposer with mock deps for testing."""
    from tests.test_synapse_proposer import (
        make_bow_embedding_fn,
        mock_synset_lookup,
        MOCK_DICTIONARY,
    )

    dict_path = os.path.join(tmp_dir, "dict.json")
    with open(dict_path, "w") as f:
        json.dump(MOCK_DICTIONARY, f)

    from synapse.proposer import SynapseEdgeProposer
    return SynapseEdgeProposer(
        dictionary_path=dict_path,
        embed_fn=make_bow_embedding_fn(),
        synset_lookup_fn=mock_synset_lookup,
        model_name="mock-bow",
        min_score=0.01,
        top_k=5,
        log_dir=os.path.join(tmp_dir, "traces"),
    )


# Minimal Synapse base JSON for SynapseStore
MOCK_SYNAPSE_JSON = {
    "synapses": {
        "love.v.01": [
            {"concept_id": "EMO.love", "raw_score": 0.9}
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════
#  Test 1-3: DiagnosticResult
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnosticResult(unittest.TestCase):

    def test_1_accessors(self):
        """DiagnosticResult provides typed access to report fields."""
        raw = make_diagnostic_report(grounding_rate=0.55, grounded=100, ungrounded=80)
        dr = DiagnosticResult(raw=raw)

        self.assertAlmostEqual(dr.grounding_rate, 0.55, places=2)
        self.assertEqual(dr.grounded, 100)
        self.assertEqual(dr.ungrounded, 80)
        self.assertEqual(len(dr.coverage_gaps), 2)
        self.assertEqual(dr.coverage_gaps[0]["verb"], "kill")

    def test_2_get_gap_verbs(self):
        """get_gap_verbs filters by min_freq."""
        raw = make_diagnostic_report()
        dr = DiagnosticResult(raw=raw)

        gaps = dr.get_gap_verbs(min_freq=5)
        verbs = [v for v, c in gaps]
        self.assertIn("kill", verbs)      # count=24 ≥ 5
        self.assertIn("defeat", verbs)    # count=10 ≥ 5

        gaps_high = dr.get_gap_verbs(min_freq=20)
        verbs_high = [v for v, c in gaps_high]
        self.assertIn("kill", verbs_high)      # 24 ≥ 20
        self.assertNotIn("defeat", verbs_high)  # 10 < 20

    def test_3_env_meta(self):
        """Environment metadata is injected and accessible (GPT §2)."""
        raw = make_diagnostic_report()
        dr = DiagnosticResult.with_env_meta(
            raw=raw,
            synapse_base_path="esde_synapses_v3.json",
            patches_loaded=["patch_v3.1.json"],
            dictionary_version="2.0",
            min_score=0.45,
            dataset="mixed",
            run_id="run_test_001",
            code_version="abc1234",
        )
        self.assertEqual(dr.env_meta["synapse_base_path"], "esde_synapses_v3.json")
        self.assertEqual(dr.env_meta["code_version"], "abc1234")
        self.assertIn("patch_v3.1.json", dr.env_meta["patches_loaded"])


# ═══════════════════════════════════════════════════════════════════════
#  Test 4-6: Diff & Audit Gate
# ═══════════════════════════════════════════════════════════════════════

class TestDiagnosticDiff(unittest.TestCase):

    def test_4_pass_condition(self):
        """PASS: gaps resolved, no regressions → exit 0."""
        before = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.55,
            grounded=100,
            ungrounded=80,
        ))
        # After: kill gap resolved, rate improved
        after = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.65,
            grounded=120,
            ungrounded=60,
            coverage_gaps=[
                # Only defeat remains
                {"type": "SYNAPSE_COVERAGE_GAP", "severity": "HIGH",
                 "verb": "defeat", "count": 10,
                 "description": "Verb 'defeat' ungrounded 10x"},
            ],
        ))

        diff = DiagnosticResult.diff(before, after)
        self.assertEqual(diff["verdict"], "PASS")
        self.assertEqual(diff["exit_code"], 0)
        self.assertIn("kill", diff["coverage_gaps"]["resolved"])
        self.assertGreater(diff["metrics"]["grounding_rate_delta"], 0)

    def test_5_warn_condition(self):
        """WARN: no improvement → exit 1."""
        before = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.55,
            grounded=100,
            ungrounded=80,
        ))
        # After: identical (no change)
        after = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.55,
            grounded=100,
            ungrounded=80,
        ))

        diff = DiagnosticResult.diff(before, after)
        self.assertEqual(diff["verdict"], "WARN")
        self.assertEqual(diff["exit_code"], 1)

    def test_6_fail_category_mismatch(self):
        """FAIL: category mismatch detected → exit 2."""
        before = DiagnosticResult(raw=make_diagnostic_report())
        after = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.60,
            grounded=110,
            ungrounded=70,
            category_mismatches=[
                {"type": "CATEGORY_MISMATCH", "severity": "HIGH",
                 "verb": "host", "atoms": ["NAT.ocean"],
                 "description": "Verb 'host' grounded to noun-category NAT.ocean",
                 "count": 3},
            ],
        ))

        diff = DiagnosticResult.diff(before, after)
        self.assertEqual(diff["verdict"], "FAIL")
        self.assertEqual(diff["exit_code"], 2)
        self.assertIn("CATEGORY_MISMATCH > 0", diff["fail_reasons"])

    def test_6b_fail_new_misground(self):
        """FAIL: new consistent misground → exit 2."""
        before = DiagnosticResult(raw=make_diagnostic_report())
        after = DiagnosticResult(raw=make_diagnostic_report(
            grounding_rate=0.60,
            grounded=110,
            ungrounded=70,
            consistent_misgrounds=[
                {"type": "CONSISTENT_MISGROUND", "severity": "HIGH",
                 "verb": "serve", "atoms": [],
                 "description": "'serve → ACT.destroy' appears 8x with low score",
                 "examples": [], "score_range": [0.1, 0.3]},
            ],
        ))

        diff = DiagnosticResult.diff(before, after)
        self.assertEqual(diff["verdict"], "FAIL")
        self.assertEqual(diff["exit_code"], 2)
        self.assertTrue(any("CONSISTENT_MISGROUND" in r for r in diff["fail_reasons"]))


# ═══════════════════════════════════════════════════════════════════════
#  Test 7-9: propose-synapse command
# ═══════════════════════════════════════════════════════════════════════

class TestProposeCommand(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Create mock synapse file
        self.synapse_path = os.path.join(self.tmp_dir, "synapse.json")
        with open(self.synapse_path, "w") as f:
            json.dump(MOCK_SYNAPSE_JSON, f)
        # Create mock dictionary
        from tests.test_synapse_proposer import MOCK_DICTIONARY
        self.dict_path = os.path.join(self.tmp_dir, "dict.json")
        with open(self.dict_path, "w") as f:
            json.dump(MOCK_DICTIONARY, f)

    def test_7_creates_run_directory(self):
        """propose-synapse creates a run directory with all expected files."""
        mock_runner = make_mock_pipeline_runner(
            lambda ds, st, ms, od: make_diagnostic_report()
        )
        proposer = make_mock_proposer(self.tmp_dir)

        exit_code, run_dir = cmd_propose_synapse(
            dataset="test",
            synapse_path=self.synapse_path,
            dictionary_path=self.dict_path,
            output_base=os.path.join(self.tmp_dir, "proposals"),
            pipeline_runner=mock_runner,
            proposer=proposer,
        )

        self.assertEqual(exit_code, 0)
        run_path = Path(run_dir)
        self.assertTrue(run_path.exists())
        self.assertTrue((run_path / "diagnostic_before.json").exists())
        self.assertTrue((run_path / "patch_candidate.json").exists())
        self.assertTrue((run_path / "proposal_report.md").exists())

    def test_8_baseline_has_env_meta(self):
        """Baseline diagnostic includes environment metadata (GPT §2)."""
        mock_runner = make_mock_pipeline_runner(
            lambda ds, st, ms, od: make_diagnostic_report()
        )
        proposer = make_mock_proposer(self.tmp_dir)

        _, run_dir = cmd_propose_synapse(
            dataset="test",
            synapse_path=self.synapse_path,
            dictionary_path=self.dict_path,
            output_base=os.path.join(self.tmp_dir, "proposals"),
            pipeline_runner=mock_runner,
            proposer=proposer,
        )

        with open(Path(run_dir) / "diagnostic_before.json") as f:
            before = json.load(f)

        self.assertIn("env_meta", before)
        self.assertEqual(before["env_meta"]["dataset"], "test")
        self.assertEqual(before["env_meta"]["patches_loaded"], [])

    def test_9_no_gaps_produces_empty_report(self):
        """When no coverage gaps exist, proposal report says so."""
        no_gap_report = make_diagnostic_report(
            coverage_gaps=[],  # No gaps
        )
        mock_runner = make_mock_pipeline_runner(
            lambda ds, st, ms, od: no_gap_report
        )

        _, run_dir = cmd_propose_synapse(
            dataset="test",
            synapse_path=self.synapse_path,
            dictionary_path=self.dict_path,
            output_base=os.path.join(self.tmp_dir, "proposals"),
            pipeline_runner=mock_runner,
        )

        report_path = Path(run_dir) / "proposal_report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text()
        self.assertIn("No coverage gaps", content)


# ═══════════════════════════════════════════════════════════════════════
#  Test 10-12: evaluate-synapse-patch command
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluateCommand(unittest.TestCase):

    def _setup_run_dir(self, before_report, after_factory, patch_entries=None):
        """
        Setup a run directory with baseline + patch, then return eval result.
        """
        tmp = tempfile.mkdtemp()
        run_dir = os.path.join(tmp, "run_test_eval")
        os.makedirs(run_dir)

        # Write baseline
        with open(os.path.join(run_dir, "diagnostic_before.json"), "w") as f:
            json.dump(before_report, f)

        # Write patch candidate
        entries = patch_entries or [
            SynapsePatchEntry(
                op="add_edge",
                edge_key="kill.v.01::EXS.death",
                synset_id="kill.v.01",
                atom="EXS.death",
                score=0.72,
                reason="test",
            ).to_dict()
        ]
        with open(os.path.join(run_dir, "patch_candidate.json"), "w") as f:
            json.dump({"patches": entries}, f)

        # Synapse base file
        synapse_path = os.path.join(tmp, "synapse.json")
        with open(synapse_path, "w") as f:
            json.dump(MOCK_SYNAPSE_JSON, f)

        # Dictionary
        dict_path = os.path.join(tmp, "dict.json")
        with open(dict_path, "w") as f:
            json.dump({"meta": {"version": "2.0"}, "concepts": {}}, f)

        mock_runner = make_mock_pipeline_runner(after_factory)

        exit_code = cmd_evaluate_synapse_patch(
            run_dir=run_dir,
            synapse_path=synapse_path,
            dictionary_path=dict_path,
            dataset="test",
            pipeline_runner=mock_runner,
        )

        return exit_code, run_dir

    def test_10_evaluate_pass(self):
        """Evaluate returns exit 0 when improvement found, no regression."""
        before = make_diagnostic_report(grounding_rate=0.55, grounded=100, ungrounded=80)
        after_report = make_diagnostic_report(
            grounding_rate=0.65, grounded=120, ungrounded=60,
            coverage_gaps=[
                {"type": "SYNAPSE_COVERAGE_GAP", "severity": "HIGH",
                 "verb": "defeat", "count": 10,
                 "description": "Verb 'defeat' ungrounded 10x"},
            ],
        )

        exit_code, run_dir = self._setup_run_dir(
            before, lambda ds, st, ms, od: after_report
        )

        self.assertEqual(exit_code, 0)  # PASS
        # Check outputs exist
        self.assertTrue(Path(run_dir, "diagnostic_after.json").exists())
        self.assertTrue(Path(run_dir, "diagnostic_diff.json").exists())
        self.assertTrue(Path(run_dir, "diagnostic_diff.md").exists())

    def test_11_evaluate_warn(self):
        """Evaluate returns exit 1 when no improvement."""
        before = make_diagnostic_report(grounding_rate=0.55)
        after_report = make_diagnostic_report(grounding_rate=0.55)  # No change

        exit_code, _ = self._setup_run_dir(
            before, lambda ds, st, ms, od: after_report
        )
        self.assertEqual(exit_code, 1)  # WARN

    def test_12_evaluate_fail(self):
        """Evaluate returns exit 2 when regression detected."""
        before = make_diagnostic_report(grounding_rate=0.55)
        after_report = make_diagnostic_report(
            grounding_rate=0.60,
            category_mismatches=[
                {"type": "CATEGORY_MISMATCH", "severity": "HIGH",
                 "verb": "host", "atoms": ["NAT.ocean"],
                 "description": "regression", "count": 3},
            ],
        )

        exit_code, _ = self._setup_run_dir(
            before, lambda ds, st, ms, od: after_report
        )
        self.assertEqual(exit_code, 2)  # FAIL


# ═══════════════════════════════════════════════════════════════════════
#  Test 13-14: Run ID & Safety
# ═══════════════════════════════════════════════════════════════════════

class TestRunIdAndSafety(unittest.TestCase):

    def test_13_run_id_collision_resistance(self):
        """GPT §1: consecutive run IDs don't collide."""
        ids = set()
        for _ in range(100):
            rid = generate_run_id("test")
            self.assertNotIn(rid, ids, f"Collision: {rid}")
            ids.add(rid)

    def test_14_no_writes_outside_run_dir(self):
        """
        GPT §4: evaluate never writes to patches/ or anywhere outside run-dir.
        """
        tmp = tempfile.mkdtemp()
        patches_dir = os.path.join(tmp, "patches")
        os.makedirs(patches_dir)

        # Record file state before
        before_files = set(os.listdir(patches_dir))

        run_dir = os.path.join(tmp, "run_test_safety")
        os.makedirs(run_dir)

        # Write baseline
        before_report = make_diagnostic_report()
        with open(os.path.join(run_dir, "diagnostic_before.json"), "w") as f:
            json.dump(before_report, f)

        # Write patch
        with open(os.path.join(run_dir, "patch_candidate.json"), "w") as f:
            json.dump({"patches": []}, f)

        # Synapse + dict
        synapse_path = os.path.join(tmp, "synapse.json")
        with open(synapse_path, "w") as f:
            json.dump(MOCK_SYNAPSE_JSON, f)
        dict_path = os.path.join(tmp, "dict.json")
        with open(dict_path, "w") as f:
            json.dump({"meta": {"version": "2.0"}, "concepts": {}}, f)

        after_report = make_diagnostic_report(grounding_rate=0.60,
            coverage_gaps=[{"type": "SYNAPSE_COVERAGE_GAP", "severity": "HIGH",
                           "verb": "defeat", "count": 10,
                           "description": "defeat"}])
        mock_runner = make_mock_pipeline_runner(
            lambda ds, st, ms, od: after_report
        )

        cmd_evaluate_synapse_patch(
            run_dir=run_dir,
            synapse_path=synapse_path,
            dictionary_path=dict_path,
            dataset="test",
            pipeline_runner=mock_runner,
        )

        # Verify patches/ was not touched
        after_files = set(os.listdir(patches_dir))
        self.assertEqual(before_files, after_files,
                         "patches/ directory was modified!")


# ═══════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
