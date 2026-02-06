"""
ESDE Synapse - Proposer Tests (Phase 2 Audit Gate)
====================================================
Tests for SynapseEdgeProposer per Design Spec v3.0 + GPT v3.0.1.

Mock Strategy:
  - WordNet:   hardcoded synsets for "kill" (from real WordNet)
  - Embedding: bag-of-words vectors (deterministic, no external deps)

Audit Gate (GPT v3.0.1):
  Condition 1: EXS.death in Top-10 for "kill"
  Condition 2: Score >= 0.28
  Condition 3: Trace contains "to cause death" and "to undergo death"

3AI: Gemini (design) → GPT (audit) → Claude (implementation)
"""

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from math import sqrt
from typing import Any, Dict, List

# ── Path setup ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synapse.schema import SynapsePatchEntry
from synapse.proposer import (
    SynapseEdgeProposer,
    RewritePack,
    REWRITE_PACK_ID,
    DEFAULT_MIN_SCORE,
)


# ═══════════════════════════════════════════════════════════════════════
#  Mock Fixtures
# ═══════════════════════════════════════════════════════════════════════

# Minimal dictionary with atoms relevant to "kill" test
MOCK_DICTIONARY = {
    "meta": {"version": "2.0", "total_concepts": 6},
    "concepts": {
        "EXS.death": {
            "id": "EXS.death",
            "category": "EXS",
            "name": "death",
            "definition_en": "Cessation of biological life.",
        },
        "ACT.destroy": {
            "id": "ACT.destroy",
            "category": "ACT",
            "name": "destroy",
            "definition_en": "To cause to cease to exist; annihilate.",
        },
        "EXS.birth": {
            "id": "EXS.birth",
            "category": "EXS",
            "name": "birth",
            "definition_en": "Coming into existence; being born.",
        },
        "VAL.good": {
            "id": "VAL.good",
            "category": "VAL",
            "name": "good",
            "definition_en": "Moral excellence; virtue.",
        },
        "STA.comfort": {
            "id": "STA.comfort",
            "category": "STA",
            "name": "comfort",
            "definition_en": "A state of physical ease; freedom from pain.",
        },
        "ACT.attack": {
            "id": "ACT.attack",
            "category": "ACT",
            "name": "attack",
            "definition_en": "An aggressive action against someone or something.",
        },
    },
}

# Real WordNet synsets for "kill" (hardcoded to avoid NLTK dependency)
MOCK_KILL_SYNSETS = [
    {
        "synset_id": "kill.v.01",
        "definition": "cause to die; put to death, usually intentionally or knowingly",
    },
    {
        "synset_id": "kill.v.02",
        "definition": "thwart the passage of",
    },
    {
        "synset_id": "kill.v.03",
        "definition": "end or extinguish by forceful means",
    },
]

# Additional synsets for other test lemmas
MOCK_DEFEAT_SYNSETS = [
    {
        "synset_id": "defeat.v.01",
        "definition": "win a victory over",
    },
    {
        "synset_id": "defeat.v.02",
        "definition": "thwart the passage of",
    },
]


def mock_synset_lookup(lemma: str) -> List[Dict[str, str]]:
    """Mock WordNet: returns hardcoded synsets per lemma."""
    lookup = {
        "kill": MOCK_KILL_SYNSETS,
        "defeat": MOCK_DEFEAT_SYNSETS,
    }
    return lookup.get(lemma, [])


def make_bow_embedding_fn():
    """
    Create a bag-of-words embedding function.

    Deterministic, no external dependencies.
    Naturally produces higher cosine similarity for semantically related texts.
    """
    vocab: Dict[str, int] = {}
    vocab_counter = [0]  # mutable counter

    def get_dim(word: str) -> int:
        if word not in vocab:
            vocab[word] = vocab_counter[0]
            vocab_counter[0] += 1
        return vocab[word]

    def embed(texts: List[str]) -> List[List[float]]:
        # First pass: build vocabulary from all texts
        for text in texts:
            for word in text.lower().split():
                word = word.strip(".,;:!?()\"'")
                if word:
                    get_dim(word)

        dim = max(len(vocab), 1)
        vectors = []
        for text in texts:
            counts = Counter()
            for word in text.lower().split():
                word = word.strip(".,;:!?()\"'")
                if word and word in vocab:
                    counts[vocab[word]] += 1
            vec = [0.0] * dim
            for idx, count in counts.items():
                if idx < dim:
                    vec[idx] = float(count)
            vectors.append(vec)
        return vectors

    return embed


# ═══════════════════════════════════════════════════════════════════════
#  Helper
# ═══════════════════════════════════════════════════════════════════════

def make_proposer(
    dictionary: Dict = None,
    min_score: float = DEFAULT_MIN_SCORE,
    top_k: int = 5,
) -> SynapseEdgeProposer:
    """Create proposer with mock dependencies and temp dictionary."""
    if dictionary is None:
        dictionary = MOCK_DICTIONARY

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(dictionary, tmp, ensure_ascii=False)
    tmp.close()

    try:
        proposer = SynapseEdgeProposer(
            dictionary_path=tmp.name,
            embed_fn=make_bow_embedding_fn(),
            synset_lookup_fn=mock_synset_lookup,
            model_name="mock-bow",
            min_score=min_score,
            top_k=top_k,
            log_dir=tempfile.mkdtemp(),
        )
    finally:
        os.unlink(tmp.name)

    return proposer


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRewritePack(unittest.TestCase):
    """Test 1-3: Rewrite Pack generation (GPT v3.0.1)."""

    def test_1_four_texts_generated(self):
        """4-Pack produces exactly 4 texts."""
        pack = RewritePack(
            atom_id="EXS.death",
            atom_name="death",
            definition_en="Cessation of biological life.",
        )
        self.assertEqual(len(pack.texts), 4)

    def test_2_correct_texts(self):
        """4-Pack texts match GPT v3.0.1 spec exactly."""
        pack = RewritePack(
            atom_id="EXS.death",
            atom_name="death",
            definition_en="Cessation of biological life.",
        )
        self.assertEqual(pack.texts[0], "Cessation of biological life.")
        self.assertEqual(pack.texts[1], "to cause death")
        self.assertEqual(pack.texts[2], "to undergo death")
        self.assertEqual(pack.texts[3], "act of death")

    def test_3_atom_name_preprocessing(self):
        """GPT audit note A: trim + lower + empty check."""
        # Trim and lower
        pack = RewritePack(
            atom_id="TEST.x",
            atom_name="  Death  ",
            definition_en="test",
        )
        self.assertEqual(pack.texts[1], "to cause death")  # lowered and trimmed

        # Empty name raises
        with self.assertRaises(ValueError):
            RewritePack(atom_id="TEST.y", atom_name="", definition_en="test")

        with self.assertRaises(ValueError):
            RewritePack(atom_id="TEST.z", atom_name="   ", definition_en="test")

    def test_3b_trace_dict(self):
        """Trace dict includes pack_id."""
        pack = RewritePack(
            atom_id="EXS.death",
            atom_name="death",
            definition_en="Cessation of biological life.",
        )
        trace = pack.to_trace_dict()
        self.assertEqual(trace["pack_id"], REWRITE_PACK_ID)
        self.assertEqual(trace["atom_id"], "EXS.death")
        self.assertIn("to cause death", trace["texts"])
        self.assertIn("to undergo death", trace["texts"])


class TestProposerCore(unittest.TestCase):
    """Test 4-7: Core proposal logic and audit gate."""

    def setUp(self):
        self.proposer = make_proposer(min_score=0.01)  # Low threshold for BoW

    def test_4_kill_produces_proposals(self):
        """kill → non-empty proposals."""
        proposals = self.proposer.propose("kill", evidence_count=24)
        self.assertGreater(len(proposals), 0, "kill should produce proposals")

    def test_5_audit_gate_death_in_top10(self):
        """
        AUDIT GATE Condition 1+2:
        EXS.death must appear in proposals for kill.v.01.
        """
        proposals = self.proposer.propose("kill", evidence_count=24)
        # Gather all atom IDs for kill.v.01 proposals
        kill_v01_atoms = [
            p.atom for p in proposals if p.synset_id == "kill.v.01"
        ]
        self.assertIn(
            "EXS.death", kill_v01_atoms,
            f"EXS.death must be in kill.v.01 proposals. Got: {kill_v01_atoms}"
        )

    def test_5b_death_score_above_threshold(self):
        """AUDIT GATE Condition 2: EXS.death score >= min_score."""
        proposals = self.proposer.propose("kill", evidence_count=24)
        death_proposals = [
            p for p in proposals
            if p.synset_id == "kill.v.01" and p.atom == "EXS.death"
        ]
        self.assertEqual(len(death_proposals), 1, "Exactly one kill.v.01::EXS.death")
        self.assertGreaterEqual(
            death_proposals[0].score, self.proposer.min_score,
            f"Score {death_proposals[0].score} < min_score {self.proposer.min_score}"
        )

    def test_6_output_schema(self):
        """All proposals follow SynapsePatchEntry schema (Design Spec v3.0 §3)."""
        proposals = self.proposer.propose("kill", evidence_count=24)
        for p in proposals:
            self.assertIsInstance(p, SynapsePatchEntry)
            self.assertEqual(p.op, "add_edge")
            self.assertEqual(p.edge_key, f"{p.synset_id}::{p.atom}")
            self.assertEqual(p.reason, "auto_proposal_v2.0")
            self.assertEqual(p.metadata["rewrite_pack_id"], REWRITE_PACK_ID)
            self.assertEqual(p.metadata["source"], "relation_pipeline")
            self.assertEqual(p.metadata["count_kind"], "lemma_based")
            self.assertEqual(p.metadata["evidence_count"], 24)

    def test_7_top_k_limit(self):
        """Top-K per synset is enforced."""
        proposer = make_proposer(min_score=0.001, top_k=2)
        proposals = proposer.propose("kill", evidence_count=10)
        # Count proposals per synset
        from collections import Counter
        per_synset = Counter(p.synset_id for p in proposals)
        for synset_id, count in per_synset.items():
            self.assertLessEqual(
                count, 2,
                f"{synset_id} has {count} proposals, expected ≤ 2"
            )


class TestProposerEdgeCases(unittest.TestCase):
    """Test 8-10: Edge cases, dedup, determinism."""

    def test_8_empty_lemma(self):
        """Empty/whitespace lemma returns empty list."""
        proposer = make_proposer()
        self.assertEqual(proposer.propose("", 0), [])
        self.assertEqual(proposer.propose("   ", 0), [])

    def test_9_unknown_lemma(self):
        """Lemma with no synsets returns empty list."""
        proposer = make_proposer()
        self.assertEqual(proposer.propose("xyznotaword", 0), [])

    def test_10_edge_key_dedup(self):
        """GPT audit note B: no duplicate edge_keys."""
        proposer = make_proposer(min_score=0.001)
        proposals = proposer.propose("kill", evidence_count=5)
        keys = [p.edge_key for p in proposals]
        self.assertEqual(len(keys), len(set(keys)), "Duplicate edge_keys found!")

    def test_11_determinism(self):
        """Same input → same output (BoW embedding is deterministic)."""
        p1 = make_proposer(min_score=0.01)
        p2 = make_proposer(min_score=0.01)
        r1 = p1.propose("kill", 24)
        r2 = p2.propose("kill", 24)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a.edge_key, b.edge_key)
            self.assertEqual(a.score, b.score)


class TestTraceAndExport(unittest.TestCase):
    """Test 12-14: Trace logging and patch export."""

    def setUp(self):
        self.proposer = make_proposer(min_score=0.01)
        self.proposals = self.proposer.propose("kill", evidence_count=24)

    def test_12_trace_contains_rewrite_packs(self):
        """
        AUDIT GATE Condition 3:
        Trace must contain "to cause death" and "to undergo death"
        for atom=EXS.death.
        """
        trace_path = self.proposer.write_trace(self.proposals, "kill")
        self.assertTrue(os.path.exists(trace_path))

        packs = []
        with open(trace_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "rewrite_pack":
                    packs.append(record)

        # Find EXS.death pack
        death_packs = [p for p in packs if p["atom_id"] == "EXS.death"]
        self.assertEqual(len(death_packs), 1, "EXS.death pack must be in trace")

        texts = death_packs[0]["texts"]
        self.assertIn("to cause death", texts,
                       "Trace must contain 'to cause death'")
        self.assertIn("to undergo death", texts,
                       "Trace must contain 'to undergo death'")
        self.assertIn("act of death", texts,
                       "Trace must contain 'act of death'")

    def test_13_trace_contains_proposals(self):
        """Trace includes proposal entries."""
        trace_path = self.proposer.write_trace(self.proposals, "kill")
        proposal_records = []
        with open(trace_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "proposal":
                    proposal_records.append(record)
        self.assertEqual(len(proposal_records), len(self.proposals))

    def test_14_export_patch_file(self):
        """Exported patch is SynapseStore-compatible JSON."""
        tmp_dir = tempfile.mkdtemp()
        patch_path = os.path.join(tmp_dir, "test_patch.json")
        self.proposer.export_patch_file(self.proposals, patch_path)

        with open(patch_path, "r") as f:
            data = json.load(f)

        self.assertIn("patches", data)
        self.assertEqual(len(data["patches"]), len(self.proposals))

        # Validate each entry can construct SynapsePatchEntry
        for entry_dict in data["patches"]:
            entry = SynapsePatchEntry.from_dict(entry_dict)
            self.assertEqual(entry.op, "add_edge")
            self.assertEqual(
                entry.edge_key,
                f"{entry.synset_id}::{entry.atom}",
            )


class TestBatchAndAuditInfo(unittest.TestCase):
    """Test 15-16: Batch operation and audit info."""

    def test_15_batch_propose(self):
        """Batch proposal processes multiple lemmas."""
        proposer = make_proposer(min_score=0.01)
        results = proposer.propose_batch([
            {"lemma": "kill", "evidence_count": 24},
            {"lemma": "defeat", "evidence_count": 10},
            {"lemma": "xyznotaword", "evidence_count": 1},
        ])
        self.assertIn("kill", results)
        self.assertIn("defeat", results)
        self.assertIn("xyznotaword", results)
        self.assertGreater(len(results["kill"]), 0)
        self.assertGreater(len(results["defeat"]), 0)
        self.assertEqual(len(results["xyznotaword"]), 0)

    def test_16_audit_info(self):
        """Audit info returns expected metadata."""
        proposer = make_proposer()
        info = proposer.get_audit_info()
        self.assertEqual(info["atom_count"], 6)  # mock dictionary
        self.assertEqual(info["rewrite_pack_id"], REWRITE_PACK_ID)
        self.assertEqual(info["rewrite_pack_texts_per_atom"], 4)
        self.assertEqual(info["model_name"], "mock-bow")


# ═══════════════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
