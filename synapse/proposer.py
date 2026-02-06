"""
ESDE Synapse - Edge Proposer (Phase 2)
=======================================
Generates candidate Synapse edges by comparing verb synsets
against ESDE Atom definitions using embedding similarity.

Design Spec: v3.0 (Gemini) + v3.0.1 amendments (GPT audit)
Input: lemma (str) + evidence_count (int) — synsets expanded internally
Output: List[SynapsePatchEntry] per synset, filtered by score & top-k

Rewrite Pack (4-Pack Strategy, GPT v3.0.1):
  1. definition_en (original)
  2. "to cause " + atom_name
  3. "to undergo " + atom_name
  4. "act of " + atom_name

3AI: Gemini (design) → GPT (audit) → Claude (implementation)
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from .schema import SynapsePatchEntry

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────

DEFAULT_MIN_SCORE: float = 0.28
DEFAULT_TOP_K: int = 5
REWRITE_PACK_ID: str = "verb_aug_v1"
DEFAULT_MODEL: str = "all-MiniLM-L6-v2"


# ─── Protocols (Dependency Injection Boundaries) ─────────────────────

class EmbeddingFn(Protocol):
    """Callable: list of texts → list of vectors."""
    def __call__(self, texts: List[str]) -> List[List[float]]: ...


class SynsetLookupFn(Protocol):
    """Callable: lemma → list of verb synset dicts."""
    def __call__(self, lemma: str) -> List[Dict[str, str]]: ...
    # Each dict: {"synset_id": "kill.v.01", "definition": "cause to die; ..."}


# ─── Default Implementations ────────────────────────────────────────

def default_synset_lookup(lemma: str) -> List[Dict[str, str]]:
    """Default WordNet synset lookup using NLTK."""
    from nltk.corpus import wordnet
    results = []
    for ss in wordnet.synsets(lemma, pos=wordnet.VERB):
        results.append({
            "synset_id": ss.name(),
            "definition": ss.definition(),
        })
    return results


def default_embedding_fn(model_name: str = DEFAULT_MODEL) -> EmbeddingFn:
    """Create default embedding function using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    def encode(texts: List[str]) -> List[List[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return encode


# ─── Rewrite Pack ────────────────────────────────────────────────────

@dataclass
class RewritePack:
    """
    4-Pack rewrite for a single Atom (GPT v3.0.1 amended).

    Texts:
      1. definition_en (original)
      2. "to cause " + atom_name
      3. "to undergo " + atom_name
      4. "act of " + atom_name
    """
    atom_id: str
    atom_name: str
    definition_en: str
    texts: List[str] = field(default_factory=list)

    def __post_init__(self):
        # GPT audit note A: atom_name preprocessing
        name = self.atom_name.strip().lower()
        if not name:
            raise ValueError(f"Empty atom_name for {self.atom_id}")
        self.texts = [
            self.definition_en,
            f"to cause {name}",
            f"to undergo {name}",
            f"act of {name}",
        ]

    def to_trace_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "atom_name": self.atom_name,
            "definition_en": self.definition_en,
            "texts": self.texts,
            "pack_id": REWRITE_PACK_ID,
        }


# ─── Internal Candidate ─────────────────────────────────────────────

@dataclass
class _ProposalCandidate:
    """Internal scoring result before filtering."""
    synset_id: str
    atom_id: str
    score: float
    evidence_count: int


# ─── Main Proposer ───────────────────────────────────────────────────

class SynapseEdgeProposer:
    """
    Phase 2: Generate candidate Synapse edges from ungrounded verb lemmas.

    Design Spec v3.0 §1-4 + GPT v3.0.1 amendments.

    Dependency Injection:
      embed_fn:          text → vectors  (default: sentence-transformers)
      synset_lookup_fn:  lemma → synsets (default: NLTK WordNet)
    """

    def __init__(
        self,
        dictionary_path: str,
        embed_fn: Optional[EmbeddingFn] = None,
        synset_lookup_fn: Optional[SynsetLookupFn] = None,
        model_name: str = DEFAULT_MODEL,
        min_score: float = DEFAULT_MIN_SCORE,
        top_k: int = DEFAULT_TOP_K,
        log_dir: str = "logs/synapse_proposal",
    ):
        # ── Load dictionary: concepts[*].name + definition_en ONLY ──
        with open(dictionary_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.atoms: Dict[str, Dict[str, str]] = {}
        for atom_id, concept in raw.get("concepts", {}).items():
            self.atoms[atom_id] = {
                "name": concept["name"],
                "definition_en": concept["definition_en"],
            }
        logger.info(f"[PROPOSER] Loaded {len(self.atoms)} atoms from {dictionary_path}")

        # ── Dependencies ──
        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            self._embed_fn = default_embedding_fn(model_name)

        if synset_lookup_fn is not None:
            self._synset_lookup_fn = synset_lookup_fn
        else:
            self._synset_lookup_fn = default_synset_lookup

        # ── Config ──
        self.model_name = model_name
        self.min_score = min_score
        self.top_k = top_k
        self.log_dir = log_dir

        # ── Pre-compute atom rewrite packs ──
        self._atom_ids: List[str] = sorted(self.atoms.keys())
        self._rewrite_packs: Dict[str, RewritePack] = {}
        for atom_id in self._atom_ids:
            info = self.atoms[atom_id]
            self._rewrite_packs[atom_id] = RewritePack(
                atom_id=atom_id,
                atom_name=info["name"],
                definition_en=info["definition_en"],
            )

        # Lazy embedding computation
        self._atom_embeddings: Optional[Dict[str, List[List[float]]]] = None
        self._embeddings_computed = False

    # ── Embedding Lifecycle ──────────────────────────────────────────

    def _ensure_embeddings(self) -> None:
        """Compute atom embeddings lazily (once)."""
        if self._embeddings_computed:
            return

        logger.info(
            f"[PROPOSER] Computing embeddings: "
            f"{len(self._atom_ids)} atoms × 4 texts = {len(self._atom_ids) * 4} vectors"
        )

        all_texts: List[str] = []
        text_index: List[Tuple[str, int]] = []

        for atom_id in self._atom_ids:
            pack = self._rewrite_packs[atom_id]
            for i, text in enumerate(pack.texts):
                all_texts.append(text)
                text_index.append((atom_id, i))

        all_vectors = self._embed_fn(all_texts)

        self._atom_embeddings = {}
        for (atom_id, _), vector in zip(text_index, all_vectors):
            if atom_id not in self._atom_embeddings:
                self._atom_embeddings[atom_id] = []
            self._atom_embeddings[atom_id].append(vector)

        self._embeddings_computed = True
        logger.info("[PROPOSER] Atom embeddings ready.")

    # ── Core Proposal Logic ──────────────────────────────────────────

    def propose(
        self,
        lemma: str,
        evidence_count: int,
    ) -> List[SynapsePatchEntry]:
        """
        Generate candidate Synapse edges for a verb lemma.

        Input:  lemma + evidence_count (synset expansion is internal)
        Output: List[SynapsePatchEntry] filtered by min_score & top_k

        Flow:
          lemma → WordNet(pos=VERB) → synsets
          for each synset:
            embed(synset.definition) vs all atom 4-packs
            score = max cosine across pack texts
            filter by min_score → sort desc → top_k
          → dedup by edge_key → return
        """
        self._ensure_embeddings()

        # ── Validate ──
        lemma = lemma.strip().lower()
        if not lemma:
            logger.warning("[PROPOSER] Empty lemma, skipping")
            return []

        # ── Synset expansion (internal — Taka directive) ──
        synsets = self._synset_lookup_fn(lemma)
        if not synsets:
            logger.info(f"[PROPOSER] No verb synsets for {lemma!r}")
            return []

        logger.info(f"[PROPOSER] {lemma!r} → {len(synsets)} verb synsets")

        # ── Score each synset against all atoms ──
        all_proposals: List[SynapsePatchEntry] = []
        seen_keys: set = set()  # GPT audit note B: dedup

        for synset_info in synsets:
            synset_id = synset_info["synset_id"]
            synset_def = synset_info["definition"]

            synset_vec = self._embed_fn([synset_def])[0]

            candidates: List[_ProposalCandidate] = []
            for atom_id in self._atom_ids:
                atom_vecs = self._atom_embeddings[atom_id]
                max_sim = max(
                    self._cosine_similarity(synset_vec, avec)
                    for avec in atom_vecs
                )
                if max_sim >= self.min_score:
                    candidates.append(_ProposalCandidate(
                        synset_id=synset_id,
                        atom_id=atom_id,
                        score=round(max_sim, 4),
                        evidence_count=evidence_count,
                    ))

            # Top-k per synset (Design Spec v3.0 §4)
            candidates.sort(key=lambda c: c.score, reverse=True)

            for cand in candidates[:self.top_k]:
                edge_key = SynapsePatchEntry.make_key(cand.synset_id, cand.atom_id)
                if edge_key in seen_keys:
                    continue
                seen_keys.add(edge_key)

                entry = SynapsePatchEntry(
                    op="add_edge",
                    edge_key=edge_key,
                    synset_id=cand.synset_id,
                    atom=cand.atom_id,
                    score=cand.score,
                    reason="auto_proposal_v2.0",
                    metadata={
                        "rewrite_pack_id": REWRITE_PACK_ID,
                        "model": self.model_name,
                        "source": "relation_pipeline",
                        "evidence_count": cand.evidence_count,
                        "count_kind": "lemma_based",
                    },
                )
                all_proposals.append(entry)

        logger.info(
            f"[PROPOSER] {lemma!r}: {len(synsets)} synsets → "
            f"{len(all_proposals)} proposals (≥{self.min_score}, top-{self.top_k})"
        )
        return all_proposals

    # ── Batch ────────────────────────────────────────────────────────

    def propose_batch(
        self,
        lemmas: List[Dict[str, Any]],
    ) -> Dict[str, List[SynapsePatchEntry]]:
        """Batch proposal: [{"lemma": str, "evidence_count": int}] → {lemma: proposals}"""
        results: Dict[str, List[SynapsePatchEntry]] = {}
        for item in lemmas:
            lemma = item["lemma"]
            count = item.get("evidence_count", 0)
            results[lemma] = self.propose(lemma, count)
        return results

    # ── Trace & Export ───────────────────────────────────────────────

    def write_trace(
        self,
        proposals: List[SynapsePatchEntry],
        lemma: str,
    ) -> str:
        """Write rewrite trace to JSONL (Design Spec v3.0 §2). Returns file path."""
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.log_dir, f"rewrite_trace_{lemma}_{ts}.jsonl")

        with open(path, "w", encoding="utf-8") as f:
            # Rewrite packs for atoms that appear in proposals
            atoms_used = sorted({p.atom for p in proposals})
            for atom_id in atoms_used:
                pack = self._rewrite_packs[atom_id]
                f.write(json.dumps({
                    "type": "rewrite_pack",
                    "lemma": lemma,
                    **pack.to_trace_dict(),
                }, ensure_ascii=False) + "\n")

            # Proposals
            for p in proposals:
                f.write(json.dumps({
                    "type": "proposal",
                    "lemma": lemma,
                    **p.to_dict(),
                }, ensure_ascii=False) + "\n")

        logger.info(f"[PROPOSER] Trace → {path}")
        return path

    def export_patch_file(
        self,
        proposals: List[SynapsePatchEntry],
        output_path: str,
    ) -> str:
        """Export proposals as SynapseStore-compatible JSON patch file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"patches": [p.to_dict() for p in proposals]},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(f"[PROPOSER] Patch → {output_path} ({len(proposals)} entries)")
        return output_path

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def get_rewrite_pack(self, atom_id: str) -> Optional[RewritePack]:
        """Get rewrite pack for an atom (for inspection/debugging)."""
        return self._rewrite_packs.get(atom_id)

    def get_audit_info(self) -> Dict[str, Any]:
        """Return audit-relevant metadata."""
        return {
            "atom_count": len(self.atoms),
            "model_name": self.model_name,
            "min_score": self.min_score,
            "top_k": self.top_k,
            "rewrite_pack_id": REWRITE_PACK_ID,
            "rewrite_pack_texts_per_atom": 4,
            "embeddings_computed": self._embeddings_computed,
        }
