#!/usr/bin/env python3
"""
ESDE Phase 8 — Projection Operators
====================================
Inter-space projection between Natural Language Space and Atom Field Space.

Implements:
  - AtomFieldEmbeddings: precomputed atom definition embeddings (z_i)
  - FieldGate (Operator B): sentence-context gating of atom candidates
  - WeakMeasurement (Operator C): Bayesian posterior with delayed collapse
  - HybridProjection (B+C): gate as likelihood in Bayesian update

Embedding backend:
  - TfidfEmbedder: sklearn TF-IDF (prototype, available everywhere)
  - MiniLMEmbedder: sentence-transformers all-MiniLM-L6-v2 (production)
  Switch via: EMBEDDER_BACKEND env var or constructor arg.

Design:
  - "Describe, do not decide" — winner=null; returns distributions, not argmax
  - Dynamic thresholds (quantile-based, no hand-tuned constants)
  - Log-space computation for numerical stability
  - Deterministic: fixed random seeds, stable sort

3AI Roles:
  - Gemini: Operator mathematical definitions
  - GPT: Implementation spec + evaluation protocol
  - Claude: Implementation

Date: 2026-03-03
Version: 0.1.0
"""

import json
import hashlib
import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Embedding Backends
# ============================================================

class SentenceEmbedder(ABC):
    """Abstract sentence embedding interface."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) array."""
        ...
    
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging/cache keys."""
        ...


class TfidfEmbedder(SentenceEmbedder):
    """
    TF-IDF based sentence embeddings (prototype).
    
    Fits a shared vocabulary from atom definitions + input sentences,
    then uses TF-IDF vectors as dense-ish representations.
    
    Limitations:
      - Lexical overlap only (no semantic understanding)
      - Must be re-fitted when corpus changes
      - Results will be substantially worse than MiniLM
    
    Upgrade path: set EMBEDDER_BACKEND=minilm
    """
    
    def __init__(self, max_features: int = 5000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        
        self._max_features = max_features
        self._target_dim = 384  # Match MiniLM for API compatibility
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=min(self._target_dim, max_features - 1), random_state=42)
        self._fitted = False
    
    def fit(self, corpus: List[str]):
        """Fit TF-IDF + SVD on a corpus (atom defs + sentences)."""
        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        actual_dim = min(self._svd.n_components, tfidf_matrix.shape[1] - 1)
        if actual_dim < self._svd.n_components:
            from sklearn.decomposition import TruncatedSVD
            self._svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True
        logger.info(f"TfidfEmbedder fitted: vocab={len(self._vectorizer.vocabulary_)}, svd_dim={self._svd.n_components}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder not fitted. Call fit() first.")
        tfidf = self._vectorizer.transform(texts)
        reduced = self._svd.transform(tfidf)
        # L2 normalize
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return reduced / norms
    
    def dim(self) -> int:
        if self._fitted:
            return self._svd.n_components
        return self._target_dim
    
    @property
    def name(self) -> str:
        return "tfidf"


class MiniLMEmbedder(SentenceEmbedder):
    """
    MiniLM sentence embeddings (production).
    
    Requires: pip install sentence-transformers
    Model: all-MiniLM-L6-v2 (384D, ~90MB)
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"MiniLMEmbedder loaded: model={model_name}, dim={self._dim}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers\n"
                "Or use EMBEDDER_BACKEND=tfidf for prototype."
            )
    
    def fit(self, corpus: List[str]):
        """No-op for pre-trained model."""
        pass
    
    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings)
    
    def dim(self) -> int:
        return self._dim
    
    @property
    def name(self) -> str:
        return "minilm"


def get_embedder(backend: str = None, **kwargs) -> SentenceEmbedder:
    """Factory for embedding backend."""
    backend = backend or os.environ.get("EMBEDDER_BACKEND", "tfidf")
    if backend == "minilm":
        return MiniLMEmbedder(**kwargs)
    elif backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown embedder backend: {backend}")


# ============================================================
# Atom Field Embeddings
# ============================================================

class AtomFieldEmbeddings:
    """
    Precomputed atom definition embeddings (z_i).
    
    Uses esde_dictionary.json to build text representations of each atom,
    then embeds them with the selected backend.
    
    Cache: cache/atom_def_emb_{backend}.npz
    """
    
    def __init__(self, dictionary_path: str, embedder: SentenceEmbedder,
                 cache_dir: str = "cache"):
        self.dictionary_path = dictionary_path
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        
        self._atom_ids: List[str] = []
        self._Z: Optional[np.ndarray] = None  # (N_atoms, dim)
        self._atom_to_idx: Dict[str, int] = {}
        
        self._load_or_compute()
    
    def _dict_hash(self) -> str:
        """Hash of dictionary file for cache invalidation."""
        with open(self.dictionary_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    
    def _cache_path(self) -> Path:
        return self.cache_dir / f"atom_def_emb_v2_{self.embedder.name}.npz"
    
    def _meta_path(self) -> Path:
        return self.cache_dir / f"atom_def_emb_v2_{self.embedder.name}_meta.json"
    
    def _load_or_compute(self):
        """Load from cache or compute embeddings."""
        cache_file = self._cache_path()
        meta_file = self._meta_path()
        dict_hash = self._dict_hash()
        
        # Try cache
        if cache_file.exists() and meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("dict_hash") == dict_hash and meta.get("backend") == self.embedder.name:
                data = np.load(cache_file)
                self._Z = data["Z"]
                self._atom_ids = meta["atom_ids"]
                self._atom_to_idx = {a: i for i, a in enumerate(self._atom_ids)}
                logger.info(f"Loaded cached atom embeddings: {len(self._atom_ids)} atoms")
                return
        
        # Compute
        self._compute(dict_hash)
    
    def _compute(self, dict_hash: str):
        """Compute atom embeddings from dictionary."""
        with open(self.dictionary_path) as f:
            raw = json.load(f)
        
        # Handle both formats: {"concepts": {...}} or direct dict
        concepts = raw.get("concepts", raw)
        if isinstance(concepts, dict) and "atoms" in concepts:
            concepts = concepts["atoms"]
        
        atom_ids = []
        texts = []
        
        for atom_id in sorted(concepts.keys()):
            entry = concepts[atom_id]
            # Build ENRICHED text: definition + triggers + examples
            # Richer text = better embeddings = better disambiguation
            parts = [atom_id]
            
            # Name/label
            if "name" in entry:
                parts.append(entry["name"])
            elif "en_label" in entry:
                parts.append(entry["en_label"])
            
            # Definition (try multiple field names)
            defn = entry.get("definition_en") or entry.get("short_definition") or entry.get("definition", "")
            if defn:
                parts.append(defn)
            
            # Triggers — critical for disambiguation
            triggers = entry.get("triggers_en", [])
            if isinstance(triggers, list) and triggers:
                parts.append("Related: " + ", ".join(triggers[:15]))
            elif isinstance(triggers, str) and triggers:
                parts.append("Related: " + triggers)
            
            # Examples — provide context
            examples = entry.get("examples_en", [])
            if isinstance(examples, list) and examples:
                parts.append("Examples: " + "; ".join(examples[:5]))
            elif isinstance(examples, str) and examples:
                parts.append("Examples: " + examples)
            
            # Kanji field (legacy support)
            if "kanji_semantic_field" in entry and isinstance(entry["kanji_semantic_field"], list):
                parts.extend(entry["kanji_semantic_field"][:3])
            
            text = ". ".join(parts)
            atom_ids.append(atom_id)
            texts.append(text)
        
        # For TF-IDF backend, we need to fit first
        if hasattr(self.embedder, 'fit') and not getattr(self.embedder, '_fitted', True):
            self.embedder.fit(texts)
        
        Z = self.embedder.embed(texts)
        
        self._atom_ids = atom_ids
        self._Z = Z
        self._atom_to_idx = {a: i for i, a in enumerate(atom_ids)}
        
        # Cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(self._cache_path(), Z=Z)
        with open(self._meta_path(), 'w') as f:
            json.dump({
                "atom_ids": atom_ids,
                "dict_hash": dict_hash,
                "backend": self.embedder.name,
                "dim": int(Z.shape[1]),
                "n_atoms": len(atom_ids),
            }, f, indent=2)
        
        logger.info(f"Computed atom embeddings: {len(atom_ids)} atoms, dim={Z.shape[1]}")
    
    @property
    def atom_ids(self) -> List[str]:
        return self._atom_ids
    
    @property
    def Z(self) -> np.ndarray:
        """(N_atoms, dim) embedding matrix."""
        return self._Z
    
    def cos_to_all(self, sentence_emb: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between a sentence embedding and all atom embeddings.
        
        Args:
            sentence_emb: (dim,) vector
        Returns:
            (N_atoms,) cosine similarities
        """
        # sentence_emb and Z are already L2 normalized
        s = sentence_emb / (np.linalg.norm(sentence_emb) + 1e-10)
        return self._Z @ s
    
    def get_embedding(self, atom_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific atom."""
        idx = self._atom_to_idx.get(atom_id)
        if idx is None:
            return None
        return self._Z[idx]


# ============================================================
# Operator B: Field-First Gate
# ============================================================

class FieldGate:
    """
    Operator B: sentence-context gating of atom candidates.
    
    Computes gravity field g_i = cos(z_s, z_i) for all atoms,
    then applies soft sigmoid gate:
        gate_i = sigmoid((g_i - tau_field) / T)
    
    tau_field is dynamic (quantile-based, no hand threshold):
        tau_field = quantile_q(g_i)
    
    Design: Synapse scores can only be reduced, never amplified.
    """
    
    def __init__(self, quantile: float = 0.85, temperature: float = 0.1):
        self.quantile = quantile
        self.temperature = temperature
    
    def compute(self, sentence_emb: np.ndarray, 
                atom_field: AtomFieldEmbeddings) -> Tuple[np.ndarray, Dict]:
        """
        Compute gate values for all atoms.
        
        Args:
            sentence_emb: (dim,) sentence embedding
            atom_field: precomputed atom embeddings
        
        Returns:
            gate_values: (N_atoms,) in (0, 1)
            stats: {tau_field, g_max, g_mean, g_std}
        """
        g = atom_field.cos_to_all(sentence_emb)
        
        # Dynamic threshold
        tau_field = float(np.quantile(g, self.quantile))
        
        # Soft sigmoid gate
        gate = _sigmoid((g - tau_field) / self.temperature)
        
        stats = {
            "tau_field": tau_field,
            "q": self.quantile,
            "T": self.temperature,
            "g_max": float(np.max(g)),
            "g_mean": float(np.mean(g)),
            "g_std": float(np.std(g)),
        }
        
        return gate, stats


# ============================================================
# Operator C: Weak Measurement (Bayesian)
# ============================================================

class WeakMeasurement:
    """
    Operator C: Bayesian posterior with delayed collapse.
    
    P(a_i | w, s) ∝ prior(a_i | w) * L(s | a_i)
    
    No argmax. Returns full distribution (winner=null).
    Collapse delayed until Island-level aggregation.
    """
    
    def __init__(self, temperature: float = 0.1, eps: float = 1e-10):
        self.temperature = temperature
        self.eps = eps
    
    def update(self, prior: Dict[str, float], 
               likelihood: Dict[str, float]) -> Dict[str, float]:
        """
        Bayesian update in log-space.
        
        Args:
            prior: {atom_id: score} from Synapse (sparse)
            likelihood: {atom_id: L_i} from field evidence
        
        Returns:
            posterior: {atom_id: P(a_i | w, s)} normalized
        """
        if not prior:
            return {}
        
        atoms = list(prior.keys())
        log_prior = np.array([np.log(prior[a] + self.eps) for a in atoms])
        log_like = np.array([np.log(likelihood.get(a, self.eps) + self.eps) for a in atoms])
        
        log_post = log_prior + log_like
        # Softmax for numerical stability
        log_post -= np.max(log_post)
        post = np.exp(log_post)
        post /= (post.sum() + self.eps)
        
        return {a: float(p) for a, p in zip(atoms, post)}


# ============================================================
# B+C Hybrid Projection
# ============================================================

class HybridProjection:
    """
    Exp-BC: Field gate as likelihood in Bayesian update.
    
    L_i := gate_i (from Operator B)
    posterior ∝ prior_i * gate_i
    
    Combines:
    - B: sentence-context filtering (delete irrelevant atoms)
    - C: probabilistic output (winner=null preserved)
    """
    
    def __init__(self, quantile: float = 0.85, temperature: float = 0.1,
                 eps: float = 1e-10):
        self.gate = FieldGate(quantile=quantile, temperature=temperature)
        self.bayes = WeakMeasurement(temperature=temperature, eps=eps)
        self.eps = eps
    
    def project(self, synapse_candidates: List[Dict],
                sentence_emb: np.ndarray,
                atom_field: AtomFieldEmbeddings,
                top_k: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Project Synapse candidates through field expansion + gate + Bayesian update.
        """
        # STEP 0: Expand candidates with field-suggested atoms
        expanded = expand_candidates_from_field(
            synapse_candidates, sentence_emb, atom_field,
            top_k_expand=15, field_weight=0.3,
        )
        
        # Build prior from expanded candidates
        prior = build_prior_from_synapse_candidates(expanded)
        
        # Compute gate
        gate_values, gate_stats = self.gate.compute(sentence_emb, atom_field)
        
        # Build likelihood dict (gate values for candidate atoms)
        atom_to_idx = atom_field._atom_to_idx
        likelihood = {}
        for atom_id in prior:
            idx = atom_to_idx.get(atom_id)
            if idx is not None:
                likelihood[atom_id] = float(gate_values[idx])
            else:
                likelihood[atom_id] = self.eps
        
        # Bayesian update
        posterior = self.bayes.update(prior, likelihood)
        
        # Sort by posterior score (descending), take top_k
        ranked = sorted(posterior.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        
        projected = []
        for atom_id, post_score in ranked:
            projected.append({
                "atom": atom_id,
                "score": post_score,
                "prior": prior.get(atom_id, 0.0),
                "gate": likelihood.get(atom_id, 0.0),
            })
        
        stats = {
            **gate_stats,
            "n_synapse_candidates": len(synapse_candidates),
            "n_expanded_candidates": len(expanded),
            "n_candidates": len(prior),
            "n_nonzero_posterior": sum(1 for v in posterior.values() if v > 0.01),
        }
        
        return projected, stats


# ============================================================
# Operator B standalone (for Exp-B mode)
# ============================================================

class FieldFirstProjection:
    """
    Exp-B: Gate-only projection (no Bayesian update).
    
    score(a_i | w, s) = Syn(w, a_i) * gate(a_i, s)
    """
    
    def __init__(self, quantile: float = 0.85, temperature: float = 0.1):
        self.gate = FieldGate(quantile=quantile, temperature=temperature)
    
    def project(self, synapse_candidates: List[Dict],
                sentence_emb: np.ndarray,
                atom_field: AtomFieldEmbeddings,
                top_k: int = 5) -> Tuple[List[Dict], Dict]:
        # Expand candidates with field-suggested atoms
        expanded = expand_candidates_from_field(
            synapse_candidates, sentence_emb, atom_field,
            top_k_expand=15, field_weight=0.3,
        )
        
        gate_values, gate_stats = self.gate.compute(sentence_emb, atom_field)
        atom_to_idx = atom_field._atom_to_idx
        
        results = []
        for cand in expanded:
            atom_id = cand.get("atom") or cand.get("concept_id", "")
            score = cand.get("score", 0.0)
            idx = atom_to_idx.get(atom_id)
            g = float(gate_values[idx]) if idx is not None else 0.0
            results.append({
                "atom": atom_id,
                "score": score * g,
                "prior": score,
                "gate": g,
            })
        
        results.sort(key=lambda x: (-x["score"], x["atom"]))
        return results[:top_k], gate_stats


# ============================================================
# Operator C standalone (for Exp-C mode)
# ============================================================

class WeakMeasurementProjection:
    """
    Exp-C: Bayesian update with sentence embedding as likelihood.
    
    L_i = exp(cos(z_s, z_i) / T)
    posterior ∝ prior_i * L_i
    """
    
    def __init__(self, temperature: float = 0.5, eps: float = 1e-10):
        self.temperature = temperature
        self.bayes = WeakMeasurement(temperature=temperature, eps=eps)
    
    def project(self, synapse_candidates: List[Dict],
                sentence_emb: np.ndarray,
                atom_field: AtomFieldEmbeddings,
                top_k: int = 5) -> Tuple[List[Dict], Dict]:
        # Expand candidates with field-suggested atoms
        expanded = expand_candidates_from_field(
            synapse_candidates, sentence_emb, atom_field,
            top_k_expand=15, field_weight=0.3,
        )
        
        prior = build_prior_from_synapse_candidates(expanded)
        
        # Compute likelihood from sentence-atom cosine similarity
        g = atom_field.cos_to_all(sentence_emb)
        atom_to_idx = atom_field._atom_to_idx
        
        likelihood = {}
        for atom_id in prior:
            idx = atom_to_idx.get(atom_id)
            if idx is not None:
                likelihood[atom_id] = float(np.exp(g[idx] / self.temperature))
            else:
                likelihood[atom_id] = 1e-10
        
        posterior = self.bayes.update(prior, likelihood)
        ranked = sorted(posterior.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        
        projected = []
        for atom_id, post_score in ranked:
            projected.append({
                "atom": atom_id,
                "score": post_score,
                "prior": prior.get(atom_id, 0.0),
                "likelihood": likelihood.get(atom_id, 0.0),
            })
        
        stats = {
            "g_max": float(np.max(g)),
            "g_mean": float(np.mean(g)),
            "T_likelihood": self.temperature,
            "n_candidates": len(prior),
        }
        
        return projected, stats


# ============================================================
# Utilities
# ============================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def build_prior_from_synapse_candidates(candidates: List[Dict]) -> Dict[str, float]:
    """
    Build normalized prior distribution from Synapse candidates.
    
    Args:
        candidates: [{"atom": "STA.wealth", "score": 0.72}, ...]
    
    Returns:
        {atom_id: normalized_score}  — sums to 1.0
    """
    if not candidates:
        return {}
    
    scores = {}
    for c in candidates:
        atom_id = c.get("atom") or c.get("concept_id", "")
        score = c.get("score", 0.0)
        if atom_id and score > 0:
            scores[atom_id] = scores.get(atom_id, 0.0) + score
    
    total = sum(scores.values())
    if total <= 0:
        return {}
    
    return {a: s / total for a, s in scores.items()}


def expand_candidates_from_field(synapse_candidates: List[Dict],
                                  sentence_emb: np.ndarray,
                                  atom_field: 'AtomFieldEmbeddings',
                                  top_k_expand: int = 25,
                                  field_weight: float = 0.5) -> List[Dict]:
    """
    Expand Synapse candidates with atoms suggested by sentence context.
    
    Critical fix: Synapse often returns only 1-2 candidates, and the correct
    atom may not be among them at all. This function adds the top atoms
    by sentence-embedding similarity, giving the Bayesian update a chance
    to surface the correct atom even when Synapse missed it.
    
    Args:
        synapse_candidates: original Synapse results
        sentence_emb: (dim,) sentence embedding
        atom_field: precomputed atom embeddings
        top_k_expand: how many field-suggested atoms to consider
        field_weight: score assigned to field-sourced candidates
                      (relative to Synapse scores)
    
    Returns:
        expanded candidate list (Synapse originals + field additions)
    """
    # Compute cosine similarity to all atoms
    cos_all = atom_field.cos_to_all(sentence_emb)
    
    # Get top-K atoms by sentence similarity
    top_indices = np.argsort(cos_all)[-top_k_expand:][::-1]
    
    # Cap Synapse scores to prevent prior dominance
    # Without cap: synapse=0.73 vs field=0.08 → 9:1 prior ratio
    # With cap:    synapse=0.40 vs field=0.08 → 5:1 prior ratio (Bayes can overcome)
    PRIOR_CAP = 0.40
    capped_synapse = []
    for c in synapse_candidates:
        cc = dict(c)
        cc["score"] = min(cc.get("score", 0), PRIOR_CAP)
        capped_synapse.append(cc)
    
    # Track existing Synapse candidates
    existing = {(c.get("atom") or c.get("concept_id", "")) for c in synapse_candidates}
    
    expanded = list(capped_synapse)
    
    for idx in top_indices:
        atom_id = atom_field.atom_ids[idx]
        if atom_id not in existing:
            sim = float(cos_all[idx])
            # Only add if similarity is positive
            if sim > 0:
                expanded.append({
                    "atom": atom_id,
                    "score": sim * field_weight,
                    "source": "field",
                })
    
    return expanded


def get_projection_operator(mode: str, **kwargs):
    """
    Factory for projection operators.
    
    Args:
        mode: "base" | "B" | "C" | "BC"
    """
    if mode == "base":
        return None
    elif mode == "B":
        return FieldFirstProjection(**kwargs)
    elif mode == "C":
        return WeakMeasurementProjection(**kwargs)
    elif mode == "BC":
        return HybridProjection(**kwargs)
    else:
        raise ValueError(f"Unknown projection mode: {mode}")