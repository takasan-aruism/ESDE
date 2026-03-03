#!/usr/bin/env python3
"""
ESDE Phase 8 — Rule-Based Molecule Generator (Phase α)
=======================================================

Generates draft molecules WITHOUT LLM.
Uses Sensor V2 candidates + spaCy SVO extraction + Glossary rules.

Three-phase architecture (A1-inspired ternary emergence):
  α: rule_generator.py   (this file — mechanical, fast)
  β: molecule_auditor.py  (LLM judge, selective)
  γ: correction           (re-gen or human review)

3AI Approval: 2026-03-02
  GPT:    Approved (score 9/10) + required POS filter, stopword, dedup, threshold
  Gemini: Approved + required Confidence Gap flag for Phase β

Design Principles:
  - No LLM dependency
  - Reuses existing modules (Sensor V2, ParserAdapter, GlossaryValidator)
  - Same output schema as MoleculeGeneratorLive (v8.3 compatible)
  - Dynamic thresholds for Phase β flagging (no hardcoded values)

Two-Pass Architecture:
  Pass 1: Generate all molecules, collect distribution statistics (no flags)
  Pass 2: Compute quantile-based thresholds from distribution → apply flags

Threshold Model (3AI approved):
  threshold = max(t_abs, quantile_85(distribution))
  - t_abs: Safety net minimum (prevents nonsensical flags on tiny batches)
  - quantile_85: 85th percentile of actual batch distribution (upper 15%)
  - Same philosophy as Phase 9 ThresholdResolver

Usage:
  from rule_generator import RuleGenerator, apply_dynamic_flags
  gen = RuleGenerator(synapse_file="esde_synapses_v3.json",
                      glossary_file="glossary_results.json")

  # Pass 1: generate (no flags)
  molecules = [gen.generate(s) for s in sentences]

  # Pass 2: flag from distribution
  flagged, thresholds = apply_dynamic_flags(molecules)
"""

import re
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple


# ============================================================
# Configuration
# ============================================================

# POS tags to keep (spaCy universal POS)
ALLOWED_POS = {"NOUN", "VERB", "ADJ", "PROPN"}

# Stopwords: function words that pollute atom mapping
STOPWORDS = {
    # Auxiliaries & modals
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Common function words that leak into content
    "get", "got", "getting",
    "make", "made", "making",
    "take", "took", "taken", "taking",
    "give", "gave", "given", "giving",
    "go", "went", "gone", "going",
    "come", "came", "coming",
    "set", "sit", "sat",
    # Pronouns & determiners that sometimes pass POS filter
    "it", "its", "there", "this", "that", "these", "those",
    "well", "also", "much", "many", "such", "very",
    "other", "new", "old", "first", "last", "early", "late",
}

# Minimum Sensor score to keep a candidate
MIN_CANDIDATE_SCORE = 0.45

# Maximum atoms per molecule
MAX_ATOMS_PER_MOLECULE = 5

# Operator mapping from syntactic structure
OPERATOR_MAP = {
    "nsubj_dobj": "▷",       # Subject acts on Object
    "nsubj_dobj_neg": "▷ ¬",  # Negated action
    "nsubj_only": "",          # Single atom (intransitive)
    "coordination": "×",       # Coordinated elements
    "default": "⊕",           # Juxtaposition (fallback)
}

# ── Dynamic Threshold Safety Nets (t_abs) ──
# These are absolute minimums — prevents flagging on trivially small values.
# Actual thresholds are max(t_abs, quantile_85(batch_distribution)).
T_ABS = {
    "atom_count": 3,           # Don't flag < 3 atoms as complex
    "operator_count": 2,       # Don't flag < 2 operators as complex formula
    "confidence_gap": 0.2,     # Don't flag < 0.2 gap as divergent
    "category_count": 3,       # Don't flag < 3 categories as cross-category
}

# Quantile level for dynamic threshold (upper 15% = 85th percentile)
FLAG_QUANTILE = 0.85

# NLP reference ranges (logged, not used for decisions)
# These are "what typical NLP systems consider" — for audit trail only.
NLP_REFERENCE_RANGES = {
    "atom_count": "3-5 typical for sentence-level semantic frames",
    "confidence_gap": "0.3-0.5 typical for embedding score spread",
    "category_count": "2-4 typical for diverse sentences",
    "audit_rate": "15-25% typical for selective audit pipelines",
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class DraftMolecule:
    """
    Draft molecule from rule-based generation.
    Same schema as MoleculeGeneratorLive output for pipeline compatibility.
    
    Flags are NOT set during generation (Pass 1).
    They are applied in Pass 2 by apply_dynamic_flags().
    """
    active_atoms: List[Dict[str, Any]]
    formula: str
    source_text: str
    
    # Audit flags (set in Pass 2 by apply_dynamic_flags)
    flags: List[str] = field(default_factory=list)
    
    # Distribution metrics (recorded in Pass 1, used in Pass 2)
    atom_count: int = 0
    operator_count: int = 0
    confidence_gap: float = 0.0
    category_count: int = 0
    
    # Metadata
    generator: str = "rule_generator_v1"
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        # Compute distribution metrics from content
        self.atom_count = len(self.active_atoms)
        self.operator_count = sum(
            1 for op in ["▷", "×", "⊕", "→", "⇒", "↺"]
            if op in self.formula
        )
        categories = set()
        for aa in self.active_atoms:
            atom = aa.get("atom", "")
            if "." in atom:
                categories.add(atom.split(".")[0])
        self.category_count = len(categories)
    
    @property
    def needs_audit(self) -> bool:
        """Whether this molecule should go to Phase β."""
        return len(self.flags) > 0
    
    def to_molecule_dict(self) -> Dict[str, Any]:
        """Convert to v8.3 compatible molecule dict."""
        return {
            "active_atoms": self.active_atoms,
            "formula": self.formula,
            "meta": {
                "generator": self.generator,
                "generator_version": "rule_v1",
                "validator_status": "draft",
                "flags": self.flags,
                "distribution_metrics": {
                    "atom_count": self.atom_count,
                    "operator_count": self.operator_count,
                    "confidence_gap": round(self.confidence_gap, 4),
                    "category_count": self.category_count,
                },
                "timestamp": self.timestamp,
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Full dict including source text."""
        d = self.to_molecule_dict()
        d["source_text"] = self.source_text[:200]
        d["needs_audit"] = self.needs_audit
        return d


@dataclass
class GenerationStats:
    """Aggregate stats for a batch run."""
    processed: int = 0
    molecules_generated: int = 0
    flagged_for_audit: int = 0
    no_candidates: int = 0
    svo_extracted: int = 0
    svo_grounded: int = 0
    
    # Formula pattern counts
    formula_patterns: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Flag distribution
    flag_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed": self.processed,
            "molecules_generated": self.molecules_generated,
            "flagged_for_audit": self.flagged_for_audit,
            "no_candidates": self.no_candidates,
            "svo_extracted": self.svo_extracted,
            "svo_grounded": self.svo_grounded,
            "audit_rate": round(self.flagged_for_audit / max(self.molecules_generated, 1), 3),
            "formula_patterns": dict(self.formula_patterns),
            "flag_counts": dict(self.flag_counts),
        }


# ============================================================
# Rule Generator
# ============================================================

class RuleGenerator:
    """
    Rule-based molecule generator. No LLM.
    
    Pipeline:
      Text → spaCy (POS + SVO) → Sensor V2 (candidates) → Rules → Draft Molecule
    """
    
    def __init__(
        self,
        synapse_file: str,
        glossary_file: str,
        synapse_patches: List[str] = None,
        min_score: float = MIN_CANDIDATE_SCORE,
        max_atoms: int = MAX_ATOMS_PER_MOLECULE,
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize generator.
        
        Args:
            synapse_file: Path to Synapse v3 JSON
            glossary_file: Path to glossary JSON
            synapse_patches: Optional list of patch files
            min_score: Minimum candidate score to keep
            max_atoms: Maximum atoms per molecule
            spacy_model: spaCy model for POS tagging and SVO extraction
        """
        self.min_score = min_score
        self.max_atoms = max_atoms
        
        # Import and init Sensor V2
        from sensor.esde_sensor_v2_modular import ESDESensorV2
        self.sensor = ESDESensorV2(
            synapse_file=synapse_file,
            glossary_file=glossary_file,
        )
        
        # Import and init ParserAdapter (SVO extraction)
        from integration.relations.parser_adapter import ParserAdapter
        self.parser = ParserAdapter(spacy_model=spacy_model)
        
        # Load glossary for axis/level lookup
        from sensor.glossary_validator import GlossaryValidator
        with open(glossary_file) as f:
            raw_glossary = json.load(f)
        self.glossary_validator = GlossaryValidator(raw_glossary)
        
        # Load spaCy for POS tagging (reuse parser's model)
        self._nlp = self.parser.nlp
        
        self.stats = GenerationStats()
    
    def generate(self, sentence: str) -> Optional[DraftMolecule]:
        """
        Generate a draft molecule from a single sentence.
        
        Args:
            sentence: Input sentence text
            
        Returns:
            DraftMolecule or None if no candidates
        """
        self.stats.processed += 1
        
        # ── Step 1: POS analysis + content word extraction ──
        doc = self._nlp(sentence)
        content_tokens = self._extract_content_tokens(doc)
        
        if not content_tokens:
            self.stats.no_candidates += 1
            return None
        
        # ── Step 2: Sensor V2 candidates (already filtered by Synapse) ──
        sensor_result = self.sensor.analyze(sentence)
        candidates = sensor_result.get("candidates", [])
        
        if not candidates:
            self.stats.no_candidates += 1
            return None
        
        # ── Step 3: Filter candidates (GPT requirements) ──
        filtered = self._filter_candidates(candidates, content_tokens)
        
        if not filtered:
            self.stats.no_candidates += 1
            return None
        
        # ── Step 4: SVO extraction ──
        extraction = self.parser.extract(sentence)
        triples = extraction.triples if extraction else []
        
        if triples:
            self.stats.svo_extracted += len(triples)
        
        # ── Step 5: Build molecule ──
        molecule = self._build_molecule(sentence, filtered, triples, doc)
        
        # ── Step 6: Compute confidence_gap (metric only, no flag decision) ──
        if candidates and filtered:
            top_sensor = max(c.get("score", 0) for c in candidates)
            bottom_used = min(c.get("score", 0) for c in filtered)
            molecule.confidence_gap = top_sensor - bottom_used
        
        self.stats.molecules_generated += 1
        
        # Track formula pattern
        pattern = re.sub(r'aa_\d+', 'X', molecule.formula) if molecule.formula else "(empty)"
        self.stats.formula_patterns[pattern] += 1
        
        return molecule
    
    def _extract_content_tokens(self, doc) -> List[Dict[str, Any]]:
        """
        Extract content tokens with POS filtering and stopword removal.
        
        Returns list of {text, lemma, pos, idx} for content words only.
        """
        tokens = []
        for token in doc:
            # POS filter
            if token.pos_ not in ALLOWED_POS:
                continue
            # Stopword filter
            if token.lemma_.lower() in STOPWORDS:
                continue
            # Skip short tokens
            if len(token.text) < 2:
                continue
            # Skip numbers
            if token.like_num:
                continue
            
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_.lower(),
                "pos": token.pos_,
                "idx": token.i,
            })
        
        return tokens
    
    def _filter_candidates(
        self,
        candidates: List[Dict],
        content_tokens: List[Dict],
    ) -> List[Dict]:
        """
        Filter and deduplicate candidates.
        
        GPT requirements:
          1. Score threshold
          2. Deduplicate same atom
          3. Limit to max_atoms
        """
        content_lemmas = {t["lemma"] for t in content_tokens}
        
        # Score filter
        scored = [c for c in candidates if c.get("score", 0) >= self.min_score]
        
        # Deduplicate by concept_id (keep highest score)
        seen = {}
        for c in scored:
            cid = c.get("concept_id", "")
            if cid not in seen or c.get("score", 0) > seen[cid].get("score", 0):
                seen[cid] = c
        
        deduped = sorted(seen.values(), key=lambda x: -x.get("score", 0))
        
        # Limit
        return deduped[:self.max_atoms]
    
    def _build_molecule(
        self,
        sentence: str,
        candidates: List[Dict],
        triples: List,
        doc,
    ) -> DraftMolecule:
        """
        Assemble draft molecule from filtered candidates + SVO structure.
        """
        active_atoms = []
        
        for i, cand in enumerate(candidates):
            atom_id = f"aa_{i+1}"
            concept_id = cand.get("concept_id", "")
            
            # Get text reference from top evidence
            text_ref = None
            top_ev = cand.get("top_evidence", {})
            if top_ev:
                text_ref = top_ev.get("token", top_ev.get("synset_id", ""))
            
            # Axis/level from glossary (first valid axis, first valid level)
            axis, level = self._lookup_axis_level(concept_id)
            
            # Span calculation (simple substring search)
            span = self._find_span(sentence, text_ref) if text_ref else None
            
            active_atoms.append({
                "id": atom_id,
                "atom": concept_id,
                "axis": axis,
                "level": level,
                "text_ref": text_ref,
                "span": span,
            })
        
        # Build formula from SVO structure
        formula = self._build_formula(active_atoms, triples, candidates, doc)
        
        return DraftMolecule(
            active_atoms=active_atoms,
            formula=formula,
            source_text=sentence,
        )
    
    def _lookup_axis_level(self, concept_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Look up first valid axis and level for an atom from glossary.
        Returns (axis, level) or (None, None).
        """
        axes = self.glossary_validator.get_valid_axes(concept_id)
        if not axes:
            return None, None
        
        # Use first axis alphabetically (deterministic)
        axis = sorted(axes)[0]
        levels = self.glossary_validator.get_valid_levels(concept_id, axis)
        
        if not levels:
            return axis, None
        
        level = sorted(levels)[0]
        return axis, level
    
    def _find_span(self, text: str, ref: str) -> Optional[List[int]]:
        """Find character span of text_ref in sentence."""
        if not ref:
            return None
        ref_lower = ref.lower()
        text_lower = text.lower()
        idx = text_lower.find(ref_lower)
        if idx >= 0:
            return [idx, idx + len(ref)]
        return None
    
    def _build_formula(
        self,
        active_atoms: List[Dict],
        triples: List,
        candidates: List[Dict],
        doc,
    ) -> str:
        """
        Build formula from SVO structure + atom assignments.
        
        Strategy:
          1. If SVO triple exists → map S/V/O to atoms → operator from structure
          2. If multiple atoms but no SVO → juxtapose with ⊕
          3. If single atom → atom ID only
        """
        if len(active_atoms) == 0:
            return ""
        
        if len(active_atoms) == 1:
            return active_atoms[0]["id"]
        
        # Try SVO-based formula
        if triples:
            formula = self._formula_from_svo(active_atoms, triples, doc)
            if formula:
                return formula
        
        # Fallback: dependency-based formula
        formula = self._formula_from_deps(active_atoms, doc)
        if formula:
            return formula
        
        # Last resort: juxtapose all atoms
        return " ⊕ ".join(a["id"] for a in active_atoms)
    
    def _formula_from_svo(
        self,
        active_atoms: List[Dict],
        triples: List,
        doc,
    ) -> Optional[str]:
        """
        Map SVO triple onto active atoms.
        
        Subject → find matching atom
        Object → find matching atom
        Verb → ▷ operator (with ¬ if negated)
        """
        # Take first triple (most prominent)
        triple = triples[0]
        
        subj_text = triple.subject.lower()
        obj_text = triple.object.lower()
        negated = triple.negated
        
        # Find atom for subject
        subj_atom = self._match_token_to_atom(subj_text, active_atoms)
        obj_atom = self._match_token_to_atom(obj_text, active_atoms)
        
        if subj_atom and obj_atom and subj_atom != obj_atom:
            self.stats.svo_grounded += 1
            op = "▷ ¬" if negated else "▷"
            return f"{subj_atom['id']} {op} {obj_atom['id']}"
        
        if subj_atom and not obj_atom:
            # Intransitive or object not grounded
            return None
        
        return None
    
    def _match_token_to_atom(
        self,
        text: str,
        active_atoms: List[Dict],
    ) -> Optional[Dict]:
        """
        Find the active atom whose text_ref best matches the given text.
        """
        text_lower = text.lower().strip()
        
        for atom in active_atoms:
            ref = atom.get("text_ref", "")
            if ref and ref.lower() in text_lower:
                return atom
            if ref and text_lower in ref.lower():
                return atom
        
        return None
    
    def _formula_from_deps(
        self,
        active_atoms: List[Dict],
        doc,
    ) -> Optional[str]:
        """
        Build formula from dependency structure when SVO fails.
        
        Looks for:
          - Subject-verb pairs → ▷
          - Coordinated nouns → ×
          - Modifier-head pairs → ⊕
        """
        if len(active_atoms) < 2:
            return None
        
        # Check for coordination (and, or)
        has_coordination = any(token.dep_ in ("conj", "cc") for token in doc)
        
        if has_coordination and len(active_atoms) == 2:
            return f"{active_atoms[0]['id']} × {active_atoms[1]['id']}"
        
        # Default: chain with ⊕
        return None
    
    # NOTE: _apply_flags removed. Flags are now applied in Pass 2
    # by the module-level apply_dynamic_flags() function.
    # This follows the 3AI-approved dynamic threshold model:
    #   threshold = max(t_abs, quantile_85(batch_distribution))
    
    def reset_stats(self):
        """Reset aggregate stats."""
        self.stats = GenerationStats()


# ============================================================
# Pass 2: Dynamic Flag Application
# ============================================================

def _quantile(values: List[float], q: float) -> float:
    """Compute quantile without numpy dependency."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = q * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def apply_dynamic_flags(
    molecules: List[DraftMolecule],
    quantile_level: float = FLAG_QUANTILE,
    t_abs: Dict[str, float] = None,
) -> Tuple[List[DraftMolecule], Dict[str, Any]]:
    """
    Pass 2: Apply flags based on batch distribution statistics.
    
    Threshold model (3AI approved, Phase 9 ThresholdResolver pattern):
      threshold = max(t_abs, quantile(distribution, quantile_level))
    
    Args:
        molecules: List of DraftMolecule from Pass 1 (no flags)
        quantile_level: Percentile for flagging (default 0.85 = upper 15%)
        t_abs: Safety net minimums (defaults to T_ABS)
    
    Returns:
        (flagged_molecules, threshold_report)
        - flagged_molecules: same list with flags applied
        - threshold_report: computed thresholds + distribution stats for audit log
    """
    if t_abs is None:
        t_abs = T_ABS
    
    if not molecules:
        return molecules, {"error": "no molecules to flag"}
    
    # ── Collect distributions ──
    atom_counts = [m.atom_count for m in molecules]
    operator_counts = [m.operator_count for m in molecules]
    confidence_gaps = [m.confidence_gap for m in molecules]
    category_counts = [m.category_count for m in molecules]
    
    # ── Compute quantile thresholds ──
    q_atom = _quantile(atom_counts, quantile_level)
    q_operator = _quantile(operator_counts, quantile_level)
    q_gap = _quantile(confidence_gaps, quantile_level)
    q_category = _quantile(category_counts, quantile_level)
    
    # ── Apply max(t_abs, quantile) ──
    t_atom = max(t_abs.get("atom_count", 3), q_atom)
    t_operator = max(t_abs.get("operator_count", 2), q_operator)
    t_gap = max(t_abs.get("confidence_gap", 0.2), q_gap)
    t_category = max(t_abs.get("category_count", 3), q_category)
    
    # ── Apply flags ──
    flagged_count = 0
    flag_counts = defaultdict(int)
    
    for m in molecules:
        flags = []
        
        if m.atom_count > t_atom:
            flags.append("COMPLEX_MOLECULE")
        
        if m.operator_count > t_operator:
            flags.append("COMPLEX_FORMULA")
        
        if m.confidence_gap > t_gap:
            flags.append("CONFIDENCE_GAP")
        
        if m.category_count > t_category:
            flags.append("CROSS_CATEGORY")
        
        m.flags = flags
        
        if flags:
            flagged_count += 1
            for f in flags:
                flag_counts[f] += 1
    
    # ── Build threshold report (for audit trail) ──
    n = len(molecules)
    report = {
        "batch_size": n,
        "quantile_level": quantile_level,
        "distributions": {
            "atom_count": {
                "mean": sum(atom_counts) / n,
                "min": min(atom_counts),
                "max": max(atom_counts),
                "quantile": round(q_atom, 2),
            },
            "operator_count": {
                "mean": sum(operator_counts) / n,
                "min": min(operator_counts),
                "max": max(operator_counts),
                "quantile": round(q_operator, 2),
            },
            "confidence_gap": {
                "mean": sum(confidence_gaps) / n,
                "min": round(min(confidence_gaps), 4),
                "max": round(max(confidence_gaps), 4),
                "quantile": round(q_gap, 4),
            },
            "category_count": {
                "mean": sum(category_counts) / n,
                "min": min(category_counts),
                "max": max(category_counts),
                "quantile": round(q_category, 2),
            },
        },
        "resolved_thresholds": {
            "atom_count": {"t_abs": t_abs.get("atom_count", 3), "quantile": round(q_atom, 2), "resolved": round(t_atom, 2)},
            "operator_count": {"t_abs": t_abs.get("operator_count", 2), "quantile": round(q_operator, 2), "resolved": round(t_operator, 2)},
            "confidence_gap": {"t_abs": t_abs.get("confidence_gap", 0.2), "quantile": round(q_gap, 4), "resolved": round(t_gap, 4)},
            "category_count": {"t_abs": t_abs.get("category_count", 3), "quantile": round(q_category, 2), "resolved": round(t_category, 2)},
        },
        "results": {
            "flagged": flagged_count,
            "total": n,
            "audit_rate": round(flagged_count / n, 4),
            "flag_counts": dict(flag_counts),
        },
        "nlp_reference_ranges": NLP_REFERENCE_RANGES,
    }
    
    return molecules, report


# ============================================================
# Batch Runner (2-Pass)
# ============================================================

def run_batch(
    sentences: List[str],
    generator: RuleGenerator,
    article_id: str = "unknown",
) -> Tuple[List[Dict], GenerationStats, Dict[str, Any]]:
    """
    Run rule generator on a list of sentences using 2-pass architecture.
    
    Pass 1: Generate all molecules (no flags)
    Pass 2: Compute dynamic thresholds → apply flags
    
    Returns:
        (molecules_dicts, stats, threshold_report)
    """
    # ── Pass 1: Generate ──
    raw_molecules = []
    
    for idx, sentence in enumerate(sentences):
        result = generator.generate(sentence)
        if result:
            result._sentence_idx = idx  # Stash for later
            raw_molecules.append(result)
    
    # ── Pass 2: Dynamic flags ──
    flagged, threshold_report = apply_dynamic_flags(raw_molecules)
    
    # Update generator stats with flag info
    for m in flagged:
        if m.needs_audit:
            generator.stats.flagged_for_audit += 1
            for f in m.flags:
                generator.stats.flag_counts[f] += 1
    
    # Convert to dicts
    molecules_out = []
    for m in flagged:
        d = m.to_dict()
        d["article_id"] = article_id
        d["sentence_idx"] = getattr(m, '_sentence_idx', -1)
        molecules_out.append(d)
    
    return molecules_out, generator.stats, threshold_report