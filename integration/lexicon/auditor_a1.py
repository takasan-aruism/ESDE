#!/usr/bin/env python3
"""
ESDE Lexicon v2 — A1 Auditor (Structural Self-Audit)
=====================================================
Runs AFTER A1 Mapper. Performs structural (not semantic) checks
on raw_scores using QwQ as a verification engine.

Design rationale:
  - Writer (mapper_a1) = generative task → "score each slot"
  - Auditor (this) = verification task → "check these specific conditions"
  - Same model, different task type → different bias direction
  - Auditor ONLY detects anomalies, NEVER fixes them
  - "describe but do not decide" — Auditor flags, Writer re-observes

Key insight (from pilot):
  - Detection works: same model CAN identify structural anomalies
  - Correction fails: same model re-produces same semantic bias
  - Solution: detect-only → re-observe with quantitative constraints

5 Structural Checks:
  C1. Distribution anomaly   — all-zero, all-high, inflation
  C2. Symmetric pair leak    — antonym words should score low except destructive
  C3. Evidence-score mismatch — evidence text vs actual high slots
  C4. Axis-generic inflation — entire axis uniformly high = generic, not atom-specific
  C5. POS coherence          — adj/adv scoring high on material/individual is suspicious

Architecture:
  JSONL (mapper output) → code pre-screen → LLM audit (flagged only)
    → PASS: keep original
    → REVISE: generate constraints → Writer re-observes

Usage:
  # Audit (detect only)
  python3 auditor_a1.py --input mapper_output/EMO_like_a1.jsonl --dictionary esde_dictionary.json

  # Audit + trigger re-observation for REVISE records
  python3 auditor_a1.py --input mapper_output/EMO_like_a1.jsonl --dictionary esde_dictionary.json --re-observe
"""

import json
import re
import math
import argparse
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor

from mapper_a1 import (
    SLOT_IDS, AXES, LLM_HOST, LLM_MODEL, LLM_TIMEOUT, LLM_TEMPERATURE
)


# ============================================================
# Configuration
# ============================================================

# Thresholds for code-side pre-screening
INFLATION_SUM_THRESHOLD = 150       # Total raw score sum — above this is suspicious
INFLATION_NONZERO_THRESHOLD = 40    # Out of 48 — too many nonzero
DEFLATION_NONZERO_THRESHOLD = 5     # Too few nonzero
HIGH_SCORE_CEILING = 8              # Slots >= this considered "high"
MAX_HIGH_SLOTS = 15                 # More than this = inflation
AXIS_MEAN_THRESHOLD = 4.0           # Average score per axis — above = generic
ANTONYM_NON_DESTRUCTIVE_MAX = 3     # Antonym word: non-destructive slots should be ≤ this

LLM_AUDIT_MAX_TOKENS = 4000
LLM_AUDIT_TEMPERATURE = 0.2        # Lower than writer — want deterministic checks
MAX_LLM_RETRIES = 2


# ============================================================
# Antonym Registry
# ============================================================

def build_antonym_set(dictionary: Dict, atom_id: str) -> set:
    """
    Build set of words that are antonyms (from symmetric pair).
    For EMO.like, these are words like 'dislike', 'aversion', etc.
    """
    atom_entry = dictionary.get(atom_id, {})
    anti_triggers = set(t.lower() for t in atom_entry.get("anti_triggers_en", []))
    return anti_triggers


# ============================================================
# Code-Side Pre-Screening (no LLM needed)
# ============================================================

@dataclass
class PreScreenResult:
    """Result of code-side structural pre-screening."""
    word: str
    pos: str
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    needs_llm_audit: bool = False
    
    def add_flag(self, flag_id: str, detail: str, severity: str = "warn"):
        self.flags.append(flag_id)
        self.details[flag_id] = {"detail": detail, "severity": severity}
        if severity in ("warn", "error"):
            self.needs_llm_audit = True


def pre_screen(record: Dict, antonym_words: set) -> PreScreenResult:
    """
    Code-side structural pre-screening. Fast, deterministic, no LLM.
    Returns flags for records that need LLM audit.
    """
    word = record["word"]
    pos = record["pos"]
    raw = record.get("raw_scores", {})
    evidence = record.get("evidence", "")
    
    result = PreScreenResult(word=word, pos=pos)
    
    scores = [raw.get(s, 0) for s in SLOT_IDS]
    
    # ── C1: Distribution Anomaly ──
    total_sum = sum(scores)
    nonzero_count = sum(1 for v in scores if v > 0)
    high_count = sum(1 for v in scores if v >= HIGH_SCORE_CEILING)
    max_score = max(scores) if scores else 0
    
    if nonzero_count <= DEFLATION_NONZERO_THRESHOLD:
        result.add_flag("C1_deflation",
            f"Only {nonzero_count}/48 nonzero slots — possible under-observation",
            severity="warn")
    
    if total_sum >= INFLATION_SUM_THRESHOLD:
        result.add_flag("C1_inflation_sum",
            f"Total sum={total_sum} exceeds {INFLATION_SUM_THRESHOLD} — score inflation",
            severity="warn")
    
    if nonzero_count >= INFLATION_NONZERO_THRESHOLD:
        result.add_flag("C1_inflation_spread",
            f"{nonzero_count}/48 nonzero — too spread, not discriminating",
            severity="info")
    
    if high_count >= MAX_HIGH_SLOTS:
        result.add_flag("C1_inflation_high",
            f"{high_count} slots >= {HIGH_SCORE_CEILING} — too many high scores",
            severity="warn")
    
    if max_score == 0:
        result.add_flag("C1_all_zero",
            "All slots are 0 — complete non-observation",
            severity="error")
    
    # ── C2: Symmetric Pair Leak ──
    if word.lower() in antonym_words:
        destructive_score = raw.get("symmetry.destructive", 0)
        non_destructive_high = []
        for s in SLOT_IDS:
            if s == "symmetry.destructive":
                continue
            v = raw.get(s, 0)
            if v > ANTONYM_NON_DESTRUCTIVE_MAX:
                non_destructive_high.append((s, v))
        
        if non_destructive_high:
            slots_str = ", ".join(f"{s}={v}" for s, v in non_destructive_high[:5])
            result.add_flag("C2_antonym_leak",
                f"Antonym word '{word}' has high non-destructive scores: {slots_str}. "
                f"destructive={destructive_score}",
                severity="warn")
        
        if destructive_score < 3:
            result.add_flag("C2_antonym_low_destructive",
                f"Antonym word '{word}' has symmetry.destructive={destructive_score} "
                f"(expected >= 3 for antonym)",
                severity="info")
    
    # ── C3: Evidence-Score Mismatch (lightweight keyword check) ──
    if evidence:
        evidence_lower = evidence.lower()
        top5_slots = sorted(raw.items(), key=lambda x: -x[1])[:5]
        top1_axis = top5_slots[0][0].split(".")[0] if top5_slots else ""
        top1_level = top5_slots[0][0].split(".")[1] if top5_slots else ""
        
        # Check if evidence mentions ANY of the top 3 axes/levels
        top3_keywords = set()
        for slot, _ in top5_slots[:3]:
            axis, level = slot.split(".")
            top3_keywords.add(axis)
            top3_keywords.add(level)
        
        evidence_mentions_top = any(kw in evidence_lower for kw in top3_keywords)
        
        if not evidence_mentions_top:
            result.add_flag("C3_evidence_mismatch",
                f"Evidence text doesn't mention any top-3 axis/level keywords. "
                f"Top3: {[s for s,_ in top5_slots[:3]]}",
                severity="info")
    
    # ── C4: Axis-Generic Inflation ──
    for axis_id, axis_def in AXES.items():
        levels = list(axis_def["levels"].keys())
        axis_scores = [raw.get(f"{axis_id}.{l}", 0) for l in levels]
        axis_mean = sum(axis_scores) / len(axis_scores) if axis_scores else 0
        
        # Check if axis is uniformly high (generic) vs peaked (specific)
        if axis_mean >= AXIS_MEAN_THRESHOLD:
            axis_max = max(axis_scores)
            axis_min = min(axis_scores)
            spread = axis_max - axis_min
            
            if spread <= 2:
                # Uniform high = generic
                result.add_flag(f"C4_generic_{axis_id}",
                    f"Axis '{axis_id}' uniformly high: mean={axis_mean:.1f}, "
                    f"spread={spread}, scores={axis_scores}",
                    severity="warn")
            # If spread > 2, the axis is peaked — OK even if mean is high
    
    # ── C5: POS Coherence ──
    if pos in ("adj", "adv"):
        material_score = raw.get("ontological.material", 0)
        if material_score >= 6:
            result.add_flag("C5_pos_material",
                f"POS={pos} but ontological.material={material_score} "
                f"— adj/adv rarely strongly material",
                severity="info")
    
    if pos == "n":
        # Nouns scoring very high on experience axes can be suspicious
        creation_score = raw.get("experience.creation", 0)
        if creation_score >= 7 and "create" not in word.lower() and "creation" not in word.lower():
            result.add_flag("C5_pos_creation",
                f"POS=n but experience.creation={creation_score} "
                f"— noun '{word}' strongly creative?",
                severity="info")
    
    return result


# ============================================================
# LLM Audit Prompt
# ============================================================

def build_audit_prompt(record: Dict, pre_flags: List[str],
                       flag_details: Dict, atom_def: str) -> str:
    """
    Build a focused structural audit prompt for QwQ.
    
    Key: This is a VERIFICATION task, not a generation task.
    The auditor checks specific conditions — it does NOT re-score.
    """
    word = record["word"]
    pos = record["pos"]
    atom = record["atom"]
    raw = record.get("raw_scores", {})
    evidence = record.get("evidence", "")
    
    # Format scores compactly (only non-zero)
    nonzero_scores = {s: v for s, v in raw.items() if v > 0}
    scores_str = json.dumps(nonzero_scores, indent=None)
    
    # Format flags
    flags_str = "\n".join(
        f"  - [{fid}] {flag_details[fid]['detail']}"
        for fid in pre_flags
    )
    
    return f"""You are a structural auditor for ESDE semantic coordinates.

## Context
- Word: "{word}" (POS: {pos})
- Atom: {atom} — "{atom_def}"
- Writer's evidence: "{evidence}"

## Writer's non-zero scores (out of 48 slots)
{scores_str}

## Pre-screening flags detected
{flags_str}

## Your Task
For EACH flag listed above, provide a verdict: CONFIRM (the flag is a real problem) or DISMISS (the flag is a false alarm, the scores are acceptable).

## RULES
1. You are a STRUCTURAL auditor. Check logical consistency, not semantic depth.
2. For C1 flags: Is the distribution shape reasonable for this word+atom combination?
3. For C2 flags: Should this antonym word really score high outside symmetry.destructive?
4. For C3 flags: Does the evidence text actually support the highest-scoring slots?
5. For C4 flags: Is the axis score pattern genuinely uniform-generic, or reasonably peaked?
6. For C5 flags: Is the POS-score combination genuinely suspicious?
7. Do NOT suggest specific score changes. You detect problems, you do not fix them.
8. Do NOT re-score any slots. Only judge whether each flag is valid.

## Output Format
Return ONLY a JSON object:
```json
{{
  "word": "{word}",
  "verdicts": {{
    "<flag_id>": {{
      "verdict": "CONFIRM" or "DISMISS",
      "reason": "1 sentence"
    }}
  }},
  "overall": "PASS" or "REVISE",
  "revision_note": "1 sentence describing the structural problem if REVISE, else null"
}}
```
"""


# ============================================================
# LLM Audit Call
# ============================================================

def call_qwq_audit(prompt: str) -> Tuple[str, float]:
    """Call QwQ for audit verdict."""
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": LLM_AUDIT_MAX_TOKENS,
        "temperature": LLM_AUDIT_TEMPERATURE,
        "stream": False,
    }
    
    t0 = time.time()
    resp = requests.post(
        f"{LLM_HOST}/chat/completions",
        json=payload,
        timeout=LLM_TIMEOUT,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    return text, elapsed


def parse_audit_response(text: str) -> dict:
    """Parse audit response JSON."""
    # Strip <think>...</think> blocks (QwQ style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip everything before </think> (Qwen3 style)
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    
    # Try markdown fence
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    
    # Try raw JSON
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end+1])
    
    raise ValueError(f"No JSON found in audit response ({len(text)} chars)")


# ============================================================
# Audit Pipeline
# ============================================================

@dataclass
class AuditRecord:
    """Complete audit result for one word."""
    word: str
    pos: str
    atom: str
    pre_screen_flags: List[str]
    pre_screen_details: Dict[str, Any]
    llm_audit: Optional[Dict] = None
    final_status: str = "PASS"  # PASS / REVISE / SKIP
    audit_elapsed_sec: float = 0.0
    timestamp: str = ""


def audit_single(record: Dict, antonym_words: set, atom_def: str,
                 dry_run: bool = False) -> AuditRecord:
    """
    Full audit pipeline for one word.
    
    1. Code pre-screen → flags
    2. If flagged → LLM audit → verdicts
    3. Aggregate → final status
    """
    word = record["word"]
    pos = record["pos"]
    atom = record.get("atom", "?")
    
    # Skip failed observations
    if record.get("status") in ("Observation_Failed", "DRY_RUN"):
        return AuditRecord(
            word=word, pos=pos, atom=atom,
            pre_screen_flags=["SKIP_FAILED"],
            pre_screen_details={},
            final_status="SKIP",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    # Step 1: Code pre-screen
    ps = pre_screen(record, antonym_words)
    
    audit = AuditRecord(
        word=word, pos=pos, atom=atom,
        pre_screen_flags=ps.flags,
        pre_screen_details=ps.details,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    if not ps.flags:
        # No flags → auto-pass
        audit.final_status = "PASS"
        return audit
    
    if not ps.needs_llm_audit:
        # Only info-level flags → pass with notes
        audit.final_status = "PASS"
        return audit
    
    if dry_run:
        prompt = build_audit_prompt(record, ps.flags, ps.details, atom_def)
        audit.final_status = "DRY_RUN"
        audit.llm_audit = {"prompt_length": len(prompt)}
        return audit
    
    # Step 2: LLM audit
    prompt = build_audit_prompt(record, ps.flags, ps.details, atom_def)
    
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            resp_text, elapsed = call_qwq_audit(prompt)
            parsed = parse_audit_response(resp_text)
            audit.audit_elapsed_sec = round(elapsed, 1)
            audit.llm_audit = parsed
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == MAX_LLM_RETRIES:
                audit.llm_audit = {"error": str(e)}
                audit.final_status = "REVISE"  # uncertain → REVISE に倒す
                return audit
        except requests.RequestException as e:
            audit.llm_audit = {"error": f"Connection: {e}"}
            audit.final_status = "REVISE"  # uncertain → REVISE に倒す
            return audit
    
    # Step 3: Aggregate verdicts
    # Policy: only explicit DISMISS clears a flag. Everything else (CONFIRM, UNCERTAIN, 
    # missing, typo) is treated as CONFIRM → REVISE に倒す (safe side).
    verdicts = parsed.get("verdicts", {})
    confirmed = [fid for fid, v in verdicts.items()
                 if v.get("verdict", "").upper() != "DISMISS"]
    
    if confirmed:
        audit.final_status = parsed.get("overall", "REVISE")
    else:
        audit.final_status = "PASS"
    
    return audit


# ============================================================
# Conditional Re-Observation (3 conditions + intersection)
# ============================================================
#
# Strategy (GPT proposal + Taka/Claude refinement):
#   1. Run 3 condition-varied observations per word
#      - Conditions derived from atom structure, not arbitrary
#   2. Intersection filter: keep only slots stable across ≥2/3 conditions
#   3. Budget normalization: enforce sum/nonzero hard limits (code-side)
#
# Why this works:
#   - Inflation = "all slots get a little score when context is vague"
#   - 3 different contexts → only truly essential slots survive all 3
#   - Evidence mandatory → blocks "vaguely related" scores
#   - Intersection + budget = mechanical, no LLM semantic judgment
#

# Hard constraints
RE_OBSERVE_MAX_SUM = 120
RE_OBSERVE_MAX_NONZERO = 25
RE_OBSERVE_FLOOR_CUT = 1.5       # Scores ≤ this → 0 after intersection
RE_OBSERVE_STABILITY_MIN = 2     # Must appear in ≥ N/3 conditions
RE_OBSERVE_SCORE_RANGE = (0, 10) # Per-slot score range

# Min mass guard: top K slots must sum to at least this
# Prevents "too thin" results where intersection killed everything
MIN_MASS_K = 3
MIN_MASS_THRESHOLD = 12


def build_condition_prompts(word: str, pos: str, atom_id: str, atom_def: str,
                             sym_pair: str, sym_def: str, category: str) -> List[Dict]:
    """
    Build 3 condition-varied observation prompts from atom structure.
    
    Conditions:
      A. Typical — most common usage in the sense of [atom]
      B. Boundary — near the symmetric pair, where [atom] vs [sym_pair] is ambiguous
      C. Specific — a narrow, concrete scenario where the word clearly means [atom]
    
    These are NOT arbitrary (legal/romantic/etc) — they are derived from
    the atom's structural position (definition + symmetric pair).
    """
    from mapper_a1 import COMPRESSED_SLOTS, SLOT_IDS
    
    conditions = [
        {
            "id": "typical",
            "label": "Typical usage",
            "instruction": (
                f'Imagine the word "{word}" used in its most COMMON, EVERYDAY sense '
                f'that relates to [{atom_id}] ("{atom_def}"). '
                f'Think of a typical sentence someone would write using "{word}" in this sense.'
            ),
        },
        {
            "id": "boundary",
            "label": "Boundary with symmetric pair",
            "instruction": (
                f'Imagine "{word}" used in a context where the meaning is CLOSE TO THE BOUNDARY '
                f'between [{atom_id}] ("{atom_def}") and [{sym_pair}] ("{sym_def}"). '
                f'The word still means [{atom_id}], but the opposite pulls slightly. '
                f'Which slots are STILL clearly active even at this boundary?'
            ),
        },
        {
            "id": "specific",
            "label": "Most specific/narrow sense",
            "instruction": (
                f'Imagine "{word}" used in its MOST SPECIFIC, CONCRETE sense of [{atom_id}]. '
                f'Pick the narrowest real-world scenario where "{word}" clearly and '
                f'unambiguously means [{atom_id}]. Which slots are essential in this narrow case?'
            ),
        },
    ]
    
    prompts = []
    for cond in conditions:
        prompt = f"""You are a semantic coordinate observer for ESDE.

## Task
Observe the word "{word}" (POS: {pos}) under a SPECIFIC CONDITION.

## Atom Context
- Atom: {atom_id} ({category})
- Definition: {atom_def}
- Symmetric pair: {sym_pair} ("{sym_def}")

## Condition: {cond['label']}
{cond['instruction']}

## 48 Semantic Slots
{COMPRESSED_SLOTS}

## RULES
1. Score ONLY slots with CLEAR, DIRECT resonance under this condition. Score 0-10 (integer).
2. MOST slots should be 0. Be strict: if you cannot name a specific reason, score 0.
3. For EVERY nonzero slot, you MUST provide 1-3 evidence words/phrases that justify it.
   If you cannot provide evidence for a slot, it MUST be 0.
4. Maximum 20 nonzero slots. If you have more, drop the weakest ones to 0.
5. Do NOT spread small scores across many slots. Prefer fewer slots with higher confidence.

## Output Format
Return ONLY a JSON object:
```json
{{
  "condition": "{cond['id']}",
  "slots": {{
    "<slot_id>": {{"score": <int>, "evidence": ["word1", "phrase2"]}},
    ...
  }},
  "context_sentence": "A short example sentence showing this word in this condition."
}}
```
Only include slots with score > 0. Respond with ONLY the JSON."""
        
        prompts.append({
            "condition_id": cond["id"],
            "label": cond["label"],
            "prompt": prompt,
        })
    
    return prompts


def parse_condition_response(text: str) -> Dict:
    """Parse a condition observation response."""
    # Strip <think>...</think> blocks (QwQ style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Strip everything before </think> (Qwen3 style)
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end+1])
    
    raise ValueError(f"No JSON in condition response ({len(text)} chars)")


def intersect_conditions(condition_results: List[Dict],
                         stability_min: int = RE_OBSERVE_STABILITY_MIN) -> Dict:
    """
    Intersection filter: keep only slots stable across conditions.
    
    Algorithm:
      1. Count how many conditions each slot appears in (score > 0)
      2. Keep slots appearing in >= stability_min conditions
      3. Score = average of nonzero scores across conditions
      4. Collect all evidence tokens
    
    This is pure code — no LLM judgment.
    """
    from mapper_a1 import SLOT_IDS
    
    # Count appearances and collect scores/evidence per slot
    slot_data = {}
    for s in SLOT_IDS:
        slot_data[s] = {"scores": [], "evidence": [], "conditions": []}
    
    for result in condition_results:
        cond_id = result.get("condition", "?")
        slots = result.get("slots", {})
        
        for slot_id, data in slots.items():
            if slot_id not in slot_data:
                continue
            score = data.get("score", 0) if isinstance(data, dict) else 0
            evidence = data.get("evidence", []) if isinstance(data, dict) else []
            
            if score > 0:
                slot_data[slot_id]["scores"].append(score)
                slot_data[slot_id]["conditions"].append(cond_id)
                if isinstance(evidence, list):
                    slot_data[slot_id]["evidence"].extend(evidence)
    
    # Intersection: keep slots appearing in >= stability_min conditions
    stable_slots = {}
    dropped_slots = {}
    
    for s in SLOT_IDS:
        n_appearances = len(slot_data[s]["scores"])
        
        if n_appearances >= stability_min:
            avg_score = sum(slot_data[s]["scores"]) / n_appearances
            stable_slots[s] = {
                "score": round(avg_score, 1),
                "appearances": n_appearances,
                "conditions": slot_data[s]["conditions"],
                "evidence": list(set(slot_data[s]["evidence"])),  # dedupe
            }
        elif n_appearances > 0:
            dropped_slots[s] = {
                "score": sum(slot_data[s]["scores"]) / n_appearances,
                "appearances": n_appearances,
                "reason": f"unstable: appeared in {n_appearances}/{len(condition_results)} conditions",
            }
    
    return {
        "stable": stable_slots,
        "dropped": dropped_slots,
        "total_conditions": len(condition_results),
    }


def budget_normalize(stable_slots: Dict, max_sum: float = RE_OBSERVE_MAX_SUM,
                     max_nonzero: int = RE_OBSERVE_MAX_NONZERO,
                     floor_cut: float = RE_OBSERVE_FLOOR_CUT) -> Dict[str, float]:
    """
    Budget normalization: enforce hard constraints on intersection result.
    
    Steps:
      1. Floor cut: scores <= floor_cut → 0
      2. If nonzero > max_nonzero: keep top N only
      3. If sum > max_sum: proportionally scale down
    
    Pure arithmetic — no semantic judgment.
    """
    from mapper_a1 import SLOT_IDS
    
    # Start with stable scores
    scores = {}
    for s in SLOT_IDS:
        if s in stable_slots:
            scores[s] = stable_slots[s]["score"]
        else:
            scores[s] = 0.0
    
    # Step 1: Floor cut
    for s in SLOT_IDS:
        if scores[s] <= floor_cut:
            scores[s] = 0.0
    
    # Step 2: If too many nonzero, keep top N
    nonzero_slots = [(s, v) for s, v in scores.items() if v > 0]
    if len(nonzero_slots) > max_nonzero:
        nonzero_slots.sort(key=lambda x: -x[1])
        keep = set(s for s, _ in nonzero_slots[:max_nonzero])
        for s in SLOT_IDS:
            if s not in keep:
                scores[s] = 0.0
    
    # Step 3: If sum too high, proportionally scale
    current_sum = sum(v for v in scores.values() if v > 0)
    if current_sum > max_sum:
        scale = max_sum / current_sum
        for s in SLOT_IDS:
            if scores[s] > 0:
                scores[s] = round(scores[s] * scale, 1)
                # Re-floor after scaling
                if scores[s] <= 1.0:
                    scores[s] = 0.0
    
    # Round to integers for consistency with original mapper
    for s in SLOT_IDS:
        scores[s] = round(scores[s])
    
    return scores


def check_min_mass(scores: Dict[str, float],
                   stable_count: int = 0,
                   evidence_count: int = 0,
                   k: int = MIN_MASS_K,
                   threshold: float = MIN_MASS_THRESHOLD) -> Dict:
    """
    Min mass guard: verify top K slots have sufficient total mass.
    
    Prevents "too thin" results where intersection + budget killed
    too aggressively, leaving a near-empty coordinate.
    
    Reason codes (when fail):
      - "top_k_sum_low"    : top K slots sum < threshold
      - "stable_slots_low" : fewer than K stable slots survived intersection
      - "evidence_sparse"  : stable slots exist but evidence coverage is thin
    
    Returns:
      {"pass": bool, "top_k_sum": float, "threshold": float,
       "top_k_slots": [...], "reason": str or null}
    """
    nonzero = [(s, v) for s, v in scores.items() if v > 0]
    nonzero.sort(key=lambda x: -x[1])
    top_k = nonzero[:k]
    top_k_sum = sum(v for _, v in top_k)
    
    passed = top_k_sum >= threshold
    reason = None
    
    if not passed:
        if len(nonzero) < k:
            reason = "stable_slots_low"
        elif evidence_count > 0 and evidence_count < len(nonzero):
            reason = "evidence_sparse"
        else:
            reason = "top_k_sum_low"
    
    return {
        "pass": passed,
        "top_k_sum": top_k_sum,
        "threshold": threshold,
        "k": k,
        "top_k_slots": [(s, v) for s, v in top_k],
        "reason": reason,
    }


def re_observe_word(record: Dict, audit_record: AuditRecord,
                    dictionary: Dict, dry_run: bool = False) -> Optional[Dict]:
    """
    Re-observe a word using 3-condition strategy + intersection.
    
    Pipeline:
      1. Generate 3 condition prompts (from atom structure)
      2. Call QwQ 3 times (different conditions)
      3. Intersect: keep slots stable across ≥2/3 conditions
      4. Budget normalize: enforce sum/nonzero limits
      5. Recompute stats (softmax, entropy, focus_rate)
    """
    from mapper_a1 import (
        call_qwq, process_raw_scores, SLOT_IDS
    )
    
    word = record["word"]
    pos = record["pos"]
    atom_id = record.get("atom", "?")
    
    atom_entry = dictionary.get(atom_id, {})
    atom_def = atom_entry.get("definition_en", "")
    sym_pair = atom_entry.get("symmetric_pair", "")
    sym_entry = dictionary.get(sym_pair, {})
    sym_def = sym_entry.get("definition_en", "")
    category = atom_entry.get("category", "")
    
    # Step 1: Build 3 condition prompts
    cond_prompts = build_condition_prompts(
        word, pos, atom_id, atom_def, sym_pair, sym_def, category
    )
    
    if dry_run:
        return {"dry_run": True, "n_prompts": len(cond_prompts),
                "prompt_lengths": [len(p["prompt"]) for p in cond_prompts]}
    
    # Step 2: Call QwQ for each condition
    condition_results = []
    total_elapsed = 0.0
    
    for cp in cond_prompts:
        cid = cp["condition_id"]
        try:
            resp_text, elapsed = call_qwq(cp["prompt"])
            total_elapsed += elapsed
            parsed = parse_condition_response(resp_text)
            parsed["condition"] = cid
            condition_results.append(parsed)
        except (json.JSONDecodeError, ValueError, requests.RequestException) as e:
            print(f"\n    ⚠ condition '{cid}' failed: {e}", end="")
            continue
    
    if len(condition_results) < 2:
        print(f"\n    ❌ Only {len(condition_results)}/3 conditions succeeded", end="")
        return None
    
    # Step 3: Intersection filter
    intersection = intersect_conditions(condition_results)
    
    # Step 4: Budget normalization
    final_scores = budget_normalize(intersection["stable"])
    
    # Step 5: Min mass guard
    evidence_slots = {s for s, data in intersection["stable"].items()
                      if s in final_scores and final_scores[s] > 0 and data.get("evidence")}
    mass_check = check_min_mass(
        final_scores,
        stable_count=len(intersection["stable"]),
        evidence_count=len(evidence_slots),
    )
    
    # Step 6: Recompute stats
    stats = process_raw_scores(final_scores)
    
    # Build result record
    new_record = dict(record)
    new_record["raw_scores"] = stats["raw_scores"]
    new_record["normalized_scores"] = stats["normalized_scores"]
    new_record["entropy_norm"] = stats["entropy_norm"]
    new_record["focus_rate"] = stats["focus_rate"]
    new_record["status"] = stats["status"]
    new_record["top5"] = stats["top5"]
    new_record["llm_elapsed_sec"] = round(total_elapsed, 1)
    new_record["re_observed"] = True
    new_record["re_observe_method"] = "3condition_intersection"
    new_record["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Audit trace
    new_record["re_observe_trace"] = {
        "conditions_attempted": len(cond_prompts),
        "conditions_succeeded": len(condition_results),
        "stable_slots": len(intersection["stable"]),
        "dropped_slots": len(intersection["dropped"]),
        "final_nonzero": sum(1 for v in final_scores.values() if v > 0),
        "final_sum": sum(final_scores.values()),
        "min_mass_guard": mass_check,
        "evidence": {
            s: data["evidence"]
            for s, data in intersection["stable"].items()
            if s in final_scores and final_scores[s] > 0
        },
        "dropped": {
            s: data["reason"]
            for s, data in intersection["dropped"].items()
        },
    }
    
    # Evidence summary
    evidence_parts = []
    for s, data in sorted(intersection["stable"].items(),
                          key=lambda x: -x[1]["score"]):
        if s in final_scores and final_scores[s] > 0:
            ev = ", ".join(data["evidence"][:3])
            evidence_parts.append(f"{s}({final_scores[s]}): {ev}")
    new_record["evidence"] = "; ".join(evidence_parts[:10])
    
    return new_record


# ============================================================
# Batch Pipeline
# ============================================================

def run_audit(input_path: str, dictionary: Dict, out_dir: str,
              dry_run: bool = False, re_observe: bool = False,
              parallel: int = 1, parallel_reobs: int = None) -> List[AuditRecord]:
    """
    Run structural audit on all records in a JSONL file.
    
    Pipeline:
      1. Pre-screen all records (code, fast)
      2. LLM audit flagged records (QwQ)
      3. If re_observe: re-run Writer on REVISE records with constraints
    """
    # Load records
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    if not records:
        print("  ⚠ No records to audit")
        return []
    
    atom_id = records[0].get("atom", "?")
    atom_entry = dictionary.get(atom_id, {})
    atom_def = atom_entry.get("definition_en", "")
    antonym_words = build_antonym_set(dictionary, atom_id)
    
    # Also add known antonym words from symmetric pair
    sym_pair = atom_entry.get("symmetric_pair", "")
    if sym_pair:
        sym_entry = dictionary.get(sym_pair, {})
        for t in sym_entry.get("triggers_en", []):
            antonym_words.add(t.lower())
    
    print(f"\n{'='*60}")
    print(f"  A1 AUDITOR: {atom_id}")
    print(f"  Records: {len(records)}")
    print(f"  Antonym words: {antonym_words or '(none detected)'}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Re-observe: {re_observe}")
    if parallel > 1:
        print(f"  Parallel: {parallel}")
    print(f"{'='*60}")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    audit_jsonl = out_path / f"{atom_id.replace('.','_')}_audit.jsonl"
    final_jsonl = out_path / f"{atom_id.replace('.','_')}_a1_final.jsonl"
    
    # Clear output files
    if not dry_run:
        for f in [audit_jsonl, final_jsonl]:
            if f.exists():
                f.unlink()
    
    results = []
    pass_count = 0
    revise_count = 0
    skip_count = 0
    failed_count = 0
    reobs_count = 0
    pass_diffuse = []  # PASS words that are Diffuse (potential REVISE candidates)
    
    # === Phase 1: Audit ===
    print(f"\n  ── Phase 1: Audit ──")
    
    def _audit_one(i_record):
        i, record = i_record
        return audit_single(record, antonym_words, atom_def, dry_run=dry_run)
    
    def _print_audit(i, record, audit):
        word = record.get("word", "?")
        pos = record.get("pos", "?")
        status = audit.final_status
        n_flags = len(audit.pre_screen_flags)
        prefix = f"  [{i+1:3d}/{len(records)}] {word:20s} ({pos}) "
        
        if status == "PASS" and n_flags == 0:
            print(f"{prefix}→ ✅ PASS (clean)")
        elif status == "PASS" and n_flags > 0:
            print(f"{prefix}→ ✅ PASS ({n_flags} flags dismissed)")
        elif status == "REVISE":
            confirmed = [f for f in audit.pre_screen_flags
                        if audit.llm_audit and isinstance(audit.llm_audit, dict) and
                        audit.llm_audit.get("verdicts", {}).get(f, {}).get("verdict", "").upper() == "CONFIRM"]
            print(f"{prefix}→ ⚠️  REVISE ({len(confirmed)} confirmed: {confirmed})")
        elif status == "SKIP":
            print(f"{prefix}→ ⏭️  SKIP")
        elif status == "AUDIT_FAILED":
            print(f"{prefix}→ ❌ AUDIT FAILED")
        elif status == "DRY_RUN":
            print(f"{prefix}→ [dry run, {n_flags} flags]")
    
    indexed = list(enumerate(records))
    
    if parallel <= 1:
        # Sequential
        for i, record in indexed:
            audit = _audit_one((i, record))
            results.append(audit)
            _print_audit(i, record, audit)
            status = audit.final_status
            if status == "PASS": pass_count += 1
            elif status == "REVISE": revise_count += 1
            elif status == "SKIP": skip_count += 1
            elif status == "AUDIT_FAILED": failed_count += 1
            if status == "PASS" and record.get("status") == "Diffuse_Observation":
                pass_diffuse.append({"word": record.get("word","?"), "pos": record.get("pos","?"),
                                     "focus_rate": record.get("focus_rate", 0)})
            if not dry_run:
                with open(audit_jsonl, 'a') as f:
                    f.write(json.dumps(asdict(audit), ensure_ascii=False) + "\n")
    else:
        # Parallel: process in chunks
        for batch_start in range(0, len(indexed), parallel):
            batch = indexed[batch_start:batch_start + parallel]
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(_audit_one, (i, records[i])) for i, _ in batch]
                batch_results = [f.result() for f in futures]
            
            for (i, record), audit in zip(batch, batch_results):
                results.append(audit)
                _print_audit(i, record, audit)
                status = audit.final_status
                if status == "PASS": pass_count += 1
                elif status == "REVISE": revise_count += 1
                elif status == "SKIP": skip_count += 1
                elif status == "AUDIT_FAILED": failed_count += 1
                if status == "PASS" and record.get("status") == "Diffuse_Observation":
                    pass_diffuse.append({"word": record.get("word","?"), "pos": record.get("pos","?"),
                                         "focus_rate": record.get("focus_rate", 0)})
                if not dry_run:
                    with open(audit_jsonl, 'a') as f:
                        f.write(json.dumps(asdict(audit), ensure_ascii=False) + "\n")
    
    # === Phase 2: Re-observe (if enabled) ===
    p2 = parallel_reobs if parallel_reobs is not None else parallel
    revise_records = [(i, r) for i, (r, a) in enumerate(zip(records, results))
                      if a.final_status == "REVISE"]
    
    if re_observe and revise_records and not dry_run:
        print(f"\n  ── Phase 2: Re-observe ({len(revise_records)} records, 3 conditions each) ──")
        if p2 > 1:
            print(f"  Parallel: {p2} (each word = 3 sequential LLM calls)")
        
        def _reobs_one(idx_record):
            idx, record = idx_record
            audit = results[idx]
            return idx, re_observe_word(record, audit, dictionary)
        
        def _print_reobs(record, new_record):
            word = record.get("word", "?")
            pos = record.get("pos", "?")
            prefix = f"  [re-obs] {word:20s} ({pos}) "
            if new_record:
                trace = new_record.get("re_observe_trace", {})
                new_sum = trace.get("final_sum", 0)
                new_nz = trace.get("final_nonzero", 0)
                new_f = new_record.get("focus_rate", 0)
                old_sum = sum(record["raw_scores"].values())
                old_nz = sum(1 for v in record["raw_scores"].values() if v > 0)
                old_f = record.get("focus_rate", 0)
                stable = trace.get("stable_slots", 0)
                dropped = trace.get("dropped_slots", 0)
                mass = trace.get("min_mass_guard", {})
                if mass.get("pass", False):
                    mass_str = "✅"
                else:
                    mass_str = f"⚠️THIN({mass.get('reason', '?')})"
                
                print(f"{prefix}→ ✅ sum {old_sum:.0f}→{new_sum:.0f}  nz {old_nz}→{new_nz}  "
                      f"F {old_f:.3f}→{new_f:.3f}  "
                      f"(stable={stable} dropped={dropped}) mass={mass_str}")
            else:
                print(f"{prefix}→ ❌ failed, keeping original")
        
        if p2 <= 1:
            # Sequential
            for idx, record in revise_records:
                _, new_record = _reobs_one((idx, record))
                _print_reobs(record, new_record)
                if new_record:
                    records[idx] = new_record
                    reobs_count += 1
        else:
            # Parallel: process in chunks
            for batch_start in range(0, len(revise_records), p2):
                batch = revise_records[batch_start:batch_start + p2]
                with ThreadPoolExecutor(max_workers=p2) as pool:
                    futures = [pool.submit(_reobs_one, (idx, record)) for idx, record in batch]
                    batch_results = [f.result() for f in futures]
                
                for (idx, record), (_, new_record) in zip(batch, batch_results):
                    _print_reobs(record, new_record)
                    if new_record:
                        records[idx] = new_record
                        reobs_count += 1
    
    # === Write final JSONL ===
    if not dry_run:
        with open(final_jsonl, 'w') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # === Summary ===
    print(f"\n  ── Audit Summary for {atom_id} ──")
    print(f"  Total: {len(results)}")
    print(f"  PASS: {pass_count}  REVISE: {revise_count}  SKIP: {skip_count}  FAILED: {failed_count}")
    if re_observe:
        print(f"  Re-observed: {reobs_count}/{revise_count}")
    
    # [GPT-1] PASS+Diffuse report
    if pass_diffuse:
        print(f"\n  ⚡ PASS but Diffuse ({len(pass_diffuse)} words — not re-observed):")
        for pd in pass_diffuse:
            print(f"    {pd['word']:25s} ({pd.get('pos','?')})  F={pd['focus_rate']:.4f}")
    
    # [GPT-2] Mass fail details with reason codes
    mass_fails_detail = []
    for record in records:
        if record.get("re_observed"):
            trace = record.get("re_observe_trace", {})
            mass = trace.get("min_mass_guard", {})
            if not mass.get("pass", True):
                mass_fails_detail.append({
                    "word": record["word"],
                    "pos": record.get("pos", "?"),
                    "reason": mass.get("reason", "unknown"),
                    "top_k_sum": mass.get("top_k_sum", 0),
                    "top_k_slots": mass.get("top_k_slots", []),
                })
    if mass_fails_detail:
        print(f"\n  🔴 Mass guard fails ({len(mass_fails_detail)}):")
        for mf in mass_fails_detail:
            slots_str = ", ".join(f"{s}={v:.0f}" for s, v in mf["top_k_slots"])
            print(f"    {mf['word']:25s} reason={mf['reason']:20s} "
                  f"top3_sum={mf['top_k_sum']:.0f}  [{slots_str}]")
    
    # [GPT-3] C4/C5 flag frequency (per this atom, category-level is in batch_report)
    flag_freq = {}
    for r in results:
        for f in r.pre_screen_flags:
            flag_freq[f] = flag_freq.get(f, 0) + 1
    
    c4c5_flags = {k: v for k, v in flag_freq.items() if k.startswith("C4_") or k.startswith("C5_")}
    
    if flag_freq:
        print(f"\n  Flag frequencies:")
        for fid, count in sorted(flag_freq.items(), key=lambda x: -x[1]):
            print(f"    {fid:30s}: {count}")
    
    if c4c5_flags:
        category = atom_id.split(".")[0]
        total_words = len(results)
        print(f"\n  C4/C5 rates for {category}:")
        for fid, count in sorted(c4c5_flags.items(), key=lambda x: -x[1]):
            rate = count / total_words * 100
            print(f"    {fid:30s}: {count}/{total_words} ({rate:.1f}%)")
    
    # [GPT-4] Rare word list (mass fail + very low nonzero after re-observe)
    rare_words = []
    for record in records:
        if record.get("re_observed"):
            trace = record.get("re_observe_trace", {})
            mass = trace.get("min_mass_guard", {})
            nz = trace.get("final_nonzero", 0)
            if not mass.get("pass", True) or nz <= 2:
                rare_words.append({
                    "word": record["word"],
                    "pos": record.get("pos", "?"),
                    "final_nz": nz,
                    "final_sum": trace.get("final_sum", 0),
                    "mass_pass": mass.get("pass", True),
                    "reason": mass.get("reason"),
                })
    if rare_words:
        print(f"\n  📚 Rare/thin words ({len(rare_words)} — Seed improvement candidates):")
        for rw in rare_words:
            mass_tag = "mass_ok" if rw["mass_pass"] else rw["reason"]
            print(f"    {rw['word']:25s} ({rw['pos']})  "
                  f"nz={rw['final_nz']}  sum={rw['final_sum']:.0f}  [{mass_tag}]")
    
    # Output files
    if not dry_run:
        print(f"\n  Output files:")
        print(f"    Audit:  {audit_jsonl}")
        print(f"    Final:  {final_jsonl}")
    
    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE A1 Auditor: Structural audit of mapper output"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Mapper output JSONL (e.g., mapper_output/EMO_like_a1.jsonl)")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="esde_dictionary.json")
    parser.add_argument("--out-dir", type=str, default="audit_output",
                        help="Output directory for audit results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pre-screen only, no LLM calls")
    parser.add_argument("--re-observe", action="store_true",
                        help="Re-run Writer on REVISE records with constraints")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent LLM requests (default: 1 = sequential)")
    parser.add_argument("--parallel-reobs", type=int, default=None,
                        help="Parallel for re-observe Phase 2 (default: same as --parallel)")
    args = parser.parse_args()
    
    # Load dictionary
    with open(args.dictionary) as f:
        d = json.load(f)
    dictionary = d.get("concepts", d)
    
    print(f"A1 Auditor Pipeline")
    print(f"  Input: {args.input}")
    print(f"  Dictionary: {args.dictionary}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Re-observe: {args.re_observe}")
    
    run_audit(
        input_path=args.input,
        dictionary=dictionary,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        re_observe=args.re_observe,
        parallel=args.parallel,
        parallel_reobs=args.parallel_reobs,
    )


if __name__ == "__main__":
    main()