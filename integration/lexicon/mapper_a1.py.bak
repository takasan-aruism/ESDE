#!/usr/bin/env python3
"""
ESDE Lexicon v2 — A1 Mapper Pipeline
=====================================
Maps Core Pool words → 48 semantic slot probability clouds via QwQ-32B.

Design principles (3AI agreed):
  - "Describe, do not decide" — winner=null always
  - LLM outputs raw scores (0-10 integer range, soft constraint)
  - Post-processing: softmax → entropy → focus_rate
  - Diffuse observations (F < 0.30) flagged for Phase 7

Architecture:
  Word → QwQ prompt → parse JSON → softmax normalize → compute stats → JSONL

Usage:
  # Single atom pilot (EMO.like)
  python3 mapper_a1.py --lexicon-entry lexicon/EMO_like.json --dictionary esde_dictionary.json

  # Dry run (print prompt, no LLM call)
  python3 mapper_a1.py --lexicon-entry lexicon/EMO_like.json --dictionary esde_dictionary.json --dry-run

  # Batch mode (multiple atoms)
  python3 mapper_a1.py --lexicon-dir lexicon/ --dictionary esde_dictionary.json --batch-size 16

Spec: A1 (pilot), 3AI design (Gemini stats / GPT schema / Claude impl)
"""

import json
import math
import os
import re
import argparse
import time
import sys
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor


# ============================================================
# Configuration
# ============================================================

LLM_HOST = os.environ.get("LLM_HOST", "http://100.107.6.119:8001/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwq32b_tp1_short8k")
LLM_TIMEOUT = 180
LLM_MAX_TOKENS = 6000
LLM_TEMPERATURE = 0.3

# QwQ uses greedy-ish sampling
LLM_TOP_P = 0.95
LLM_TOP_K = 20

SOFTMAX_TAU = 1.0           # Temperature for softmax normalization
DIFFUSE_THRESHOLD = 0.30    # Focus rate below this = Diffuse_Observation
MAX_RETRIES = 2             # JSON parse retries

# ============================================================
# 48 Slot Definitions (compressed, 1-line each)
# ============================================================

AXES = {
    "temporal": {
        "name": "Temporal Condition",
        "summary": "Where in time does this manifest?",
        "levels": {
            "emergence":      "First appearance; coming into being",
            "indication":     "Early signs suggesting presence",
            "influence":      "Active effect on change over time",
            "transformation": "Fundamental change in nature",
            "establishment":  "Becoming stable and recognized",
            "continuation":   "Ongoing persistence through time",
            "permanence":     "Enduring beyond normal timeframes",
        },
    },
    "scale": {
        "name": "Scale Condition",
        "summary": "At what scale does this operate?",
        "levels": {
            "individual":  "Single person or entity",
            "community":   "Small group or local community",
            "society":     "Large social structure or nation",
            "ecosystem":   "Interconnected system of entities",
            "stellar":     "Planetary or stellar scope",
            "cosmic":      "Universal or existence-level scope",
        },
    },
    "epistemological": {
        "name": "Epistemological Condition",
        "summary": "How is this known or understood?",
        "levels": {
            "perception":      "Direct sensory awareness",
            "identification":  "Naming and categorizing",
            "understanding":   "Grasping underlying structure",
            "experience":      "Lived, embodied knowing",
            "creation":        "Generating new knowledge",
        },
    },
    "ontological": {
        "name": "Ontological Condition",
        "summary": "At what layer of existence does this operate?",
        "levels": {
            "material":       "Physical, tangible layer",
            "informational":  "Data and information layer",
            "relational":     "Relationships between entities",
            "structural":     "Patterns and organizational forms",
            "semantic":       "Meaning and value layer",
        },
    },
    "interconnection": {
        "name": "Interconnection Condition",
        "summary": "How does this connect with other entities?",
        "levels": {
            "independent":  "Existing independently",
            "catalytic":    "Triggering change in others",
            "chained":      "Propagating through cause-effect chains",
            "synchronous":  "Multiple entities changing together",
            "resonant":     "Deep mutual resonance",
        },
    },
    "resonance": {
        "name": "Resonance Depth Condition",
        "summary": "How deeply does the connection touch?",
        "levels": {
            "superficial":  "Surface-level, temporary",
            "structural":   "Pattern-level correspondence",
            "essential":    "Touching the core essence",
            "existential":  "Fundamental, existence-level",
        },
    },
    "symmetry": {
        "name": "Symmetry Relation Condition",
        "summary": "What relational dynamic is at play?",
        "levels": {
            "destructive":    "Dark side, excess, pathology",
            "inclusive":      "Absorbing, welcoming inward",
            "transformative": "Catalyzing metamorphosis",
            "generative":    "Producing something new",
            "cyclical":      "Recurring rhythmic patterns",
        },
    },
    "lawfulness": {
        "name": "Lawfulness Condition",
        "summary": "How predictable is this phenomenon?",
        "levels": {
            "predictable": "Expected, follows known patterns",
            "emergent":    "Surprising, arising unexpectedly",
            "contingent":  "Dependent on conditions",
            "necessary":   "Inevitable, cannot not occur",
        },
    },
    "experience": {
        "name": "Experiential Quality Condition",
        "summary": "What is the quality of experiencing this?",
        "levels": {
            "discovery":      "Encountering the unknown",
            "creation":       "Bringing something new into being",
            "comprehension":  "Arriving at deep understanding",
        },
    },
    "value_generation": {
        "name": "Value Generation Condition",
        "summary": "What kind of value is produced?",
        "levels": {
            "functional":  "Practical utility",
            "aesthetic":   "Beauty, elegance, sensory appreciation",
            "ethical":     "Moral dimension, right/wrong",
            "sacred":      "Transcendent, beyond ordinary value",
        },
    },
}

# Build ordered slot list
SLOT_IDS = []
for axis_id, axis in AXES.items():
    for level_id in axis["levels"]:
        SLOT_IDS.append(f"{axis_id}.{level_id}")

assert len(SLOT_IDS) == 48, f"Expected 48 slots, got {len(SLOT_IDS)}"


# ============================================================
# Prompt Builder
# ============================================================

def build_compressed_slot_text() -> str:
    """Build the compressed 48-slot definition for the prompt."""
    lines = []
    for axis_id, axis in AXES.items():
        lines.append(f"\n{axis_id} — {axis['summary']}")
        for level_id, desc in axis["levels"].items():
            lines.append(f"  {axis_id}.{level_id}: {desc}")
    return "\n".join(lines)


COMPRESSED_SLOTS = build_compressed_slot_text()


def build_a1_prompt(word: str, pos: str, atom_id: str, atom_def: str,
                    sym_pair: str, category: str) -> str:
    """
    Build the A1 mapper prompt for a single word.
    
    Key design: "You are an observer, not a classifier."
    """
    slot_json_template = ", ".join(f'"{s}": <score>' for s in SLOT_IDS[:3])
    
    return f"""You are a semantic coordinate observer for ESDE.

## Task
Given the word "{word}" (POS: {pos}) in the context of atom [{atom_id}] ("{atom_def}"),
score how strongly this word resonates with each of the 48 semantic slots.

## Atom Context
- Atom: {atom_id} ({category})
- Definition: {atom_def}
- Symmetric pair: {sym_pair}

## 48 Semantic Slots
{COMPRESSED_SLOTS}

## RULES
1. You are an OBSERVER, not a classifier. Do NOT pick a winner.
2. Score each slot from 0 to 10 (integer). 0 = no resonance. 10 = strong resonance.
3. Most slots will be 0-2. A few will be higher. This is expected.
4. Think about how "{word}" manifests in real text when used in the sense of [{atom_id}].
5. Do NOT force high scores. If the word barely relates to a slot, score it 0 or 1.
6. The symmetric pair [{sym_pair}] is the opposite. If the word pulls toward it, keep scores low.

## Output Format
Return ONLY a JSON object with this structure (no other text):
```json
{{
  "word": "{word}",
  "pos": "{pos}",
  "atom": "{atom_id}",
  "raw_scores": {{
    {slot_json_template}, ...
  }},
  "evidence": "1-2 sentences: WHY does this word resonate with the slots you scored highest?"
}}
```

All 48 slots must appear in raw_scores. Respond with ONLY the JSON."""


# ============================================================
# LLM Interface
# ============================================================

def call_qwq(prompt: str, system: str = "") -> Tuple[str, float]:
    """Call QwQ-32B and return (response_text, elapsed_seconds)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
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


def parse_qwq_response(text: str) -> dict:
    """
    Extract JSON from LLM response.
    Handles both:
      - QwQ format:   <think>...</think> then JSON
      - Qwen3 format: thinking text...</think> then JSON (no opening <think>)
    """
    # Strip <think>...</think> blocks (QwQ style)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Strip everything before </think> (Qwen3 style — no opening <think> tag)
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    
    # Try to find JSON block in markdown fences
    m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    
    # Try to find raw JSON object
    m = re.search(r'\{[^{}]*"raw_scores"[^{}]*\{.*?\}.*?\}', text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    
    # Last resort: find first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return json.loads(text[start:end+1])
    
    raise ValueError(f"No JSON found in response ({len(text)} chars)")


# ============================================================
# Post-processing: Softmax + Entropy + Focus Rate
# ============================================================

def softmax(raw: Dict[str, float], tau: float = SOFTMAX_TAU) -> Dict[str, float]:
    """
    Softmax normalization over 48 slots.
    p_i = exp(s_i / tau) / sum(exp(s_j / tau))
    """
    values = [raw.get(s, 0.0) for s in SLOT_IDS]
    max_v = max(values)  # for numerical stability
    exps = [math.exp((v - max_v) / tau) for v in values]
    total = sum(exps)
    
    return {s: round(e / total, 6) for s, e in zip(SLOT_IDS, exps)}


def shannon_entropy(probs: Dict[str, float]) -> float:
    """H = -sum(p_i * log(p_i)), normalized by log(K)."""
    h = 0.0
    for p in probs.values():
        if p > 1e-12:
            h -= p * math.log(p)
    h_max = math.log(len(SLOT_IDS))  # log(48)
    return round(h / h_max, 4) if h_max > 0 else 0.0


def focus_rate(entropy_norm: float) -> float:
    """F = 1 - H_norm. 1.0 = sharp, 0.0 = fog."""
    return round(1.0 - entropy_norm, 4)


def classify_status(f: float) -> str:
    """Classify observation status based on focus rate."""
    if f < DIFFUSE_THRESHOLD:
        return "Diffuse_Observation"
    return "OK"


def process_raw_scores(raw: Dict[str, float]) -> Dict[str, Any]:
    """Full post-processing pipeline: raw → normalized → entropy → focus → status."""
    # Ensure all 48 slots present
    clean_raw = {}
    for s in SLOT_IDS:
        v = raw.get(s, 0)
        try:
            clean_raw[s] = float(v)
        except (ValueError, TypeError):
            clean_raw[s] = 0.0
    
    normalized = softmax(clean_raw)
    h = shannon_entropy(normalized)
    f = focus_rate(h)
    status = classify_status(f)
    
    # Top 5 slots (for quick inspection)
    top5 = sorted(normalized.items(), key=lambda x: -x[1])[:5]
    
    return {
        "raw_scores": clean_raw,
        "normalized_scores": normalized,
        "entropy_norm": h,
        "focus_rate": f,
        "status": status,
        "top5": [{"slot": s, "p": p} for s, p in top5],
    }


# ============================================================
# Single Word Pipeline
# ============================================================

def map_single_word(word: str, pos: str, atom_id: str, atom_def: str,
                    sym_pair: str, category: str,
                    dry_run: bool = False) -> Dict[str, Any]:
    """
    Full A1 pipeline for one word.
    Returns the complete observation record.
    """
    prompt = build_a1_prompt(word, pos, atom_id, atom_def, sym_pair, category)
    
    if dry_run:
        return {
            "word": word,
            "pos": pos,
            "atom": atom_id,
            "status": "DRY_RUN",
            "prompt_length": len(prompt),
            "prompt_preview": prompt[:500],
        }
    
    # Call LLM with retries
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response_text, elapsed = call_qwq(prompt)
            parsed = parse_qwq_response(response_text)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            print(f"    ⚠ Attempt {attempt}/{MAX_RETRIES} parse failed: {last_error}")
            if attempt == MAX_RETRIES:
                return {
                    "word": word,
                    "pos": pos,
                    "atom": atom_id,
                    "status": "Observation_Failed",
                    "error": last_error,
                    "raw_response_length": len(response_text) if 'response_text' in dir() else 0,
                }
        except requests.RequestException as e:
            return {
                "word": word,
                "pos": pos,
                "atom": atom_id,
                "status": "Observation_Failed",
                "error": f"LLM connection error: {e}",
            }
    
    # Post-process
    raw_scores = parsed.get("raw_scores", {})
    stats = process_raw_scores(raw_scores)
    
    record = {
        "word": word,
        "pos": pos,
        "atom": atom_id,
        "raw_scores": stats["raw_scores"],
        "normalized_scores": stats["normalized_scores"],
        "entropy_norm": stats["entropy_norm"],
        "focus_rate": stats["focus_rate"],
        "status": stats["status"],
        "top5": stats["top5"],
        "evidence": parsed.get("evidence", ""),
        "llm_elapsed_sec": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    return record


# ============================================================
# Batch Pipeline
# ============================================================

def load_lexicon_entry(path: str) -> Dict[str, Any]:
    """Load a single lexicon entry JSON."""
    with open(path) as f:
        return json.load(f)


def load_dictionary(path: str) -> Dict[str, Any]:
    """Load esde_dictionary.json → concepts dict."""
    with open(path) as f:
        d = json.load(f)
    return d.get("concepts", d)


def get_core_words(entry: Dict) -> List[Dict[str, str]]:
    """Extract core pool words from lexicon entry."""
    return entry.get("core_pool", {}).get("words", [])


def run_atom(entry: Dict, dictionary: Dict, dry_run: bool = False,
             out_path: Optional[str] = None, parallel: int = 1) -> List[Dict]:
    """Run A1 mapper for all core words of a single atom."""
    atom_id = entry["atom"]
    atom_def_entry = dictionary.get(atom_id, {})
    atom_def = atom_def_entry.get("definition_en", entry.get("definition", ""))
    sym_pair = atom_def_entry.get("symmetric_pair", "")
    category = atom_def_entry.get("category", entry.get("category", "?"))
    
    core_words = get_core_words(entry)
    if not core_words:
        print(f"  ⚠ No core words for {atom_id}")
        return []
    
    print(f"\n{'='*60}")
    print(f"  A1 MAPPER: {atom_id}")
    print(f"  Definition: {atom_def}")
    print(f"  Symmetric pair: {sym_pair}")
    print(f"  Core words: {len(core_words)}")
    if parallel > 1:
        print(f"  Parallel: {parallel}")
    print(f"{'='*60}")
    
    results = []
    ok = 0
    diffuse = 0
    failed = 0
    
    def _process_one(i_w):
        """Worker: map a single word (thread-safe, no shared state)."""
        i, w = i_w
        word = w.get("w", w.get("lemma", ""))
        pos = w.get("pos", "n")
        if not word:
            return None
        return map_single_word(
            word=word, pos=pos, atom_id=atom_id,
            atom_def=atom_def, sym_pair=sym_pair,
            category=category, dry_run=dry_run,
        )
    
    def _print_record(i, record):
        """Print status line for one record."""
        word = record.get("word", "?")
        pos = record.get("pos", "?")
        status = record.get("status", "?")
        prefix = f"  [{i+1:3d}/{len(core_words)}] {word:20s} ({pos}) "
        if status == "OK":
            f_val = record.get("focus_rate", 0)
            top1 = record.get("top5", [{}])[0]
            print(f"{prefix}→ F={f_val:.2f}  top={top1.get('slot','?')} ({top1.get('p',0):.3f})")
        elif status == "Diffuse_Observation":
            f_val = record.get("focus_rate", 0)
            print(f"{prefix}→ 🌫️ DIFFUSE F={f_val:.2f}")
        elif status == "DRY_RUN":
            print(f"{prefix}→ [dry run, prompt={record['prompt_length']} chars]")
        else:
            print(f"{prefix}→ ❌ {record.get('error','?')[:60]}")
    
    # --- Main loop: sequential or parallel ---
    indexed_words = [(i, w) for i, w in enumerate(core_words)]
    
    if parallel <= 1:
        # Sequential (original behavior)
        for i_w in indexed_words:
            record = _process_one(i_w)
            if record is None:
                continue
            results.append(record)
            _print_record(i_w[0], record)
            status = record.get("status", "?")
            if status == "OK": ok += 1
            elif status == "Diffuse_Observation": diffuse += 1
            elif status not in ("DRY_RUN",): failed += 1
            if out_path and not dry_run:
                with open(out_path, 'a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        # Parallel: process in chunks
        for batch_start in range(0, len(indexed_words), parallel):
            batch = indexed_words[batch_start:batch_start + parallel]
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = [pool.submit(_process_one, iw) for iw in batch]
                batch_results = [f.result() for f in futures]
            
            # Print and write in order
            for (i, _), record in zip(batch, batch_results):
                if record is None:
                    continue
                results.append(record)
                _print_record(i, record)
                status = record.get("status", "?")
                if status == "OK": ok += 1
                elif status == "Diffuse_Observation": diffuse += 1
                elif status not in ("DRY_RUN",): failed += 1
                if out_path and not dry_run:
                    with open(out_path, 'a') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Summary
    print(f"\n  ── Summary for {atom_id} ──")
    print(f"  Total: {len(results)}  OK: {ok}  Diffuse: {diffuse}  Failed: {failed}")
    
    if results and not dry_run:
        focus_rates = [r["focus_rate"] for r in results if r.get("focus_rate") is not None]
        if focus_rates:
            avg_f = sum(focus_rates) / len(focus_rates)
            min_f = min(focus_rates)
            max_f = max(focus_rates)
            print(f"  Focus rate: avg={avg_f:.3f}  min={min_f:.3f}  max={max_f:.3f}")
    
    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESDE A1 Mapper: Core Pool → 48 slot probability clouds"
    )
    parser.add_argument("--lexicon-entry", type=str,
                        help="Single lexicon entry JSON (e.g., lexicon/EMO_like.json)")
    parser.add_argument("--lexicon-dir", type=str,
                        help="Directory of lexicon entries (batch mode)")
    parser.add_argument("--dictionary", type=str, required=True,
                        help="esde_dictionary.json (or v2)")
    parser.add_argument("--out-dir", type=str, default="mapper_output",
                        help="Output directory for JSONL files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling LLM")
    parser.add_argument("--atoms", type=str, nargs="*",
                        help="Filter: only process these atom IDs")
    parser.add_argument("--limit", type=int,
                        help="Limit number of words per atom (for testing)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of concurrent LLM requests (default: 1 = sequential)")
    args = parser.parse_args()
    
    if not args.lexicon_entry and not args.lexicon_dir:
        parser.error("Provide --lexicon-entry or --lexicon-dir")
    
    dictionary = load_dictionary(args.dictionary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect entries
    entries = []
    if args.lexicon_entry:
        entries.append(load_lexicon_entry(args.lexicon_entry))
    elif args.lexicon_dir:
        for f in sorted(Path(args.lexicon_dir).glob("*.json")):
            if f.name.startswith("_"):
                continue
            entry = load_lexicon_entry(str(f))
            if entry.get("status") == "merged":
                continue
            if args.atoms and entry.get("atom") not in args.atoms:
                continue
            entries.append(entry)
    
    print(f"A1 Mapper Pipeline")
    print(f"  Entries: {len(entries)}")
    print(f"  Output: {out_dir}")
    if args.parallel > 1:
        print(f"  Parallel: {args.parallel}")
    print(f"  LLM: {LLM_HOST} ({LLM_MODEL})")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    
    all_results = []
    
    for entry in entries:
        atom_id = entry.get("atom", "?")
        
        # Apply word limit for testing
        if args.limit:
            core = entry.get("core_pool", {})
            if "words" in core:
                core["words"] = core["words"][:args.limit]
        
        out_path = out_dir / f"{atom_id.replace('.', '_')}_a1.jsonl"
        
        # Clear output file (fresh run)
        if not args.dry_run and out_path.exists():
            out_path.unlink()
        
        results = run_atom(
            entry=entry,
            dictionary=dictionary,
            dry_run=args.dry_run,
            out_path=str(out_path),
            parallel=args.parallel,
        )
        all_results.extend(results)
    
    # Final summary
    if all_results and not args.dry_run:
        print(f"\n{'='*60}")
        print(f"  A1 PIPELINE COMPLETE")
        print(f"{'='*60}")
        total = len(all_results)
        ok = sum(1 for r in all_results if r.get("status") == "OK")
        diffuse = sum(1 for r in all_results if r.get("status") == "Diffuse_Observation")
        failed = sum(1 for r in all_results if r.get("status") == "Observation_Failed")
        print(f"  Total: {total}  OK: {ok}  Diffuse: {diffuse}  Failed: {failed}")
        print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()