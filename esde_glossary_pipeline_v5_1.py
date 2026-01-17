"""
ESDE Glossary Pipeline v5.1 (Semantic Auditor)

Changes from v5.0:
  - Auditor now receives REFERENCE DEFINITIONS for all 48 levels
  - Semantic mismatch check is explicit and critical
  - Definition vs level meaning validation is primary audit focus

Architecture:
  1. Writer (QwQ) → Generate glossary JSON
  2. Auditor (QwQ) → Semantic + structural validation
  3. Fixer (QwQ) → Apply fix_instructions if needed
"""
import requests
import json
import re
import os
import sys
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple


VERSION = "5.1.0"
CHECKPOINT_INTERVAL = 10
MAX_REGENERATE_ATTEMPTS = 2
MAX_FIX_ATTEMPTS = 1

# Immutable axis definitions
AXES_LABELS = {
    "temporal": ["emergence", "indication", "influence", "transformation", "establishment", "continuation", "permanence"],
    "scale": ["individual", "community", "society", "ecosystem", "stellar", "cosmic"],
    "epistemological": ["perception", "identification", "understanding", "experience", "creation"],
    "ontological": ["material", "informational", "relational", "structural", "semantic"],
    "interconnection": ["independent", "catalytic", "chained", "synchronous", "resonant"],
    "resonance": ["superficial", "structural", "essential", "existential"],
    "symmetry": ["destructive", "inclusive", "transformative", "generative", "cyclical"],
    "lawfulness": ["predictable", "emergent", "contingent", "necessary"],
    "experience": ["discovery", "creation", "comprehension"],
    "value_generation": ["functional", "aesthetic", "ethical", "sacred"]
}

# REFERENCE DEFINITIONS for semantic validation
# Auditor compares Writer's definitions against these
LEVEL_REFERENCE = {
    "temporal": {
        "emergence": "Initial appearance; the moment something first comes into being or becomes noticeable",
        "indication": "Early signs or signals that suggest presence or future development",
        "influence": "Active effect on development or change over time",
        "transformation": "Fundamental change in form, nature, or character",
        "establishment": "Becoming stable, fixed, or firmly settled",
        "continuation": "Ongoing persistence through time without fundamental change",
        "permanence": "Lasting indefinitely; unchanging and enduring state"
    },
    "scale": {
        "individual": "Single person, entity, or discrete unit",
        "community": "Local group of interconnected individuals",
        "society": "Large-scale organized human social system",
        "ecosystem": "Interconnected system of living and non-living components",
        "stellar": "Star system or astronomical scale",
        "cosmic": "Universe-wide or fundamental reality scale"
    },
    "epistemological": {
        "perception": "Direct sensory or intuitive awareness",
        "identification": "Recognition and categorization of what is perceived",
        "understanding": "Comprehension of meaning, causes, and relationships",
        "experience": "Direct lived engagement and accumulated knowledge",
        "creation": "Generation of new knowledge or meaning"
    },
    "ontological": {
        "material": "Physical substance and tangible matter",
        "informational": "Data, patterns, and encoded content",
        "relational": "Connections and relationships between entities",
        "structural": "Organization, arrangement, and systemic form",
        "semantic": "Meaning, significance, and interpretive content"
    },
    "interconnection": {
        "independent": "Self-contained; not dependent on or affecting others",
        "catalytic": "Triggering or accelerating change in others without being changed",
        "chained": "Sequential cause-effect relationships",
        "synchronous": "Simultaneous coordination without direct causation",
        "resonant": "Mutual amplification through shared frequency or pattern"
    },
    "resonance": {
        "superficial": "Surface-level effect; easily changed or removed",
        "structural": "Affecting organization and arrangement",
        "essential": "Touching core nature or fundamental character",
        "existential": "Concerning existence, being, and ultimate meaning"
    },
    "symmetry": {
        "destructive": "Breaking down, eliminating, or reducing",
        "inclusive": "Incorporating, accepting, or embracing",
        "transformative": "Converting from one form to another",
        "generative": "Creating, producing, or bringing forth new",
        "cyclical": "Recurring patterns; death and rebirth"
    },
    "lawfulness": {
        "predictable": "Following known rules; outcomes can be anticipated",
        "emergent": "Arising from complexity; not reducible to components",
        "contingent": "Dependent on conditions; could be otherwise",
        "necessary": "Could not be otherwise; logically or metaphysically required"
    },
    "experience": {
        "discovery": "Finding or uncovering what already exists",
        "creation": "Bringing into existence something new",
        "comprehension": "Grasping meaning and achieving understanding"
    },
    "value_generation": {
        "functional": "Practical utility and instrumental usefulness",
        "aesthetic": "Beauty, harmony, and sensory appreciation",
        "ethical": "Moral worth and right action",
        "sacred": "Transcendent significance and ultimate value"
    }
}

VALID_AXIS_NAMES = set(AXES_LABELS.keys())
VALID_TOP_LEVEL_KEYS = {"concept_id", "name", "axes", "_meta"}


class ESDEGlossaryPipeline:
    """
    Writer → Auditor → Fixer pipeline with semantic validation.
    """

    def __init__(self, api_base: str, model_name: str,
                 dictionary_path: str = "esde_dictionary.json",
                 output_dir: str = "glossary_output_v5"):
        
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, "raw")
        self.audit_dir = os.path.join(output_dir, "audits")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.audit_dir, exist_ok=True)
        
        with open(dictionary_path, "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)
        
        self.concepts = self.dictionary.get("concepts", {})
        self.concept_ids = sorted(self.concepts.keys())
        
        self.results: Dict[str, Any] = {}
        self.processed_count = 0
        
        self.checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        self.results_path = os.path.join(output_dir, "glossary_results.json")
        
        self.stats = {
            "ok": 0, "pass_with_warn": 0, "fixed": 0,
            "failed_regenerate": 0, "failed_fix": 0, 
            "failed_llm": 0, "failed_parse": 0
        }
        
        print(f"[GLOSSARY PIPELINE v{VERSION}]")
        print(f"  API: {self.api_base}")
        print(f"  Model: {self.model_name}")
        print(f"  Total: {len(self.concept_ids)} concepts")
        print(f"  Semantic validation: ENABLED (48 level references)")

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _call_llm(self, prompt: str, role: str, timeout: int = 180) -> Tuple[Optional[str], str]:
        system_prompts = {
            "writer": "You are a glossary generator. Output valid JSON only. No explanation.",
            "auditor": "You are a strict semantic auditor. Output audit result JSON only. No explanation.",
            "fixer": "You are a JSON fixer. Apply only the specified fixes. Output corrected JSON only."
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompts.get(role, system_prompts["writer"])},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 8192
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'], ""
        except requests.exceptions.Timeout:
            return None, "timeout"
        except Exception as e:
            return None, f"llm_error:{str(e)[:50]}"

    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text)
        cleaned = re.sub(r'^```json\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^```\s*\n?', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        candidates = []
        depth = 0
        start = -1
        
        for i, char in enumerate(cleaned):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(cleaned[start:i+1])
                        candidates.append((len(cleaned[start:i+1]), parsed))
                    except json.JSONDecodeError:
                        pass
                    start = -1
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        
        return None

    def _save_raw(self, concept_id: str, stage: str, attempt: int, text: str) -> str:
        safe_id = concept_id.replace(".", "_")
        filename = f"{safe_id}_{stage}_a{attempt}.txt"
        filepath = os.path.join(self.raw_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text or "")
        return filename

    # ========================================
    # WRITER
    # ========================================
    
    def _writer_prompt(self, concept_id: str, concept_data: Dict) -> str:
        name = concept_data.get("name", concept_id.split(".")[-1])
        definition = concept_data.get("definition_en", "")
        
        # Include reference definitions so Writer knows correct meanings
        ref_json = json.dumps(LEVEL_REFERENCE, indent=2)
        
        return f"""Generate ESDE glossary entry for this concept.

RULES:
1. Output JSON only. No explanation.
2. Use ONLY levels from the reference below.
3. Your definition_en must ALIGN with the reference meaning for each level.
4. Each level needs: definition_en, triggers_en (list), anti_triggers_en (list)
5. All text must be English (ASCII only).
6. Include only applicable levels (typically 3-8 total).

REFERENCE DEFINITIONS (your definitions must align with these meanings):
{ref_json}

Concept to Process:
- ID: {concept_id}
- Name: {name}
- Definition: {definition}

Output format:
{{
  "concept_id": "{concept_id}",
  "name": "{name}",
  "axes": {{
    "temporal": {{
      "emergence": {{
        "definition_en": "How {name} relates to emergence (initial appearance)...",
        "triggers_en": ["...", "..."],
        "anti_triggers_en": ["...", "..."]
      }}
    }}
  }}
}}"""

    def _call_writer(self, concept_id: str, concept_data: Dict, attempt: int) -> Tuple[Optional[Dict], str]:
        prompt = self._writer_prompt(concept_id, concept_data)
        raw_text, error = self._call_llm(prompt, "writer")
        
        self._save_raw(concept_id, "writer", attempt, raw_text or f"ERROR:{error}")
        
        if error:
            return None, f"writer_{error}"
        
        parsed = self._extract_json(raw_text)
        if parsed is None:
            return None, "writer_parse_failed"
        
        return parsed, ""

    # ========================================
    # AUDITOR (Semantic Focus)
    # ========================================
    
    def _auditor_prompt(self, concept_id: str, writer_json: Dict) -> str:
        ref_json = json.dumps(LEVEL_REFERENCE, indent=2)
        writer_str = json.dumps(writer_json, indent=2)
        
        return f"""Audit this ESDE glossary JSON with SEMANTIC VALIDATION.

## REFERENCE DEFINITIONS (Ground Truth)
{ref_json}

## JSON TO AUDIT (concept: {concept_id})
{writer_str}

## AUDIT RULES

### FATAL (status=REGENERATE) - Structure broken:
- JSON parse error
- Invalid top-level keys (allowed: concept_id, name, axes, _meta only)
- Missing or non-dict "axes"
- Axis name not in valid 10 axes
- Axis data leaked to top level

### FAIL (status=FIX) - Fixable errors:
- Level name not in valid levels for that axis
- Missing required keys: definition_en, triggers_en, anti_triggers_en
- Non-ASCII characters in English fields
- **SEMANTIC MISMATCH**: definition_en contradicts or is unrelated to the reference meaning

### WARN (status=PASS with warnings):
- Definition too short (<15 chars) or too generic
- Triggers list empty or has fewer than 2 items
- Minor semantic drift (related but imprecise)

## CRITICAL: SEMANTIC VALIDATION
For EACH level in the JSON, compare definition_en against the REFERENCE.
- If definition describes the OPPOSITE meaning → FAIL (semantic_contradiction)
- If definition is UNRELATED to the level meaning → FAIL (semantic_mismatch)
- If definition is WEAK but related → WARN (semantic_drift)
- If definition correctly captures the meaning for this concept → OK

## OUTPUT (JSON only)
{{
  "concept_id": "{concept_id}",
  "status": "PASS|FIX|REGENERATE",
  "severity": "OK|WARN|FAIL|FATAL",
  "issues": [
    {{
      "code": "SEMANTIC_MISMATCH|SEMANTIC_CONTRADICTION|MISSING_KEY|INVALID_LEVEL|...",
      "severity": "FATAL|FAIL|WARN",
      "path": "$.axes.temporal.emergence.definition_en",
      "message": "Definition describes permanence, not emergence",
      "expected": "Initial appearance meaning",
      "actual": "What was written"
    }}
  ],
  "fix_instructions": [
    {{
      "op": "REWRITE",
      "path": "$.axes.temporal.emergence.definition_en",
      "reason": "Must describe initial appearance, not final state"
    }}
  ],
  "metrics": {{
    "axis_count": 0,
    "level_count": 0,
    "semantic_ok": 0,
    "semantic_warn": 0,
    "semantic_fail": 0
  }}
}}"""

    def _call_auditor(self, concept_id: str, writer_json: Dict, attempt: int) -> Tuple[Optional[Dict], str]:
        prompt = self._auditor_prompt(concept_id, writer_json)
        raw_text, error = self._call_llm(prompt, "auditor")
        
        self._save_raw(concept_id, "auditor", attempt, raw_text or f"ERROR:{error}")
        
        if raw_text:
            safe_id = concept_id.replace(".", "_")
            audit_path = os.path.join(self.audit_dir, f"{safe_id}_a{attempt}.json")
            with open(audit_path, "w", encoding="utf-8") as f:
                f.write(raw_text)
        
        if error:
            return None, f"auditor_{error}"
        
        parsed = self._extract_json(raw_text)
        if parsed is None:
            return None, "auditor_parse_failed"
        
        return parsed, ""

    # ========================================
    # FIXER
    # ========================================
    
    def _fixer_prompt(self, concept_id: str, writer_json: Dict, 
                      fix_instructions: List[Dict], issues: List[Dict]) -> str:
        writer_str = json.dumps(writer_json, indent=2)
        fixes_str = json.dumps(fix_instructions, indent=2)
        issues_str = json.dumps(issues, indent=2)
        ref_json = json.dumps(LEVEL_REFERENCE, indent=2)
        
        return f"""Fix the JSON according to the audit results.

## REFERENCE DEFINITIONS (Use these to fix semantic issues)
{ref_json}

## ORIGINAL JSON
{writer_str}

## ISSUES FOUND
{issues_str}

## FIX INSTRUCTIONS
{fixes_str}

## RULES
1. Fix ALL listed issues
2. For SEMANTIC issues: rewrite definition_en to match the reference meaning
3. Keep the concept-specific context while fixing the semantic alignment
4. Output the complete corrected JSON only

## OUTPUT
Corrected JSON with all issues fixed:"""

    def _call_fixer(self, concept_id: str, writer_json: Dict, 
                    fix_instructions: List[Dict], issues: List[Dict], 
                    attempt: int) -> Tuple[Optional[Dict], str]:
        prompt = self._fixer_prompt(concept_id, writer_json, fix_instructions, issues)
        raw_text, error = self._call_llm(prompt, "fixer")
        
        self._save_raw(concept_id, "fixer", attempt, raw_text or f"ERROR:{error}")
        
        if error:
            return None, f"fixer_{error}"
        
        parsed = self._extract_json(raw_text)
        if parsed is None:
            return None, "fixer_parse_failed"
        
        return parsed, ""

    # ========================================
    # LOCAL VALIDATION (Fast pre-check)
    # ========================================
    
    def _local_validate(self, data: Dict, concept_id: str) -> Tuple[str, List[Dict]]:
        issues = []
        
        if not isinstance(data, dict):
            return "REGENERATE", [{"code": "NOT_DICT", "severity": "FATAL"}]
        
        extra_keys = set(data.keys()) - VALID_TOP_LEVEL_KEYS
        axis_leak = extra_keys & VALID_AXIS_NAMES
        if axis_leak:
            issues.append({
                "code": "TOP_LEVEL_AXIS_LEAK",
                "severity": "FATAL",
                "path": f"$.{list(axis_leak)[0]}"
            })
            return "REGENERATE", issues
        
        if "axes" not in data:
            return "REGENERATE", [{"code": "MISSING_AXES", "severity": "FATAL"}]
        
        if not isinstance(data["axes"], dict):
            return "REGENERATE", [{"code": "AXES_NOT_DICT", "severity": "FATAL"}]
        
        for axis_name in data["axes"].keys():
            if axis_name not in VALID_AXIS_NAMES:
                issues.append({
                    "code": "INVALID_AXIS_NAME",
                    "severity": "FATAL",
                    "path": f"$.axes.{axis_name}"
                })
        
        if any(i["severity"] == "FATAL" for i in issues):
            return "REGENERATE", issues
        
        for axis_name, axis_data in data["axes"].items():
            if not isinstance(axis_data, dict):
                issues.append({
                    "code": "AXIS_NOT_DICT",
                    "severity": "FAIL",
                    "path": f"$.axes.{axis_name}"
                })
                continue
            
            valid_levels = AXES_LABELS.get(axis_name, [])
            for level_name, level_data in axis_data.items():
                if level_name not in valid_levels:
                    issues.append({
                        "code": "INVALID_LEVEL",
                        "severity": "FAIL",
                        "path": f"$.axes.{axis_name}.{level_name}"
                    })
                
                if isinstance(level_data, dict):
                    for req_key in ["definition_en", "triggers_en", "anti_triggers_en"]:
                        if req_key not in level_data:
                            issues.append({
                                "code": "MISSING_KEY",
                                "severity": "FAIL",
                                "path": f"$.axes.{axis_name}.{level_name}.{req_key}"
                            })
        
        if any(i["severity"] == "FAIL" for i in issues):
            return "FIX", issues
        
        return "PASS", issues

    # ========================================
    # PIPELINE
    # ========================================
    
    def process_concept(self, concept_id: str) -> Dict[str, Any]:
        concept_data = self.concepts.get(concept_id, {})
        
        result = {
            "concept_id": concept_id,
            "status": "pending",
            "final_json": None,
            "pipeline_log": [],
            "audit_summary": None,
            "_meta": {
                "version": VERSION,
                "timestamp_utc": self._now_utc()
            }
        }
        
        writer_json = None
        
        for attempt in range(1, MAX_REGENERATE_ATTEMPTS + 1):
            result["pipeline_log"].append(f"writer_a{attempt}")
            
            writer_json, writer_error = self._call_writer(concept_id, concept_data, attempt)
            
            if writer_error:
                result["pipeline_log"].append(f"err:{writer_error}")
                continue
            
            # Local validation (fast structural check)
            local_status, local_issues = self._local_validate(writer_json, concept_id)
            result["pipeline_log"].append(f"local:{local_status}")
            
            if local_status == "REGENERATE":
                continue
            
            # Auditor (semantic validation)
            result["pipeline_log"].append("auditor")
            audit_result, audit_error = self._call_auditor(concept_id, writer_json, attempt)
            
            if audit_error:
                result["pipeline_log"].append(f"err:{audit_error}")
                # Fallback: if local passed, use it
                if local_status == "PASS":
                    result["status"] = "ok_no_audit"
                    result["final_json"] = writer_json
                    return result
                continue
            
            audit_status = audit_result.get("status", "PASS")
            audit_severity = audit_result.get("severity", "OK")
            result["pipeline_log"].append(f"audit:{audit_status}/{audit_severity}")
            result["audit_summary"] = audit_result.get("metrics")
            
            if audit_status == "REGENERATE":
                continue
            
            if audit_status == "FIX":
                fix_instructions = audit_result.get("fix_instructions", [])
                issues = audit_result.get("issues", [])
                
                if fix_instructions:
                    result["pipeline_log"].append("fixer")
                    fixed_json, fix_error = self._call_fixer(
                        concept_id, writer_json, fix_instructions, issues, attempt
                    )
                    
                    if fix_error:
                        result["pipeline_log"].append(f"err:{fix_error}")
                    elif fixed_json:
                        result["status"] = "fixed"
                        result["final_json"] = fixed_json
                        result["fixes_applied"] = len(fix_instructions)
                        return result
                
                result["status"] = "pass_with_issues"
                result["final_json"] = writer_json
                result["issues"] = issues
                return result
            
            # PASS
            result["status"] = "ok" if audit_severity == "OK" else "pass_with_warn"
            result["final_json"] = writer_json
            if audit_severity == "WARN":
                result["warnings"] = [i for i in audit_result.get("issues", []) 
                                      if i.get("severity") == "WARN"]
            return result
        
        result["status"] = "failed_regenerate"
        result["final_json"] = writer_json
        return result

    def _count_levels(self, data: Optional[Dict]) -> int:
        if not data or "axes" not in data:
            return 0
        return sum(
            len(levels) if isinstance(levels, dict) else 0
            for levels in data.get("axes", {}).values()
        )

    def save_checkpoint(self):
        self.stats = {
            "ok": 0, "ok_no_audit": 0, "pass_with_warn": 0, "fixed": 0, 
            "pass_with_issues": 0, "failed_regenerate": 0, "failed_exception": 0
        }
        for r in self.results.values():
            status = r.get("status", "failed_exception")
            if status in self.stats:
                self.stats[status] += 1
            elif "failed" in status:
                self.stats["failed_regenerate"] += 1
            else:
                self.stats["failed_exception"] += 1
        
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": VERSION,
                "timestamp_utc": self._now_utc(),
                "processed_count": self.processed_count,
                "total_count": len(self.concept_ids),
                "stats": self.stats
            }, f, indent=2)
        
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": VERSION,
                "timestamp_utc": self._now_utc(),
                "stats": self.stats,
                "glossary": self.results
            }, f, indent=2, ensure_ascii=False)
        
        ok_total = sum(self.stats.get(k, 0) for k in ["ok", "ok_no_audit", "pass_with_warn", "fixed"])
        failed = self.stats.get("failed_regenerate", 0) + self.stats.get("failed_exception", 0)
        print(f"  [SAVE] {self.processed_count}/{len(self.concept_ids)} | ok={ok_total} fail={failed}")

    def load_checkpoint(self) -> int:
        if not os.path.exists(self.checkpoint_path):
            return 0
        
        with open(self.checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                data = json.load(f)
                self.results = data.get("glossary", {})
                self.stats = data.get("stats", self.stats)
        
        resume_index = checkpoint.get("processed_count", 0)
        print(f"[RESUME] {resume_index}/{len(self.concept_ids)}")
        return resume_index

    def run(self, resume: bool = False):
        start_index = self.load_checkpoint() if resume else 0
        self.processed_count = start_index
        total = len(self.concept_ids)
        
        print(f"\n{'='*60}")
        print(f"PIPELINE v{VERSION}: {start_index} → {total}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for i in range(start_index, total):
            concept_id = self.concept_ids[i]
            print(f"\n[{i+1}/{total}] {concept_id}")
            
            try:
                result = self.process_concept(concept_id)
                self.results[concept_id] = result
                
                status = result["status"]
                levels = self._count_levels(result.get("final_json"))
                log_summary = " → ".join(result.get("pipeline_log", [])[-4:])
                
                audit_metrics = result.get("audit_summary", {})
                sem_ok = audit_metrics.get("semantic_ok", "?")
                sem_warn = audit_metrics.get("semantic_warn", "?")
                sem_fail = audit_metrics.get("semantic_fail", "?")
                
                print(f"    {status} lvl={levels} sem={sem_ok}/{sem_warn}/{sem_fail} | {log_summary}")
                
            except Exception as e:
                self.results[concept_id] = {
                    "concept_id": concept_id,
                    "status": "failed_exception",
                    "error": str(e)
                }
                print(f"    EXCEPTION: {e}")
            
            self.processed_count = i + 1
            
            if self.processed_count % CHECKPOINT_INTERVAL == 0:
                self.save_checkpoint()
            
            time.sleep(0.3)
        
        self.save_checkpoint()
        
        elapsed = time.time() - start_time
        ok_total = sum(self.stats.get(k, 0) for k in ["ok", "ok_no_audit", "pass_with_warn", "fixed"])
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {ok_total}/{total} ({100*ok_total/total:.1f}%)")
        print(f"Stats: {self.stats}")
        print(f"Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    HOST = "http://100.107.6.119:8001/v1"
    MODEL = "qwq32b_tp2_long32k_existing"
    
    resume = "--resume" in sys.argv
    
    pipeline = ESDEGlossaryPipeline(
        api_base=HOST,
        model_name=MODEL,
        dictionary_path="esde_dictionary.json",
        output_dir="glossary_output_v5"
    )
    
    pipeline.run(resume=resume)
