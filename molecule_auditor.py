#!/usr/bin/env python3
"""
ESDE Phase 8 — Molecule Auditor (Phase β)
===========================================

LLM-based selective judge for flagged draft molecules.
Only called on molecules flagged by rule_generator.py.

Pattern: Same as auditor_a1.py (A1 pipeline)
  - Input: Draft molecule + original text
  - Output: PASS / FAIL + one-line reason
  - Lightweight prompt (judge-only, no generation)

3AI Approval: 2026-03-02

Usage:
  from molecule_auditor import MoleculeAuditor
  auditor = MoleculeAuditor()
  verdict = auditor.audit(draft_molecule_dict)
"""

import json
import re
import requests
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone


# ============================================================
# Configuration
# ============================================================

DEFAULT_LLM_HOST = "http://100.107.6.119:8001/v1"
DEFAULT_LLM_MODEL = "qwq32b_tp2_fp16_8k_b8"
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_MAX_TOKENS = 2048


# ============================================================
# Prompt
# ============================================================

AUDIT_SYSTEM_PROMPT = """You are the ESDE Molecule Auditor.

You judge whether a draft semantic molecule correctly represents the meaning of the source sentence.

A molecule consists of:
- active_atoms: concepts from the ESDE ontology (326 atoms)
- formula: operators connecting atoms (▷=action, ×=connection, ⊕=juxtaposition, ¬=negation)

Your job: Judge if the atoms and formula capture the core meaning of the sentence.

Rules:
- Respond ONLY with a JSON object: {"verdict": "PASS" or "FAIL", "reason": "one sentence"}
- PASS = atoms are reasonable and formula captures the structural relationship
- FAIL = atoms are wrong, formula is misleading, or critical meaning is lost
- Be lenient on axis/level (these are refinements, not core meaning)
- Be strict on atom selection and formula structure

CRITICAL: Return ONLY JSON. No reasoning, no explanation, just the JSON."""

AUDIT_USER_TEMPLATE = """Source sentence: "{sentence}"

Draft molecule:
  Atoms: {atoms_summary}
  Formula: {formula}
  Flags: {flags}

Is this molecule a reasonable semantic representation? Return JSON only."""


# ============================================================
# Data Structures
# ============================================================

@dataclass
class AuditVerdict:
    """Result of Phase β audit."""
    verdict: str          # "PASS" or "FAIL"
    reason: str           # One-line explanation
    llm_called: bool      # Whether LLM was actually called
    error: Optional[str] = None  # Error message if LLM failed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "llm_called": self.llm_called,
            "error": self.error,
        }


# ============================================================
# Auditor
# ============================================================

class MoleculeAuditor:
    """
    LLM-based judge for flagged molecules.
    
    Only audits molecules with needs_audit=True.
    Non-flagged molecules automatically PASS.
    """
    
    def __init__(
        self,
        llm_host: str = DEFAULT_LLM_HOST,
        llm_model: str = DEFAULT_LLM_MODEL,
        llm_timeout: int = DEFAULT_LLM_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.llm_host = llm_host
        self.llm_model = llm_model
        self.llm_timeout = llm_timeout
        self.max_tokens = max_tokens
        
        self.stats = {
            "total": 0,
            "auto_pass": 0,
            "llm_called": 0,
            "llm_pass": 0,
            "llm_fail": 0,
            "llm_error": 0,
        }
    
    def audit(self, molecule_dict: Dict[str, Any]) -> AuditVerdict:
        """
        Audit a draft molecule.
        
        Args:
            molecule_dict: Output from DraftMolecule.to_dict()
            
        Returns:
            AuditVerdict
        """
        self.stats["total"] += 1
        
        # Auto-pass non-flagged molecules
        if not molecule_dict.get("needs_audit", False):
            self.stats["auto_pass"] += 1
            return AuditVerdict(
                verdict="PASS",
                reason="No flags — auto-approved",
                llm_called=False,
            )
        
        # Build audit prompt
        atoms_summary = self._format_atoms(molecule_dict.get("active_atoms", []))
        formula = molecule_dict.get("formula", "")
        flags = molecule_dict.get("meta", {}).get("flags", [])
        sentence = molecule_dict.get("source_text", "")
        
        user_prompt = AUDIT_USER_TEMPLATE.format(
            sentence=sentence[:300],
            atoms_summary=atoms_summary,
            formula=formula,
            flags=", ".join(flags) if flags else "none",
        )
        
        # Call LLM
        self.stats["llm_called"] += 1
        
        try:
            response = self._call_llm(user_prompt)
            verdict = self._parse_verdict(response)
            
            if verdict.verdict == "PASS":
                self.stats["llm_pass"] += 1
            else:
                self.stats["llm_fail"] += 1
            
            return verdict
            
        except Exception as e:
            self.stats["llm_error"] += 1
            # On error, conservatively PASS (don't block pipeline)
            return AuditVerdict(
                verdict="PASS",
                reason="LLM error — conservative pass",
                llm_called=True,
                error=str(e),
            )
    
    def _format_atoms(self, active_atoms: List[Dict]) -> str:
        """Format atoms for prompt (compact)."""
        parts = []
        for aa in active_atoms:
            aid = aa.get("id", "?")
            atom = aa.get("atom", "?")
            ref = aa.get("text_ref", "")
            parts.append(f'{aid}={atom}("{ref}")')
        return ", ".join(parts)
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call LLM API."""
        url = f"{self.llm_host}/chat/completions"
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.llm_timeout,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _parse_verdict(self, response_text: str) -> AuditVerdict:
        """
        Parse LLM response into verdict.
        Handles QwQ's <think>...</think> blocks.
        """
        # Strip thinking blocks
        text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        text = text.strip()
        
        # Try to find JSON
        # Pattern 1: raw JSON
        json_match = re.search(r'\{[^{}]*\}', text)
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                verdict = data.get("verdict", "PASS").upper()
                reason = data.get("reason", "no reason given")
                
                if verdict not in ("PASS", "FAIL"):
                    verdict = "PASS"  # Conservative
                
                return AuditVerdict(
                    verdict=verdict,
                    reason=reason[:200],
                    llm_called=True,
                )
            except json.JSONDecodeError:
                pass
        
        # Fallback: look for PASS/FAIL keywords
        upper = text.upper()
        if "FAIL" in upper:
            return AuditVerdict(
                verdict="FAIL",
                reason="(parsed from text: FAIL keyword found)",
                llm_called=True,
            )
        
        # Conservative: PASS
        return AuditVerdict(
            verdict="PASS",
            reason="(parse fallback — no clear verdict found)",
            llm_called=True,
        )
    
    def batch_audit(
        self,
        molecule_dicts: List[Dict[str, Any]],
        max_workers: int = 8,
    ) -> List[Tuple[Dict[str, Any], AuditVerdict]]:
        """
        Audit multiple molecules in parallel.
        
        Uses ThreadPoolExecutor to send concurrent requests
        to the LLM server (which supports batch processing).
        
        Args:
            molecule_dicts: List of molecule dicts from DraftMolecule.to_dict()
            max_workers: Number of concurrent threads (match server batch size)
            
        Returns:
            List of (molecule_dict, AuditVerdict) pairs
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(molecule_dicts)
        
        def _audit_one(idx: int, mol: Dict) -> Tuple[int, AuditVerdict]:
            verdict = self.audit(mol)
            return idx, verdict
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_audit_one, i, mol): i
                for i, mol in enumerate(molecule_dicts)
            }
            
            done = 0
            total = len(futures)
            for future in as_completed(futures):
                idx, verdict = future.result()
                results[idx] = (molecule_dicts[idx], verdict)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"      Audit: {done}/{total}")
        
        return results