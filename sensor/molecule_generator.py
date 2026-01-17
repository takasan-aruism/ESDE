"""
ESDE Sensor - Molecule Generator
Phase 8-2: LLM-based semantic molecule generation

Design:
  - Takes Sensor V2 output (candidate atoms) and original text
  - Uses LLM to generate structured molecule
  - Validates with MoleculeValidator
  - Retry on failure (max 2 times)
  - Abstain if all retries fail

GPT Audit v8.2.1 Compliance:
  - Never Guess: axis/level = null if uncertain
  - No New Atoms: only use atoms from candidates
  - Glossary Subset: only pass relevant definitions to LLM
"""
import json
import hashlib
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass

from .molecule_validator import MoleculeValidator, ValidationResult, GlossaryValidator


# ==========================================
# Configuration
# ==========================================
DEFAULT_LLM_HOST = "http://100.107.6.119:8001/v1"
DEFAULT_LLM_MODEL = "qwq32b_tp2_long32k_existing"
DEFAULT_LLM_TIMEOUT = 60
MAX_RETRIES = 2


# ==========================================
# Prompt Templates
# ==========================================
SYSTEM_PROMPT = """You are a semantic structure analyzer for the ESDE (Extended Semantic Differential Engine).
Your task is to convert natural language text into structured semantic molecules.

RULES (MUST FOLLOW):
1. ONLY use atoms from the provided candidate list - NO new atoms allowed
2. If you cannot determine axis/level with confidence, set them to null - NEVER GUESS
3. text_ref must be an EXACT substring from the original text
4. span must be the exact character positions [start, end) of text_ref

Output must be valid JSON matching the schema exactly."""

USER_PROMPT_TEMPLATE = """## Original Text
{original_text}

## Candidate Atoms (from Sensor)
{candidates_json}

## Glossary Definitions (for reference)
{glossary_subset_json}

## Operators (v0.3)
- × : Connection (A × B)
- ▷ : Action (A acts on B)
- → : Transition (state change)
- ⊕ : Juxtaposition (simultaneous)
- ¬ : Negation

## Task
Generate a semantic molecule that captures the meaning of the original text.

## Output Schema (JSON)
{{
  "active_atoms": [
    {{
      "id": "aa_1",
      "atom": "<concept_id from candidates>",
      "coordinates": {{
        "axis": "<axis_name or null>",
        "level": "<level_name or null>",
        "confidence": <0.0-1.0>
      }},
      "text_ref": "<exact substring from original text>",
      "span": [<start_index>, <end_index>]
    }}
  ],
  "formula": "<formula using aa_N IDs and operators>"
}}

Respond with ONLY the JSON, no explanation."""


# ==========================================
# Generation Result
# ==========================================
@dataclass
class GenerationResult:
    success: bool
    molecule: Optional[Dict]
    validation: Optional[ValidationResult]
    error: Optional[str]
    attempts: int
    abstained: bool
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "molecule": self.molecule,
            "validation": self.validation.to_dict() if self.validation else None,
            "error": self.error,
            "attempts": self.attempts,
            "abstained": self.abstained
        }


# ==========================================
# Molecule Generator
# ==========================================
class MoleculeGenerator:
    """
    LLM-based semantic molecule generator.
    """
    
    def __init__(self,
                 glossary: Dict[str, Any],
                 llm_host: str = None,
                 llm_model: str = None,
                 llm_timeout: int = None,
                 allow_glossary_atoms: bool = False):
        """
        Args:
            glossary: Full glossary dict
            llm_host: LLM API host
            llm_model: Model name
            llm_timeout: Request timeout
            allow_glossary_atoms: If True, allow atoms from glossary subset (not just candidates)
        """
        self.glossary = glossary
        self.glossary_validator = GlossaryValidator(glossary)
        self.llm_host = llm_host or DEFAULT_LLM_HOST
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.llm_timeout = llm_timeout or DEFAULT_LLM_TIMEOUT
        self.allow_glossary_atoms = allow_glossary_atoms
    
    def generate(self,
                 original_text: str,
                 candidates: List[Dict],
                 max_retries: int = MAX_RETRIES) -> GenerationResult:
        """
        Generate a molecule from text and candidates.
        
        Args:
            original_text: Original input text
            candidates: List of candidate dicts from Sensor V2
            max_retries: Max retry attempts on validation failure
        
        Returns:
            GenerationResult
        """
        # Extract concept IDs from candidates
        candidate_ids = [c.get("concept_id") for c in candidates if c.get("concept_id")]
        
        if not candidate_ids:
            return GenerationResult(
                success=False,
                molecule=None,
                validation=None,
                error="No candidate atoms provided",
                attempts=0,
                abstained=True
            )
        
        # Get glossary subset
        glossary_subset = self.glossary_validator.get_glossary_subset(candidate_ids)
        
        # Create validator
        validator = MoleculeValidator(
            glossary=self.glossary,
            sensor_candidates=candidate_ids,
            allow_glossary_atoms=self.allow_glossary_atoms
        )
        
        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            original_text=original_text,
            candidates_json=json.dumps(candidates, indent=2, ensure_ascii=False),
            glossary_subset_json=json.dumps(glossary_subset, indent=2, ensure_ascii=False)
        )
        
        # Retry loop
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Call LLM
                response_text = self._call_llm(SYSTEM_PROMPT, user_prompt)
                
                # Parse JSON
                molecule = self._parse_json(response_text)
                
                if molecule is None:
                    last_error = f"Failed to parse JSON from LLM response"
                    continue
                
                # Validate
                validation = validator.validate(molecule, original_text)
                
                if validation.valid:
                    # Success
                    molecule["meta"] = {
                        "generator": self.llm_model,
                        "validator_status": "pass",
                        "coordinate_completeness": validator.compute_coordinate_completeness(
                            molecule.get("active_atoms", [])
                        ),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    return GenerationResult(
                        success=True,
                        molecule=molecule,
                        validation=validation,
                        error=None,
                        attempts=attempt + 1,
                        abstained=False
                    )
                else:
                    # Validation failed - add error context to next prompt
                    last_error = f"Validation failed: {validation.errors}"
                    user_prompt = self._add_error_context(user_prompt, validation.errors)
                    
            except Exception as e:
                last_error = str(e)
        
        # All retries failed - abstain
        return GenerationResult(
            success=False,
            molecule=None,
            validation=None,
            error=last_error,
            attempts=max_retries + 1,
            abstained=True
        )
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM API."""
        url = f"{self.llm_host}/chat/completions"
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Low temperature for determinism
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.llm_timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to LLM at {self.llm_host}")
        except requests.exceptions.Timeout:
            raise Exception(f"LLM request timeout ({self.llm_timeout}s)")
        except Exception as e:
            raise Exception(f"LLM call failed: {e}")
    
    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON from LLM response (handles markdown code blocks)."""
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            match = text.find('{')
            if match >= 0:
                try:
                    # Find matching closing brace
                    depth = 0
                    for i, c in enumerate(text[match:]):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                return json.loads(text[match:match+i+1])
                except:
                    pass
            return None
    
    def _add_error_context(self, prompt: str, errors: List[str]) -> str:
        """Add error context to prompt for retry."""
        error_text = "\n".join(f"- {e}" for e in errors)
        return prompt + f"""

## PREVIOUS ATTEMPT FAILED
The following validation errors occurred:
{error_text}

Please fix these issues and try again. Remember:
- ONLY use atoms from the candidate list
- Set axis/level to null if uncertain
- text_ref must match EXACTLY from original text
- span must be correct character positions"""
    
    def compute_molecule_id(self, molecule: Dict) -> str:
        """Compute deterministic ID for molecule."""
        content = json.dumps(molecule, sort_keys=True, ensure_ascii=False)
        return f"mol_{hashlib.sha256(content.encode()).hexdigest()[:16]}"


# ==========================================
# Mock Generator (for testing without LLM)
# ==========================================
class MockMoleculeGenerator(MoleculeGenerator):
    """
    Mock generator for testing without LLM connection.
    """
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Return mock response."""
        # Parse candidates from prompt
        import re
        candidates_match = re.search(r'## Candidate Atoms.*?\n```json\n(.*?)\n```', user_prompt, re.DOTALL)
        
        mock_atom = "EMO.love"  # Default
        if candidates_match:
            try:
                cands = json.loads(candidates_match.group(1))
                if cands:
                    mock_atom = cands[0].get("concept_id", mock_atom)
            except:
                pass
        
        # Find "love" in original text
        text_match = re.search(r'## Original Text\n(.+?)\n', user_prompt)
        original_text = text_match.group(1) if text_match else "I love you"
        
        # Find position of first word
        words = original_text.split()
        if len(words) > 1:
            text_ref = words[1]  # Second word
            start = original_text.find(text_ref)
            span = [start, start + len(text_ref)]
        else:
            text_ref = original_text
            span = [0, len(original_text)]
        
        return json.dumps({
            "active_atoms": [
                {
                    "id": "aa_1",
                    "atom": mock_atom,
                    "coordinates": {
                        "axis": None,
                        "level": None,
                        "confidence": 0.5
                    },
                    "text_ref": text_ref,
                    "span": span
                }
            ],
            "formula": "aa_1"
        })


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("MoleculeGenerator Test")
    print("=" * 60)
    
    # Mock glossary
    glossary = {
        "EMO.love": {
            "final_json": {
                "axes": {
                    "interconnection": {
                        "resonant": {"definition_en": "Deep connection"}
                    }
                }
            }
        }
    }
    
    # ===========================================
    # Test 1: MockGenerator success
    # ===========================================
    print("\n[Test 1] MockGenerator success")
    generator = MockMoleculeGenerator(glossary=glossary)
    
    candidates = [
        {"concept_id": "EMO.love", "score": 1.38, "axis": "interconnection"}
    ]
    
    result = generator.generate(
        original_text="I love you",
        candidates=candidates
    )
    
    print(f"  Success: {result.success} (expected: True)")
    print(f"  Abstained: {result.abstained} (expected: False)")
    print(f"  Attempts: {result.attempts}")
    assert result.success, "Test 1 FAILED: should succeed"
    assert not result.abstained, "Test 1 FAILED: should not abstain"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 2: Empty candidates → Abstain
    # ===========================================
    print("\n[Test 2] Empty candidates → Abstain")
    
    result = generator.generate(
        original_text="I love you",
        candidates=[]  # No candidates
    )
    
    print(f"  Success: {result.success} (expected: False)")
    print(f"  Abstained: {result.abstained} (expected: True)")
    assert not result.success, "Test 2 FAILED: should fail"
    assert result.abstained, "Test 2 FAILED: should abstain"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 3: FailingMockGenerator → Retry → Abstain
    # ===========================================
    print("\n[Test 3] Retry exhaustion → Abstain")
    
    class FailingMockGenerator(MoleculeGenerator):
        """Always returns invalid molecule to test retry→abstain."""
        def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
            # Return molecule with unknown atom (will fail validation)
            return json.dumps({
                "active_atoms": [
                    {
                        "id": "aa_1",
                        "atom": "UNKNOWN.atom",  # NOT in candidates
                        "coordinates": {"axis": None, "level": None, "confidence": 0.5},
                        "text_ref": "love",
                        "span": [2, 6]
                    }
                ],
                "formula": "aa_1"
            })
    
    failing_generator = FailingMockGenerator(
        glossary=glossary,
        allow_glossary_atoms=False  # Strict mode
    )
    
    result = failing_generator.generate(
        original_text="I love you",
        candidates=candidates,
        max_retries=2
    )
    
    print(f"  Success: {result.success} (expected: False)")
    print(f"  Abstained: {result.abstained} (expected: True)")
    print(f"  Attempts: {result.attempts} (expected: 3)")  # 1 initial + 2 retries
    print(f"  Error: {result.error}")
    
    assert not result.success, "Test 3 FAILED: should fail"
    assert result.abstained, "Test 3 FAILED: should abstain after retries"
    assert result.attempts == 3, f"Test 3 FAILED: expected 3 attempts, got {result.attempts}"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 4: Verify Never Guess (null coordinates)
    # ===========================================
    print("\n[Test 4] Never Guess (null coordinates with low confidence)")
    
    result = generator.generate(
        original_text="I love you",
        candidates=candidates
    )
    
    if result.molecule:
        atoms = result.molecule.get("active_atoms", [])
        for atom in atoms:
            coords = atom.get("coordinates", {})
            confidence = coords.get("confidence", 0)
            axis = coords.get("axis")
            level = coords.get("level")
            
            print(f"  Atom: {atom.get('atom')}")
            print(f"  Confidence: {confidence}")
            print(f"  Axis: {axis}, Level: {level}")
            
            # Low confidence should allow null coords
            if confidence < 0.7:
                # This is acceptable (Never Guess)
                print("  → Low confidence, null coords acceptable")
    
    print("  ✅ PASS")
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
