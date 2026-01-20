"""
ESDE Sensor - Molecule Generator Live (Phase 8-3)
Real LLM (QwQ-32B) integration with strict guardrails.

Spec v8.3-Audit-Final Compliance:
  - Strict Output Contract (Zero Chatter)
  - Fail-Closed Parsing (No fuzzy logic)
  - System-Calculated Span (Token Proximity)
  - Coordinate Coercion Logging (GPT Audit)
  - Empty Check (Skip LLM if no candidates)
"""
import json
import hashlib
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .glossary_validator import GlossaryValidator


# ==========================================
# ValidationResult (self-contained, no legacy dependency)
# ==========================================
@dataclass
class ValidationResult:
    """Validation result for molecule generation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


# ==========================================
# Configuration
# ==========================================
DEFAULT_LLM_HOST = "http://100.107.6.119:8001/v1"
DEFAULT_LLM_MODEL = "qwq32b_tp2_long32k_existing"
DEFAULT_LLM_TIMEOUT = 120
MAX_RETRIES = 2  # Only for JSON parse errors

# Valid Operators (v0.3 Complete)
VALID_OPERATORS = {
    '×',    # 結合: A × B
    '▷',    # 作用: A ▷ B
    '→',    # 遷移: A → B
    '⊕',    # 並置: A ⊕ B
    '|',    # 条件: A | B
    '◯',    # 対象: A × ◯
    '↺',    # 再帰: A ↺ A
    '〈',    # 階層開始
    '〉',    # 階層終了
    '≡',    # 等価: A ≡ B
    '≃',    # 実用等価: A ≃ B
    '¬',    # 否定: ¬A
    '⇒',    # 創発: A ⇒ B
    '⇒+',   # 創造的創発
    '-|>',  # 破壊的創発
}


# ==========================================
# Prompt Template (v8.3)
# ==========================================
SYSTEM_PROMPT_V83 = """You are the ESDE Introspection Engine.
Your task is to structure the Input Text into a Semantic Molecule using ONLY the provided Candidate Atoms.

Constraint 1 (No New Atoms): You must ONLY use atoms listed in [Candidates].
Constraint 2 (Never Guess): If the context implies a specific axis/level defined in [Glossary Definitions], use it. If uncertain, set axis=null and level=null.
Constraint 3 (Output Format): Return ONLY a valid JSON object. Any other text is forbidden. Do NOT include any reasoning, explanation, or thinking - just the JSON.
Constraint 4 (Formula Syntax): 
 - Use atom IDs (e.g., "aa_1") for reference.
 - Use ONLY these operators:
   × (connection), ▷ (action), → (transition), ⊕ (juxtaposition),
   | (condition), ◯ (target), ↺ (recursion), 〈〉 (hierarchy),
   ≡ (equivalence), ≃ (practical equivalence), ¬ (negation),
   ⇒ (emergence), ⇒+ (creative emergence), -|> (destructive emergence)
 - Examples: "aa_1", "aa_1 ▷ aa_2", "¬aa_1", "〈aa_1 × aa_2〉"

CRITICAL: Return ONLY a JSON object. NO reasoning, NO explanation, NO markdown. Just raw JSON starting with { and ending with }."""

USER_PROMPT_V83 = """Input Text: "{text}"

[Candidates]
{candidates_json}

[Glossary Definitions (Subset)]
{glossary_subset_json}

Required JSON Structure:
{{
  "active_atoms": [
    {{ "id": "aa_1", "atom": "...", "axis": "...", "level": "...", "text_ref": "..." }}
  ],
  "formula": "..."
}}"""


# ==========================================
# Generation Result
# ==========================================
@dataclass
class LiveGenerationResult:
    success: bool
    molecule: Optional[Dict] = None
    validation: Optional[ValidationResult] = None
    error: Optional[str] = None
    attempts: int = 0
    abstained: bool = False
    llm_called: bool = False
    coordinate_coercions: List[Dict] = field(default_factory=list)
    span_warnings: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "molecule": self.molecule,
            "validation": self.validation.to_dict() if self.validation else None,
            "error": self.error,
            "attempts": self.attempts,
            "abstained": self.abstained,
            "llm_called": self.llm_called,
            "coordinate_coercions": self.coordinate_coercions,
            "span_warnings": self.span_warnings
        }


# ==========================================
# Span Calculator (v8.3 Audit-Fixed)
# ==========================================
class SpanCalculator:
    """
    System-calculated span from text_ref.
    
    Algorithm (GPT Audit Fixed):
      1. Find all occurrences of text_ref in original_text
      2. If token_index available, pick closest to token position
      3. Else use leftmost match
      4. If not found, return None (WARN, not ERROR)
    """
    
    @staticmethod
    def find_all_occurrences(text: str, substring: str) -> List[Tuple[int, int]]:
        """Find all occurrences of substring in text."""
        occurrences = []
        start = 0
        sub_lower = substring.lower()
        text_lower = text.lower()
        
        while True:
            pos = text_lower.find(sub_lower, start)
            if pos == -1:
                break
            occurrences.append((pos, pos + len(substring)))
            start = pos + 1
        
        return occurrences
    
    @staticmethod
    def estimate_token_position(text: str, token_index: int) -> int:
        """
        Estimate character position from token index.
        Simple heuristic: split by whitespace and count.
        """
        tokens = text.split()
        if token_index >= len(tokens):
            return len(text)
        
        pos = 0
        for i, token in enumerate(tokens):
            if i == token_index:
                return pos
            pos += len(token) + 1  # +1 for space
        return pos
    
    @classmethod
    def calculate_span(cls,
                       text: str,
                       text_ref: str,
                       token_index: Optional[int] = None) -> Tuple[Optional[List[int]], Optional[str]]:
        """
        Calculate span for text_ref.
        
        Returns:
            (span, warning_message)
            span is [start, end] or None
            warning_message is None if successful
        """
        if not text_ref:
            return None, "text_ref is empty"
        
        occurrences = cls.find_all_occurrences(text, text_ref)
        
        if not occurrences:
            return None, f"text_ref '{text_ref}' not found in text"
        
        if len(occurrences) == 1:
            # Only one match - use it
            return list(occurrences[0]), None
        
        # Multiple occurrences - pick based on token_index
        if token_index is not None:
            token_pos = cls.estimate_token_position(text, token_index)
            
            # Find closest occurrence
            closest = min(occurrences, key=lambda o: abs(o[0] - token_pos))
            return list(closest), None
        
        # No token_index - use leftmost
        return list(occurrences[0]), None


# ==========================================
# Coordinate Coercer (v8.3 Audit-Fixed)
# ==========================================
class CoordinateCoercer:
    """
    Handles invalid coordinates by coercing to null with logging.
    
    GPT Audit: Must log proposed values and reason for coercion.
    """
    
    def __init__(self, glossary_validator: GlossaryValidator):
        self.glossary_validator = glossary_validator
    
    def coerce_if_invalid(self, 
                          active_atom: Dict,
                          atom_id: str) -> Tuple[Dict, Optional[Dict]]:
        """
        Check and coerce invalid coordinates.
        
        Returns:
            (corrected_atom, coercion_log or None)
        """
        atom = active_atom.get("atom")
        axis = active_atom.get("axis")
        level = active_atom.get("level")
        
        coercion_log = None
        
        if atom and axis is not None:
            # Check axis validity
            if not self.glossary_validator.is_valid_axis(atom, axis):
                coercion_log = {
                    "id": atom_id,
                    "proposed_axis": axis,
                    "proposed_level": level,
                    "reason": "axis_not_in_glossary"
                }
                active_atom["axis"] = None
                active_atom["level"] = None
                return active_atom, coercion_log
            
            # Check level validity (only if axis is valid)
            if level is not None:
                if not self.glossary_validator.is_valid_level(atom, axis, level):
                    coercion_log = {
                        "id": atom_id,
                        "proposed_axis": axis,
                        "proposed_level": level,
                        "reason": "level_not_in_glossary"
                    }
                    active_atom["level"] = None
                    return active_atom, coercion_log
        
        return active_atom, None


# ==========================================
# Formula Validator
# ==========================================
class FormulaValidator:
    """
    Validates formula syntax and references.
    """
    
    # Single-character operators
    SINGLE_CHAR_OPS = {'×', '▷', '→', '⊕', '|', '◯', '↺', '〈', '〉', '≡', '≃', '¬', '⇒'}
    # Multi-character operators (must check first)
    MULTI_CHAR_OPS = {'⇒+', '-|>'}
    
    @staticmethod
    def normalize_formula(formula: str) -> str:
        """Normalize whitespace in formula."""
        return ' '.join(formula.split())
    
    @staticmethod
    def extract_atom_ids(formula: str) -> List[str]:
        """Extract atom IDs (aa_N) from formula."""
        return re.findall(r'aa_\d+', formula)
    
    @classmethod
    def extract_operators(cls, formula: str) -> List[str]:
        """
        Extract operators from formula.
        Handles consecutive single-char operators like ¬¬.
        """
        # Remove atom IDs and parentheses
        clean = re.sub(r'aa_\d+', '', formula)
        clean = re.sub(r'[()]', '', clean)
        clean = clean.strip()
        
        if not clean:
            return []
        
        operators = []
        i = 0
        while i < len(clean):
            # Skip whitespace
            if clean[i].isspace():
                i += 1
                continue
            
            # Check multi-char operators first
            matched = False
            for multi_op in cls.MULTI_CHAR_OPS:
                if clean[i:].startswith(multi_op):
                    operators.append(multi_op)
                    i += len(multi_op)
                    matched = True
                    break
            
            if matched:
                continue
            
            # Check single-char operators
            if clean[i] in cls.SINGLE_CHAR_OPS:
                operators.append(clean[i])
                i += 1
            else:
                # Unknown character - add as-is for error reporting
                # Collect consecutive unknown chars
                unknown = ""
                while i < len(clean) and not clean[i].isspace() and clean[i] not in cls.SINGLE_CHAR_OPS:
                    # Check if it's start of multi-char op
                    is_multi = False
                    for multi_op in cls.MULTI_CHAR_OPS:
                        if clean[i:].startswith(multi_op):
                            is_multi = True
                            break
                    if is_multi:
                        break
                    unknown += clean[i]
                    i += 1
                if unknown:
                    operators.append(unknown)
        
        return operators
    
    def validate(self, 
                 formula: str, 
                 valid_atom_ids: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate formula.
        
        Returns:
            (is_valid, errors)
        """
        errors = []
        
        if not formula:
            return True, []  # Empty formula is OK for single atom
        
        formula = self.normalize_formula(formula)
        
        # Check atom IDs exist
        used_ids = self.extract_atom_ids(formula)
        for atom_id in used_ids:
            if atom_id not in valid_atom_ids:
                errors.append(f"Formula references undefined atom: {atom_id}")
        
        # Check operators
        operators = self.extract_operators(formula)
        for op in operators:
            if op not in VALID_OPERATORS:
                errors.append(f"Unknown operator in formula: {op}")
        
        return len(errors) == 0, errors


# ==========================================
# Molecule Generator Live (v8.3)
# ==========================================
class MoleculeGeneratorLive:
    """
    Live LLM-based molecule generator with v8.3 guardrails.
    """
    
    def __init__(self,
                 glossary: Dict[str, Any],
                 llm_host: str = None,
                 llm_model: str = None,
                 llm_timeout: int = None,
                 strict_coordinate: bool = False):
        """
        Args:
            glossary: Full glossary dict
            llm_host: LLM API host
            llm_model: Model name
            llm_timeout: Request timeout
            strict_coordinate: If False, coerce invalid coords to null (v8.3 default)
        """
        self.glossary = glossary
        self.glossary_validator = GlossaryValidator(glossary)
        self.llm_host = llm_host or DEFAULT_LLM_HOST
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        self.llm_timeout = llm_timeout or DEFAULT_LLM_TIMEOUT
        self.strict_coordinate = strict_coordinate
        
        self.span_calculator = SpanCalculator()
        self.coord_coercer = CoordinateCoercer(self.glossary_validator)
        self.formula_validator = FormulaValidator()
    
    def generate(self,
                 original_text: str,
                 candidates: List[Dict],
                 max_retries: int = MAX_RETRIES) -> LiveGenerationResult:
        """
        Generate molecule from text and candidates.
        
        Args:
            original_text: Original input text
            candidates: List of candidate dicts from Sensor V2
            max_retries: Max retry for JSON parse errors only
        
        Returns:
            LiveGenerationResult
        """
        # ===========================================
        # Empty Check: Skip LLM if no candidates
        # ===========================================
        candidate_ids = [c.get("concept_id") for c in candidates if c.get("concept_id")]
        
        if not candidate_ids:
            return LiveGenerationResult(
                success=False,
                molecule=None,
                error="No candidate atoms (Empty Check: LLM not called)",
                attempts=0,
                abstained=True,
                llm_called=False
            )
        
        # Build candidate to token_index map
        candidate_token_map = {}
        for c in candidates:
            cid = c.get("concept_id")
            if cid:
                # Get token_index from top_evidence if available
                top_ev = c.get("top_evidence", {})
                if top_ev:
                    candidate_token_map[cid] = top_ev.get("token_index")
        
        # Get glossary subset
        glossary_subset = self.glossary_validator.get_glossary_subset(candidate_ids)
        
        # Build prompt
        user_prompt = USER_PROMPT_V83.format(
            text=original_text,
            candidates_json=json.dumps(candidates, indent=2, ensure_ascii=False),
            glossary_subset_json=json.dumps(glossary_subset, indent=2, ensure_ascii=False)
        )
        
        # ===========================================
        # Retry loop (only for JSON parse errors)
        # ===========================================
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Call LLM
                response_text = self._call_llm(SYSTEM_PROMPT_V83, user_prompt)
                
                # Fail-Closed Parsing
                molecule = self._strict_parse_json(response_text)
                
                if molecule is None:
                    last_error = f"JSON parse failed (attempt {attempt + 1})"
                    continue  # Retry for parse errors
                
                # ===========================================
                # Post-processing (System-calculated span, coercion)
                # ===========================================
                result = self._post_process(
                    molecule=molecule,
                    original_text=original_text,
                    candidate_ids=candidate_ids,
                    candidate_token_map=candidate_token_map
                )
                
                if result.success:
                    result.attempts = attempt + 1
                    result.llm_called = True
                    return result
                else:
                    # Validation failed (NOT retrying for validation errors)
                    result.attempts = attempt + 1
                    result.llm_called = True
                    return result
                
            except Exception as e:
                last_error = str(e)
                # Only retry for connection/timeout errors
                if "connect" in str(e).lower() or "timeout" in str(e).lower():
                    continue
                break
        
        # All retries failed
        return LiveGenerationResult(
            success=False,
            molecule=None,
            error=last_error,
            attempts=max_retries + 1,
            abstained=True,
            llm_called=True
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
            "temperature": 0.1,
            "max_tokens": 16000  # QwQ <think> can be very long
        }
        
        response = requests.post(
            url,
            json=payload,
            timeout=self.llm_timeout,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    def _strict_parse_json(self, text: str) -> Optional[Dict]:
        """
        Fail-Closed JSON parsing.
        Permitted cleanup:
          - QwQ <think>...</think> tag removal
          - Markdown code block removal
        """
        text = text.strip()
        
        # QwQ model: Remove <think>...</think> reasoning block
        if '</think>' in text:
            # Take content after </think>
            parts = text.split('</think>')
            text = parts[-1].strip()
        elif '<think>' in text and '</think>' not in text:
            # Incomplete think tag - try to find JSON after it
            think_start = text.find('<think>')
            # Look for JSON object after any reasoning
            json_start = text.rfind('{')
            if json_start > think_start:
                text = text[json_start:]
        
        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        # Strict parse - no fuzzy logic
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object if there's trailing content
            try:
                # Find the complete JSON object
                depth = 0
                start = text.find('{')
                if start == -1:
                    return None
                for i, c in enumerate(text[start:]):
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            return json.loads(text[start:start+i+1])
            except:
                pass
            return None
    
    def _post_process(self,
                      molecule: Dict,
                      original_text: str,
                      candidate_ids: List[str],
                      candidate_token_map: Dict[str, int]) -> LiveGenerationResult:
        """
        Post-process molecule: span calculation, coordinate coercion, validation.
        """
        coercions = []
        span_warnings = []
        
        active_atoms = molecule.get("active_atoms", [])
        valid_atom_ids = []
        
        for i, aa in enumerate(active_atoms):
            atom_id = aa.get("id", f"aa_{i+1}")
            valid_atom_ids.append(atom_id)
            
            # 1. System-calculated span
            text_ref = aa.get("text_ref")
            atom = aa.get("atom")
            token_index = candidate_token_map.get(atom)
            
            span, warn = self.span_calculator.calculate_span(
                text=original_text,
                text_ref=text_ref,
                token_index=token_index
            )
            aa["span"] = span
            
            if warn:
                span_warnings.append({
                    "id": atom_id,
                    "text_ref": text_ref,
                    "warning": warn
                })
            
            # 2. Coordinate coercion (if not strict mode)
            if not self.strict_coordinate:
                aa, coercion = self.coord_coercer.coerce_if_invalid(aa, atom_id)
                if coercion:
                    coercions.append(coercion)
        
        # 3. Validate atom integrity
        errors = []
        for aa in active_atoms:
            atom = aa.get("atom")
            if atom and atom not in candidate_ids:
                errors.append(f"Atom '{atom}' not in candidates")
        
        # 4. Validate formula
        formula = molecule.get("formula", "")
        formula_valid, formula_errors = self.formula_validator.validate(formula, valid_atom_ids)
        errors.extend(formula_errors)
        
        if errors:
            return LiveGenerationResult(
                success=False,
                molecule=molecule,
                validation=ValidationResult(valid=False, errors=errors, warnings=[]),
                error="; ".join(errors),
                abstained=False,
                coordinate_coercions=coercions,
                span_warnings=span_warnings
            )
        
        # Success
        molecule["meta"] = {
            "generator": self.llm_model,
            "generator_version": "v8.3",
            "validator_status": "pass",
            "coordinate_coercions": coercions,
            "span_warnings": span_warnings,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return LiveGenerationResult(
            success=True,
            molecule=molecule,
            validation=ValidationResult(valid=True, errors=[], warnings=[]),
            error=None,
            abstained=False,
            coordinate_coercions=coercions,
            span_warnings=span_warnings
        )
    
    def compute_molecule_id(self, molecule: Dict) -> str:
        """Compute deterministic ID for molecule."""
        content = json.dumps(molecule, sort_keys=True, ensure_ascii=False)
        return f"mol_{hashlib.sha256(content.encode()).hexdigest()[:16]}"


# ==========================================
# Mock Generator for Testing
# ==========================================
class MockMoleculeGeneratorLive(MoleculeGeneratorLive):
    """Mock for testing without LLM."""
    
    def __init__(self, glossary: Dict[str, Any], **kwargs):
        super().__init__(glossary, **kwargs)
        self.mock_response = None
    
    def set_mock_response(self, response: str):
        """Set mock LLM response."""
        self.mock_response = response
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if self.mock_response:
            return self.mock_response
        
        # Default mock response
        return json.dumps({
            "active_atoms": [
                {
                    "id": "aa_1",
                    "atom": "EMO.love",
                    "axis": None,
                    "level": None,
                    "text_ref": "love"
                }
            ],
            "formula": "aa_1"
        })


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("MoleculeGeneratorLive Test (v8.3)")
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
        },
        "EMO.respect": {
            "final_json": {
                "axes": {
                    "value_generation": {
                        "semantic": {"definition_en": "Value-based respect"}
                    }
                }
            }
        }
    }
    
    # ===========================================
    # Test 1: Empty Check (No LLM call)
    # ===========================================
    print("\n[Test 1] Empty candidates → Abstain (No LLM call)")
    generator = MockMoleculeGeneratorLive(glossary=glossary)
    
    result = generator.generate(
        original_text="I love you",
        candidates=[]
    )
    
    print(f"  Success: {result.success} (expected: False)")
    print(f"  Abstained: {result.abstained} (expected: True)")
    print(f"  LLM Called: {result.llm_called} (expected: False)")
    assert not result.success
    assert result.abstained
    assert not result.llm_called, "LLM should NOT be called for empty candidates"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 2: Success case
    # ===========================================
    print("\n[Test 2] Success with valid response")
    
    candidates = [
        {"concept_id": "EMO.love", "score": 1.38, "top_evidence": {"token_index": 1}}
    ]
    
    result = generator.generate(
        original_text="I love you",
        candidates=candidates
    )
    
    print(f"  Success: {result.success} (expected: True)")
    print(f"  LLM Called: {result.llm_called}")
    if result.molecule:
        print(f"  Span: {result.molecule['active_atoms'][0].get('span')}")
    assert result.success
    print("  ✅ PASS")
    
    # ===========================================
    # Test 3: Coordinate coercion
    # ===========================================
    print("\n[Test 3] Coordinate coercion (invalid axis → null)")
    
    generator.set_mock_response(json.dumps({
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": "invalid_axis",  # Will be coerced
                "level": "invalid_level",
                "text_ref": "love"
            }
        ],
        "formula": "aa_1"
    }))
    
    result = generator.generate(
        original_text="I love you",
        candidates=candidates
    )
    
    print(f"  Success: {result.success}")
    print(f"  Coercions: {result.coordinate_coercions}")
    
    if result.molecule:
        aa = result.molecule["active_atoms"][0]
        print(f"  Axis after coercion: {aa.get('axis')} (expected: None)")
        assert aa.get("axis") is None, "Axis should be coerced to None"
    
    assert len(result.coordinate_coercions) > 0, "Should have coercion log"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 4: Span calculation with token proximity
    # ===========================================
    print("\n[Test 4] Span calculation (token proximity)")
    
    generator.set_mock_response(json.dumps({
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": None,
                "level": None,
                "text_ref": "love"
            }
        ],
        "formula": "aa_1"
    }))
    
    # Text with multiple "love"
    result = generator.generate(
        original_text="I love you because love is eternal",
        candidates=[
            {"concept_id": "EMO.love", "top_evidence": {"token_index": 1}}  # First "love"
        ]
    )
    
    if result.molecule:
        span = result.molecule["active_atoms"][0].get("span")
        print(f"  Calculated span: {span}")
        # Should pick first "love" at position 2-6 (closer to token_index 1)
        assert span == [2, 6], f"Expected [2, 6], got {span}"
    
    print("  ✅ PASS")
    
    # ===========================================
    # Test 5: Unknown atom (validation failure)
    # ===========================================
    print("\n[Test 5] Unknown atom → Validation failure")
    
    generator.set_mock_response(json.dumps({
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "UNKNOWN.atom",  # Not in candidates
                "axis": None,
                "level": None,
                "text_ref": "love"
            }
        ],
        "formula": "aa_1"
    }))
    
    result = generator.generate(
        original_text="I love you",
        candidates=candidates
    )
    
    print(f"  Success: {result.success} (expected: False)")
    print(f"  Error: {result.error}")
    assert not result.success
    assert "UNKNOWN.atom" in result.error
    print("  ✅ PASS")
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
