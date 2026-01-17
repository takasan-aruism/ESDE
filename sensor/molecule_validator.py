"""
ESDE Sensor - Molecule Validator
Phase 8-2: Validates generated molecules for integrity

GPT Audit v8.2.1 Compliance:
  - Atom Integrity: 2-tier (Sensor candidates + optional Glossary subset)
  - Coordinate Valid: axis/level must exist in Glossary
  - Evidence Linkage: span must match text_ref exactly
  - Syntax Check: operator validity, bracket matching
"""
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass


# ==========================================
# Operator Definitions (v0.3 Complete)
# ==========================================
VALID_OPERATORS = {
    '×',   # 結合: Connection
    '▷',   # 作用: Action
    '→',   # 遷移: Transition
    '⊕',   # 並置: Juxtaposition
    '|',   # 条件: Condition
    '◯',   # 対象: Target
    '↺',   # 再帰: Recursion
    '〈',   # 階層開始: Hierarchy open
    '〉',   # 階層終了: Hierarchy close
    '≡',   # 等価: Equivalence
    '≃',   # 実用等価: Practical Equivalence
    '¬',   # 否定: Negation
    '⇒',   # 創発: Emergence
    '⇒+',  # 創造的創発: Creative Emergence
    '-|>', # 破壊的創発: Destructive Emergence
}


# ==========================================
# Validation Result
# ==========================================
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __bool__(self):
        return self.valid
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


# ==========================================
# Glossary Loader (for validation)
# ==========================================
class GlossaryValidator:
    """
    Validates atoms and coordinates against Glossary definitions.
    """
    
    def __init__(self, glossary: Dict[str, Any]):
        # Handle nested glossary structure (glossary_results.json format)
        if "glossary" in glossary and isinstance(glossary["glossary"], dict):
            self.glossary = glossary["glossary"]
        else:
            self.glossary = glossary
        self._build_indexes()
    
    def _build_indexes(self):
        """Build lookup indexes."""
        self.valid_atoms: Set[str] = set()
        self.atom_axes: Dict[str, Set[str]] = {}  # atom -> set of valid axes
        self.axis_levels: Dict[str, Dict[str, Set[str]]] = {}  # atom -> axis -> set of valid levels
        
        for concept_id, content in self.glossary.items():
            # Skip non-dict entries
            if not isinstance(content, dict):
                continue
            
            self.valid_atoms.add(concept_id)
            
            final = content.get("final_json", content)
            if not isinstance(final, dict):
                continue
            
            axes = final.get("axes", {})
            if not isinstance(axes, dict):
                continue
            
            self.atom_axes[concept_id] = set()
            self.axis_levels[concept_id] = {}
            
            for axis_name, levels in axes.items():
                self.atom_axes[concept_id].add(axis_name)
                
                if isinstance(levels, dict):
                    self.axis_levels[concept_id][axis_name] = set(levels.keys())
    
    def is_valid_atom(self, atom: str) -> bool:
        """Check if atom exists in glossary."""
        return atom in self.valid_atoms
    
    def is_valid_axis(self, atom: str, axis: str) -> bool:
        """Check if axis exists for this atom."""
        if atom not in self.atom_axes:
            return False
        return axis in self.atom_axes[atom]
    
    def is_valid_level(self, atom: str, axis: str, level: str) -> bool:
        """Check if level exists for this atom+axis."""
        if atom not in self.axis_levels:
            return False
        if axis not in self.axis_levels[atom]:
            return False
        return level in self.axis_levels[atom][axis]
    
    def get_valid_axes(self, atom: str) -> Set[str]:
        """Get valid axes for an atom."""
        return self.atom_axes.get(atom, set())
    
    def get_valid_levels(self, atom: str, axis: str) -> Set[str]:
        """Get valid levels for an atom+axis."""
        return self.axis_levels.get(atom, {}).get(axis, set())
    
    def get_glossary_subset(self, atoms: List[str]) -> Dict[str, Any]:
        """
        Extract glossary subset for given atoms.
        Used to create lightweight context for LLM.
        """
        subset = {}
        for atom in atoms:
            if atom in self.glossary:
                content = self.glossary[atom]
                final = content.get("final_json", content)
                if isinstance(final, dict) and "axes" in final:
                    subset[atom] = {"axes": final["axes"]}
        return subset


# ==========================================
# Molecule Validator
# ==========================================
class MoleculeValidator:
    """
    Validates generated molecules for integrity.
    
    Checks:
      1. Atom Integrity: atoms must be in allowed set
      2. Operator Valid: operators must be in v0.3 spec
      3. Syntax Check: bracket matching
      4. Coordinate Valid: axis/level must exist in Glossary
      5. Evidence Linkage: span must match text_ref exactly
    """
    
    def __init__(self,
                 glossary: Dict[str, Any],
                 sensor_candidates: List[str] = None,
                 allow_glossary_atoms: bool = False):
        """
        Args:
            glossary: Full glossary dict
            sensor_candidates: List of concept_ids from Sensor V2 output
            allow_glossary_atoms: If True, allow any atom in glossary subset
        """
        self.glossary_validator = GlossaryValidator(glossary)
        self.sensor_candidates = set(sensor_candidates or [])
        self.allow_glossary_atoms = allow_glossary_atoms
        
        # Build allowed atoms set
        self.allowed_atoms = self.sensor_candidates.copy()
        if allow_glossary_atoms:
            self.allowed_atoms.update(self.glossary_validator.valid_atoms)
    
    def validate(self, 
                 molecule: Dict[str, Any],
                 original_text: str) -> ValidationResult:
        """
        Validate a molecule structure.
        
        Args:
            molecule: Generated molecule dict with active_atoms, formula
            original_text: Original input text
        
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        active_atoms = molecule.get("active_atoms", [])
        formula = molecule.get("formula", "")
        
        # 1. Atom Integrity
        atom_errors = self._check_atom_integrity(active_atoms)
        errors.extend(atom_errors)
        
        # 2. Operator Valid
        op_errors = self._check_operator_validity(formula)
        errors.extend(op_errors)
        
        # 3. Syntax Check
        syntax_errors = self._check_syntax(formula)
        errors.extend(syntax_errors)
        
        # 4. Coordinate Valid
        coord_errors, coord_warnings = self._check_coordinate_validity(active_atoms)
        errors.extend(coord_errors)
        warnings.extend(coord_warnings)
        
        # 5. Evidence Linkage
        evidence_errors = self._check_evidence_linkage(active_atoms, original_text)
        errors.extend(evidence_errors)
        
        # 6. Coverage Policy (null coordinates with high confidence)
        coverage_warnings = self._check_coverage_policy(active_atoms)
        warnings.extend(coverage_warnings)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _check_atom_integrity(self, active_atoms: List[Dict]) -> List[str]:
        """
        Check that all atoms are in allowed set.
        
        GPT Audit v8.2.1: 2-tier check
          - Tier 1: Sensor candidates (always allowed)
          - Tier 2: Glossary atoms (if ALLOW_GLOSSARY_ATOMS=True)
        """
        errors = []
        for aa in active_atoms:
            atom = aa.get("atom")
            if not atom:
                errors.append(f"Active atom missing 'atom' field: {aa.get('id')}")
                continue
            
            if atom not in self.allowed_atoms:
                if self.allow_glossary_atoms:
                    errors.append(f"Atom '{atom}' not in Glossary")
                else:
                    errors.append(f"Atom '{atom}' not in Sensor candidates: {list(self.sensor_candidates)}")
        
        return errors
    
    def _check_operator_validity(self, formula: str) -> List[str]:
        """Check that all operators in formula are valid."""
        errors = []
        
        # Extract operators from formula
        # Pattern: non-alphanumeric sequences that aren't whitespace or brackets
        tokens = re.findall(r'[^\w\s\(\)\[\]〈〉]+', formula)
        
        for token in tokens:
            # Check if it's a known operator or part of one
            if token not in VALID_OPERATORS:
                # Check if it's a substring issue
                is_valid = False
                for op in VALID_OPERATORS:
                    if token in op or op in token:
                        is_valid = True
                        break
                if not is_valid:
                    errors.append(f"Unknown operator: '{token}'")
        
        return errors
    
    def _check_syntax(self, formula: str) -> List[str]:
        """Check bracket matching and basic syntax."""
        errors = []
        
        # Check bracket pairs
        bracket_pairs = [('(', ')'), ('[', ']'), ('〈', '〉')]
        
        for open_b, close_b in bracket_pairs:
            open_count = formula.count(open_b)
            close_count = formula.count(close_b)
            if open_count != close_count:
                errors.append(f"Unmatched brackets: {open_b}={open_count}, {close_b}={close_count}")
        
        return errors
    
    def _check_coordinate_validity(self, active_atoms: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Check that coordinates are valid in Glossary.
        
        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        
        for aa in active_atoms:
            atom = aa.get("atom")
            coords = aa.get("coordinates", {})
            axis = coords.get("axis")
            level = coords.get("level")
            
            if not atom:
                continue
            
            # Check if atom exists in glossary
            if not self.glossary_validator.is_valid_atom(atom):
                errors.append(f"Atom '{atom}' not found in Glossary")
                continue
            
            # Check axis if specified
            if axis is not None:
                if not self.glossary_validator.is_valid_axis(atom, axis):
                    valid_axes = self.glossary_validator.get_valid_axes(atom)
                    errors.append(f"Axis '{axis}' not valid for '{atom}'. Valid: {valid_axes}")
                    continue
                
                # Check level if specified
                if level is not None:
                    if not self.glossary_validator.is_valid_level(atom, axis, level):
                        valid_levels = self.glossary_validator.get_valid_levels(atom, axis)
                        errors.append(f"Level '{level}' not valid for '{atom}@{axis}'. Valid: {valid_levels}")
        
        return errors, warnings
    
    def _check_evidence_linkage(self, active_atoms: List[Dict], original_text: str) -> List[str]:
        """
        Check that text_ref matches span exactly.
        
        GPT Audit v8.2.1:
          - span must be in range [0, len(text))
          - text_ref == text[span[0]:span[1]]
        """
        errors = []
        text_len = len(original_text)
        
        for aa in active_atoms:
            text_ref = aa.get("text_ref")
            span = aa.get("span")
            
            if text_ref is None or span is None:
                # text_ref and span are optional
                continue
            
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                errors.append(f"Invalid span format for '{aa.get('id')}': {span}")
                continue
            
            start, end = span
            
            # Check range
            if start < 0 or end > text_len or start >= end:
                errors.append(f"Span out of range for '{aa.get('id')}': {span} (text_len={text_len})")
                continue
            
            # Check exact match
            actual_text = original_text[start:end]
            if actual_text != text_ref:
                errors.append(f"text_ref mismatch for '{aa.get('id')}': "
                            f"expected '{text_ref}', got '{actual_text}' at span {span}")
        
        return errors
    
    def _check_coverage_policy(self, active_atoms: List[Dict]) -> List[str]:
        """
        Check for null coordinates with high confidence (contradictory).
        """
        warnings = []
        CONFIDENCE_THRESHOLD = 0.7
        
        for aa in active_atoms:
            coords = aa.get("coordinates", {})
            confidence = coords.get("confidence", 0.0)
            axis = coords.get("axis")
            level = coords.get("level")
            
            if confidence >= CONFIDENCE_THRESHOLD:
                if axis is None or level is None:
                    warnings.append(
                        f"High confidence ({confidence}) but null coordinates for '{aa.get('atom')}' "
                        f"(axis={axis}, level={level})"
                    )
        
        return warnings
    
    def compute_coordinate_completeness(self, active_atoms: List[Dict]) -> float:
        """
        Compute ratio of atoms with complete coordinates.
        
        Returns:
            0.0 to 1.0 (1.0 = all atoms have axis and level)
        """
        if not active_atoms:
            return 0.0
        
        complete_count = 0
        for aa in active_atoms:
            coords = aa.get("coordinates", {})
            if coords.get("axis") is not None and coords.get("level") is not None:
                complete_count += 1
        
        return complete_count / len(active_atoms)


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("MoleculeValidator Test")
    print("=" * 60)
    
    # Mock glossary
    glossary = {
        "EMO.love": {
            "final_json": {
                "axes": {
                    "interconnection": {
                        "resonant": {"definition_en": "Deep connection"},
                        "independent": {"definition_en": "Self-love"}
                    },
                    "ontological": {
                        "relational": {"definition_en": "Relational love"}
                    }
                }
            }
        }
    }
    
    original_text = "I love you"
    
    # ===========================================
    # Test 1: Valid molecule (PASS expected)
    # ===========================================
    print("\n[Test 1] Valid molecule")
    validator = MoleculeValidator(
        glossary=glossary,
        sensor_candidates=["EMO.love"],
        allow_glossary_atoms=True
    )
    
    molecule_valid = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "coordinates": {
                    "axis": "interconnection",
                    "level": "resonant",
                    "confidence": 0.95
                },
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_valid, original_text)
    print(f"  Valid: {result.valid} (expected: True)")
    print(f"  Errors: {result.errors}")
    assert result.valid, "Test 1 FAILED: should be valid"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 2: Unknown Atom (NG expected)
    # ===========================================
    print("\n[Test 2] Unknown Atom (not in candidates)")
    validator_strict = MoleculeValidator(
        glossary=glossary,
        sensor_candidates=["EMO.love"],  # Only EMO.love allowed
        allow_glossary_atoms=False
    )
    
    molecule_unknown_atom = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.hate",  # NOT in candidates
                "coordinates": {"axis": None, "level": None, "confidence": 0.5},
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator_strict.validate(molecule_unknown_atom, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 2 FAILED: should be invalid (unknown atom)"
    assert any("EMO.hate" in e for e in result.errors), "Test 2 FAILED: error should mention EMO.hate"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 3: Bad span - out of range (NG expected)
    # ===========================================
    print("\n[Test 3] Bad span (out of range)")
    
    molecule_bad_span = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "coordinates": {"axis": None, "level": None, "confidence": 0.5},
                "text_ref": "love",
                "span": [100, 110]  # Way out of range
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_bad_span, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 3 FAILED: should be invalid (bad span)"
    assert any("out of range" in e.lower() or "span" in e.lower() for e in result.errors)
    print("  ✅ PASS")
    
    # ===========================================
    # Test 4: Bad span - text_ref mismatch (NG expected)
    # ===========================================
    print("\n[Test 4] Bad span (text_ref mismatch)")
    
    molecule_mismatch = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "coordinates": {"axis": None, "level": None, "confidence": 0.5},
                "text_ref": "hate",  # Does NOT match text[2:6]
                "span": [2, 6]       # This is "love"
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_mismatch, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 4 FAILED: should be invalid (text_ref mismatch)"
    assert any("mismatch" in e.lower() for e in result.errors)
    print("  ✅ PASS")
    
    # ===========================================
    # Test 5: Invalid axis (NG expected)
    # ===========================================
    print("\n[Test 5] Invalid axis (not in glossary)")
    
    molecule_bad_axis = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "coordinates": {
                    "axis": "nonexistent_axis",  # Invalid axis
                    "level": "resonant",
                    "confidence": 0.9
                },
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_bad_axis, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 5 FAILED: should be invalid (bad axis)"
    assert any("axis" in e.lower() for e in result.errors)
    print("  ✅ PASS")
    
    # ===========================================
    # Test 6: Invalid level (NG expected)
    # ===========================================
    print("\n[Test 6] Invalid level (not in glossary)")
    
    molecule_bad_level = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "coordinates": {
                    "axis": "interconnection",
                    "level": "nonexistent_level",  # Invalid level
                    "confidence": 0.9
                },
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_bad_level, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 6 FAILED: should be invalid (bad level)"
    assert any("level" in e.lower() for e in result.errors)
    print("  ✅ PASS")
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)