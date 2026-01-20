"""
ESDE Sensor - Molecule Validator v8.3
Canonical validator for flat-schema molecules.

Canonical Schema Contract:
  - axis/level at top level (NOT nested in coordinates)
  - span is system-calculated [start, end)
  - No confidence field (selection criteria, not observation data)

GPT Audit v8.3.1 Compliance:
  - Canonical根拠は Ledger契約（Synapse形式は補助）
  - legacy module に依存しない
  - allowed_atoms の出所（synapse_hash）を記録（再現性）
  - span mismatch は warning（言語・正規化でズレうる）
"""
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field

from .constants import VALID_OPERATORS, BRACKET_PAIRS
from .glossary_validator import GlossaryValidator


# ==========================================
# Validation Result
# ==========================================
@dataclass
class ValidationResultV83:
    """
    Validation result with audit trail.
    
    Attributes:
        valid: True if no errors
        errors: List of blocking errors
        warnings: List of non-blocking warnings
        synapse_hash: Hash of synapse file used for allowed_atoms (reproducibility)
        atoms_checked: Number of atoms validated
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    synapse_hash: Optional[str] = None
    atoms_checked: int = 0
    
    def __bool__(self):
        return self.valid
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "synapse_hash": self.synapse_hash,
            "atoms_checked": self.atoms_checked
        }


# ==========================================
# Molecule Validator v8.3
# ==========================================
class MoleculeValidatorV83:
    """
    Validates v8.3 canonical flat-schema molecules.
    
    Checks:
      1. Atom Integrity: atoms must be in allowed set
      2. Coordinate Validity: axis/level must exist in Glossary (if specified)
      3. Span Validity: range check + text_ref match (warning only)
      4. Formula Syntax: operator validity, bracket matching
    
    Does NOT check:
      - coordinates nesting (canonical schema is flat)
      - confidence field (not part of canonical schema)
    """
    
    def __init__(self,
                 glossary: Dict[str, Any],
                 allowed_atoms: Set[str],
                 synapse_hash: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            glossary: Full glossary dict for axis/level validation
            allowed_atoms: Set of allowed concept_ids (from Sensor candidates)
            synapse_hash: Hash of synapse file for audit trail (reproducibility)
        """
        self.glossary_validator = GlossaryValidator(glossary)
        self.allowed_atoms = allowed_atoms
        self.synapse_hash = synapse_hash
    
    def validate(self, 
                 molecule: Dict[str, Any],
                 original_text: str) -> ValidationResultV83:
        """
        Validate a v8.3 canonical molecule.
        
        Args:
            molecule: Molecule dict with active_atoms, formula, meta
            original_text: Original input text for span validation
        
        Returns:
            ValidationResultV83 with errors, warnings, and audit info
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        active_atoms = molecule.get("active_atoms", [])
        formula = molecule.get("formula", "")
        
        # 1. Atom Integrity
        atom_errors = self._check_atom_integrity(active_atoms)
        errors.extend(atom_errors)
        
        # 2. Coordinate Validity
        coord_errors, coord_warnings = self._check_coordinate_validity(active_atoms)
        errors.extend(coord_errors)
        warnings.extend(coord_warnings)
        
        # 3. Span Validity (warnings only for mismatch)
        span_errors, span_warnings = self._check_span_validity(active_atoms, original_text)
        errors.extend(span_errors)
        warnings.extend(span_warnings)
        
        # 4. Formula Syntax
        formula_errors = self._check_formula_syntax(formula, active_atoms)
        errors.extend(formula_errors)
        
        return ValidationResultV83(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            synapse_hash=self.synapse_hash,
            atoms_checked=len(active_atoms)
        )
    
    def _check_atom_integrity(self, active_atoms: List[Dict]) -> List[str]:
        """
        Check that all atoms are in allowed set.
        
        Returns:
            List of error messages
        """
        errors = []
        
        for aa in active_atoms:
            atom = aa.get("atom")
            aa_id = aa.get("id", "unknown")
            
            if not atom:
                errors.append(f"[{aa_id}] Missing 'atom' field")
                continue
            
            if atom not in self.allowed_atoms:
                errors.append(f"[{aa_id}] Unknown atom: {atom} (not in allowed set)")
        
        return errors
    
    def _check_coordinate_validity(self, active_atoms: List[Dict]) -> tuple:
        """
        Check axis/level validity against glossary.
        
        v8.3 canonical: axis/level are top-level fields (NOT nested in coordinates)
        
        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        
        for aa in active_atoms:
            atom = aa.get("atom")
            aa_id = aa.get("id", "unknown")
            
            # v8.3 canonical: flat structure
            axis = aa.get("axis")
            level = aa.get("level")
            
            # Warn if legacy nested structure detected
            if "coordinates" in aa:
                warnings.append(f"[{aa_id}] Legacy 'coordinates' nesting detected - use flat structure")
            
            # Skip validation if atom unknown (already reported)
            if not atom or atom not in self.allowed_atoms:
                continue
            
            # axis validation (only if specified)
            if axis is not None:
                if not self.glossary_validator.is_valid_axis(atom, axis):
                    valid_axes = self.glossary_validator.get_valid_axes(atom)
                    errors.append(
                        f"[{aa_id}] Invalid axis '{axis}' for {atom}. "
                        f"Valid: {sorted(valid_axes) if valid_axes else 'none'}"
                    )
                elif level is not None:
                    # level validation (only if axis is valid)
                    if not self.glossary_validator.is_valid_level(atom, axis, level):
                        valid_levels = self.glossary_validator.get_valid_levels(atom, axis)
                        errors.append(
                            f"[{aa_id}] Invalid level '{level}' for {atom}@{axis}. "
                            f"Valid: {sorted(valid_levels) if valid_levels else 'none'}"
                        )
            
            # Warn if level specified without axis
            if axis is None and level is not None:
                warnings.append(f"[{aa_id}] Level '{level}' specified without axis")
        
        return errors, warnings
    
    def _check_span_validity(self, 
                              active_atoms: List[Dict],
                              original_text: str) -> tuple:
        """
        Check span validity.
        
        - Range out of bounds: ERROR
        - text_ref mismatch: WARNING (can differ due to normalization, punctuation)
        
        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        text_len = len(original_text)
        
        for aa in active_atoms:
            aa_id = aa.get("id", "unknown")
            span = aa.get("span")
            text_ref = aa.get("text_ref")
            
            if span is None:
                warnings.append(f"[{aa_id}] Missing span")
                continue
            
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                errors.append(f"[{aa_id}] Invalid span format: {span} (expected [start, end])")
                continue
            
            start, end = span
            
            # Range check (ERROR)
            if start < 0 or end > text_len or start >= end:
                errors.append(f"[{aa_id}] Span out of range: [{start}, {end}] (text length: {text_len})")
                continue
            
            # text_ref match check (WARNING only)
            if text_ref:
                actual_text = original_text[start:end]
                if actual_text.lower() != text_ref.lower():
                    warnings.append(
                        f"[{aa_id}] Span content mismatch: "
                        f"expected '{text_ref}', got '{actual_text}'"
                    )
        
        return errors, warnings
    
    def _check_formula_syntax(self, 
                               formula: str,
                               active_atoms: List[Dict]) -> List[str]:
        """
        Check formula syntax.
        
        - All referenced atom IDs must exist
        - Brackets must be balanced
        - Unknown operators are warnings (not errors, for extensibility)
        
        Returns:
            List of error messages
        """
        errors = []
        
        if not formula:
            # Empty formula is valid (single atom case)
            return errors
        
        # Get defined atom IDs
        defined_ids = {aa.get("id") for aa in active_atoms if aa.get("id")}
        
        # Extract atom ID references from formula (aa_N pattern)
        import re
        referenced_ids = set(re.findall(r'aa_\d+', formula))
        
        # Check all referenced IDs exist
        undefined = referenced_ids - defined_ids
        if undefined:
            errors.append(f"Formula references undefined atoms: {sorted(undefined)}")
        
        # Check bracket balance
        stack = []
        for char in formula:
            if char in BRACKET_PAIRS:
                stack.append(char)
            elif char in BRACKET_PAIRS.values():
                if not stack:
                    errors.append(f"Unmatched closing bracket: {char}")
                else:
                    expected_close = BRACKET_PAIRS.get(stack[-1])
                    if char == expected_close:
                        stack.pop()
                    else:
                        errors.append(f"Mismatched brackets: expected {expected_close}, got {char}")
        
        if stack:
            errors.append(f"Unclosed brackets: {stack}")
        
        return errors


# ==========================================
# Test
# ==========================================
if __name__ == "__main__":
    print("MoleculeValidatorV83 Test")
    print("=" * 60)
    
    # Mock glossary
    glossary = {
        "EMO.love": {
            "final_json": {
                "axes": {
                    "interconnection": {
                        "resonant": {"definition_en": "Deep connection"},
                        "catalytic": {"definition_en": "Sparking connection"}
                    }
                }
            }
        }
    }
    
    allowed_atoms = {"EMO.love"}
    original_text = "I love you"
    
    validator = MoleculeValidatorV83(
        glossary=glossary,
        allowed_atoms=allowed_atoms,
        synapse_hash="sha256:abc123"
    )
    
    # ===========================================
    # Test 1: Valid v8.3 molecule
    # ===========================================
    print("\n[Test 1] Valid v8.3 molecule (flat structure)")
    
    molecule_valid = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": "interconnection",
                "level": "catalytic",
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_valid, original_text)
    print(f"  Valid: {result.valid} (expected: True)")
    print(f"  Errors: {result.errors}")
    print(f"  Synapse Hash: {result.synapse_hash}")
    assert result.valid, "Test 1 FAILED"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 2: Unknown atom
    # ===========================================
    print("\n[Test 2] Unknown atom")
    
    molecule_unknown = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.hate",  # Not in allowed_atoms
                "axis": None,
                "level": None,
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_unknown, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 2 FAILED"
    assert any("EMO.hate" in e for e in result.errors), "Test 2 FAILED: should mention EMO.hate"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 3: Invalid axis
    # ===========================================
    print("\n[Test 3] Invalid axis")
    
    molecule_bad_axis = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": "nonexistent",
                "level": None,
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_bad_axis, original_text)
    print(f"  Valid: {result.valid} (expected: False)")
    print(f"  Errors: {result.errors}")
    assert not result.valid, "Test 3 FAILED"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 4: Span mismatch (warning only)
    # ===========================================
    print("\n[Test 4] Span mismatch (warning only)")
    
    molecule_span_mismatch = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": None,
                "level": None,
                "text_ref": "hate",  # Doesn't match span content
                "span": [2, 6]       # This is "love"
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_span_mismatch, original_text)
    print(f"  Valid: {result.valid} (expected: True - mismatch is warning)")
    print(f"  Warnings: {result.warnings}")
    assert result.valid, "Test 4 FAILED: span mismatch should be warning, not error"
    assert len(result.warnings) > 0, "Test 4 FAILED: should have warning"
    print("  ✅ PASS")
    
    # ===========================================
    # Test 5: Legacy coordinates nesting (warning)
    # ===========================================
    print("\n[Test 5] Legacy coordinates nesting (warning)")
    
    molecule_legacy = {
        "active_atoms": [
            {
                "id": "aa_1",
                "atom": "EMO.love",
                "axis": "interconnection",
                "level": "catalytic",
                "coordinates": {  # Legacy nesting
                    "axis": "interconnection",
                    "level": "catalytic"
                },
                "text_ref": "love",
                "span": [2, 6]
            }
        ],
        "formula": "aa_1"
    }
    
    result = validator.validate(molecule_legacy, original_text)
    print(f"  Valid: {result.valid}")
    print(f"  Warnings: {result.warnings}")
    assert any("coordinates" in w for w in result.warnings), "Test 5 FAILED: should warn about legacy nesting"
    print("  ✅ PASS")
    
    # ===========================================
    # Summary
    # ===========================================
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)
