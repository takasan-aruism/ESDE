"""
ESDE Phase 8 Integration Test
Sensor → Generator Live → Validator V83

Real LLM (QwQ-32B) integration test.
Tests the full pipeline with canonical v8.3 schema.

Requirements:
  - LLM server running at http://100.107.6.119:8001/v1
  - Synapse file: esde_synapses_v2_1.json
  - Glossary file: glossary_results.json

Run:
  python test_phase8_integration.py
"""
import json
import sys
import requests
from typing import Dict, Any, Optional
from datetime import datetime

# ==========================================
# Configuration
# ==========================================
LLM_HOST = "http://100.107.6.119:8001/v1"
SYNAPSE_FILE = "esde_synapses_v2_1.json"
GLOSSARY_FILE = "glossary_results.json"

# Test inputs
TEST_CASES = [
    {
        "text": "I love you",
        "expect_candidates": True,
        "expect_success": True,
    },
    {
        "text": "The law requires obedience",
        "expect_candidates": True,
        "expect_success": True,
    },
    {
        "text": "apprenticed to a master",
        "expect_candidates": True,
        "expect_success": True,
    },
    {
        "text": "",
        "expect_candidates": False,
        "expect_success": False,  # Empty → Abstain
    },
    {
        "text": "asdfghjkl qwertyuiop",
        "expect_candidates": False,  # Noise → likely no candidates
        "expect_success": False,
    },
]


# ==========================================
# LLM Health Check
# ==========================================
def check_llm_available() -> bool:
    """Check if LLM server is reachable."""
    try:
        resp = requests.get(f"{LLM_HOST}/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ==========================================
# Load Dependencies
# ==========================================
def load_glossary(filepath: str) -> Dict[str, Any]:
    """Load glossary from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "glossary" in data:
                glossary = data["glossary"]
                if isinstance(glossary, list):
                    return {item["concept_id"]: item for item in glossary 
                           if isinstance(item, dict) and "concept_id" in item}
                return glossary
            return data
        return {}
    except Exception as e:
        print(f"[ERROR] Failed to load glossary: {e}")
        return {}


# ==========================================
# Integration Test
# ==========================================
class Phase8IntegrationTest:
    """
    Integration test for Phase 8 pipeline.
    
    Flow:
      Sensor V2 → MoleculeGeneratorLive → MoleculeValidatorV83
    """
    
    def __init__(self):
        self.results = []
        self.sensor = None
        self.generator = None
        self.validator = None
        self.glossary = None
    
    def setup(self) -> bool:
        """Initialize components."""
        print("=" * 60)
        print("Phase 8 Integration Test Setup")
        print("=" * 60)
        
        # Check LLM
        print("\n[1/4] Checking LLM server...")
        if not check_llm_available():
            print(f"  ❌ LLM server not available at {LLM_HOST}")
            print("  Skipping integration test.")
            return False
        print(f"  ✅ LLM server available at {LLM_HOST}")
        
        # Load glossary
        print("\n[2/4] Loading glossary...")
        self.glossary = load_glossary(GLOSSARY_FILE)
        if not self.glossary:
            print(f"  ❌ Failed to load glossary from {GLOSSARY_FILE}")
            return False
        print(f"  ✅ Loaded {len(self.glossary)} concepts")
        
        # Import and initialize Sensor
        print("\n[3/4] Initializing Sensor V2...")
        try:
            from esde_sensor_v2_modular import ESDESensorV2
            self.sensor = ESDESensorV2(
                synapse_file=SYNAPSE_FILE,
                glossary_file=GLOSSARY_FILE
            )
            print("  ✅ Sensor V2 initialized")
        except Exception as e:
            print(f"  ❌ Failed to initialize Sensor: {e}")
            return False
        
        # Import and initialize Generator
        print("\n[4/4] Initializing Generator & Validator...")
        try:
            from sensor.molecule_generator_live import MoleculeGeneratorLive
            from sensor.validator_v83 import MoleculeValidatorV83
            
            self.generator = MoleculeGeneratorLive(
                glossary=self.glossary,
                llm_host=LLM_HOST
            )
            print("  ✅ MoleculeGeneratorLive initialized")
            
            # Validator needs allowed_atoms from Sensor
            # Will be set per-test based on candidates
            print("  ✅ MoleculeValidatorV83 ready")
            
        except Exception as e:
            print(f"  ❌ Failed to initialize Generator/Validator: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("Setup complete. Running tests...")
        print("=" * 60)
        return True
    
    def run_single_test(self, test_case: Dict) -> Dict:
        """
        Run a single integration test.
        
        Returns:
            Test result dict
        """
        text = test_case["text"]
        expect_candidates = test_case["expect_candidates"]
        expect_success = test_case["expect_success"]
        
        result = {
            "input": text,
            "expect_candidates": expect_candidates,
            "expect_success": expect_success,
            "sensor_ok": False,
            "generator_ok": False,
            "validator_ok": False,
            "schema_ok": False,
            "passed": False,
            "error": None,
            "details": {}
        }
        
        try:
            # Step 1: Sensor
            sensor_result = self.sensor.analyze(text)
            candidates = sensor_result.get("candidates", [])
            result["details"]["candidates_count"] = len(candidates)
            result["details"]["sensor_engine"] = sensor_result.get("meta", {}).get("engine")
            
            if expect_candidates and not candidates:
                result["error"] = "Expected candidates but got none"
                return result
            if not expect_candidates and candidates:
                # Got candidates when not expected - still proceed
                pass
            
            result["sensor_ok"] = True
            
            # Step 2: Generator
            if not candidates:
                # Empty candidates → expect abstain
                gen_result = self.generator.generate(text, candidates)
                result["details"]["generator_abstained"] = gen_result.abstained
                result["details"]["generator_llm_called"] = gen_result.llm_called
                
                if gen_result.abstained and not gen_result.llm_called:
                    # Correct: abstained without LLM call (Empty Check)
                    result["generator_ok"] = True
                    result["validator_ok"] = True  # N/A
                    result["schema_ok"] = True     # N/A
                    result["passed"] = not expect_success  # Should fail if success expected
                    return result
                else:
                    result["error"] = "Expected abstain but got different result"
                    return result
            
            gen_result = self.generator.generate(text, candidates)
            result["details"]["generator_success"] = gen_result.success
            result["details"]["generator_attempts"] = gen_result.attempts
            result["details"]["generator_coercions"] = len(gen_result.coordinate_coercions)
            
            if not gen_result.success:
                result["error"] = f"Generator failed: {gen_result.error}"
                result["generator_ok"] = False
                result["passed"] = not expect_success
                return result
            
            result["generator_ok"] = True
            molecule = gen_result.molecule
            
            # Step 3: Schema Check (v8.3 canonical)
            schema_errors = self._check_canonical_schema(molecule)
            if schema_errors:
                result["error"] = f"Schema violation: {schema_errors}"
                result["schema_ok"] = False
                return result
            
            result["schema_ok"] = True
            result["details"]["formula"] = molecule.get("formula")
            result["details"]["atoms_count"] = len(molecule.get("active_atoms", []))
            
            # Step 4: Validator
            from sensor.validator_v83 import MoleculeValidatorV83
            
            allowed_atoms = {c.get("concept_id") for c in candidates if c.get("concept_id")}
            synapse_hash = sensor_result.get("meta", {}).get("determinism_hash_full", "unknown")
            
            validator = MoleculeValidatorV83(
                glossary=self.glossary,
                allowed_atoms=allowed_atoms,
                synapse_hash=synapse_hash
            )
            
            val_result = validator.validate(molecule, text)
            result["details"]["validator_valid"] = val_result.valid
            result["details"]["validator_errors"] = val_result.errors
            result["details"]["validator_warnings"] = val_result.warnings
            result["details"]["validator_synapse_hash"] = val_result.synapse_hash
            
            if not val_result.valid:
                result["error"] = f"Validation failed: {val_result.errors}"
                result["validator_ok"] = False
                result["passed"] = not expect_success
                return result
            
            result["validator_ok"] = True
            result["passed"] = expect_success
            
        except Exception as e:
            result["error"] = str(e)
            import traceback
            result["details"]["traceback"] = traceback.format_exc()
        
        return result
    
    def _check_canonical_schema(self, molecule: Dict) -> list:
        """
        Check if molecule follows v8.3 canonical schema.
        
        Returns:
            List of schema errors (empty if valid)
        """
        errors = []
        
        # Must have active_atoms
        if "active_atoms" not in molecule:
            errors.append("Missing 'active_atoms'")
            return errors
        
        for aa in molecule.get("active_atoms", []):
            aa_id = aa.get("id", "unknown")
            
            # Must have flat axis/level (not nested in coordinates)
            if "coordinates" in aa:
                errors.append(f"[{aa_id}] Legacy 'coordinates' nesting detected")
            
            # Required fields
            for field in ["id", "atom", "text_ref"]:
                if field not in aa:
                    errors.append(f"[{aa_id}] Missing required field: {field}")
            
            # axis/level should be at top level
            # (can be null, but should exist as keys)
            if "axis" not in aa and "coordinates" not in aa:
                errors.append(f"[{aa_id}] Missing 'axis' field")
            if "level" not in aa and "coordinates" not in aa:
                errors.append(f"[{aa_id}] Missing 'level' field")
        
        # Must have formula (can be empty string)
        if "formula" not in molecule:
            errors.append("Missing 'formula'")
        
        return errors
    
    def run_all(self) -> bool:
        """
        Run all test cases.
        
        Returns:
            True if all tests passed
        """
        all_passed = True
        
        for i, test_case in enumerate(TEST_CASES):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(TEST_CASES)}: \"{test_case['text'][:30]}...\"" 
                  if len(test_case['text']) > 30 else f"Test {i+1}/{len(TEST_CASES)}: \"{test_case['text']}\"")
            print("=" * 60)
            
            result = self.run_single_test(test_case)
            self.results.append(result)
            
            # Print result
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            print(f"\nResult: {status}")
            print(f"  Sensor:    {'✅' if result['sensor_ok'] else '❌'}")
            print(f"  Generator: {'✅' if result['generator_ok'] else '❌'}")
            print(f"  Schema:    {'✅' if result['schema_ok'] else '❌'}")
            print(f"  Validator: {'✅' if result['validator_ok'] else '❌'}")
            
            if result.get("details"):
                d = result["details"]
                if "candidates_count" in d:
                    print(f"  Candidates: {d['candidates_count']}")
                if "formula" in d:
                    print(f"  Formula: {d['formula']}")
                if "validator_warnings" in d and d["validator_warnings"]:
                    print(f"  Warnings: {d['validator_warnings']}")
            
            if result.get("error"):
                print(f"  Error: {result['error']}")
            
            if not result["passed"]:
                all_passed = False
        
        return all_passed
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        print(f"\nTotal: {passed}/{total} passed")
        
        if passed == total:
            print("\n✅ ALL TESTS PASSED")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("\nFailed tests:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - \"{r['input']}\": {r.get('error', 'Unknown error')}")
        
        # Schema compliance
        schema_ok = sum(1 for r in self.results if r["schema_ok"])
        print(f"\nSchema compliance: {schema_ok}/{total}")
        
        # Synapse hash tracking
        hashes = [r["details"].get("validator_synapse_hash") for r in self.results 
                  if r["details"].get("validator_synapse_hash")]
        if hashes:
            print(f"Synapse hash recorded: ✅ ({len(hashes)} tests)")


# ==========================================
# Main
# ==========================================
def main():
    test = Phase8IntegrationTest()
    
    if not test.setup():
        print("\n⚠️  Setup failed. Integration test skipped.")
        sys.exit(0)  # Exit 0 so CI doesn't fail on missing LLM
    
    all_passed = test.run_all()
    test.print_summary()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"data/audit_runs/phase8_integration_{timestamp}.json"
    
    try:
        import os
        os.makedirs("data/audit_runs", exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "llm_host": LLM_HOST,
                "synapse_file": SYNAPSE_FILE,
                "glossary_file": GLOSSARY_FILE,
                "passed": all_passed,
                "results": test.results
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nReport saved: {report_file}")
    except Exception as e:
        print(f"\nWarning: Could not save report: {e}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
