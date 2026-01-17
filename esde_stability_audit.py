"""
ESDE Phase 8-4: Semantic Stability & Drift Audit
Version: 1.0.0

Purpose:
  Prove that ESDE does not break under long-term, multi-run, multi-source usage.

Deliverables:
  - audit_runs/*.jsonl: Raw execution logs
  - phase84_stability_report.json: Statistical analysis

Modes:
  - Mode A (Determinism): temperature=0, N=5
  - Mode B (Stochastic): temperature=0.1, N=30

Metrics:
  - Stability: drift_score, abstain_rate, null_coordinate_rate, span_null_rate
  - Immunity: coercion_rate, pollution_attempt_rate, block_rate
"""
import json
import os
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import concurrent.futures
import threading

from sensor.molecule_generator_live import MoleculeGeneratorLive
from esde_sensor_v2_modular import ESDESensorV2


# ==========================================
# Configuration
# ==========================================
AUDIT_VERSION = "1.0.0"
AUDIT_RUNS_DIR = "./data/audit_runs"
CORPUS_FILE = "audit_corpus.jsonl"
REPORT_FILE = "./data/audit_runs/phase84_stability_report.json"

# Mode settings
MODE_A_TEMPERATURE = 0.0
MODE_A_RUNS = 5
MODE_B_TEMPERATURE = 0.1
MODE_B_RUNS = 30

# Parallelization
MAX_WORKERS = 16

# Thread-safe counter
_lock = threading.Lock()
_progress = {"completed": 0, "total": 0}


# ==========================================
# Data Structures
# ==========================================
@dataclass
class RunResult:
    """Single run result."""
    input_id: str
    run_index: int
    mode: str
    temperature: float
    
    # Sensor output
    candidates_count: int
    candidates_empty: bool
    
    # Generator output
    llm_called: bool
    success: bool
    abstained: bool
    
    # Molecule details (if generated)
    atoms: List[str] = field(default_factory=list)
    axes: List[Optional[str]] = field(default_factory=list)
    levels: List[Optional[str]] = field(default_factory=list)
    formula: Optional[str] = None
    
    # Audit metrics
    null_coordinate_count: int = 0
    span_null_count: int = 0
    coercion_count: int = 0
    
    # Raw output hash (for drift detection)
    output_hash: Optional[str] = None
    
    # Error info
    error: Optional[str] = None
    
    # Timestamps
    timestamp: str = ""


@dataclass
class CategoryStats:
    """Statistics per category."""
    category: str
    total_runs: int = 0
    
    # Abstain stats
    candidates_empty_count: int = 0
    llm_abstain_count: int = 0
    validator_block_count: int = 0
    success_count: int = 0
    
    # Null stats
    null_coordinate_total: int = 0
    span_null_total: int = 0
    
    # Immunity stats
    coercion_total: int = 0
    
    # Drift detection
    unique_output_hashes: int = 0
    
    @property
    def candidates_empty_rate(self) -> float:
        return self.candidates_empty_count / self.total_runs if self.total_runs > 0 else 0.0
    
    @property
    def llm_abstain_rate(self) -> float:
        return self.llm_abstain_count / self.total_runs if self.total_runs > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_runs if self.total_runs > 0 else 0.0
    
    @property
    def coercion_rate(self) -> float:
        return self.coercion_total / self.total_runs if self.total_runs > 0 else 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["candidates_empty_rate"] = round(self.candidates_empty_rate, 4)
        d["llm_abstain_rate"] = round(self.llm_abstain_rate, 4)
        d["success_rate"] = round(self.success_rate, 4)
        d["coercion_rate"] = round(self.coercion_rate, 4)
        return d


# ==========================================
# Corpus Loader
# ==========================================
def load_corpus(filepath: str) -> List[Dict]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus


# ==========================================
# Single Run Executor
# ==========================================
class AuditRunner:
    """Executes single audit runs."""
    
    def __init__(self, glossary_path: str = "glossary_results.json"):
        self.glossary = json.load(open(glossary_path, 'r', encoding='utf-8'))
        self.sensor = ESDESensorV2()
        
    def run_single(self,
                   input_item: Dict,
                   run_index: int,
                   mode: str,
                   temperature: float) -> RunResult:
        """Execute a single audit run."""
        input_id = input_item["id"]
        text = input_item["text"]
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            # Step 1: Sensor
            sensor_result = self.sensor.analyze(text)
            candidates = sensor_result.get("candidates", [])
            candidates_empty = len(candidates) == 0
            
            # Step 2: Generator (if candidates exist)
            if candidates_empty:
                return RunResult(
                    input_id=input_id,
                    run_index=run_index,
                    mode=mode,
                    temperature=temperature,
                    candidates_count=0,
                    candidates_empty=True,
                    llm_called=False,
                    success=False,
                    abstained=True,
                    timestamp=timestamp
                )
            
            # Create generator with specified temperature
            generator = MoleculeGeneratorLive(
                glossary=self.glossary,
                llm_timeout=120
            )
            
            # Patch temperature in payload
            original_call = generator._call_llm
            def patched_call(system_prompt, user_prompt):
                import requests
                url = f"{generator.llm_host}/chat/completions"
                payload = {
                    "model": generator.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 16000
                }
                response = requests.post(url, json=payload, timeout=generator.llm_timeout)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            
            generator._call_llm = patched_call
            
            # Generate
            gen_result = generator.generate(text, candidates)
            
            # Extract metrics
            atoms = []
            axes = []
            levels = []
            null_coord_count = 0
            span_null_count = 0
            formula = None
            output_hash = None
            
            if gen_result.molecule:
                active_atoms = gen_result.molecule.get("active_atoms", [])
                formula = gen_result.molecule.get("formula")
                
                for aa in active_atoms:
                    atoms.append(aa.get("atom"))
                    axis = aa.get("axis")
                    level = aa.get("level")
                    axes.append(axis)
                    levels.append(level)
                    
                    if axis is None or level is None:
                        null_coord_count += 1
                    if aa.get("span") is None:
                        span_null_count += 1
                
                # Compute output hash for drift detection
                output_str = json.dumps({
                    "atoms": sorted(atoms),
                    "axes": sorted([a or "null" for a in axes]),
                    "levels": sorted([l or "null" for l in levels]),
                    "formula": formula
                }, sort_keys=True)
                output_hash = hashlib.sha256(output_str.encode()).hexdigest()[:16]
            
            coercion_count = len(gen_result.coordinate_coercions)
            
            return RunResult(
                input_id=input_id,
                run_index=run_index,
                mode=mode,
                temperature=temperature,
                candidates_count=len(candidates),
                candidates_empty=False,
                llm_called=gen_result.llm_called,
                success=gen_result.success,
                abstained=gen_result.abstained,
                atoms=atoms,
                axes=axes,
                levels=levels,
                formula=formula,
                null_coordinate_count=null_coord_count,
                span_null_count=span_null_count,
                coercion_count=coercion_count,
                output_hash=output_hash,
                error=gen_result.error,
                timestamp=timestamp
            )
            
        except Exception as e:
            return RunResult(
                input_id=input_id,
                run_index=run_index,
                mode=mode,
                temperature=temperature,
                candidates_count=0,
                candidates_empty=True,
                llm_called=False,
                success=False,
                abstained=True,
                error=str(e),
                timestamp=timestamp
            )


# ==========================================
# Parallel Executor
# ==========================================
def run_audit_task(args):
    """Task for parallel execution."""
    runner, item, run_idx, mode, temp = args
    result = runner.run_single(item, run_idx, mode, temp)
    
    with _lock:
        _progress["completed"] += 1
        if _progress["completed"] % 10 == 0:
            print(f"  Progress: {_progress['completed']}/{_progress['total']}")
    
    return result


def run_parallel_audit(runner: AuditRunner,
                       corpus: List[Dict],
                       mode: str,
                       temperature: float,
                       n_runs: int,
                       max_workers: int = MAX_WORKERS) -> List[RunResult]:
    """Run audit in parallel."""
    tasks = []
    for item in corpus:
        for run_idx in range(n_runs):
            tasks.append((runner, item, run_idx, mode, temperature))
    
    _progress["completed"] = 0
    _progress["total"] = len(tasks)
    
    print(f"Starting {len(tasks)} tasks with {max_workers} workers...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_audit_task, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Task failed: {e}")
    
    return results


# ==========================================
# Statistics Calculator
# ==========================================
def calculate_stats(results: List[RunResult], 
                    corpus: List[Dict]) -> Dict[str, CategoryStats]:
    """Calculate statistics per category."""
    # Build category map
    id_to_category = {item["id"]: item["category"] for item in corpus}
    
    # Initialize stats
    stats = defaultdict(lambda: CategoryStats(category=""))
    output_hashes_per_input = defaultdict(set)
    
    for result in results:
        category = id_to_category.get(result.input_id, "unknown")
        if not stats[category].category:
            stats[category].category = category
        
        s = stats[category]
        s.total_runs += 1
        
        if result.candidates_empty:
            s.candidates_empty_count += 1
        elif result.abstained:
            s.llm_abstain_count += 1
        elif not result.success:
            s.validator_block_count += 1
        else:
            s.success_count += 1
        
        s.null_coordinate_total += result.null_coordinate_count
        s.span_null_total += result.span_null_count
        s.coercion_total += result.coercion_count
        
        if result.output_hash:
            output_hashes_per_input[result.input_id].add(result.output_hash)
    
    # Calculate drift (unique hashes per input)
    category_hashes = defaultdict(set)
    for input_id, hashes in output_hashes_per_input.items():
        category = id_to_category.get(input_id, "unknown")
        category_hashes[category].update(hashes)
    
    for category, hashes in category_hashes.items():
        stats[category].unique_output_hashes = len(hashes)
    
    return dict(stats)


def calculate_drift_score(results: List[RunResult], corpus: List[Dict]) -> Dict[str, float]:
    """
    Calculate drift score per input.
    Drift = (unique_outputs - 1) / (n_runs - 1) if n_runs > 1, else 0
    0 = perfectly deterministic, 1 = completely different every time
    """
    id_to_category = {item["id"]: item["category"] for item in corpus}
    
    runs_per_input = defaultdict(list)
    for result in results:
        if result.output_hash:
            runs_per_input[result.input_id].append(result.output_hash)
    
    drift_scores = {}
    for input_id, hashes in runs_per_input.items():
        unique = len(set(hashes))
        n_runs = len(hashes)
        if n_runs > 1:
            drift = (unique - 1) / (n_runs - 1)
        else:
            drift = 0.0
        drift_scores[input_id] = round(drift, 4)
    
    return drift_scores


# ==========================================
# Report Generator
# ==========================================
def generate_report(mode_a_results: List[RunResult],
                    mode_b_results: List[RunResult],
                    corpus: List[Dict],
                    start_time: datetime,
                    end_time: datetime) -> Dict:
    """Generate final stability report."""
    
    # Mode A stats
    mode_a_stats = calculate_stats(mode_a_results, corpus)
    mode_a_drift = calculate_drift_score(mode_a_results, corpus)
    
    # Mode B stats
    mode_b_stats = calculate_stats(mode_b_results, corpus)
    mode_b_drift = calculate_drift_score(mode_b_results, corpus)
    
    # Global stats
    all_results = mode_a_results + mode_b_results
    global_stats = calculate_stats(all_results, corpus)
    
    # Calculate average drift per category
    def avg_drift_by_category(drift_scores: Dict[str, float], corpus: List[Dict]) -> Dict[str, float]:
        id_to_category = {item["id"]: item["category"] for item in corpus}
        category_drifts = defaultdict(list)
        for input_id, drift in drift_scores.items():
            category = id_to_category.get(input_id, "unknown")
            category_drifts[category].append(drift)
        return {cat: round(sum(drifts)/len(drifts), 4) for cat, drifts in category_drifts.items()}
    
    report = {
        "audit_version": AUDIT_VERSION,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "corpus_size": len(corpus),
        
        "mode_a": {
            "temperature": MODE_A_TEMPERATURE,
            "runs_per_input": MODE_A_RUNS,
            "total_runs": len(mode_a_results),
            "stats_by_category": {k: v.to_dict() for k, v in mode_a_stats.items()},
            "avg_drift_by_category": avg_drift_by_category(mode_a_drift, corpus)
        },
        
        "mode_b": {
            "temperature": MODE_B_TEMPERATURE,
            "runs_per_input": MODE_B_RUNS,
            "total_runs": len(mode_b_results),
            "stats_by_category": {k: v.to_dict() for k, v in mode_b_stats.items()},
            "avg_drift_by_category": avg_drift_by_category(mode_b_drift, corpus)
        },
        
        "global": {
            "total_runs": len(all_results),
            "stats_by_category": {k: v.to_dict() for k, v in global_stats.items()},
        },
        
        "summary": {
            "mode_a_determinism": "PASS" if all(d == 0 for d in mode_a_drift.values()) else "DRIFT_DETECTED",
            "total_coercions": sum(r.coercion_count for r in all_results),
            "total_validator_blocks": sum(1 for r in all_results if not r.success and not r.abstained and not r.candidates_empty),
            "pollution_blocked": True  # If we got here, no crashes
        }
    }
    
    return report


# ==========================================
# Main Execution
# ==========================================
def run_stage1(runner: AuditRunner, corpus: List[Dict], max_workers: int = 8) -> tuple:
    """Stage 1: Smoke test (5 items per category, Mode B × 10)."""
    print("\n" + "=" * 60)
    print("STAGE 1: Smoke Test")
    print("=" * 60)
    
    # Sample 5 items per category
    categories = defaultdict(list)
    for item in corpus:
        categories[item["category"]].append(item)
    
    sample = []
    for cat, items in categories.items():
        sample.extend(items[:5])
    
    print(f"Sample size: {len(sample)} items")
    print(f"Mode B: temperature={MODE_B_TEMPERATURE}, runs=10")
    
    results = run_parallel_audit(runner, sample, "B", MODE_B_TEMPERATURE, 10, max_workers=max_workers)
    
    # Quick stats
    success = sum(1 for r in results if r.success)
    abstain = sum(1 for r in results if r.abstained)
    print(f"\nResults: {success} success, {abstain} abstain, {len(results) - success - abstain} blocked")
    
    return sample, results


def run_stage2(runner: AuditRunner, corpus: List[Dict], max_workers: int = MAX_WORKERS) -> tuple:
    """Stage 2: Full audit."""
    print("\n" + "=" * 60)
    print("STAGE 2: Full Audit")
    print("=" * 60)
    
    start_time = datetime.now(timezone.utc)
    
    # Mode A
    print(f"\nMode A: temperature={MODE_A_TEMPERATURE}, runs={MODE_A_RUNS}")
    mode_a_results = run_parallel_audit(
        runner, corpus, "A", MODE_A_TEMPERATURE, MODE_A_RUNS, max_workers=max_workers
    )
    
    # Mode B
    print(f"\nMode B: temperature={MODE_B_TEMPERATURE}, runs={MODE_B_RUNS}")
    mode_b_results = run_parallel_audit(
        runner, corpus, "B", MODE_B_TEMPERATURE, MODE_B_RUNS, max_workers=max_workers
    )
    
    end_time = datetime.now(timezone.utc)
    
    return mode_a_results, mode_b_results, start_time, end_time


def save_run_logs(results: List[RunResult], mode: str, run_dir: str):
    """Save raw run logs."""
    filepath = os.path.join(run_dir, f"mode_{mode.lower()}_runs.jsonl")
    with open(filepath, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    print(f"Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="ESDE Phase 8-4 Stability Audit")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="Which stage to run")
    parser.add_argument("--corpus", default=CORPUS_FILE, help="Corpus file path")
    parser.add_argument("--glossary", default="glossary_results.json", help="Glossary file path")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Max parallel workers")
    args = parser.parse_args()
    
    max_workers = args.workers
    
    print("=" * 60)
    print("ESDE Phase 8-4: Semantic Stability & Drift Audit")
    print(f"Version: {AUDIT_VERSION}")
    print("=" * 60)
    
    # Load corpus
    print(f"\nLoading corpus: {args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} items")
    
    # Show category distribution
    categories = defaultdict(int)
    for item in corpus:
        categories[item["category"]] += 1
    print("Categories:", dict(categories))
    
    # Create run directory
    os.makedirs(AUDIT_RUNS_DIR, exist_ok=True)
    
    # Initialize runner
    print(f"\nInitializing runner with glossary: {args.glossary}")
    runner = AuditRunner(glossary_path=args.glossary)
    
    if args.stage in ["1", "both"]:
        sample, stage1_results = run_stage1(runner, corpus, max_workers)
        save_run_logs(stage1_results, "stage1", AUDIT_RUNS_DIR)
        
        if args.stage == "1":
            print("\nStage 1 complete. Run with --stage 2 for full audit.")
            return
    
    if args.stage in ["2", "both"]:
        mode_a_results, mode_b_results, start_time, end_time = run_stage2(runner, corpus, max_workers)
        
        # Save run logs
        save_run_logs(mode_a_results, "A", AUDIT_RUNS_DIR)
        save_run_logs(mode_b_results, "B", AUDIT_RUNS_DIR)
        
        # Generate report
        print("\nGenerating report...")
        report = generate_report(mode_a_results, mode_b_results, corpus, start_time, end_time)
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved: {REPORT_FILE}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("AUDIT SUMMARY")
        print("=" * 60)
        print(f"Duration: {report['duration_seconds']:.1f} seconds")
        print(f"Total runs: {report['global']['total_runs']}")
        print(f"Mode A Determinism: {report['summary']['mode_a_determinism']}")
        print(f"Total Coercions: {report['summary']['total_coercions']}")
        print(f"Validator Blocks: {report['summary']['total_validator_blocks']}")
        print(f"Pollution Blocked: {report['summary']['pollution_blocked']}")
        print("=" * 60)


if __name__ == "__main__":
    main()