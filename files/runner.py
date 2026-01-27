"""
ESDE Migration Phase 3: Statistics Pipeline Runner
===================================================
Orchestrates W2/W3 statistics pipeline with ExecutionContext.

This runner provides:
  - ExecutionContext generation with policy metadata
  - Coordinated W2 aggregation and W3 calculation
  - Scope-based file isolation
  - Metadata propagation for W6 export

Usage:
    from statistics.runner import StatisticsPipelineRunner
    from statistics.policies import StandardConditionPolicy
    from substrate import SubstrateRegistry
    
    # Create runner with policy
    runner = StatisticsPipelineRunner(
        registry=registry,
        policy=policy,
        analysis_scope_id="run_20260125",  # Optional, auto-generated if not provided
    )
    
    # Run W2 aggregation
    w2_result = runner.run_w2(articles)
    
    # Run W3 calculation
    w3_result = runner.run_w3(w1_stats)
    
    # Get execution context for W6
    context = runner.get_execution_context()

Migration Phase 3 P0 Compliance:
  P0-MIG3-1: Policy mode requires analysis_scope_id (auto-generated if not provided)
  P0-MIG3-2: Path traversal prevention via utils validation
  
Spec: Migration Phase 3 v0.3.1
"""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

from .utils import (
    ExecutionContext,
    generate_scope_id,
    resolve_stats_dir,
    resolve_w2_paths,
    resolve_w3_output_dir,
    compute_policy_signature_hash,
    validate_scope_id,
    ScopeValidationError,
    UTILS_VERSION,
)

if TYPE_CHECKING:
    from .policies.base import BaseConditionPolicy
    from .schema import W1GlobalStats
    from .schema_w2 import W2GlobalStats
    from .schema_w3 import W3Record
    from substrate import SubstrateRegistry


# ==========================================
# Constants
# ==========================================

RUNNER_VERSION = "0.3.1"
DEFAULT_BASE_DIR = "data/stats"


# ==========================================
# Pipeline Result
# ==========================================

@dataclass
class W2PipelineResult:
    """Result of W2 aggregation pipeline."""
    success: bool
    articles_processed: int = 0
    total_records: int = 0
    total_conditions: int = 0
    records_path: str = ""
    conditions_path: str = ""
    analysis_scope_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "articles_processed": self.articles_processed,
            "total_records": self.total_records,
            "total_conditions": self.total_conditions,
            "records_path": self.records_path,
            "conditions_path": self.conditions_path,
            "analysis_scope_id": self.analysis_scope_id,
            "errors": self.errors,
        }


@dataclass
class W3PipelineResult:
    """Result of W3 calculation pipeline."""
    success: bool
    conditions_calculated: int = 0
    output_dir: str = ""
    analysis_scope_id: Optional[str] = None
    records: List["W3Record"] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "conditions_calculated": self.conditions_calculated,
            "output_dir": self.output_dir,
            "analysis_scope_id": self.analysis_scope_id,
            "record_ids": [r.analysis_id for r in self.records],
            "errors": self.errors,
        }


# ==========================================
# Statistics Pipeline Runner
# ==========================================

class StatisticsPipelineRunner:
    """
    Orchestrates W2/W3 statistics pipeline with ExecutionContext.
    
    This runner manages:
      - ExecutionContext creation and propagation
      - Scope-based file isolation
      - W2 aggregation with policy support
      - W3 calculation with scope isolation
    """
    
    def __init__(
        self,
        registry: Optional["SubstrateRegistry"] = None,
        policy: Optional["BaseConditionPolicy"] = None,
        analysis_scope_id: Optional[str] = None,
        base_dir: str = DEFAULT_BASE_DIR,
        auto_generate_scope: bool = True,
    ):
        """
        Initialize Statistics Pipeline Runner.
        
        Args:
            registry: SubstrateRegistry instance (optional)
            policy: BaseConditionPolicy instance (optional)
            analysis_scope_id: Scope identifier (auto-generated if not provided and policy is set)
            base_dir: Base directory for statistics data
            auto_generate_scope: If True, auto-generate scope_id when policy is set but scope is not
        
        Raises:
            ScopeValidationError: If policy is set, auto_generate_scope is False, and scope_id is not provided
        """
        self.registry = registry
        self.policy = policy
        self.base_dir = base_dir
        
        # Handle scope_id
        if policy and not analysis_scope_id:
            if auto_generate_scope:
                analysis_scope_id = generate_scope_id()
                print(f"[StatisticsPipelineRunner] Auto-generated scope_id: {analysis_scope_id}")
            else:
                raise ScopeValidationError(
                    "Policy mode requires analysis_scope_id when auto_generate_scope is False"
                )
        
        # Validate scope_id if provided
        if analysis_scope_id:
            validate_scope_id(analysis_scope_id)
        
        self.analysis_scope_id = analysis_scope_id
        
        # Create ExecutionContext
        self._context = self._create_execution_context()
        
        # Track pipeline state
        self._w2_result: Optional[W2PipelineResult] = None
        self._w3_result: Optional[W3PipelineResult] = None
    
    def _create_execution_context(self) -> ExecutionContext:
        """Create ExecutionContext with current configuration."""
        return ExecutionContext(
            analysis_scope_id=self.analysis_scope_id,
            policy=self.policy,
            signature_source="policy" if self.policy else "legacy",
        )
    
    @property
    def execution_context(self) -> ExecutionContext:
        """Get the current ExecutionContext."""
        return self._context
    
    @property
    def is_policy_mode(self) -> bool:
        """Check if running in policy mode."""
        return self.policy is not None
    
    def get_resolved_paths(self) -> Dict[str, str]:
        """Get all resolved paths for current configuration."""
        stats_dir = resolve_stats_dir(self.base_dir, self.policy, self.analysis_scope_id)
        w2_paths = resolve_w2_paths(self.base_dir, self.policy, self.analysis_scope_id)
        w3_dir = resolve_w3_output_dir(self.base_dir, self.policy, self.analysis_scope_id)
        
        return {
            "stats_dir": stats_dir,
            "w2_records_path": w2_paths["records_path"],
            "w2_conditions_path": w2_paths["conditions_path"],
            "w3_output_dir": w3_dir,
        }
    
    def run_w2(
        self,
        articles: List,
        save: bool = True,
    ) -> W2PipelineResult:
        """
        Run W2 aggregation pipeline.
        
        Args:
            articles: List of ArticleRecord instances
            save: If True, save results to files
            
        Returns:
            W2PipelineResult with aggregation results
        """
        from .w2_aggregator import W2Aggregator
        
        result = W2PipelineResult(
            success=False,
            analysis_scope_id=self.analysis_scope_id,
        )
        
        try:
            # Create W2Aggregator with current configuration
            aggregator = W2Aggregator(
                registry=self.registry,
                policy=self.policy,
                analysis_scope_id=self.analysis_scope_id,
                base_dir=self.base_dir,
            )
            
            # Process articles
            batch_result = aggregator.process_batch(articles)
            
            result.articles_processed = batch_result["articles_processed"]
            result.total_records = batch_result["total_records"]
            result.total_conditions = batch_result["total_conditions"]
            result.records_path = aggregator.records_path
            result.conditions_path = aggregator.conditions_path
            
            # Save if requested
            if save:
                aggregator.save()
            
            result.success = True
            
            # Store aggregator stats for W3
            self._w2_stats = aggregator.stats
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"[StatisticsPipelineRunner] W2 error: {e}")
        
        self._w2_result = result
        return result
    
    def run_w3(
        self,
        w1_stats: "W1GlobalStats",
        w2_stats: Optional["W2GlobalStats"] = None,
        save: bool = True,
    ) -> W3PipelineResult:
        """
        Run W3 calculation pipeline.
        
        Args:
            w1_stats: W1GlobalStats (global distribution)
            w2_stats: W2GlobalStats (optional, uses cached from run_w2 if not provided)
            save: If True, save results to files
            
        Returns:
            W3PipelineResult with calculation results
        """
        from .w3_calculator import W3Calculator
        
        result = W3PipelineResult(
            success=False,
            analysis_scope_id=self.analysis_scope_id,
        )
        
        # Use provided w2_stats or cached from run_w2
        if w2_stats is None:
            if hasattr(self, '_w2_stats'):
                w2_stats = self._w2_stats
            else:
                result.errors.append("No W2 stats available. Run run_w2() first or provide w2_stats.")
                return result
        
        try:
            # Create W3Calculator with current configuration
            calculator = W3Calculator(
                w1_stats=w1_stats,
                w2_stats=w2_stats,
                policy=self.policy,
                analysis_scope_id=self.analysis_scope_id,
                base_dir=self.base_dir,
            )
            
            # Calculate all conditions
            records = calculator.calculate_all()
            
            result.conditions_calculated = len(records)
            result.output_dir = calculator.output_dir
            result.records = records
            
            # Save if requested
            if save:
                calculator.save_all(records)
            
            result.success = True
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"[StatisticsPipelineRunner] W3 error: {e}")
        
        self._w3_result = result
        return result
    
    def get_w6_metadata(self) -> Dict[str, Any]:
        """
        Get metadata dict for W6 export.
        
        Returns:
            Dict with policy_id, policy_version, policy_signature_hash, etc.
        """
        return self._context.to_metadata_dict()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline state."""
        return {
            "runner_version": RUNNER_VERSION,
            "utils_version": UTILS_VERSION,
            "is_policy_mode": self.is_policy_mode,
            "analysis_scope_id": self.analysis_scope_id,
            "policy_id": self.policy.policy_id if self.policy else None,
            "policy_version": self.policy.version if self.policy else None,
            "base_dir": self.base_dir,
            "resolved_paths": self.get_resolved_paths(),
            "w2_result": self._w2_result.to_dict() if self._w2_result else None,
            "w3_result": self._w3_result.to_dict() if self._w3_result else None,
            "execution_context": self._context.to_metadata_dict(),
        }


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Migration Phase 3: StatisticsPipelineRunner Test")
    print("=" * 60)
    
    import tempfile
    
    # Mock Policy
    class MockPolicy:
        policy_id = "test_runner_policy"
        version = "v1.0"
        target_keys = ["legacy:source_type"]
        
        def compute_signature(self, record):
            return "mock_signature_" + "a" * 48
        
        def extract_factors(self, record):
            return {"mock": "factors"}
    
    # Test 1: Legacy mode (no policy)
    print("\n[Test 1] Legacy mode (no policy)")
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StatisticsPipelineRunner(base_dir=tmpdir)
        
        print(f"  is_policy_mode: {runner.is_policy_mode}")
        print(f"  analysis_scope_id: {runner.analysis_scope_id}")
        
        paths = runner.get_resolved_paths()
        print(f"  stats_dir: {paths['stats_dir']}")
        
        assert not runner.is_policy_mode
        assert runner.analysis_scope_id is None
        print("  ✅ PASS")
    
    # Test 2: Policy mode with auto-generated scope
    print("\n[Test 2] Policy mode with auto-generated scope")
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StatisticsPipelineRunner(
            policy=MockPolicy(),
            base_dir=tmpdir,
            auto_generate_scope=True,
        )
        
        print(f"  is_policy_mode: {runner.is_policy_mode}")
        print(f"  analysis_scope_id: {runner.analysis_scope_id}")
        
        paths = runner.get_resolved_paths()
        print(f"  w2_records_path: {paths['w2_records_path']}")
        
        assert runner.is_policy_mode
        assert runner.analysis_scope_id is not None
        assert runner.analysis_scope_id.startswith("run_")
        assert "policies/test_runner_policy/v1.0" in paths['w2_records_path']
        print("  ✅ PASS")
    
    # Test 3: Policy mode with explicit scope
    print("\n[Test 3] Policy mode with explicit scope")
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StatisticsPipelineRunner(
            policy=MockPolicy(),
            analysis_scope_id="my_custom_scope",
            base_dir=tmpdir,
        )
        
        print(f"  analysis_scope_id: {runner.analysis_scope_id}")
        
        metadata = runner.get_w6_metadata()
        print(f"  W6 metadata keys: {list(metadata.keys())}")
        
        assert runner.analysis_scope_id == "my_custom_scope"
        assert metadata["policy_id"] == "test_runner_policy"
        assert metadata["analysis_scope_id"] == "my_custom_scope"
        print("  ✅ PASS")
    
    # Test 4: Policy mode without scope and auto_generate=False (should fail)
    print("\n[Test 4] Policy mode without scope (auto_generate=False)")
    try:
        runner = StatisticsPipelineRunner(
            policy=MockPolicy(),
            auto_generate_scope=False,
            # analysis_scope_id not provided!
        )
        print("  ❌ FAIL: Should have raised ScopeValidationError")
    except ScopeValidationError as e:
        print(f"  ✅ Correctly rejected: {str(e)[:50]}...")
    
    # Test 5: ExecutionContext metadata
    print("\n[Test 5] ExecutionContext metadata for W6")
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = StatisticsPipelineRunner(
            policy=MockPolicy(),
            analysis_scope_id="w6_test_scope",
            base_dir=tmpdir,
        )
        
        ctx = runner.execution_context
        metadata = ctx.to_metadata_dict()
        
        print(f"  policy_id: {metadata['policy_id']}")
        print(f"  policy_version: {metadata['policy_version']}")
        print(f"  signature_source: {metadata['signature_source']}")
        print(f"  policy_signature_hash: {metadata['policy_signature_hash'][:20]}...")
        
        assert metadata["policy_id"] == "test_runner_policy"
        assert metadata["signature_source"] == "policy"
        print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
