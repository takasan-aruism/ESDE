"""
ESDE Substrate Layer: Context Registry
======================================

Append-only JSONL storage for ContextRecords.

Philosophy: "Write once, read many, delete never."

Design Principles:
  - Append-only (INV-SUB-005)
  - Deduplication by context_id
  - In-memory cache for fast lookup
  - Canonical export order (INV-SUB-007)

Invariants:
  INV-SUB-001: Upper Read-Only (upper layers can only read)
  INV-SUB-005: Append-Only (no updates, no deletes)
  INV-SUB-007: Canonical Export (deterministic output order)

Spec: Substrate Layer v0.1.0
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
from datetime import datetime, timezone

from .schema import ContextRecord, create_context_record

# ==========================================
# Constants
# ==========================================

REGISTRY_VERSION = "v0.1.0"
DEFAULT_REGISTRY_PATH = "data/substrate/context_registry.jsonl"

# P1-SUB-003: Canonical file writing settings
# Fixed encoding and newline to prevent hash drift across environments
FILE_ENCODING = "utf-8"
FILE_NEWLINE = "\n"  # Unix-style, never \r\n


# ==========================================
# SubstrateRegistry
# ==========================================

class SubstrateRegistry:
    """
    Append-only registry for ContextRecords.
    
    Storage:
      - JSONL format (one record per line)
      - Sorted by context_id for canonical export
      - Deduplication: same context_id = no re-write
    
    Caching:
      - In-memory cache for fast lookup
      - Lazy loading on first access
    
    Invariants:
      INV-SUB-005: Append-only (register() only, no update/delete)
      INV-SUB-007: Canonical export order
    
    Usage:
        registry = SubstrateRegistry()
        
        # Register a new record
        record = create_context_record(...)
        context_id = registry.register(record)
        
        # Retrieve by ID
        record = registry.get(context_id)
        
        # Query by criteria
        records = registry.query_by_trace_key("html:has_h1", True)
    """
    
    def __init__(self, storage_path: str = DEFAULT_REGISTRY_PATH):
        """
        Initialize SubstrateRegistry.
        
        Args:
            storage_path: Path to JSONL storage file
        """
        self.storage_path = Path(storage_path)
        self._cache: Dict[str, ContextRecord] = {}
        self._loaded = False
        self._stats = {
            "total_records": 0,
            "register_calls": 0,
            "dedup_hits": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    
    # ==========================================
    # Storage Operations
    # ==========================================
    
    def _ensure_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> None:
        """
        Load all records into memory cache.
        
        This is called lazily on first access.
        P1-SUB-003: Fixed encoding (UTF-8) for cross-platform determinism
        """
        if self._loaded:
            return
        
        if not self.storage_path.exists():
            self._loaded = True
            return
        
        try:
            # P1-SUB-003: Explicit encoding for cross-platform determinism
            with open(self.storage_path, "r", encoding=FILE_ENCODING) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = ContextRecord.from_jsonl(line)
                        self._cache[record.context_id] = record
                    except Exception as e:
                        # Log error but continue loading
                        print(f"[SubstrateRegistry] Warning: Line {line_num} parse error: {e}")
        except Exception as e:
            print(f"[SubstrateRegistry] Warning: Failed to load cache: {e}")
        
        self._stats["total_records"] = len(self._cache)
        self._loaded = True
    
    def _append_to_storage(self, record: ContextRecord) -> None:
        """
        Append a single record to storage file.
        
        INV-SUB-005: Append-only
        P1-SUB-003: Fixed encoding (UTF-8) and newline (LF)
        """
        self._ensure_directory()
        
        # P1-SUB-003: Explicit encoding and newline for cross-platform determinism
        with open(self.storage_path, "a", encoding=FILE_ENCODING, newline="") as f:
            f.write(record.to_jsonl() + FILE_NEWLINE)
    
    # ==========================================
    # Public API: Write
    # ==========================================
    
    def register(self, record: ContextRecord) -> str:
        """
        Register a ContextRecord.
        
        If a record with the same context_id already exists,
        the existing record is returned (deduplication).
        
        INV-SUB-005: Append-only (no update)
        
        Args:
            record: ContextRecord to register
            
        Returns:
            context_id of the registered (or existing) record
        """
        self._load_cache()
        self._stats["register_calls"] += 1
        
        # Deduplication check
        if record.context_id in self._cache:
            self._stats["dedup_hits"] += 1
            return record.context_id
        
        # Append to storage
        self._append_to_storage(record)
        
        # Update cache
        self._cache[record.context_id] = record
        self._stats["total_records"] += 1
        
        return record.context_id
    
    def register_traces(
        self,
        traces: Dict[str, Any],
        retrieval_path: Optional[str] = None,
        capture_version: str = "v1.0",
        observed_at: Optional[str] = None,
    ) -> str:
        """
        Convenience method to register traces directly.
        
        Creates a ContextRecord and registers it.
        
        Args:
            traces: Raw traces dict
            retrieval_path: URL, file path, or None
            capture_version: Version of trace extraction logic
            observed_at: When observation occurred
            
        Returns:
            context_id of the registered record
        """
        record = create_context_record(
            traces=traces,
            retrieval_path=retrieval_path,
            capture_version=capture_version,
            observed_at=observed_at,
        )
        return self.register(record)
    
    # ==========================================
    # Public API: Read
    # ==========================================
    
    def get(self, context_id: str) -> Optional[ContextRecord]:
        """
        Retrieve a ContextRecord by ID.
        
        Args:
            context_id: The context_id to look up
            
        Returns:
            ContextRecord if found, None otherwise
        """
        self._load_cache()
        
        record = self._cache.get(context_id)
        if record:
            self._stats["cache_hits"] += 1
        else:
            self._stats["cache_misses"] += 1
        
        return record
    
    def exists(self, context_id: str) -> bool:
        """
        Check if a context_id exists in the registry.
        
        Args:
            context_id: The context_id to check
            
        Returns:
            True if exists, False otherwise
        """
        self._load_cache()
        return context_id in self._cache
    
    def count(self) -> int:
        """
        Get total number of records.
        
        Returns:
            Number of records in registry
        """
        self._load_cache()
        return len(self._cache)
    
    # ==========================================
    # Public API: Query
    # ==========================================
    
    def query_by_trace_key(
        self,
        key: str,
        value: Any,
    ) -> List[ContextRecord]:
        """
        Query records by trace key-value pair.
        
        Args:
            key: Trace key (e.g., "html:has_h1")
            value: Expected value
            
        Returns:
            List of matching ContextRecords
        """
        self._load_cache()
        
        results = []
        for record in self._cache.values():
            if record.traces.get(key) == value:
                results.append(record)
        
        return results
    
    def query_by_namespace(self, namespace: str) -> List[ContextRecord]:
        """
        Query records that have any trace in the given namespace.
        
        Args:
            namespace: Namespace prefix (e.g., "html")
            
        Returns:
            List of matching ContextRecords
        """
        self._load_cache()
        prefix = f"{namespace}:"
        
        results = []
        for record in self._cache.values():
            for key in record.traces.keys():
                if key.startswith(prefix):
                    results.append(record)
                    break
        
        return results
    
    def iterate_all(self) -> Iterator[ContextRecord]:
        """
        Iterate over all records in canonical order.
        
        INV-SUB-007: Canonical Export (sorted by context_id)
        
        Yields:
            ContextRecord instances in sorted order
        """
        self._load_cache()
        
        for context_id in sorted(self._cache.keys()):
            yield self._cache[context_id]
    
    # ==========================================
    # Export
    # ==========================================
    
    def export_canonical(self, output_path: str) -> int:
        """
        Export all records to a new file in canonical order.
        
        INV-SUB-007: Canonical Export
        P1-SUB-003: Fixed encoding (UTF-8) and newline (LF)
        
        This creates a clean, sorted copy of the registry.
        Useful for:
          - Creating deterministic snapshots
          - Compacting the storage file
          - Sharing registry state
        
        Args:
            output_path: Path to output file
            
        Returns:
            Number of records exported
        """
        self._load_cache()
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        # P1-SUB-003: Explicit encoding and newline for cross-platform determinism
        with open(output, "w", encoding=FILE_ENCODING, newline="") as f:
            for record in self.iterate_all():
                f.write(record.to_jsonl() + FILE_NEWLINE)
                count += 1
        
        return count
    
    # ==========================================
    # Stats & Info
    # ==========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dict with usage statistics
        """
        self._load_cache()
        
        return {
            "version": REGISTRY_VERSION,
            "storage_path": str(self.storage_path),
            "total_records": self._stats["total_records"],
            "register_calls": self._stats["register_calls"],
            "dedup_hits": self._stats["dedup_hits"],
            "cache_hits": self._stats["cache_hits"],
            "cache_misses": self._stats["cache_misses"],
            "loaded": self._loaded,
        }
    
    def get_namespace_summary(self) -> Dict[str, int]:
        """
        Get count of records per namespace.
        
        Returns:
            Dict mapping namespace to record count
        """
        self._load_cache()
        
        summary: Dict[str, int] = {}
        
        for record in self._cache.values():
            namespaces_seen = set()
            for key in record.traces.keys():
                if ":" in key:
                    ns = key.split(":")[0]
                    namespaces_seen.add(ns)
            
            for ns in namespaces_seen:
                summary[ns] = summary.get(ns, 0) + 1
        
        return dict(sorted(summary.items()))


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("Substrate Registry Test")
    print("=" * 60)
    
    # Create temporary directory for tests
    test_dir = tempfile.mkdtemp(prefix="substrate_test_")
    test_path = os.path.join(test_dir, "test_registry.jsonl")
    
    try:
        # Test 1: Basic registration
        print("\n[Test 1] Basic registration")
        registry = SubstrateRegistry(storage_path=test_path)
        
        record = create_context_record(
            traces={"html:tag_count": 42, "text:char_count": 1000},
            retrieval_path="https://example.com/article",
        )
        
        context_id = registry.register(record)
        print(f"  Registered: {context_id}")
        assert registry.count() == 1
        print("  ✅ Registration successful")
        
        # Test 2: Deduplication
        print("\n[Test 2] Deduplication")
        id2 = registry.register(record)  # Same record
        assert id2 == context_id
        assert registry.count() == 1  # Still 1
        print(f"  Dedup hit: {id2 == context_id}")
        print("  ✅ Deduplication works")
        
        # Test 3: Retrieval
        print("\n[Test 3] Retrieval")
        retrieved = registry.get(context_id)
        assert retrieved is not None
        assert retrieved.context_id == context_id
        print(f"  Retrieved: {retrieved.context_id}")
        print("  ✅ Retrieval works")
        
        # Test 4: Non-existent ID
        print("\n[Test 4] Non-existent ID")
        not_found = registry.get("nonexistent_id")
        assert not_found is None
        print("  ✅ Returns None for missing ID")
        
        # Test 5: Multiple records
        print("\n[Test 5] Multiple records")
        for i in range(5):
            registry.register_traces(
                traces={"test:index": i, "test:value": f"value_{i}"},
                retrieval_path=f"https://example.com/page{i}",
            )
        print(f"  Total records: {registry.count()}")
        assert registry.count() == 6  # 1 + 5
        print("  ✅ Multiple registrations work")
        
        # Test 6: Query by trace key
        print("\n[Test 6] Query by trace key")
        results = registry.query_by_trace_key("test:index", 2)
        print(f"  Found {len(results)} records with test:index=2")
        assert len(results) == 1
        print("  ✅ Query by trace key works")
        
        # Test 7: Query by namespace
        print("\n[Test 7] Query by namespace")
        test_records = registry.query_by_namespace("test")
        html_records = registry.query_by_namespace("html")
        print(f"  'test' namespace: {len(test_records)} records")
        print(f"  'html' namespace: {len(html_records)} records")
        assert len(test_records) == 5
        assert len(html_records) == 1
        print("  ✅ Query by namespace works")
        
        # Test 8: Canonical export
        print("\n[Test 8] Canonical export")
        export_path = os.path.join(test_dir, "export.jsonl")
        count = registry.export_canonical(export_path)
        print(f"  Exported {count} records")
        
        # Verify export is sorted
        with open(export_path, "r") as f:
            ids = []
            for line in f:
                data = json.loads(line)
                ids.append(data["context_id"])
        assert ids == sorted(ids), "Export should be sorted"
        print("  ✅ Canonical export sorted correctly")
        
        # Test 9: Persistence (reload from file)
        print("\n[Test 9] Persistence")
        registry2 = SubstrateRegistry(storage_path=test_path)
        assert registry2.count() == 6
        retrieved2 = registry2.get(context_id)
        assert retrieved2 is not None
        print(f"  Reloaded {registry2.count()} records")
        print("  ✅ Persistence works")
        
        # Test 10: Stats
        print("\n[Test 10] Stats")
        stats = registry.get_stats()
        print(f"  Stats: {stats}")
        print("  ✅ Stats work")
        
        # Test 11: Namespace summary
        print("\n[Test 11] Namespace summary")
        summary = registry.get_namespace_summary()
        print(f"  Summary: {summary}")
        assert "test" in summary
        assert "html" in summary
        print("  ✅ Namespace summary works")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\n[Cleanup] Removed {test_dir}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
