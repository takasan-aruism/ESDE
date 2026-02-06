"""
ESDE Synapse - SynapseStore
============================
Single source of truth for Synapse data across all ESDE subsystems.

Design Spec v2.1 (Gemini design, GPT audit, Claude implementation)

Consumers (GO Condition — GPT Audit Gate):
  - Phase 8 Sensor  (sensor/loader_synapse.py → SynapseLoader)
  - Observation C    (integration/relations/relation_logger.py → SynapseGrounder)
  - Phase 7 Engine   (esde_engine/loaders.py → SynapseLoader)
  ALL must use SynapseStore. If either bypasses it, tests MUST fail.
  Patch effects are reflected in molecule generation AND relation extraction
  simultaneously.

Overlay Rules (§2):
  1. Load order: Base JSON → Patch v3.1 → Patch v3.2 → ...
  2. edge_key = "{synset_id}::{atom_id}" identifies each edge uniquely
  3. disable_edge ALWAYS wins (tombstone: kept in memory, excluded from results)
  4. add_edge duplicates: last-one-wins (score/metadata updated)
  5. All conflicts logged as [OVERLAY_CONFLICT] at DEBUG level

Audit Checklist (Phase 1 gate):
  1. load(base, patches) returns deterministic results
  2. Tombstone: disabled edge_key never appears in get_edges()
  3. [OVERLAY_CONFLICT] logged on conflict
  4. Patch-free behavior = exact current behavior (zero regression)
  5. Same store usable by Phase 8 Sensor and Obs C
"""

import json
import hashlib
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from copy import deepcopy

from .schema import SynapsePatchEntry

logger = logging.getLogger(__name__)


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file (truncated)."""
    if not os.path.exists(filepath):
        return "file_not_found"
    try:
        with open(filepath, 'rb') as f:
            return f"sha256:{hashlib.sha256(f.read()).hexdigest()[:16]}"
    except Exception:
        return "hash_error"


class SynapseStore:
    """
    Unified Synapse data store with overlay patch support.

    Provides the same interface as the legacy SynapseLoader for
    backward compatibility, plus patch overlay capabilities.

    Usage (no patches — backward compatible):
        store = SynapseStore()
        store.load("esde_synapses_v3.json")
        edges = store.get_edges("kill.v.01")

    Usage (with patches):
        store = SynapseStore()
        store.load("esde_synapses_v3.json", patches=["patches/v3.1.json"])
        edges = store.get_edges("kill.v.01")  # includes patched edges
    """

    def __init__(self):
        # Core data: synset_id → List[edge_dict]
        # This is the RESOLVED view after overlay application.
        self.synapses: Dict[str, List[Dict]] = {}

        # Base metadata from the JSON _meta field
        self.meta: Dict[str, Any] = {}

        # Tombstone set: edge_keys that are disabled
        self._tombstones: Set[str] = set()

        # Audit: track which patches were applied
        self._applied_patches: List[str] = []
        self._patch_stats: Dict[str, int] = {
            "edges_added": 0,
            "edges_disabled": 0,
            "conflicts_overwrite": 0,
            "conflicts_disable_wins": 0,
        }

        # File hashes for audit trail
        self._base_hash: str = "not_loaded"
        self._patch_hashes: List[str] = []

        # Loading state
        self._loaded: bool = False
        self._base_path: Optional[str] = None

    # ==========================================
    # Loading
    # ==========================================

    def load(
        self,
        base_path: str,
        patches: Optional[List[str]] = None,
    ) -> bool:
        """
        Load base Synapse JSON and apply overlay patches.

        Args:
            base_path: Path to base Synapse JSON (e.g., esde_synapses_v3.json)
            patches: Optional list of patch file paths, applied in order.

        Returns:
            True if load succeeded.

        Audit: Deterministic — same inputs always produce same output.
        """
        # Reset state
        self.synapses = {}
        self.meta = {}
        self._tombstones = set()
        self._applied_patches = []
        self._patch_stats = {
            "edges_added": 0,
            "edges_disabled": 0,
            "conflicts_overwrite": 0,
            "conflicts_disable_wins": 0,
        }
        self._patch_hashes = []
        self._loaded = False
        self._base_path = base_path

        # Step 1: Load base JSON
        if not os.path.exists(base_path):
            logger.error("[SynapseStore] Base file not found: %s", base_path)
            return False

        try:
            self._base_hash = _compute_file_hash(base_path)
            with open(base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.synapses = deepcopy(data.get("synapses", {}))
            self.meta = data.get("_meta", {})
            logger.info(
                "[SynapseStore] Loaded base: %d synsets from %s",
                len(self.synapses), base_path,
            )
        except Exception as e:
            logger.error("[SynapseStore] Error loading base: %s", e)
            return False

        # Step 2: Build edge index from base (for conflict detection)
        # edge_key → (synset_id, index_in_list)
        self._edge_index: Dict[str, tuple] = {}
        for synset_id, edges in self.synapses.items():
            for i, edge in enumerate(edges):
                atom = edge.get("concept_id", "")
                key = SynapsePatchEntry.make_key(synset_id, atom)
                self._edge_index[key] = (synset_id, i)

        # Step 3: Apply patches in order
        if patches:
            for patch_path in patches:
                self._apply_patch_file(patch_path)

        # Step 4: Purge tombstoned edges from the resolved view
        self._purge_tombstones()

        self._loaded = True
        logger.info(
            "[SynapseStore] Ready: %d synsets, %d patches applied, "
            "%d tombstones, %d conflicts",
            len(self.synapses),
            len(self._applied_patches),
            len(self._tombstones),
            self._patch_stats["conflicts_overwrite"]
            + self._patch_stats["conflicts_disable_wins"],
        )
        return True

    def _apply_patch_file(self, patch_path: str) -> None:
        """Apply a single patch file (JSONL or JSON array)."""
        if not os.path.exists(patch_path):
            logger.warning("[SynapseStore] Patch file not found: %s", patch_path)
            return

        patch_hash = _compute_file_hash(patch_path)
        self._patch_hashes.append(patch_hash)

        entries = self._read_patch_file(patch_path)
        for entry in entries:
            self._apply_entry(entry)

        self._applied_patches.append(patch_path)
        logger.info(
            "[SynapseStore] Applied patch: %s (%d entries)",
            patch_path, len(entries),
        )

    def _read_patch_file(self, path: str) -> List[SynapsePatchEntry]:
        """
        Read patch entries from file.

        Supports two formats:
          - JSONL: one JSON object per line
          - JSON: array of objects, or {"patches": [...]}
        """
        entries = []
        path_str = str(path)

        try:
            with open(path_str, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return entries

            # Strategy: try full-document JSON first; on failure, try JSONL.
            parsed_as_json = False
            try:
                data = json.loads(content)
                parsed_as_json = True
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict) and "patches" in data:
                    items = data["patches"]
                elif isinstance(data, dict):
                    items = [data]
                else:
                    items = []
                for item in items:
                    try:
                        entries.append(SynapsePatchEntry.from_dict(item))
                    except (KeyError, ValueError) as e:
                        logger.warning(
                            "[SynapseStore] Skipping invalid patch entry: %s", e
                        )
            except json.JSONDecodeError:
                pass

            if not parsed_as_json:
                # JSONL: one JSON object per line
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        entries.append(SynapsePatchEntry.from_dict(item))
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(
                            "[SynapseStore] Skipping invalid JSONL line: %s", e
                        )

        except Exception as e:
            logger.error("[SynapseStore] Error reading patch %s: %s", path, e)

        return entries

    def _apply_entry(self, entry: SynapsePatchEntry) -> None:
        """
        Apply a single patch entry to the store.

        Conflict Resolution (Design Spec v2.1 §2):
          - disable_edge: ALWAYS wins. Adds to tombstone set.
          - add_edge on existing key: last-one-wins (overwrite).
          - add_edge on tombstoned key: tombstone wins, add is ignored.
        """
        key = entry.edge_key
        existed = key in self._edge_index

        if entry.op == "disable_edge":
            self._tombstones.add(key)
            self._patch_stats["edges_disabled"] += 1

            if existed:
                self._patch_stats["conflicts_disable_wins"] += 1
                logger.debug(
                    "[OVERLAY_CONFLICT] disable_edge wins over existing: %s",
                    key,
                )
            return

        # op == "add_edge"
        if key in self._tombstones:
            # Tombstone takes priority — disable_edge ALWAYS wins
            self._patch_stats["conflicts_disable_wins"] += 1
            logger.debug(
                "[OVERLAY_CONFLICT] tombstone blocks add_edge: %s "
                "(disabled edge cannot be re-added via patch)",
                key,
            )
            return

        new_edge = entry.to_synapse_edge()
        synset_id = entry.synset_id

        if existed:
            # Overwrite existing edge (last-one-wins)
            old_synset_id, old_idx = self._edge_index[key]
            if old_synset_id == synset_id:
                old_edge = self.synapses[old_synset_id][old_idx]
                logger.debug(
                    "[OVERLAY_CONFLICT] overwrite edge %s: "
                    "score %.3f → %.3f, reason: %s",
                    key,
                    old_edge.get("raw_score", 0),
                    entry.score,
                    entry.reason,
                )
                self.synapses[old_synset_id][old_idx] = new_edge
            else:
                # Edge key references different synset — shouldn't happen
                # but handle gracefully by appending to correct synset
                logger.warning(
                    "[OVERLAY_CONFLICT] edge_key %s: synset mismatch "
                    "(index=%s, entry=%s). Appending to entry synset.",
                    key, old_synset_id, synset_id,
                )
                if synset_id not in self.synapses:
                    self.synapses[synset_id] = []
                self.synapses[synset_id].append(new_edge)

            self._patch_stats["conflicts_overwrite"] += 1
        else:
            # New edge — append
            if synset_id not in self.synapses:
                self.synapses[synset_id] = []
            self.synapses[synset_id].append(new_edge)
            self._patch_stats["edges_added"] += 1

        # Update index
        if synset_id not in self.synapses:
            # Shouldn't reach here, but defensive
            self.synapses[synset_id] = [new_edge]
        idx = len(self.synapses[synset_id]) - 1
        if not existed:
            self._edge_index[key] = (synset_id, idx)

    def _purge_tombstones(self) -> None:
        """Remove tombstoned edges from the resolved view."""
        if not self._tombstones:
            return

        for synset_id in list(self.synapses.keys()):
            original = self.synapses[synset_id]
            filtered = []
            for edge in original:
                atom = edge.get("concept_id", "")
                key = SynapsePatchEntry.make_key(synset_id, atom)
                if key not in self._tombstones:
                    filtered.append(edge)
            if filtered:
                self.synapses[synset_id] = filtered
            else:
                del self.synapses[synset_id]

    # ==========================================
    # Query Interface (backward-compatible)
    # ==========================================

    def get_edges(self, synset_id: str) -> List[Dict]:
        """
        Get edges for a synset ID.

        Returns list of edge dicts, each with at least:
          {"concept_id": "...", "raw_score": float, ...}

        Tombstoned edges are excluded.
        Compatible with legacy SynapseLoader.get_edges().
        """
        return self.synapses.get(synset_id, [])

    def has_synset(self, synset_id: str) -> bool:
        """Check if synset exists (with non-tombstoned edges)."""
        return synset_id in self.synapses and len(self.synapses[synset_id]) > 0

    def get_all_concept_ids(self) -> set:
        """Get all unique concept IDs in resolved synapses."""
        concept_ids = set()
        for edges in self.synapses.values():
            for edge in edges:
                cid = edge.get("concept_id")
                if cid:
                    concept_ids.add(cid)
        return concept_ids

    def is_tombstoned(self, edge_key: str) -> bool:
        """Check if an edge_key has been disabled."""
        return edge_key in self._tombstones

    # ==========================================
    # Audit Interface
    # ==========================================

    def get_file_hash(self) -> str:
        """Get base file hash for audit trail."""
        return self._base_hash

    def get_meta_top_k(self) -> Optional[int]:
        """Get global_top_k from meta (backward compat)."""
        config = self.meta.get("config", {})
        return config.get("global_top_k")

    def get_audit_info(self) -> Dict[str, Any]:
        """
        Get full audit trail for this store instance.

        Includes: base hash, patch hashes, patch stats,
        tombstone count, total synset count.
        """
        return {
            "base_path": self._base_path,
            "base_hash": self._base_hash,
            "patches_applied": list(self._applied_patches),
            "patch_hashes": list(self._patch_hashes),
            "patch_stats": dict(self._patch_stats),
            "tombstone_count": len(self._tombstones),
            "synset_count": len(self.synapses),
            "total_edges": sum(len(v) for v in self.synapses.values()),
        }

    def get_synapse_dict(self) -> Dict[str, List[Dict]]:
        """
        Get the raw synapses dict.

        For consumers that need direct dict access (e.g., SynapseGrounder).
        Returns the resolved view (tombstones already purged).
        """
        return self.synapses

    # ==========================================
    # Singleton support (optional)
    # ==========================================

    _instance: Optional["SynapseStore"] = None

    @classmethod
    def get_instance(
        cls,
        base_path: Optional[str] = None,
        patches: Optional[List[str]] = None,
    ) -> "SynapseStore":
        """
        Get or create a singleton instance.

        On first call, loads with given paths.
        Subsequent calls return the same instance (ignoring args).
        Use reset_instance() to force reload.
        """
        if cls._instance is None:
            cls._instance = cls()
            if base_path:
                cls._instance.load(base_path, patches=patches)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
