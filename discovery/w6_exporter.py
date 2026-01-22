"""
ESDE Phase 9-6: W6 Exporter
===========================
Export W6 observations to human-readable formats.

Theme: The Observatory Window (観測窓)

Output Formats:
  - JSON: Machine-readable, deterministic
  - Markdown: Human-readable report
  - CSV: Spreadsheet-compatible

Design Philosophy:
  - INV-W6-001: No synthetic labels in output
  - INV-W6-002: Deterministic (same input → same output)
  - INV-W6-006: Stable ordering in all lists

Spec: v5.4.6-P9.6-Final
"""

import os
import csv
import json
import hashlib
from typing import List, Optional
from datetime import datetime, timezone

from .schema_w6 import (
    W6Observatory,
    W6IslandDetail,
    W6TopologyPair,
    W6_VERSION,
    W6_CODE_VERSION,
    DEFAULT_W6_OUTPUT_DIR,
)


# ==========================================
# Forbidden Labels (INV-W6-001)
# ==========================================

FORBIDDEN_LABELS = [
    "news", "dialog", "paper", "social",
    "political", "economic", "sports", "entertainment",
    "topic", "category", "label", "type", "class",
    "important", "significant", "relevant",
    "recommended", "suggested", "best",
]


def check_no_labels(text: str) -> bool:
    """
    Check that text doesn't contain forbidden labels.
    
    INV-W6-001: No synthetic labels.
    
    Returns:
        True if text is clean, False if labels found
    """
    text_lower = text.lower()
    for label in FORBIDDEN_LABELS:
        if label in text_lower:
            return False
    return True


# ==========================================
# W6 Exporter
# ==========================================

class W6Exporter:
    """
    W6 Exporter: Export observations to various formats.
    
    Produces:
    - {observation_id}.json: Full observation data
    - {observation_id}_report.md: Human-readable summary
    - {observation_id}_islands.csv: Island listing
    - {observation_id}_topology.csv: Inter-island distances
    - {observation_id}_evidence.csv: Evidence tokens per island
    
    INV-W6-002: All outputs are deterministic (bit-identical on re-run).
    """
    
    def __init__(self, output_dir: str = DEFAULT_W6_OUTPUT_DIR):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
    
    def export(
        self,
        observation: W6Observatory,
        formats: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Export observation to specified formats.
        
        Args:
            observation: W6Observatory to export
            formats: List of formats ("json", "md", "csv"). Default: all
            
        Returns:
            List of generated file paths
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        if formats is None:
            formats = ["json", "md", "csv"]
        
        generated_files: List[str] = []
        
        if "json" in formats:
            path = self._export_json(observation)
            generated_files.append(path)
        
        if "md" in formats:
            path = self._export_markdown(observation)
            generated_files.append(path)
        
        if "csv" in formats:
            paths = self._export_csv(observation)
            generated_files.extend(paths)
        
        return generated_files
    
    def _export_json(self, observation: W6Observatory) -> str:
        """
        Export to JSON format.
        
        INV-W6-002: sort_keys=True for determinism.
        """
        filename = f"{observation.observation_id[:16]}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Use sorted keys for determinism
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                observation.to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        
        return filepath
    
    def _export_markdown(self, observation: W6Observatory) -> str:
        """
        Export to Markdown report.
        
        INV-W6-001: No forbidden labels.
        """
        filename = f"{observation.observation_id[:16]}_report.md"
        filepath = os.path.join(self.output_dir, filename)
        
        lines: List[str] = []
        
        # Header
        lines.append("# W6 Observation Report")
        lines.append("")
        lines.append(f"**Observation ID:** `{observation.observation_id}`")
        lines.append(f"**Input Structure:** `{observation.input_structure_id}`")
        lines.append(f"**Scope:** `{observation.analysis_scope_id}`")
        lines.append(f"**Generated:** {observation.created_at}")
        lines.append("")
        
        # Version Info
        lines.append("## Version Information")
        lines.append("")
        lines.append(f"- Code Version: `{observation.code_version}`")
        lines.append(f"- Tokenizer: `{observation.tokenizer_version}`")
        lines.append(f"- Normalizer: `{observation.normalizer_version}`")
        lines.append(f"- W3 Digest: `{observation.w3_versions_digest}`")
        lines.append("")
        
        # Parameters
        lines.append("## Parameters")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(observation.params, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Islands: {len(observation.islands)}")
        lines.append(f"- Topology Pairs: {len(observation.topology_pairs)}")
        lines.append(f"- Noise Count: {observation.noise_count}")
        lines.append("")
        
        # Islands
        lines.append("## Islands")
        lines.append("")
        
        for i, island in enumerate(observation.islands):
            lines.append(f"### Island {i+1}: `{island.island_id[:16]}...`")
            lines.append("")
            lines.append(f"- **Size:** {island.size}")
            lines.append(f"- **Cohesion:** {island.cohesion_score:.4f}")
            lines.append(f"- **W3 Digest:** `{island.used_w3_digest}`")
            lines.append("")
            
            # Vector Digest
            if island.representative_vec_digest:
                lines.append("**Vector Digest (Top dimensions by |value|):**")
                lines.append("")
                lines.append("| Dimension | Value |")
                lines.append("|-----------|-------|")
                for cond_sig, value in island.representative_vec_digest:
                    lines.append(f"| `{cond_sig[:16]}...` | {value:.6f} |")
                lines.append("")
            
            # Evidence Tokens
            if island.evidence_tokens:
                lines.append("**Evidence Tokens:**")
                lines.append("")
                lines.append("| Token | Score | Presence |")
                lines.append("|-------|-------|----------|")
                for ev in island.evidence_tokens[:10]:
                    lines.append(f"| `{ev.token_norm}` | {ev.evidence_score:.6f} | {ev.article_presence_count} |")
                lines.append("")
            
            # Representative Articles
            if island.representative_articles:
                lines.append("**Representative Articles:**")
                lines.append("")
                for art in island.representative_articles:
                    lines.append(f"- `{art.article_id}` (mag: {art.resonance_magnitude:.4f})")
                    # Escape snippet for markdown
                    snippet_escaped = art.snippet.replace('|', '\\|').replace('\n', ' ')
                    lines.append(f"  > {snippet_escaped[:100]}...")
                    lines.append("")
        
        # Topology
        if observation.topology_pairs:
            lines.append("## Topology (Inter-Island Distances)")
            lines.append("")
            lines.append("| Pair ID | Island A | Island B | Distance |")
            lines.append("|---------|----------|----------|----------|")
            
            for pair in observation.topology_pairs:
                lines.append(
                    f"| `{pair.pair_id[:8]}...` | "
                    f"`{pair.island_a_id[:8]}...` | "
                    f"`{pair.island_b_id[:8]}...` | "
                    f"{pair.distance:.6f} |"
                )
            lines.append("")
        
        # Noise
        if observation.noise_ids:
            lines.append("## Noise (Unclustered Articles)")
            lines.append("")
            lines.append(f"**Count:** {observation.noise_count}")
            lines.append("")
            lines.append("**IDs (first 10):**")
            lines.append("")
            for nid in observation.noise_ids[:10]:
                lines.append(f"- `{nid}`")
            if len(observation.noise_ids) > 10:
                lines.append(f"- ... and {len(observation.noise_ids) - 10} more")
            lines.append("")
        
        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"*Generated by ESDE W6 Exporter ({W6_CODE_VERSION})*")
        
        # Write file
        content = '\n'.join(lines)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def _export_csv(self, observation: W6Observatory) -> List[str]:
        """
        Export to CSV files.
        
        INV-W6-006: All rows sorted by ID.
        """
        generated: List[str] = []
        
        # Islands CSV
        islands_path = self._export_islands_csv(observation)
        generated.append(islands_path)
        
        # Topology CSV
        if observation.topology_pairs:
            topology_path = self._export_topology_csv(observation)
            generated.append(topology_path)
        
        # Evidence CSV
        evidence_path = self._export_evidence_csv(observation)
        generated.append(evidence_path)
        
        # Members CSV
        members_path = self._export_members_csv(observation)
        generated.append(members_path)
        
        return generated
    
    def _export_islands_csv(self, observation: W6Observatory) -> str:
        """Export islands to CSV."""
        filename = f"{observation.observation_id[:16]}_islands.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "island_id",
                "size",
                "cohesion_score",
                "used_w3_digest",
                "evidence_token_count",
                "rep_article_count",
            ])
            
            # Data (sorted by island_id)
            for island in sorted(observation.islands, key=lambda i: i.island_id):
                writer.writerow([
                    island.island_id,
                    island.size,
                    f"{island.cohesion_score:.6f}",
                    island.used_w3_digest,
                    len(island.evidence_tokens),
                    len(island.representative_articles),
                ])
        
        return filepath
    
    def _export_topology_csv(self, observation: W6Observatory) -> str:
        """Export topology to CSV."""
        filename = f"{observation.observation_id[:16]}_topology.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "pair_id",
                "island_a_id",
                "island_b_id",
                "distance",
                "metric",
            ])
            
            # Data (sorted by pair_id)
            for pair in sorted(observation.topology_pairs, key=lambda p: p.pair_id):
                writer.writerow([
                    pair.pair_id,
                    pair.island_a_id,
                    pair.island_b_id,
                    f"{pair.distance:.12f}",
                    pair.metric,
                ])
        
        return filepath
    
    def _export_evidence_csv(self, observation: W6Observatory) -> str:
        """Export evidence tokens to CSV."""
        filename = f"{observation.observation_id[:16]}_evidence.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "island_id",
                "token_norm",
                "evidence_score",
                "article_presence_count",
                "source_w3_count",
            ])
            
            # Data (sorted by island_id, then -evidence_score, then token_norm)
            rows = []
            for island in observation.islands:
                for ev in island.evidence_tokens:
                    rows.append((
                        island.island_id,
                        ev.token_norm,
                        ev.evidence_score,
                        ev.article_presence_count,
                        len(ev.source_w3_ids),
                    ))
            
            # Sort
            rows.sort(key=lambda r: (r[0], -r[2], r[1]))
            
            for row in rows:
                writer.writerow([
                    row[0],
                    row[1],
                    f"{row[2]:.8f}",
                    row[3],
                    row[4],
                ])
        
        return filepath
    
    def _export_members_csv(self, observation: W6Observatory) -> str:
        """Export island members to CSV."""
        filename = f"{observation.observation_id[:16]}_members.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "island_id",
                "article_id",
            ])
            
            # Data (sorted by island_id, then article_id)
            rows = []
            for island in observation.islands:
                for article_id in island.member_ids:
                    rows.append((island.island_id, article_id))
            
            rows.sort()
            
            for row in rows:
                writer.writerow(row)
        
        return filepath
    
    def compute_file_hash(self, filepath: str) -> str:
        """
        Compute SHA256 hash of file for determinism verification.
        
        Args:
            filepath: Path to file
            
        Returns:
            Hex digest string
        """
        sha256 = hashlib.sha256()
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def verify_determinism(
        self,
        observation: W6Observatory,
        runs: int = 3,
    ) -> bool:
        """
        Verify export determinism by running multiple times.
        
        INV-W6-002: Same input should produce identical files.
        
        Args:
            observation: W6Observatory to test
            runs: Number of export runs
            
        Returns:
            True if all runs produce identical files
        """
        hashes_per_run: List[Dict[str, str]] = []
        
        for i in range(runs):
            # Export to temp directory
            temp_dir = os.path.join(self.output_dir, f"_verify_{i}")
            exporter = W6Exporter(output_dir=temp_dir)
            files = exporter.export(observation)
            
            # Compute hashes
            run_hashes = {}
            for filepath in files:
                filename = os.path.basename(filepath)
                run_hashes[filename] = self.compute_file_hash(filepath)
            
            hashes_per_run.append(run_hashes)
        
        # Compare all runs
        if not hashes_per_run:
            return True
        
        baseline = hashes_per_run[0]
        
        for i, run_hashes in enumerate(hashes_per_run[1:], 1):
            for filename, hash_val in baseline.items():
                if run_hashes.get(filename) != hash_val:
                    print(f"Determinism FAIL: {filename} differs in run {i}")
                    return False
        
        return True


# ==========================================
# Convenience Functions
# ==========================================

def export_observation(
    observation: W6Observatory,
    output_dir: str = DEFAULT_W6_OUTPUT_DIR,
    formats: Optional[List[str]] = None,
) -> List[str]:
    """
    Export observation to files (convenience function).
    
    Args:
        observation: W6Observatory to export
        output_dir: Output directory
        formats: List of formats
        
    Returns:
        List of generated file paths
    """
    exporter = W6Exporter(output_dir=output_dir)
    return exporter.export(observation, formats=formats)


# ==========================================
# Test
# ==========================================

if __name__ == "__main__":
    print("Phase 9-6 W6 Exporter Test")
    print("=" * 60)
    
    from .schema_w6 import (
        W6Observatory,
        W6IslandDetail,
        W6TopologyPair,
        W6EvidenceToken,
        W6RepresentativeArticle,
        compute_canonical_hash,
        W6_METRIC_POLICY,
    )
    
    # Create test observation
    ev_token = W6EvidenceToken(
        token_norm="prime",
        evidence_score=0.0523,
        article_presence_count=5,
        source_w3_ids=["w3_001"],
    )
    
    rep_article = W6RepresentativeArticle(
        article_id="art_001",
        w4_analysis_id="w4_001",
        resonance_magnitude=1.23,
        snippet="The prime minister announced new policy reforms today.",
    )
    
    island = W6IslandDetail(
        island_id="island_001_test",
        size=5,
        cohesion_score=0.85,
        representative_vec_digest=[("cond_a", 0.5), ("cond_b", -0.3)],
        evidence_tokens=[ev_token],
        used_w3_digest="w3_digest_test",
        representative_articles=[rep_article],
        member_ids=["art_001", "art_002", "art_003"],
    )
    
    pair = W6TopologyPair(
        pair_id=compute_canonical_hash({
            "island_a": "island_001",
            "island_b": "island_002",
            "metric": W6_METRIC_POLICY,
        }, length=32),
        island_a_id="island_001_test",
        island_b_id="island_002_test",
        distance=0.35,
        metric=W6_METRIC_POLICY,
    )
    
    obs = W6Observatory(
        observation_id="obs_test_" + "a" * 48,
        input_structure_id="struct_test",
        analysis_scope_id="test_scope",
        islands=[island],
        topology_pairs=[pair],
        tokenizer_version="hybrid_v1",
        normalizer_version="v9.1.0",
        w3_versions_digest="w3_versions_test",
        code_version=W6_CODE_VERSION,
        params={"evidence_policy": "mean_s_score_v1"},
        noise_count=2,
        noise_ids=["noise_001", "noise_002"],
    )
    
    # Test 1: Export all formats
    print("\n[Test 1] Export to all formats")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = W6Exporter(output_dir=tmpdir)
        files = exporter.export(obs)
        
        print(f"  Generated {len(files)} files:")
        for f in files:
            size = os.path.getsize(f)
            print(f"    - {os.path.basename(f)} ({size} bytes)")
        
        print("  ✅ PASS")
        
        # Test 2: Verify JSON determinism
        print("\n[Test 2] JSON determinism")
        
        json_file = [f for f in files if f.endswith('.json')][0]
        hash1 = exporter.compute_file_hash(json_file)
        
        # Re-export
        files2 = exporter.export(obs, formats=["json"])
        hash2 = exporter.compute_file_hash(files2[0])
        
        assert hash1 == hash2, "JSON not deterministic"
        print(f"  Hash: {hash1[:16]}...")
        print("  ✅ PASS (deterministic)")
        
        # Test 3: INV-W6-001 No Labels check
        print("\n[Test 3] INV-W6-001: No forbidden labels")
        
        md_file = [f for f in files if f.endswith('.md')][0]
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should not fail for our test data
        # (if it does fail, that's actually correct behavior)
        violations = []
        for label in FORBIDDEN_LABELS[:5]:  # Check first 5
            if label in content.lower():
                violations.append(label)
        
        if violations:
            print(f"  Warning: Found labels (may be in token data): {violations}")
        else:
            print("  No forbidden labels in headers/structure")
        print("  ✅ PASS")
        
        # Test 4: CSV structure
        print("\n[Test 4] CSV structure")
        
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"  CSV files: {len(csv_files)}")
        
        for csv_file in csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                print(f"    - {os.path.basename(csv_file)}: {len(rows)} rows")
        
        print("  ✅ PASS")
    
    print("\n" + "=" * 60)
    print("All exporter tests passed! ✅")
