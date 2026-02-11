#!/usr/bin/env python3
"""
ESDE Lexicon — Proposal Generator (Lexicon Constitution v1.0)
==============================================================
Scans core_report.csv → generates proposals per the 3AI-approved rules.

Constitution v1.0 Priority:
  Pattern A (Merge)  > Pattern D (Subsume) > Pattern B/C (Couple)

Pattern A: pair_jaccard >= 0.75, same category, size_ratio <= 0.25
Pattern D: pair_jaccard >= 0.60, same category, NOT Pattern A
Pattern B/C: pair_jaccard >= 0.50, different category

All proposals are auto_status: flagged. Taka approval required.

Usage:
  python3 wn_proposal_gen.py --report core_report.csv --out proposals.json
"""

import json
import csv
import argparse
from collections import defaultdict
from pathlib import Path


def load_report(path: str) -> list:
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_pair_jaccard(rows: list) -> dict:
    """
    Build pair_jaccard map using max(A→B, B→A) rule.
    Returns: { (atomA, atomB): { pair_jaccard, j_ab, j_ba, ... } }
    Pairs are canonical: sorted tuple.
    """
    # Index by atom_id
    by_id = {r['atom_id']: r for r in rows}
    
    # Collect all top1 relationships
    edges = []
    for r in rows:
        a = r['atom_id']
        b = r.get('top1_overlap_atom', '')
        if not b or b not in by_id:
            continue
        j = float(r['top1_jaccard'])
        if j < 0.40:  # pre-filter
            continue
        edges.append((a, b, j))
    
    # Build pair map
    pairs = {}
    for a, b, j_ab in edges:
        pair_key = tuple(sorted([a, b]))
        if pair_key not in pairs:
            pairs[pair_key] = {
                'atom_a': pair_key[0],
                'atom_b': pair_key[1],
                'j_ab': 0,
                'j_ba': 0,
            }
        
        if a == pair_key[0]:
            pairs[pair_key]['j_ab'] = max(pairs[pair_key]['j_ab'], j_ab)
        else:
            pairs[pair_key]['j_ba'] = max(pairs[pair_key]['j_ba'], j_ab)
    
    # Also check reverse direction
    for a, b, j_ab in edges:
        pair_key = tuple(sorted([a, b]))
        if a == pair_key[1]:
            pairs[pair_key]['j_ba'] = max(pairs[pair_key]['j_ba'], j_ab)
        else:
            pairs[pair_key]['j_ab'] = max(pairs[pair_key]['j_ab'], j_ab)
    
    # Compute pair_jaccard = max(j_ab, j_ba)
    for pk, pv in pairs.items():
        pv['pair_jaccard'] = max(pv['j_ab'], pv['j_ba'])
        pv['bidirectional'] = (pv['j_ab'] > 0 and pv['j_ba'] > 0 
                                and abs(pv['j_ab'] - pv['j_ba']) < 0.01)
    
    return pairs


def classify_pair(pair_data: dict, row_a: dict, row_b: dict) -> dict:
    """
    Apply Constitution v1.0 rules to classify a pair.
    Priority: A > D > B/C
    """
    j = pair_data['pair_jaccard']
    cat_a = row_a['category']
    cat_b = row_b['category']
    same_cat = (cat_a == cat_b)
    
    core_a = int(row_a['core_count'])
    core_b = int(row_b['core_count'])
    max_core = max(core_a, core_b)
    
    # Size ratio for Pattern A
    size_diff_ratio = abs(core_a - core_b) / max_core if max_core > 0 else 0
    
    # Unique ratios for Pattern D hints
    uniq_a = float(row_a['unique_ratio_pct'])
    uniq_b = float(row_b['unique_ratio_pct'])
    
    result = {
        'atom_a': pair_data['atom_a'],
        'atom_b': pair_data['atom_b'],
        'pair_jaccard': j,
        'bidirectional': pair_data['bidirectional'],
        'category_a': cat_a,
        'category_b': cat_b,
        'same_category': same_cat,
        'core_count_a': core_a,
        'core_count_b': core_b,
        'size_diff_ratio': round(size_diff_ratio, 3),
        'unique_ratio_a': uniq_a,
        'unique_ratio_b': uniq_b,
        'auto_status': 'flagged',
        'approved': False,
    }
    
    # === Pattern A: Merge (相転移) ===
    # pair_jaccard >= 0.75, same category, size_ratio <= 0.25
    if j >= 0.75 and same_cat and size_diff_ratio <= 0.25:
        result['pattern'] = 'A_MERGE'
        result['priority'] = 1
        result['action'] = 'merge_as_redefinition'
        result['rationale'] = (
            f'Near-synonym within {cat_a}. '
            f'J={j:.3f}, size_ratio={size_diff_ratio:.3f}. '
            f'Constitution §2: Merge as Re-definition.'
        )
        # Determine primary/secondary
        if core_a >= core_b:
            result['primary_id'] = pair_data['atom_a']
            result['alias_id'] = pair_data['atom_b']
        else:
            result['primary_id'] = pair_data['atom_b']
            result['alias_id'] = pair_data['atom_a']
        return result
    
    # === Pattern D: Subsume (包含) ===
    # pair_jaccard >= 0.60, same category, NOT Pattern A
    if j >= 0.60 and same_cat:
        result['pattern'] = 'D_SUBSUME'
        result['priority'] = 2
        result['action'] = 'subsume_hierarchy'
        
        # Mechanical hints per §3.1
        size_ratio_check = size_diff_ratio >= 0.15  # i.e. < 0.85 symmetry
        child_low_unique = min(uniq_a, uniq_b) < 10
        parent_large = max_core >= 100
        
        needs_human = size_ratio_check or (child_low_unique and parent_large)
        result['needs_human'] = needs_human
        result['subsume_hints'] = {
            'size_ratio_asymmetric': size_ratio_check,
            'child_low_unique': child_low_unique,
            'parent_large_core': parent_large,
        }
        
        # Determine parent/child (larger core = parent)
        if core_a >= core_b:
            result['parent_id'] = pair_data['atom_a']
            result['child_id'] = pair_data['atom_b']
        else:
            result['parent_id'] = pair_data['atom_b']
            result['child_id'] = pair_data['atom_a']
        
        result['rationale'] = (
            f'Same category ({cat_a}), J={j:.3f} but size/unique asymmetry. '
            f'Constitution §3: Subsume hierarchy.'
        )
        return result
    
    # === Pattern B/C: Couple / Relation (共鳴) ===
    # pair_jaccard >= 0.50, different category
    if j >= 0.50 and not same_cat:
        result['pattern'] = 'B_COUPLE'
        result['priority'] = 3
        result['action'] = 'register_couple'
        result['rationale'] = (
            f'Cross-category overlap ({cat_a}↔{cat_b}), J={j:.3f}. '
            f'Constitution §4: Structural resonance, Phase 9 bypass.'
        )
        result['couple_data'] = {
            'target_atom_from_a': pair_data['atom_b'],
            'target_atom_from_b': pair_data['atom_a'],
            'overlap_count': int(row_a.get('top1_overlap_count', 0)),
            'directionality': 'bidirectional' if pair_data['bidirectional'] else 'unidirectional',
        }
        return result
    
    # === Below threshold but still notable ===
    if j >= 0.40:
        result['pattern'] = 'MONITOR'
        result['priority'] = 4
        result['action'] = 'log_and_watch'
        result['rationale'] = f'J={j:.3f}, below action threshold. Monitor only.'
        return result
    
    return result


def main():
    parser = argparse.ArgumentParser(description='ESDE Lexicon Proposal Generator (Constitution v1.0)')
    parser.add_argument('--report', required=True, help='core_report.csv path')
    parser.add_argument('--out', default='proposals.json', help='Output proposals JSON')
    args = parser.parse_args()
    
    rows = load_report(args.report)
    by_id = {r['atom_id']: r for r in rows}
    
    print(f"Loaded {len(rows)} atoms from {args.report}")
    
    # Compute pair Jaccards
    pairs = compute_pair_jaccard(rows)
    print(f"Found {len(pairs)} unique pairs with J > 0.40")
    
    # Classify each pair
    proposals = []
    for pair_key, pair_data in sorted(pairs.items(), key=lambda x: -x[1]['pair_jaccard']):
        a, b = pair_key
        if a not in by_id or b not in by_id:
            continue
        
        result = classify_pair(pair_data, by_id[a], by_id[b])
        if result.get('pattern'):
            proposals.append(result)
    
    # Sort by priority then jaccard
    proposals.sort(key=lambda x: (x.get('priority', 99), -x.get('pair_jaccard', 0)))
    
    # Output
    output = {
        'meta': {
            'constitution_version': '1.0',
            'source': args.report,
            'total_proposals': len(proposals),
            'by_pattern': {},
        },
        'proposals': proposals,
    }
    
    # Count by pattern
    for p in proposals:
        pat = p.get('pattern', 'unknown')
        output['meta']['by_pattern'][pat] = output['meta']['by_pattern'].get(pat, 0) + 1
    
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  LEXICON CONSTITUTION v1.0 — PROPOSAL REPORT")
    print(f"{'='*70}")
    
    for pat_name in ['A_MERGE', 'D_SUBSUME', 'B_COUPLE', 'MONITOR']:
        pat_props = [p for p in proposals if p.get('pattern') == pat_name]
        if not pat_props:
            continue
        
        icon = {'A_MERGE': '🔴', 'D_SUBSUME': '🟠', 'B_COUPLE': '🔵', 'MONITOR': '⚪'}
        section = {
            'A_MERGE': '§2 Merge as Re-definition (相転移)',
            'D_SUBSUME': '§3 Subsume (包含)',
            'B_COUPLE': '§4 Couple / Relation (共鳴)',
            'MONITOR': 'Monitor (監視)',
        }
        
        print(f"\n  {icon.get(pat_name, '?')} {section.get(pat_name, pat_name)} — {len(pat_props)} proposals")
        print(f"  {'-'*60}")
        
        for p in pat_props:
            j = p['pair_jaccard']
            a = p['atom_a']
            b = p['atom_b']
            bi = '↔' if p.get('bidirectional') else '→'
            
            if pat_name == 'A_MERGE':
                print(f"    {a:25s} {bi} {b:25s}  J={j:.3f}  primary={p.get('primary_id','?')}")
            elif pat_name == 'D_SUBSUME':
                nh = ' ⚠️HUMAN' if p.get('needs_human') else ''
                print(f"    {a:25s} {bi} {b:25s}  J={j:.3f}  parent={p.get('parent_id','?')}{nh}")
            elif pat_name == 'B_COUPLE':
                print(f"    {a:25s} {bi} {b:25s}  J={j:.3f}  ({p['category_a']}↔{p['category_b']})")
            else:
                print(f"    {a:25s} {bi} {b:25s}  J={j:.3f}")
    
    print(f"\n  Total: {len(proposals)} proposals")
    print(f"  All proposals are auto_status=flagged, awaiting Taka approval.")
    print(f"\n  ✅ Saved: {args.out}")


if __name__ == "__main__":
    main()
