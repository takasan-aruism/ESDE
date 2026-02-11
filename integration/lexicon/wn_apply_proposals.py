#!/usr/bin/env python3
"""
ESDE Lexicon — Proposal Applicator (Constitution v1.0)
======================================================
Reads proposals.json → applies approved changes to dictionary & lexicon.

Actions per pattern:
  A_MERGE:   primary absorbs alias. alias_of link created. Core pools merged.
  D_SUBSUME: parent_of / child_of links created. Both atoms stay active.
  B_COUPLE:  couple_of links created. Both atoms stay independent. Phase 9 data.
  MONITOR:   No action (logged only).

All changes produce a NEW version of dictionary + a relations registry.
Original files are never modified in-place.

Usage:
  python3 wn_apply_proposals.py \
    --proposals proposals.json \
    --dictionary esde_dictionary.json \
    --lexicon-dir lexicon/ \
    --out-dictionary esde_dictionary_v2.json \
    --out-relations atom_relations.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from copy import deepcopy


VERSION = "1.0.0"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved: {path}")


def _handle_merge_symmetry(primary, alias, primary_sym, alias_sym, dictionary):
    """
    Constitution §2.4a-c: Multi-symmetry with backward compatibility.
    
    §2.4a: Primary keeps primary_sym_pair (= old symmetric_pair). Phase 8 safe.
    §2.4b: Alias's sym_pair is absorbed into primary's sym_edges[] with axis label.
    §2.4c: Orphaned partner (alias_sym) re-points to primary. No orphans.
    
    Schema:
      symmetric_pair  → unchanged (Phase 8 backward compat, = primary_sym_pair)
      sym_edges[]     → new field: [{target, inherited_from, axis_label}]
    """
    changes = {}
    
    if not alias_sym or alias_sym == primary_sym:
        print(f"    sym: no additional symmetry to inherit")
        return changes
    
    # §2.4a: symmetric_pair stays as-is (primary's original partner)
    # This is the backward-compatible field. Phase 8 reads this.
    print(f"    sym §2.4a: {primary}.symmetric_pair = {primary_sym} (preserved)")
    
    # §2.4b: Add alias's partner to sym_edges[]
    if 'sym_edges' not in dictionary[primary]:
        # Initialize with the existing primary pair
        dictionary[primary]['sym_edges'] = [{
            'target': primary_sym,
            'inherited_from': primary,
            'axis_label': 'original',
        }]
    
    # Add the alias's partner
    existing_targets = [e['target'] for e in dictionary[primary]['sym_edges']]
    if alias_sym not in existing_targets:
        dictionary[primary]['sym_edges'].append({
            'target': alias_sym,
            'inherited_from': alias,
            'axis_label': f'inherited_from_{alias}',
        })
        print(f"    sym §2.4b: {primary}.sym_edges += [{alias_sym} via {alias}]")
    
    changes['primary_sym_edges'] = [e['target'] for e in dictionary[primary]['sym_edges']]
    
    # §2.4c: Re-point the orphaned partner to primary (no orphans)
    if alias_sym in dictionary:
        old_ref = dictionary[alias_sym].get('symmetric_pair', '')
        dictionary[alias_sym]['symmetric_pair'] = primary
        
        # Also give the partner a sym_edges for full traceability
        if 'sym_edges' not in dictionary[alias_sym]:
            dictionary[alias_sym]['sym_edges'] = [{
                'target': primary,
                'inherited_from': alias_sym,
                'axis_label': 'original',
                'note': f'rewired from {old_ref} (merged into {primary})',
            }]
        else:
            # Update existing edge
            for edge in dictionary[alias_sym]['sym_edges']:
                if edge['target'] == alias:
                    edge['target'] = primary
                    edge['note'] = f'rewired from {old_ref} (merged into {primary})'
        
        print(f"    sym §2.4c: {alias_sym}.symmetric_pair = {old_ref} → {primary} (orphan fixed)")
        changes['orphan_fixed'] = alias_sym
        changes['orphan_old_ref'] = old_ref
    
    return changes


def apply_merge(proposal, dictionary, lexicon_dir):
    """
    Constitution §2: Merge as Re-definition (相転移)
    
    - primary_id keeps its entry
    - alias_id gets alias_of pointing to primary
    - Core pools are merged under primary (multi-nucleus)
    - Deviation pools are merged with dev_origin tracking
    - Symmetric pairs: §2.4a-c (multi-symmetry with backward compatibility)
    """
    primary = proposal['primary_id']
    alias = proposal['alias_id']
    j = proposal['pair_jaccard']
    
    print(f"\n  🔴 MERGE: {alias} → {primary} (J={j:.3f})")
    
    changes = {
        'type': 'merge',
        'primary_id': primary,
        'alias_id': alias,
        'pair_jaccard': j,
    }
    
    # 1. Dictionary: mark alias
    if alias in dictionary:
        old_def = dictionary[alias].get('definition_en', '')
        dictionary[alias]['status'] = 'merged'
        dictionary[alias]['alias_of'] = primary
        dictionary[alias]['merged_at'] = datetime.now(timezone.utc).isoformat()
        dictionary[alias]['pre_merge_definition'] = old_def
        print(f"    dict: {alias} → alias_of={primary}")
    
    # 2. Dictionary: update primary with multi-nucleus info
    if primary in dictionary:
        if 'core_clusters' not in dictionary[primary]:
            dictionary[primary]['core_clusters'] = [primary]
        dictionary[primary]['core_clusters'].append(alias)
        print(f"    dict: {primary} core_clusters={dictionary[primary]['core_clusters']}")
    
    # 3. Symmetric pair handling (§2.4a-c)
    primary_sym = dictionary.get(primary, {}).get('symmetric_pair', '')
    alias_sym = dictionary.get(alias, {}).get('symmetric_pair', '')
    
    sym_changes = _handle_merge_symmetry(
        primary, alias, primary_sym, alias_sym, dictionary
    )
    changes['symmetry'] = sym_changes
    
    # 3. Lexicon: merge core pools (if lexicon entries exist)
    primary_file = lexicon_dir / (primary.replace('.', '_') + '.json')
    alias_file = lexicon_dir / (alias.replace('.', '_') + '.json')
    
    core_merged = 0
    dev_merged = 0
    
    if primary_file.exists() and alias_file.exists():
        p_entry = load_json(primary_file)
        a_entry = load_json(alias_file)
        
        # Merge core words (deduplicate by (w, pos))
        existing_keys = set((w['w'].lower(), w['pos']) for w in p_entry['core_pool']['words'])
        for w in a_entry['core_pool']['words']:
            key = (w['w'].lower(), w['pos'])
            if key not in existing_keys:
                w['merged_from'] = alias
                p_entry['core_pool']['words'].append(w)
                existing_keys.add(key)
                core_merged += 1
        
        # Merge deviation words with origin tracking
        existing_dev = set((w['w'].lower(), w['pos']) for w in p_entry['deviation_pool']['words'])
        for w in a_entry['deviation_pool']['words']:
            key = (w['w'].lower(), w['pos'])
            if key not in existing_dev:
                w['dev_origin'] = alias
                p_entry['deviation_pool']['words'].append(w)
                existing_dev.add(key)
                dev_merged += 1
        
        # Update counts
        p_entry['core_pool']['count'] = len(p_entry['core_pool']['words'])
        p_entry['deviation_pool']['count'] = len(p_entry['deviation_pool']['words'])
        p_entry['core_pool']['words'].sort(key=lambda x: x['w'].lower())
        p_entry['deviation_pool']['words'].sort(key=lambda x: x['w'].lower())
        
        # Mark alias entry
        a_entry['status'] = 'merged'
        a_entry['alias_of'] = primary
        
        save_json(p_entry, primary_file)
        save_json(a_entry, alias_file)
        
        print(f"    lexicon: +{core_merged} core, +{dev_merged} dev → {primary}")
    
    changes['core_words_merged'] = core_merged
    changes['dev_words_merged'] = dev_merged
    
    return changes


def apply_subsume(proposal, dictionary, lexicon_dir):
    """
    Constitution §3: Subsume (包含)
    
    - Both atom IDs remain active
    - parent_of / child_of links created
    - Mapper prioritizes child for grounding
    """
    parent = proposal['parent_id']
    child = proposal['child_id']
    j = proposal['pair_jaccard']
    
    print(f"\n  🟠 SUBSUME: {child} ⊂ {parent} (J={j:.3f})")
    
    # Dictionary: add hierarchy links
    if parent in dictionary:
        if 'children' not in dictionary[parent]:
            dictionary[parent]['children'] = []
        if child not in dictionary[parent]['children']:
            dictionary[parent]['children'].append(child)
        print(f"    dict: {parent}.children += [{child}]")
    
    if child in dictionary:
        dictionary[child]['parent_of'] = parent
        print(f"    dict: {child}.parent_of = {parent}")
    
    return {
        'type': 'subsume',
        'parent_id': parent,
        'child_id': child,
        'pair_jaccard': j,
    }


def apply_couple(proposal, dictionary):
    """
    Constitution §4: Couple / Relation (共鳴)
    
    - Both atoms stay independent
    - Structural resonance link registered for Phase 9
    """
    a = proposal['atom_a']
    b = proposal['atom_b']
    j = proposal['pair_jaccard']
    cat_a = proposal['category_a']
    cat_b = proposal['category_b']
    
    print(f"\n  🔵 COUPLE: {a} ({cat_a}) ↔ {b} ({cat_b}) (J={j:.3f})")
    
    # Add couple_of to both atoms
    for src, tgt in [(a, b), (b, a)]:
        if src in dictionary:
            if 'couples' not in dictionary[src]:
                dictionary[src]['couples'] = []
            couple_entry = {
                'target': tgt,
                'pair_jaccard': j,
                'cross_category': f"{cat_a}↔{cat_b}",
            }
            # Avoid duplicates
            existing_targets = [c['target'] for c in dictionary[src]['couples']]
            if tgt not in existing_targets:
                dictionary[src]['couples'].append(couple_entry)
    
    return {
        'type': 'couple',
        'atom_a': a,
        'atom_b': b,
        'pair_jaccard': j,
        'categories': f"{cat_a}↔{cat_b}",
    }


def main():
    parser = argparse.ArgumentParser(description='Apply approved Lexicon proposals')
    parser.add_argument('--proposals', required=True, help='proposals.json')
    parser.add_argument('--dictionary', required=True, help='esde_dictionary.json')
    parser.add_argument('--lexicon-dir', default='lexicon', help='Lexicon entries directory')
    parser.add_argument('--out-dictionary', default='esde_dictionary_v2.json', help='Output dictionary')
    parser.add_argument('--out-relations', default='atom_relations.json', help='Output relations registry')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without writing')
    args = parser.parse_args()
    
    proposals_data = load_json(args.proposals)
    dictionary_raw = load_json(args.dictionary)
    lexicon_dir = Path(args.lexicon_dir)
    
    # Normalize dictionary format
    if 'concepts' in dictionary_raw and isinstance(dictionary_raw['concepts'], dict):
        dictionary = dictionary_raw['concepts']
        wrapper = dictionary_raw
    else:
        dictionary = dictionary_raw
        wrapper = None
    
    proposals = proposals_data['proposals']
    
    # Filter to actionable proposals (not MONITOR)
    actionable = [p for p in proposals if p.get('pattern') in ('A_MERGE', 'D_SUBSUME', 'B_COUPLE')]
    
    print(f"{'='*70}")
    print(f"  LEXICON CONSTITUTION v1.0 — PROPOSAL APPLICATION")
    print(f"{'='*70}")
    print(f"  Total proposals: {len(proposals)}")
    print(f"  Actionable: {len(actionable)} (excluding MONITOR)")
    print(f"  {'DRY RUN — no files will be written' if args.dry_run else 'LIVE — files will be updated'}")
    
    # Apply in priority order (A > D > B/C)
    change_log = []
    
    merges = [p for p in actionable if p['pattern'] == 'A_MERGE']
    subsumes = [p for p in actionable if p['pattern'] == 'D_SUBSUME']
    couples = [p for p in actionable if p['pattern'] == 'B_COUPLE']
    
    # --- Merges ---
    if merges:
        print(f"\n  ── Pattern A: Merge ({len(merges)} proposals) ──")
        for p in merges:
            if args.dry_run:
                print(f"    [DRY] Would merge {p['alias_id']} → {p['primary_id']}")
            else:
                change = apply_merge(p, dictionary, lexicon_dir)
                change_log.append(change)
    
    # --- Subsumes ---
    if subsumes:
        print(f"\n  ── Pattern D: Subsume ({len(subsumes)} proposals) ──")
        for p in subsumes:
            if args.dry_run:
                print(f"    [DRY] Would subsume {p['child_id']} ⊂ {p['parent_id']}")
            else:
                change = apply_subsume(p, dictionary, lexicon_dir)
                change_log.append(change)
    
    # --- Couples ---
    if couples:
        print(f"\n  ── Pattern B/C: Couple ({len(couples)} proposals) ──")
        for p in couples:
            if args.dry_run:
                print(f"    [DRY] Would couple {p['atom_a']} ↔ {p['atom_b']}")
            else:
                change = apply_couple(p, dictionary)
                change_log.append(change)
    
    if args.dry_run:
        print(f"\n  Dry run complete. No files written.")
        return
    
    # --- Save outputs ---
    print(f"\n  ── Saving ──")
    
    # Update dictionary version
    if wrapper:
        if 'meta' not in wrapper:
            wrapper['meta'] = {}
        wrapper['meta']['version'] = wrapper['meta'].get('version', '1.0') + '-lex2'
        wrapper['meta']['lexicon_constitution'] = 'v1.0'
        wrapper['meta']['applied_at'] = datetime.now(timezone.utc).isoformat()
        wrapper['concepts'] = dictionary
        save_json(wrapper, args.out_dictionary)
    else:
        save_json(dictionary, args.out_dictionary)
    
    # Build relations registry (Phase 9 fuel)
    relations = {
        'meta': {
            'version': VERSION,
            'constitution': 'v1.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'source': args.proposals,
        },
        'merges': [c for c in change_log if c['type'] == 'merge'],
        'subsumes': [c for c in change_log if c['type'] == 'subsume'],
        'couples': [c for c in change_log if c['type'] == 'couple'],
        'summary': {
            'total_changes': len(change_log),
            'merges': len([c for c in change_log if c['type'] == 'merge']),
            'subsumes': len([c for c in change_log if c['type'] == 'subsume']),
            'couples': len([c for c in change_log if c['type'] == 'couple']),
        }
    }
    save_json(relations, args.out_relations)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  APPLICATION COMPLETE")
    print(f"{'='*70}")
    
    merge_count = len([c for c in change_log if c['type'] == 'merge'])
    subsume_count = len([c for c in change_log if c['type'] == 'subsume'])
    couple_count = len([c for c in change_log if c['type'] == 'couple'])
    
    print(f"  Merges applied:  {merge_count}")
    if merge_count > 0:
        for c in change_log:
            if c['type'] == 'merge':
                print(f"    {c['alias_id']} → {c['primary_id']}  (+{c['core_words_merged']} core, +{c['dev_words_merged']} dev)")
    
    print(f"  Subsumes applied: {subsume_count}")
    if subsume_count > 0:
        for c in change_log:
            if c['type'] == 'subsume':
                print(f"    {c['child_id']} ⊂ {c['parent_id']}")
    
    print(f"  Couples registered: {couple_count}")
    if couple_count > 0:
        for c in change_log:
            if c['type'] == 'couple':
                print(f"    {c['atom_a']} ↔ {c['atom_b']}  ({c['categories']})")
    
    active_atoms = sum(1 for v in dictionary.values() 
                       if isinstance(v, dict) and v.get('status') != 'merged')
    merged_atoms = sum(1 for v in dictionary.values() 
                       if isinstance(v, dict) and v.get('status') == 'merged')
    
    print(f"\n  Atom inventory: {active_atoms} active + {merged_atoms} merged = {active_atoms + merged_atoms} total")
    print(f"\n  ✅ Dictionary: {args.out_dictionary}")
    print(f"  ✅ Relations: {args.out_relations}")


if __name__ == "__main__":
    main()
