"""
ESDE Engine v5.3.2 - CLI Entry Point

Usage:
    python -m esde_engine
"""

from .config import VERSION
from .engine import ESDEEngine


def main():
    print("=" * 60)
    print(f"ESDE Engine v{VERSION}")
    print("Phase 7A+: Multi-Hypothesis Routing")
    print("=" * 60)
    
    engine = ESDEEngine()
    if not engine.initialize():
        print("Failed to initialize. Exiting.")
        return
    
    print("\nReady. Enter text (or 'quit'):\n")
    
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if not text:
            continue
        
        result = engine.process(text)
        print("\n--- Result ---")
        print(f"Tokens (all): {result.get('tokens', [])}")
        print(f"Tokens (active): {result.get('active_tokens', [])}")
        print(f"Synsets: {result.get('synsets_found', 0)}")
        
        meta = result.get("_meta", {})
        print(f"Hits: direct={meta.get('direct_hits', 0)}, proxy={meta.get('proxy_hits', 0)}")
        print(f"Skipped (D): {meta.get('skipped_count', 0)}")
        
        print("\nTop Concepts:")
        for c in result.get("top_concepts", []):
            print(f"  {c['concept_id']}: {c['score']}")
        
        print("\nTop Levels:")
        for l in result.get("top_levels", []):
            print(f"  {l['concept_id']} -> {l['axis']}.{l['level']}: {l['score']}")
        
        # Routing Decisions (Multi-Hypothesis)
        routing_decisions = result.get("routing_decisions", [])
        if routing_decisions:
            print("\nRouting Decisions:")
            for rd in routing_decisions:
                action = rd.get("action", "unknown")
                token = rd.get("token", "?")
                margin = rd.get("margin", 0)
                entropy = rd.get("entropy", 0)
                reason = rd.get("reason", "")
                hyp = rd.get("hypotheses", {})
                
                if action == "abstain":
                    scores_str = " ".join([f"{r}:{s:.2f}" for r, s in sorted(hyp.items())])
                    print(f"  {token}: abstain ({reason}) [{scores_str}] margin={margin:.2f} entropy={entropy:.2f} (queued)")
                else:
                    winner = rd.get("winner", "?")
                    print(f"  {token}: {action} (winner={winner}) margin={margin:.2f} entropy={entropy:.2f}")
        
        # Routing Queue output (legacy display)
        rq = result.get("routing_queue", {})
        
        typos = rq.get("typo_candidates", [])
        if typos:
            print("\nRoute A (Typo Candidates):")
            for t in typos:
                info = t.get("typo_info", {})
                print(f"  {t['token']} -> {info.get('suggestion')} (conf={info.get('confidence')})")
        
        proper_nouns = rq.get("proper_noun_candidates", [])
        if proper_nouns:
            print("\nRoute B (Proper Noun Candidates):")
            for p in proper_nouns:
                print(f"  {p['token']}: {p.get('synsets', [])[:3]}")
        
        molecules = rq.get("molecule_candidates", [])
        if molecules:
            print("\nRoute C (Molecule Candidates):")
            for m in molecules:
                synsets = m.get("synsets", [])
                if synsets:
                    print(f"  {m['token']}: {m['reason']} | {synsets[:3]}")
                else:
                    print(f"  {m['token']}: {m['reason']}")
        
        abstain = result.get("abstain_summary", {})
        if any(v > 0 for v in abstain.values()):
            print("\nAbstain Summary:")
            for reason, count in abstain.items():
                if count > 0:
                    print(f"  {reason}: {count}")
        
        # Unknown Queue summary
        queue_summary = result.get("queue_summary", {})
        if queue_summary:
            print("\nUnknown Queue:")
            print(f"  queued_records: {queue_summary.get('queued_records', 0)}")
            print(f"  deduped_records: {queue_summary.get('deduped_records', 0)}")
            print(f"  path: {queue_summary.get('path', 'N/A')}")
            print(f"  run_id: {queue_summary.get('run_id', 'N/A')}")
        
        evidence = result.get("evidence", [])
        variance_ev = [e for e in evidence if "variance" in e]
        if variance_ev:
            print("\nHigh Variance (per-synset):")
            for e in variance_ev[:3]:
                v = e["variance"]
                print(f"  {e['token']} ({e['synset']}): margin={v['margin']}, entropy={v['entropy']} -> {e['decision']}")
        
        print()


if __name__ == "__main__":
    main()
