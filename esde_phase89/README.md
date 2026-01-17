# ESDE Phase 8-9 統合手順

## Theme: Semantic Monitor & Continuous Existence (生命の鼓動を視る)

Phase 8-9は、Phase 8の最終フェーズとして「可視化」と「長時間検証」を実装する。

---

## 新規パッケージ

```
esde/
├── monitor/                    # [NEW] Phase 8-9
│   ├── __init__.py
│   └── semantic_monitor.py     # TUIダッシュボード
│
├── runner/                     # [NEW] Phase 8-9
│   ├── __init__.py
│   └── long_run.py             # Long-Run検証ハーネス
│
├── esde_cli.py                 # [NEW] CLIエントリポイント
│
├── feedback/                   # [既存] Phase 8-8
├── pipeline/                   # [既存] Phase 8-8
├── index/                      # [既存] Phase 8-7
└── ledger/                     # [既存] Phase 8-5/8-6
```

## 依存関係

```bash
pip install rich  # TUIダッシュボード用（オプション）
```

richがない場合はプレーンテキストモードで動作します。

---

## CLI使用方法

```bash
# 単発観測
python esde_cli.py observe "I love you"

# TUIモニター起動
python esde_cli.py monitor --steps 100

# Long-Runテスト
python esde_cli.py longrun --steps 50 --check-interval 5

# ステータス表示
python esde_cli.py status
```

---

## Semantic Monitor

TUIダッシュボードで以下を表示：

| Panel | 内容 |
|-------|------|
| Header | Version, Uptime, Ledger Seq, Alert Count |
| Live Feed | Input, Target Atom, Rigidity, Strategy, Formula |
| Rankings | Top Rigid (R→1.0), Top Volatile (R→0.0) |
| Stats | Index Size, Direction Balance |

```python
from monitor import SemanticMonitor

monitor = SemanticMonitor()
monitor.update(pipeline_result)
monitor.update_from_index(index)
monitor.render()
```

---

## Long-Run Runner

長時間稼働検証：

```python
from runner import LongRunRunner, print_report

runner = LongRunRunner(
    pipeline=pipeline,
    ledger=ledger,
    index=index,
    health_check_interval=10
)

report = runner.run(steps=100)
print_report(report)
```

### LongRunReport

```python
@dataclass
class LongRunReport:
    total_steps: int
    completed_steps: int
    successful_observations: int
    abstained: int
    total_alerts: int
    alert_atoms: List[str]
    ledger_valid: bool
    validation_checks: int
    errors: List[Dict]
    stopped_early: bool
```

### エラー処理方針（GPT監査v8.9.1）

**Fail Fast**: エラー発生時は即座に停止しreportを返す。

---

## テスト結果

```
============================================================
ESDE Phase 8-9: Monitor & Long-Run Test Suite
============================================================

Rich library available: True

Monitor State Init: ✅ PASS
Monitor Update: ✅ PASS
Monitor Update with Alert: ✅ PASS
Monitor Render: ✅ PASS
LongRun Basic (steps=10): ✅ PASS
LongRun Alert Count: ✅ PASS
LongRun Abstain Handling: ✅ PASS
LongRun Report Structure: ✅ PASS
T895: Ledger Invariance: ✅ PASS
CLI Observe (Mock): ✅ PASS

Results: 10/10 passed
✅ ALL TESTS PASSED - Phase 8-9 Ready
```

---

## Phase 8 完了サマリー

| Phase | テーマ | Tests |
|-------|--------|-------|
| 8-1 | Sensor V2 + Modular | ✅ |
| 8-2 | Molecule Generator/Validator | ✅ |
| 8-3 | Live LLM Integration | ✅ |
| 8-4 | Stability Audit | ✅ |
| 8-5 | Ephemeral Ledger | 11/11 ✅ |
| 8-6 | Persistent Ledger | 13/13 ✅ |
| 8-7 | Semantic Index | 13/13 ✅ |
| 8-8 | Feedback Loop | 14/14 ✅ |
| **8-9** | **Monitor & Long-Run** | **10/10** ✅ |

---

## Phase 9 への展望

Phase 8で構築した内省エンジンv1は完成。

Phase 9候補：
- **9-A: Reboot Implementation** - Alert → 自動Reboot
- **9-B: Energy Function** - v3.0数学モデル実装
- **9-C: Visualization** - WebUIダッシュボード
- **9-D: Application** - 実ユースケース
