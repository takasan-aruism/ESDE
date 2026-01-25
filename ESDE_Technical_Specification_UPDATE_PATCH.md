# ESDE Technical Specification - Update Patch v5.4.8-MIG.2

**適用日**: 2026-01-25  
**変更内容**: Migration Phase 2追加、Phase 9-7削除、バージョン更新

---

## 適用手順

以下の変更を ESDE_Technical_Specification.md に適用してください。

---

## 1. ファイル先頭のバージョン更新

**検索**: `Version: 5.4.` で始まる行

**変更前**:
```
**Version**: 5.4.7-SUB.1
```

**変更後**:
```
**Version**: 5.4.8-MIG.2
**Updated**: 2026-01-25
```

---

## 2. Section 15.11 "Next Steps" テーブルの更新

**検索**: `### 15.11 Next Steps`

**変更前** (最後の数行):
```markdown
| 17 | ~~Phase 9-5 W5 Structural Condensation~~ | ✅ **PASS** |
| 18 | ~~Phase 9-6 W6 Structural Observation~~ | ✅ **PASS** |
| 19 | Phase 9-7 W7 Judgment Layer | **Next** |
```

**変更後**:
```markdown
| 17 | ~~Phase 9-5 W5 Structural Condensation~~ | ✅ **PASS** |
| 18 | ~~Phase 9-6 W6 Structural Observation~~ | ✅ **PASS** |
| 19 | ~~Substrate Layer (v0.1.0)~~ | ✅ **PASS** |
| 20 | ~~Migration Phase 2 (v0.2.1)~~ | ✅ **PASS** |
| 21 | Phase 10 | **Next** |

**Note:** Phase 9 is complete at W6. Next development phase is Phase 10.
```

---

## 3. Section 15.12 "Development Workflow" の更新

**検索**: `### 15.12 Development Workflow`

**変更前**:
```markdown
All:     Phase 9-7 Scope Decision  ← Current
```

**変更後**:
```markdown
All:     Phase 10 Scope Decision  ← Current
```

---

## 4. Section 16 "Version History" テーブルに追記

**検索**: `## 16. Version History`

テーブルの最後（`5.4.6-P9.6` の行の後）に以下を追加:

```markdown
| **5.4.7-SUB.1** | **2026-01-23** | **Substrate Layer (Context Fabric)** |
| **5.4.8-MIG.2** | **2026-01-25** | **Migration Phase 2 (Policy-Based Statistics)** |
```

---

## 5. 新規セクション追加: Migration Phase 2

**挿入位置**: Phase 9 セクションの後、または Section 15 の適切な場所

```markdown
### Migration Phase 2: Policy-Based Statistics (v0.2.1)

Migration Phase 2 introduces Policy-based condition signature generation, bridging Substrate Layer with W2 statistics.

#### Components

| File | Purpose |
|------|---------|
| `statistics/policies/__init__.py` | Package exports |
| `statistics/policies/base.py` | BaseConditionPolicy abstract class |
| `statistics/policies/standard.py` | StandardConditionPolicy implementation |
| `statistics/w2_aggregator.py` | Updated with registry/policy support |

#### P0 Requirements

| ID | Requirement | Status |
|----|-------------|--------|
| P0-MIG-1 | Policy ID in hash (collision prevention) | ✅ |
| P0-MIG-2 | Type preservation (no str() coercion) | ✅ |
| P0-MIG-3 | Canonical JSON (Substrate unified) | ✅ |
| P0-MIG-4 | Missing key tracking (explicit list) | ✅ |

#### Data Flow

```
ArticleRecord
    ↓ substrate_ref
SubstrateRegistry.get()
    ↓
ContextRecord.traces
    ↓
StandardConditionPolicy.compute_signature()
    ↓
64-char SHA256 condition signature
    ↓
W2Aggregator statistics
```

#### Legacy Fallback

When `substrate_ref` is None or registry lookup fails:
- W2Aggregator uses `_compute_legacy_hash()`
- Extracts factors from `ArticleRecord.source_meta`
- Uses Substrate-unified canonical JSON (no spaces)

#### Test Results

| Test Suite | Result |
|------------|--------|
| test_migration_phase2.py | 12/12 ✅ |
| test_e2e_gateway_substrate_w2.py | 4/4 ✅ |
| Regression (existing tests) | 3/3 ✅ |
```

---

## 6. ディレクトリ構造に追加

**検索**: ファイル構造を示すセクション（`statistics/` の部分）

以下を追加:

```markdown
├── statistics/
│   ├── policies/                    # [Migration Phase 2]
│   │   ├── __init__.py              # Package exports
│   │   ├── base.py                  # BaseConditionPolicy
│   │   └── standard.py              # StandardConditionPolicy
```

---

## 7. "Phase 9-7" の全削除

ファイル全体で以下を検索し、削除または修正:

- `Phase 9-7` → 削除または `Phase 10` に変更
- `W7` → 削除（W6で終了のため）
- `Judgment Layer` → 削除

---

## 変更サマリー

| 項目 | 変更内容 |
|------|----------|
| バージョン | 5.4.7-SUB.1 → 5.4.8-MIG.2 |
| Phase 9-7 | 全削除 |
| Next Steps #19 | Phase 9-7 → Substrate Layer (完了) |
| Next Steps #20 | 新規: Migration Phase 2 (完了) |
| Next Steps #21 | 新規: Phase 10 (Next) |
| Development Workflow | Phase 9-7 → Phase 10 |
| Version History | 2行追加 |
| 新規セクション | Migration Phase 2 詳細 |

---

*パッチ作成: Claude (Implementation Engineer)*  
*日付: 2026-01-25*
