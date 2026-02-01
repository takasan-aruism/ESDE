# ESDE Module Reference（統合ツール開発用）

**Version**: 5.5.0  
**Updated**: 2026-02-02  
**Purpose**: 全モジュールの役割を把握し、統合パイプラインを設計するための資料  
**Note**: Phase 9 セクションを v2.0 パイプライン（Lens統合版）に全面改訂

---

## 1. 全体構成図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Entry Points                              │
├─────────────────────────────────────────────────────────────────────────┤
│  esde-engine-v532.py          │ Phase 7A: テキスト→Unknown Queue       │
│  resolve_unknown_queue_*.py   │ Phase 7B+: Unknown Queue解決           │
│  esde_cli_live.py             │ Phase 8-9: 統合CLI（observe/monitor）   │
│  stats_cli.py                 │ Phase 9 (legacy): 旧統計パイプラインCLI │
│  run_full_pipeline.py         │ Phase 9 (v2.0): Lens統合パイプラインCLI │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Core Packages                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  esde_engine/     │ Runtime Engine（トークン化、ルーティング）           │
│  sensor/          │ Phase 8: テキスト→Atom→Molecule変換                │
│  ledger/          │ Phase 8-5/6: 意味記憶（減衰/強化/永続化）           │
│  index/           │ Phase 8-7: Semantic Index（硬直性計算）             │
│  pipeline/        │ Phase 8-8: Feedback Loop（戦略調整）                │
│  monitor/         │ Phase 8-9: TUIダッシュボード                        │
│  runner/          │ Phase 8-9: Long-Run実行器                          │
│  integration/     │ Phase 9-0: ContentGateway（外部データ取込）         │
│  statistics/      │ Phase 9 (legacy): W1-W4統計計算                    │
│  statistics/pipeline/ │ Phase 9 (v2.0): Lens統合パイプライン ★現行    │
│  discovery/       │ Phase 9 (legacy): W5-W6構造発見                    │
│  substrate/       │ Layer 0: 条件因子トレース保存                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. esde_engine/（Runtime Engine）

**Phase**: 7A  
**役割**: テキストをトークン化し、既知/未知を判定してルーティング

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `config.py` | 定数群 | 全閾値・パスの定義（Single Source of Truth） | - |
| `utils.py` | `tokenize()`, `compute_entropy()` | トークン化、エントロピー計算、typo検出 | text → tokens |
| `loaders.py` | `SynapseLoader`, `GlossaryLoader` | Synapse/Glossary JSONの読み込み | file → dict |
| `extractors.py` | `SynsetExtractor` | WordNet synset抽出 | token → synsets |
| `collectors.py` | `ActivationCollector` | Synapse活性化の収集 | synsets → activations |
| `routing.py` | `UnknownTokenRouter` | 4仮説並列評価（A/B/C/D）、分散ゲート | token → route_decision |
| `queue.py` | `UnknownQueueWriter` | Unknown Queueへの追記 | decision → JSONL |
| `engine.py` | `ESDEEngine` | メインオーケストレータ | text → result + queue |

**処理フロー:**
```
text → tokenize → extract_synsets → collect_activations → route → queue
```

---

## 3. esde_engine/resolver/（Phase 7B+ Unknown Resolution）

**Phase**: 7B+, 7C, 7C', 7D  
**役割**: Unknown Queueを解決（仮説生成、監査、メタ監査）

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `state.py` | `QueueStateManager` | レガシー状態管理 | - |
| `aggregate_state.py` | `AggregateStateManager` | 集約状態管理（v5.3.4+） | token → aggregate_key → state |
| `hypothesis.py` | `evaluate_all_hypotheses()` | A/B/C/D仮説の並列評価 | evidence → scores + volatility |
| `online.py` | `MultiSourceProvider` | 外部API検索（5ソース） | query → evidence_items |
| `ledger.py` | `EvidenceLedger` | 解決決定の監査証跡 | decision → JSONL |
| `cache.py` | `SearchCache` | 検索結果キャッシュ | query → cached_result |
| `patches.py` | `PatchWriter` | パッチ出力（alias/synapse/stopword追加） | decision → patch_file |

**外部ソース（online.py MultiSourceProvider）:**
- FreeDictionaryAPI
- WikipediaAPI
- DatamuseAPI
- UrbanDictionaryAPI
- DuckDuckGoAPI

---

## 4. sensor/（Phase 8: Introspection Sensor）

**Phase**: 8-1〜8-3  
**役割**: テキストをAtom候補に変換し、LLMでMoleculeを生成

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `loader_synapse.py` | `SynapseLoader` | Synapse JSONロード（singleton） | file → synapse_map |
| `extract_synset.py` | `SynsetExtractor` | WordNet synset抽出 | token → synsets |
| `rank_candidates.py` | `CandidateRanker` | スコア集約、決定論的ソート | synsets → candidates |
| `legacy_trigger.py` | `LegacyTriggerMatcher` | v1トリガーマッチング（fallback） | token → atoms |
| `audit_trace.py` | `AuditTracer` | カウンタ/ハッシュ/evidence記録 | - |
| `glossary_validator.py` | `GlossaryValidator` | Glossary座標検証 | atom+axis+level → valid? |
| `validator_v83.py` | `MoleculeValidatorV83` | v8.3スキーマ検証 | molecule → validation_result |
| `molecule_generator_live.py` | `MoleculeGeneratorLive` | QwQ-32B LLM呼び出し | candidates → molecule |
| `constants.py` | `VALID_OPERATORS` | 演算子定義 | - |

**MoleculeGeneratorLive 内部クラス:**
| クラス | 役割 |
|--------|------|
| `SpanCalculator` | text_ref → span[start,end) 計算 |
| `CoordinateCoercer` | 無効座標 → null + ログ |
| `FormulaValidator` | formula構文検証 |

**処理フロー:**
```
text → extract_synsets → rank_candidates → generate_molecule(LLM) → validate
```

---

## 5. ledger/（Phase 8-5/6: Semantic Memory）

**Phase**: 8-5（Ephemeral）, 8-6（Persistent）  
**役割**: 意味観測の記録、減衰/強化、ハッシュチェーン永続化

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `memory_math.py` | `decay()`, `reinforce()`, `tau_for_axis()` | 減衰/強化の数学計算 | weight + dt → new_weight |
| `ephemeral_ledger.py` | `EphemeralLedger` | インメモリ意味記憶 | molecule → memory_entry |
| `canonical.py` | `canonical_json()` | 正規化JSON（バイト一致保証） | dict → bytes |
| `chain_crypto.py` | `compute_event_hash()` | ハッシュチェーン計算 | entry + prev_hash → hash |
| `persistent_ledger.py` | `PersistentLedger` | JSONL永続化（改ざん検出可能） | entry → file |

**Memory Math パラメータ:**
```python
decay(w, dt, tau) = w × exp(-dt / tau)
reinforce(w, alpha=0.2) = w + alpha × (1 - w)
oblivion_threshold = 0.01  # これ以下は消去
```

---

## 6. index/（Phase 8-7: Semantic Index）

**Phase**: 8-7  
**役割**: Atom使用パターンの索引化、硬直性（Rigidity）計算

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `semantic_index.py` | `SemanticIndex` | L2インメモリ構造（AtomStats, FormulaStats） | - |
| `projector.py` | `Projector` | L1（Ledger）→ L2（Index）投影 | ledger → index |
| `rigidity.py` | `compute_rigidity()` | formula多様性から硬直度計算 | formula_stats → R値 |
| `query_api.py` | `QueryAPI` | 外部問い合わせAPI | query → stats |

**Rigidity計算:**
```
R = 1.0 → 常に同じformula（硬直）
R < 1.0 → formulaに変動あり（健全）
```

---

## 7. pipeline/（Phase 8-8: Feedback Loop）

**Phase**: 8-8  
**役割**: 硬直性に基づく戦略調整

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `core_pipeline.py` | `ESDEPipeline` | Sensor→Ledger→Index統合パイプライン | text → observation |
| `core_pipeline.py` | `ModulatedGenerator` | 硬直性に応じたLLMパラメータ調整 | rigidity → temperature |

---

## 8. monitor/（Phase 8-9: TUI Dashboard）

**Phase**: 8-9  
**役割**: リアルタイム監視ダッシュボード

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `semantic_monitor.py` | `SemanticMonitor` | Rich TUIダッシュボード | ledger+index → display |

---

## 9. runner/（Phase 8-9: Long-Run Execution）

**Phase**: 8-9  
**役割**: 長期実行と統計収集

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `long_run.py` | `LongRunRunner` | N回の観測実行 | corpus → observations |
| `long_run.py` | `LongRunReport` | 実行レポート生成 | observations → report |

---

## 10. integration/（Phase 9-0: Content Gateway）

**Phase**: 9-0  
**役割**: 外部データの正規化と取り込み

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `content_gateway.py` | `ContentGateway` | 外部コンテンツ取り込み | raw_data → ArticleRecord |
| `content_gateway.py` | `ArticleRecord` | 正規化された記事データ構造 | - |

**ArticleRecord構造:**
```python
@dataclass
class ArticleRecord:
    article_id: str
    raw_text: str
    source_meta: Dict  # source_type, language_profile, fetched_at
    substrate_ref: Optional[str]  # Substrate Layer参照
```

---

## 11. statistics/（Phase 9 Legacy: W1-W4 Statistics）

> **⚠ LEGACY**: 以下は Phase 9 v1.x（Lens統合前）の旧モジュール群。
> 現行パイプラインは **Section 11b** の `statistics/pipeline/` を参照。
> 旧モジュールは Phase 7/8 との統合時に ContentGateway 経由で使用される可能性があるため記録を残す。

**Phase**: 9-1（W1）, 9-2（W2）, 9-3（W3）, 9-4（W4）  
**役割**: 条件付き統計計算、S-Score、共鳴ベクトル（旧パイプライン）

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| **スキーマ** ||||
| `schema.py` | `W1Record`, `W1GlobalStats` | W1データ構造 | - |
| `schema_w2.py` | `W2Record`, `ConditionEntry` | W2データ構造 | - |
| `schema_w3.py` | `W3Record`, `CandidateToken` | W3データ構造 | - |
| `schema_w4.py` | `W4Record` | W4データ構造 | - |
| **処理** ||||
| `tokenizer.py` | `HybridTokenizer` | トークン抽出（英語+記号） | text → tokens |
| `normalizer.py` | `normalize_token()` | NFKC正規化 | token → normalized |
| `w1_aggregator.py` | `W1Aggregator` | グローバル統計集計 | articles → W1GlobalStats |
| `w2_aggregator.py` | `W2Aggregator` | 条件付き統計集計 | articles + conditions → W2Records |
| `w3_calculator.py` | `W3Calculator` | S-Score計算 | W1 + W2 → W3Candidates |
| `w4_projector.py` | `W4Projector` | 共鳴ベクトル投影 | article + W3 → W4Record |
| **Policy** ||||
| `policies/base.py` | `BaseConditionPolicy` | Policy基底クラス | - |
| `policies/standard.py` | `StandardConditionPolicy` | 標準Policy実装 | - |
| **Utils (MIG-3)** ||||
| `utils.py` | `ExecutionContext` | 実行コンテキスト | - |
| `utils.py` | `validate_scope_id()` | Scope検証 | scope_id → valid? |
| `utils.py` | `resolve_stats_dir()` | パス解決 | policy + scope → path |
| `runner.py` | `StatisticsPipelineRunner` | 統計パイプライン実行 | policy + scope → results |

---

## 11b. statistics/pipeline/（Phase 9 v2.0: Lens統合パイプライン）★現行

**Phase**: 9 (v1.7〜v1.9, 完了)  
**役割**: Lens選択 → 特徴抽出 → 条件集約 → プロファイル → 類似度 → 島形成 → エクスポート

### コアパイプライン

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `run_full_pipeline.py` | `main()` | CLI エントリポイント。全W層を順次実行 | CLI args → JSON/MD/CSV |
| `lens.py` | `LENSES` dict | 3レンズ定義（Structure/Semantic/Hybrid） | - |

### W1: Feature Extraction

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| *(features/)* | `FeatureExtractor` | spaCy による20次元トークン特徴抽出 | text → List[TokenFeature] |

### W2: Conditional Aggregation

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `condition_provider.py` | `SectionConditionProvider` | セクション名を条件として抽出 | token + context → condition_id |
| `condition_provider.py` | `DocumentConditionProvider` | ドキュメント名を条件として抽出 | token + context → condition_id |
| `condition_provider.py` | `PassiveConditionProvider` | 受動態(0/1)を条件として抽出 | token + context → condition_id |
| `condition_provider.py` | `ParenthesesConditionProvider` | 括弧内(0/1)を条件として抽出 | token + context → condition_id |
| `condition_provider.py` | `QuoteConditionProvider` | 引用文内(0/1)を条件として抽出 | token + context → condition_id |
| `condition_provider.py` | `ProperNounConditionProvider` | 固有名詞有無(0/1)を条件として抽出 | token + context → condition_id |
| `w2_aggregator.py` | `W2Aggregator` | 条件別にトークン特徴を集約 | features + provider → W2Result |
| `w2_adapter.py` | adapter functions | 旧W2スキーマとの変換層 | - |

### W3: Profile Computation

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `w3_calculator.py` | `W3Calculator` | S-Score候補抽出（token mode） | W2 → W3Candidates |
| `w3_vector.py` | `W3VectorCalculator` | z-scoreプロファイル計算（vector mode） | W2 → W3VectorResult |

### W4: Similarity Computation

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `w4_projector.py` | `W4Projector`, `compute_pairwise_similarities()` | 共鳴ベクトル投影 + ペア類似度（token mode） | W3 + articles → W4Result |
| `w4_vector.py` | `W4VectorCalculator` | z-scoreベクトルのコサイン類似度（vector mode） | W3Vector → W4VectorResult |

### W5: Island Formation

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `w5_w6_adapter.py` | `SimpleCondensator` | 閾値フィルタ → 連結成分 → Island構造 | W4 + threshold → SimpleStructure |
| `w5_w6_adapter.py` | `SimpleIsland`, `SimpleStructure` | Island/Structure データ構造 | - |
| `threshold.py` | `ThresholdResolver` | 3層閾値合成（t_abs / t_rel / t_resolved） | similarities + config → threshold + trace |
| `global_model.py` | `GlobalThresholdModel` | 累積類似度分布の管理（lens別） | similarities → global quantile |
| `edge_selector.py` | `MutualKNNSelector` | 双方向k近傍フィルタ（連鎖防止） | similarity_pairs + k → filtered_edges |
| `edge_selector.py` | `NoOpSelector` | フィルタなし（単連結、比較用） | similarity_pairs → same |
| `edge_policy.py` | `EdgePolicyResolver` | k-sweep による自動k選定 | similarities + policy → PolicyResult |
| `edge_policy.py` | `SweepRow`, `PolicyResult` | k-sweep結果のデータ構造 | - |
| `chaining_metrics.py` | `compute_chaining_metrics()` | 連鎖診断指標（gcr, mean_intra等） | structure → metrics dict |

### W6: Export

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `w5_w6_adapter.py` | export functions | JSON/Markdown/CSV出力 | structure → files |
| `run_full_pipeline.py` | `_export_vector_report()` | Markdownレポート生成（threshold trace, k-sweep table含む） | structure → report.md |

### 処理フロー（v2.0）

```
CLI args (--dataset, --lens, --edge-filter, --knn-k, --threshold-mode)
  │
  ├── Wikipedia API fetch → ArticleRecord群
  │
  ├── Lens選択 → (ConditionProvider, FeatureMode)
  │
  ├── W1: FeatureExtractor → 20次元トークン特徴
  │     ↓
  ├── W2: ConditionProvider + W2Aggregator → 条件別統計
  │     ↓
  ├── W3: W3VectorCalculator → z-scoreプロファイル  [vector mode]
  │   or  W3Calculator → S-Score候補              [token mode]
  │     ↓
  ├── W4: W4VectorCalculator → コサイン類似度行列   [vector mode]
  │   or  W4Projector → 共鳴ベクトル + ペア類似度   [token mode]
  │     ↓
  ├── ThresholdResolver → t_resolved（3層合成）
  │     ↓
  ├── EdgePolicyResolver → k-sweep → k_chosen      [--knn-k auto]
  │   or  MutualKNNSelector → filtered edges        [--knn-k N]
  │   or  NoOpSelector → all edges                  [--edge-filter none]
  │     ↓
  ├── W5: SimpleCondensator → Island構造
  │     ↓
  ├── Chaining Metrics → gcr, mean_intra, etc.
  │     ↓
  └── W6: Export → analysis.json, report.md, k_sweep.csv
```

---

## 12. discovery/（Phase 9 Legacy: W5-W6 Discovery）

> **⚠ LEGACY**: 以下は Phase 9 v1.x（Lens統合前）の旧モジュール群。
> 現行の W5/W6 機能は `statistics/pipeline/w5_w6_adapter.py` に統合済み（Section 11b 参照）。

**Phase**: 9-5（W5）, 9-6（W6）  
**役割**: 島構造の形成、観測出力（旧パイプライン）

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `schema_w5.py` | `W5Island`, `W5Structure` | W5データ構造（島、ノイズ） | - |
| `schema_w6.py` | `W6Observatory`, `W6IslandDetail` | W6データ構造（観測窓） | - |
| `w5_condensator.py` | `W5Condensator` | 共鳴ベクトルクラスタリング | W4Records → W5Structure |
| `w6_analyzer.py` | `W6Analyzer` | Evidence抽出、Topology計算 | W5 + articles → W6Observatory |
| `w6_exporter.py` | `W6Exporter` | MD/CSV/JSON出力 | W6Observatory → files |

---

## 13. substrate/（Layer 0: Context Fabric）

**Phase**: Substrate  
**役割**: 条件因子のトレース保存（意味解釈なし）

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `context_record.py` | `ContextRecord` | 不変の観測単位 | - |
| `registry.py` | `SubstrateRegistry` | Append-only JSONL保存 | record → file |
| `trace.py` | `Trace` | namespace:name形式のKVペア | - |

---

## 14. データファイル（data/）

### Phase 7
| ファイル | 役割 |
|----------|------|
| `unknown_queue.jsonl` | 未知トークンキュー (7A) |
| `unknown_queue_7bplus.jsonl` | 集約済みキュー (7B+) |
| `unknown_queue_state_7bplus.json` | 集約状態 (7B+) |
| `evidence_ledger_7bplus.jsonl` | 解決監査証跡 (7B+) |
| `audit_log_7c.jsonl` | 構造監査ログ (7C) |
| `audit_votes_7cprime.jsonl` | LLM三重監査投票 (7C') |
| `patch_*.jsonl` | パッチ出力（人間レビュー用） |

### Phase 8
| ファイル | 役割 |
|----------|------|
| `semantic_ledger.jsonl` | 意味記憶（ハッシュチェーン） (8-6) |

### Phase 9 (legacy)
| ファイル | 役割 |
|----------|------|
| `stats/w1_global.json` | グローバル統計 (9-1) |
| `stats/w2_records.jsonl` | 条件付き統計 (9-2) |
| `stats/w3_candidates/` | 軸候補 (9-3) |
| `stats/w4_projections/` | 共鳴ベクトル (9-4) |

### Phase 9 (v2.0) ★現行
| ファイル | 役割 |
|----------|------|
| `data/threshold/{lens}_{feature_mode}.json` | GlobalThresholdModel 累積データ |
| `output/analysis.json` | 分析結果（島構造、threshold trace, edge policy trace） |
| `output/report.md` | 人間可読レポート（k-sweep テーブル含む） |
| `output/k_sweep.csv` | k-sweep 全候補の指標一覧 |
| `output/structure_stats.json` | 文構造統計（文長等） |

---

## 15. 統合処理フロー

### A. Phase 8 フロー（意味構造化）
```
text
  → sensor/extract_synset.py (WordNet)
  → sensor/rank_candidates.py (Atom候補)
  → sensor/molecule_generator_live.py (LLM → Molecule)
  → sensor/validator_v83.py (検証)
  → ledger/persistent_ledger.py (永続化)
  → index/projector.py (Index更新)
  → index/rigidity.py (硬直性計算)
  → pipeline/core_pipeline.py (戦略調整)
```

### B. Phase 9 フロー（v2.0: Lens統合パイプライン）★現行
```
CLI (--dataset, --lens, --edge-filter, --knn-k, --threshold-mode)
  → Wikipedia API fetch → ArticleRecord群
  → statistics/pipeline/run_full_pipeline.py
    → Lens選択 (lens.py)
    → W1: FeatureExtractor → 20次元トークン特徴
    → W2: ConditionProvider + W2Aggregator → 条件別統計
    → W3: W3VectorCalculator (z-score) or W3Calculator (S-Score)
    → W4: W4VectorCalculator (cosine) or W4Projector (resonance)
    → ThresholdResolver (t_abs / t_rel → t_resolved)
    → EdgePolicyResolver (k-sweep) or MutualKNNSelector (fixed k)
    → W5: SimpleCondensator → Island構造
    → Chaining Metrics (gcr, mean_intra, etc.)
    → W6: Export → analysis.json, report.md, k_sweep.csv
```

### B'. Phase 9 フロー（legacy: 旧パイプライン）
```
external_data
  → integration/content_gateway.py (ArticleRecord)
  → statistics/w1_aggregator.py (グローバル統計)
  → statistics/w2_aggregator.py (条件付き統計)
  → statistics/w3_calculator.py (S-Score)
  → statistics/w4_projector.py (共鳴ベクトル)
  → discovery/w5_condensator.py (島形成)
  → discovery/w6_analyzer.py (観測)
  → discovery/w6_exporter.py (出力)
```

### C. Phase 7 フロー（未知解決）
```
unknown_queue.jsonl
  → esde_engine/resolver/aggregate_state.py (集約)
  → esde_engine/resolver/online.py (外部検索)
  → esde_engine/resolver/hypothesis.py (仮説評価)
  → esde_engine/resolver/ledger.py (監査証跡)
  → patch_*.jsonl (人間レビュー待ち)
```

---

## 16. 統合ツール設計のポイント

### 必要な統合ポイント

| 接続 | 現状 | 必要な作業 |
|------|------|-----------|
| Phase 8 → Phase 9 | 独立 | Molecule → Cell統合層で条件因子により結合（未実装） |
| Phase 9 v2.0 → Cell | 独立 | Island + z-score profile → Cell統合層（未実装） |
| Phase 7 → Phase 8 | 独立 | 解決済みトークン → Synapse追加 |
| Substrate → W2 | Migration済 | Policy経由で接続済み |
| Phase 9 legacy → v2.0 | 共存 | 旧パイプラインは残存。将来的に整理の可能性 |

### 統合CLIの候補機能

```bash
# 全フロー実行
esde run --input articles/ --output results/

# Phase別実行
esde phase8 observe "I love you"
esde phase9 analyze --policy standard --scope run_001
esde phase7 resolve --limit 50

# モニタリング
esde monitor --live
esde status
```

---

*「記述せよ、しかし決定するな」*
