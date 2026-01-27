# ESDE Module Reference（統合ツール開発用）

**Version**: 5.4.8-MIG.3  
**Purpose**: 全モジュールの役割を把握し、統合パイプラインを設計するための資料

---

## 1. 全体構成図

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLI Entry Points                              │
├─────────────────────────────────────────────────────────────────────────┤
│  esde-engine-v532.py          │ Phase 7A: テキスト→Unknown Queue       │
│  resolve_unknown_queue_*.py   │ Phase 7B+: Unknown Queue解決           │
│  esde_cli_live.py             │ Phase 8-9: 統合CLI（observe/monitor）   │
│  stats_cli.py                 │ Phase 9: 統計パイプラインCLI            │
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
│  statistics/      │ Phase 9-1〜4: W1-W4統計計算                        │
│  discovery/       │ Phase 9-5〜6: W5-W6構造発見                        │
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

## 11. statistics/（Phase 9-1〜9-4: W1-W4 Statistics）

**Phase**: 9-1（W1）, 9-2（W2）, 9-3（W3）, 9-4（W4）  
**役割**: 条件付き統計計算、S-Score、共鳴ベクトル

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

**S-Score計算:**
```
S(token|condition) = log(P(token|condition) / P(token|global))
正: その条件で特徴的
負: その条件で希少
```

---

## 12. discovery/（Phase 9-5〜9-6: W5-W6 Discovery）

**Phase**: 9-5（W5）, 9-6（W6）  
**役割**: 島構造の形成、観測出力

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

| ファイル | Phase | 役割 |
|----------|-------|------|
| `unknown_queue.jsonl` | 7A | 未知トークンキュー |
| `unknown_queue_7bplus.jsonl` | 7B+ | 集約済みキュー |
| `unknown_queue_state_7bplus.json` | 7B+ | 集約状態 |
| `evidence_ledger_7bplus.jsonl` | 7B+ | 解決監査証跡 |
| `audit_log_7c.jsonl` | 7C | 構造監査ログ |
| `audit_votes_7cprime.jsonl` | 7C' | LLM三重監査投票 |
| `semantic_ledger.jsonl` | 8-6 | 意味記憶（ハッシュチェーン） |
| `patch_*.jsonl` | 7B+ | パッチ出力（人間レビュー用） |
| `stats/w1_global.json` | 9-1 | グローバル統計 |
| `stats/w2_records.jsonl` | 9-2 | 条件付き統計 |
| `stats/w3_candidates/` | 9-3 | 軸候補 |
| `stats/w4_projections/` | 9-4 | 共鳴ベクトル |

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

### B. Phase 9 フロー（軸発見）
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
| Phase 8 → Phase 9 | 独立 | Molecule → W1/W2入力として接続 |
| Phase 7 → Phase 8 | 独立 | 解決済みトークン → Synapse追加 |
| Substrate → W2 | Migration済 | Policy経由で接続済み |

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
