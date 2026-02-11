# ESDE Module Reference（統合ツール開発用）

**Version**: 5.7.0  
**Updated**: 2026-02-11  
**Note**: Synapse Expansion Phase 1-3 完了 + 実走 v3.2 まで完了 + **Lexicon v2 Pipeline 完成 + Constitution v1.0 確定**。Phase 9 v2.0 パイプライン + Observation C + Cell Architecture v2.2

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
│  run_relations.py             │ Obs C: Relation Pipeline CLI           │
│  synapse/cli.py               │ Synapse Exp: propose / evaluate CLI     │
│  lexicon_wn/wn_*.py           │ Lexicon v2: WordNet語彙供給パイプライン  │
│  [cell_integrator.py]         │ Phase 10: Cell統合（未実装）             │
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
│  integration/relations/ │ Obs C: SVO抽出 → Atom接地（Phase 8↔9橋渡し）│
│  harvester/       │ Obs C 基盤: Wikipedia fetch + ローカルキャッシュ    │
│  statistics/      │ Phase 9 (legacy): W1-W4統計計算                    │
│  statistics/pipeline/ │ Phase 9 (v2.0): Lens統合パイプライン ★現行    │
│  discovery/       │ Phase 9 (legacy): W5-W6構造発見                    │
│  cell/            │ Phase 10: Cell統合（Molecule+Island結合）未実装    │
│  synapse/         │ Synapse Expansion: Store + Proposer + CLI（Phase 1-3）│
│  patches/         │ Synapse パッチファイル格納                          │
│  lexicon_wn/      │ Lexicon v2: WordNet語彙供給 + Core/Dev分離 + Constitution│
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

## 10b. integration/relations/（Observation C: Relation Pipeline）

**Phase**: Observation C（Phase 8 ↔ Phase 9 橋渡し）  
**役割**: テキストから SVO トリプルを抽出し、動詞を Synapse 経由で Atom に接地

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `__init__.py` | — | パッケージ定義 | — |
| `parser_adapter.py` | `ParserAdapter` | spaCy 依存構造解析 → SVO 抽出 | text → `ExtractionResult` |
| `parser_adapter.py` | `SVOTriple` | SVO データ構造 | — |
| `parser_adapter.py` | `ExtractionResult` | 抽出結果コンテナ | — |
| `relation_logger.py` | `SynapseGrounder` | WordNet VERB → Synapse → Atom 候補 | verb_lemma → candidates[] |
| `relation_logger.py` | `RelationLogger` | SVO + Grounding → Edge JSONL | triples → edges[] |
| `relation_logger.py` | `aggregate_entity_graph()` | Edge → エンティティグラフ集約 | edges → entity_graph.json |
| `relation_logger.py` | `aggregate_section_profile()` | Edge → セクション別プロファイル | edges → section_profile.json |
| `run_relations.py` | `main()` | CLI ランナー + 診断レポート生成 | dataset → diagnostic_report |

### 処理フロー

```
Text (Wikipedia sections)
  │
  ▼
ParserAdapter.extract()
  │ spaCy dep parse → SVO triples
  │ 受動態検出、否定検出、接続詞展開
  ▼
RelationLogger.process_section()
  │
  ├─ Light Verb? → UNGROUNDED_LIGHTVERB (edge preserved)
  │
  └─ SynapseGrounder.ground_verb()
       │ verb_lemma → WordNet synsets (POS=VERB)
       │ → Synapse lookup → raw candidates
       │ → POS Guard (block NAT/MAT/PRP/SPA)
       │ → Score Threshold (min_score=0.45)
       ▼
     _build_edge() → edge dict
  │
  ▼
Aggregation
  ├─ aggregate_entity_graph() → UI facing
  └─ aggregate_section_profile() → Phase 9 Lens input
```

### Grounding Filters (v0.2.0)

| 定数 | 値 | 役割 |
|------|---|------|
| `LIGHT_VERB_STOPLIST` | 13 verbs (have, make, do, ...) | 軽動詞の Atom 付与抑制 |
| `POS_GUARD_BLOCKED_CATEGORIES` | {NAT, MAT, PRP, SPA} | 動詞に不適切な Atom カテゴリ除外 |
| `DEFAULT_MIN_SCORE` | 0.45 | 最低スコア閾値（CLI --min-score で可変） |

### CLI

```bash
python -m integration.relations.run_relations --dataset mixed --synapse esde_synapses_v3.json [--min-score 0.45] [--synapse-patches patches/synapse_v3.1.json patches/synapse_v3.2.json]
```

### 出力ファイル

| ファイル | 内容 |
|----------|------|
| `{article}_edges.jsonl` | 生エッジ（1行1トリプル） |
| `{article}_graph.json` | エンティティグラフ（ノード + 集約エッジ） |
| `diagnostic_report.json` | 機械可読診断レポート |
| `diagnostic_report.md` | 人間可読診断レポート |

---

## 10c. harvester/（データ収集・キャッシュ）

**Phase**: Observation C 基盤  
**役割**: Wikipedia 記事のフェッチとローカルキャッシュ

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `__init__.py` | — | パッケージ定義 | — |
| `fetcher.py` | `WikipediaFetcher` | Wikipedia API アクセス | title → raw_response |
| `distiller.py` | `TextDistiller` | Raw → テキスト + 構造統計 | raw → sections[] |
| `storage.py` | `ArtifactStorage` | Artifact + dataset ファイル管理 | — |
| `cli.py` | `main()` | CLI（harvest, list） | — |

### データセット

| Dataset | Articles | Domain |
|---------|----------|--------|
| mixed | 15 | 武将5 + 学者5 + 都市5 |
| warlords | 10 | 武将10 |

### 設計原則

- "Fetch once, analyze many times"
- 2層保存: Artifact (raw) + Distilled (text)
- Substrate-compatible traces

---

## 10d. tests/test_relations.py

| テスト | 内容 | 依存 |
|--------|------|------|
| Test 1: parser_adapter | SVO 抽出、受動態、否定 | spaCy |
| Test 2: batch_extraction | 複数セクション一括処理 | spaCy |
| Test 3: logger_raw | Synapse なしモード | — |
| Test 4: logger_grounded | Synapse ありモード + フィルタログ | --synapse |
| Test 5: aggregation | entity_graph + section_profile | — |
| Test 6: jsonl_output | JSONL 書き込み/読み戻し | — |
| Test 7: full_pipeline | End-to-end | --synapse |

---

## 10e. synapse/（Synapse Expansion Data Layer）★v5.6.0 新設, v5.6.1 更新

**Phase**: Synapse Expansion（Phase 1-3 完了 + 実走 v3.2 まで完了）  
**役割**: Synapse データの単一ソース。Base JSON + Overlay patch を統合管理。候補 Edge 自動生成・評価 CLI を含む  
**Design Spec**: Synapse Expansion via Phase 7, v2.1（Gemini 設計 → GPT 監査 §1-§5）

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `__init__.py` | - | パッケージ定義（SynapseStore, SynapsePatchEntry export） | - |
| `schema.py` | `SynapsePatchEntry` | パッチエントリのデータモデル（edge_key 付き） | - |
| `store.py` | `SynapseStore` | Overlay 統合ストア（tombstone, conflict log） | Base JSON + patches → 解決済み辞書 |
| `proposer.py` | `SynapseEdgeProposer` | Coverage Gap 動詞 → 候補 Edge 生成（4-Pack Rewrite） | lemma → patch_candidates |
| `proposer.py` | `RewritePack` | 4-Pack 中間状態（synset_id, definition, embedding, scored_atoms） | - |
| `diagnostic.py` | `DiagnosticResult` | 診断レポートの型付きラッパー（diff + Audit Gate） | diagnostic_json → structured access |
| `cli.py` | `cmd_propose_synapse()` | CLI Command A: 診断 → 提案 → ベースライン保存 | dataset → run_dir |
| `cli.py` | `cmd_evaluate_synapse_patch()` | CLI Command B: パッチ評価 → Audit Gate 判定 | run_dir + patch → PASS/WARN/FAIL |

**Overlay ルール（Design Spec v2.1 §2）:**

| ルール | 動作 |
|--------|------|
| 適用順序 | Base JSON → Patch v3.1 → Patch v3.2 ... |
| edge_key | `{synset_id}::{atom_id}` で一意識別 |
| disable_edge | 常勝（tombstone: re-add 不可） |
| add_edge 重複 | 後勝ち（スコア・メタデータ更新） |
| 衝突ログ | `[OVERLAY_CONFLICT]` で DEBUG 出力 |

**GO 条件（3消費者の統合）:**

| 消費者 | 現行ファイル | 移行方式 |
|--------|------------|----------|
| Phase 8 Sensor | `sensor/loader_synapse.py` | SynapseStore にデリゲーション |
| Observation C | `integration/relations/relation_logger.py` | `SynapseGrounder.from_store()` |
| Phase 7 Engine | `esde_engine/loaders.py` | SynapseStore にデリゲーション |

**テスト:** `tests/test_synapse_store.py`（10テスト、監査チェックリスト全通過）

## 10f. patches/（Synapse パッチファイル格納）★v5.6.0 新設

**Phase**: Synapse Expansion  
**役割**: SynapseStore が読み込むパッチファイルの格納先

| パッチ | 状態 | 内容 |
|--------|------|------|
| `synapse_v3.1.json` | ✓ 適用済 | 42 edges, 7 gap 解消 (write, serve, join, hold, attack, kill, occupy). Rate 55.2%→61.0% |
| `synapse_v3.2.json` | ✓ 適用済 | 27 edges, 4 gap 解消 (cross, found, introduce, represent). Rate 61.0%→63.0% |

**フォーマット:** JSON（`{"patches": [...]}`）または JSONL（1行1エントリ）を自動判別

### テスト

| ファイル | テスト数 | 内容 |
|----------|---------|------|
| `tests/test_synapse_store.py` | 10 | SynapseStore Overlay ルール、tombstone、衝突解決 |
| `tests/test_phase3_cli.py` | — | CLI propose/evaluate の統合テスト |

---

## 10g. lexicon_wn/（Lexicon v2: WordNet-Based Vocabulary Supply）★v5.7.0 新設

**Phase**: Lexicon v2  
**役割**: 326 Atom の語彙を WordNet から自動供給し、Core/Deviation に分離して統計監査する

### Pipeline 概要

```
esde_dictionary.json (326 atoms 定義)
        │
        ▼
┌─────────────────┐
│ wn_auto_seed.py  │  Step 1: 各 atom の WordNet seed synset を自動生成
└────────┬────────┘
         │ seeds.json
         ▼
┌─────────────────────┐
│ wn_batch_expand.py   │  Step 2: 全 atom を WordNet 展開 (12 relations)
└────────┬────────────┘
         │ expanded/*.json (326 files)
         ▼
┌─────────────────────┐
│ wn_lexicon_entry.py  │  Step 3: Core/Deviation 分離 → Lexicon Entry 生成
└────────┬────────────┘
         │ lexicon/*.json (326 files) + _summary.json
         ▼
┌────────────────────────────┐
│ wn_cross_stats.py          │  Step 4a: 全体統計 (full expansion)
│ wn_core_stats.py           │  Step 4b: Core-only 統計 (Mapper's world)
└────────┬───────────────────┘
         │ report.csv / core_report.csv
         ▼
┌─────────────────────┐
│ wn_proposal_gen.py   │  Step 5: Constitution v1.0 に基づく Proposal 自動生成
└────────┬────────────┘
         │ proposals.json
         ▼
    Taka 審査 → 承認/棄却
```

### ファイル一覧

| ファイル | クラス/関数 | 役割 | 入力→出力 |
|----------|------------|------|-----------|
| `wn_auto_seed.py` | — | Seed 自動生成 | esde_dictionary.json → seeds.json |
| `wn_batch_expand.py` | — | 326 atom 一括 WordNet 展開 | seeds.json → expanded/*.json |
| `wn_lexicon_entry.py` | — | Core/Deviation 分離 | expanded/*.json → lexicon/*.json |
| `wn_cross_stats.py` | — | 全体統計（10カラム GPT レポート） | expanded/*.json → report.csv |
| `wn_core_stats.py` | — | Core-only 統計 | lexicon/*.json → core_report.csv |
| `wn_proposal_gen.py` | — | Proposal 自動生成（Constitution v1.0） | core_report.csv → proposals.json |
| `wn_max_expand.py` | — | 単一 atom 詳細展開（デバッグ用） | atom_id → 詳細 JSON |
| `wn_lexicon.py` | — | 単一 atom パイプライン（レガシー） | — |

### 展開ステップ（12 relations）

| Step | WordNet Relation | Pool | 説明 |
|------|-----------------|------|------|
| 0_seed | Seed lemmas | Core | 定義の核 |
| 2_hypernym_d1 | 上位語 depth=1 | Deviation | 汎用的すぎる |
| 3_hyponym_d1 | 下位語 depth=1 | Core | 直接の具体化 |
| 4_hyponym_d2 | 下位語 depth=2 | Deviation | 深すぎる |
| 5_hyponym_d3 | 下位語 depth=3 | Deviation | さらに深い |
| 6_derivational | 派生形 | Core | 品詞違い同概念 |
| 7_similar_to | 類語（adj） | Core | 同義語圏 |
| 8_also_see | 関連語 | Deviation | 弱いリンク |
| 9_antonym | 対義語（seed のみ） | Core | 対称ペア境界 |
| 10_sibling | 同親語 | Deviation | **主要汚染源＆情報源** |
| 11_pertainym | 関連形 | Deviation | 散発的 |
| 12_verb_group | 動詞群 | Deviation | 散発的 |

### 統計レポート カラム定義（GPT 設計 10 カラム）

| カラム | 意味 | 健全条件 |
|--------|------|----------|
| total_keys / core_count | 語数 | > 0 |
| unique_ratio_pct | その atom 固有の語の割合 | > 5% |
| mean_atoms_per_word (APW) | 1語が平均何 atom に出現 | < 8 |
| pos_n/v/adj_pct | 品詞分布 | バランス |
| generic_at_5/10/20pct | N% 以上の atom に出る語数 | 少ないほど良い |
| top1_jaccard | 最も重なる atom との Jaccard | < 0.4 |
| sym_overlap_keys | 対称ペアとの共有語数 | 少ないほど良い |

### Constitution v1.0 処理ルール

| Pattern | 条件 | 処置 | 該当数 |
|---------|------|------|--------|
| 🔴 A_MERGE | J≥0.75, 同カテゴリ, サイズ近似 | alias 化 + 多核 Core | 3 |
| 🟠 D_SUBSUME | J≥0.60, 同カテゴリ, 非対称 | parent/child 階層 | 1 |
| 🔵 B_COUPLE | J≥0.50, 異カテゴリ | Phase 9 バイパス | 6 |
| ⚪ MONITOR | J 0.40-0.50 | ログのみ | 7 |

全 proposal は `auto_status: flagged`。Taka 承認必須（「記述せよ、決定するな」）。

### 3AI 役割分担

| AI | 担当 | 成果物 |
|----|------|--------|
| **Claude** | アーキテクチャ・実装 | Pipeline スクリプト群、Core/Dev 分離ロジック、Proposal 生成 |
| **Gemini** | 統計・運用リアリティ | 10 カラムレポート設計、Constitution 閾値設定、カテゴリ別分析 |
| **GPT** | 監査・ガバナンス | Constitution v1.0 最終稿、双方向 Jaccard ルール、処理優先順位 |

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

## 14. cell/（Phase 10: Cell Architecture）★設計段階

**Phase**: 10（Phase 8 + Phase 9 統合）  
**役割**: 条件因子による Molecule と Island の結合、階層構造形成  
**Status**: アーキテクチャ設計完了、実装未着手

### 14.1 Cell アーキテクチャ概念

**核心原理:**
- Phase 8（強い意味系）と Phase 9（弱い意味系）は **別々の系** であり、混ぜてはならない
- **条件因子（Condition Factor）** が「引力」として機能し、両者を結合する
- 物理学的アナロジー：原子核（Molecule）+ 電子（Island）→ 原子（Cell）

**光学的アナロジー（Lens系統）:**
```
顕微鏡の対物レンズ:
  4x → 組織全体の構造
  40x → 個々の細胞  
  100x → 細胞内小器官

ESDE Lens:
  Structure → Wikipedia テンプレート構造（Hub/Narrative/Institutional）
  Semantic → 主題の意味的類似性（戦争/哲学/都市）
  Hybrid → セクション内の意味的偏り（書き方の個性）
  
  k パラメータ = 焦点距離:
    k 大（広角）→ 大域的構造
    k 小（望遠）→ 局所的クラスタ
```

### 14.2 階層構造

| 層 | 定義 | 生成元 | 粒度 | 実装状況 |
|----|------|--------|------|----------|
| **Atom** | 326個の最小意味単位（163対称ペア） | Foundation Layer | 固定 | ✓ 完了 |
| **Molecule** | セグメント単位の意味構造（Atom + Formula） | Phase 8 | セグメント | ✓ 完了 |
| **Island** | 書き方が統計的に類似したセクション群のクラスタ | Phase 9 (W5) | セクション | ✓ 完了 |
| **Cell** | 条件因子で結合された Molecule + Island | Phase 10 | 可変 | ⚠ 未実装 |
| **Organ** | 上位条件因子でグループ化された Cell 群 | Phase 10 | 可変 | ⚠ 未実装 |
| **Ecosystem** | 全体の意味構造 + LLM による言語化 | Phase 10 | 全体 | ⚠ 未実装 |

### 14.3 Cell スキーマ（v2.0設計案）

```python
@dataclass
class Cell:
    """
    条件因子で結合された Molecule + Island。
    
    Phase 8 と Phase 9 の観測結果を並列で保持。
    両者は混ぜない。
    """
    
    # Identity
    cell_id: str
    
    # 結合に使用した条件因子
    binding_factor: Dict[str, Any]
    # e.g., {"section_name": "early_life", "article_id": "cao_cao"}
    
    # Phase 8 からの観測（強い意味）
    molecules: List[Molecule]
    
    # Phase 9 からの観測（弱い意味）
    z_score_profile: Optional[Dict[str, float]]  # 20次元 z-score
    island_membership: Optional[str]  # 所属 island_id（noise なら None）
    cohesion_score: Optional[float]   # 島内結束度
    
    # Lens 情報（どのレンズで観測したか）
    lens_name: str   # "structure" / "semantic" / "hybrid"
    k_used: int      # Mutual-kNN の k（焦点距離）
    
    # メタデータ
    source_segment: Optional[str]
    created_at: str
```

### 14.4 条件因子の定義

Phase 9 v2.0 で確定した条件因子は **テキストの内部構造** から抽出される:

| ConditionProvider | 抽出条件 | 例 | 使用Lens |
|-------------------|----------|----|---------| 
| `SectionConditionProvider` | セクション名 | "cao_cao__early_life" | Structure, Hybrid |
| `DocumentConditionProvider` | ドキュメント名 | "cao_cao" | Semantic |
| `PassiveConditionProvider` | 受動態の有無 | "passive_1" / "passive_0" | Hybrid |
| `QuoteConditionProvider` | 引用文内の有無 | "quote_1" / "quote_0" | Structure |
| `ParenthesesConditionProvider` | 括弧内の有無 | "paren_1" / "paren_0" | Structure |
| `ProperNounConditionProvider` | 固有名詞の有無 | "propernoun_1" / "propernoun_0" | Semantic |

### 14.5 統合処理フロー（設計段階）

```
Phase 8 Output: Molecule群 + segment_id（動的条件因子）
     │
     │ 条件因子で引く
     ▼
┌────────────────────────────────────────┐
│  Cell Integrator（未実装）               │
│                                        │
│  binding_factor = {"section_name":     │
│    "early_life", "article_id":         │
│    "cao_cao"}                          │
│                                        │
│  molecules = [Phase 8 Molecule群]      │
│  z_score_profile = [Phase 9 Profile]   │
│  island_membership = "island_42"       │
│                                        │
└────────────────────────────────────────┘
     │ 上位条件因子でグルーピング
     ▼
Phase 9 Output: Island群 + セクション別 z-score profile
```

### 14.6 実装されている統合ポイント

| 接続 | 現状 | 実装ファイル |
|------|------|-------------|
| Obs C → Phase 9 | ✓ 接続済 | `relation_logger.py: aggregate_section_profile()` |
| Obs C → Phase 8 | ✓ 共有Synapse | `SynapseGrounder` が同じ `esde_synapses_v3.json` を使用 |
| Harvester → Phase 9 | ✓ 接続済 | `run_full_pipeline.py: load_dataset()` |
| Harvester → Obs C | ✓ 接続済 | `run_relations.py` がHarvesterキャッシュを読込 |
| Phase 9 → Substrate | ✓ 接続済 | `run_full_pipeline.py` による trace 記録 |

### 14.7 未実装の統合作業

| 統合 | 必要な作業 | 優先度 |
|------|-----------|--------|
| Phase 8 → Cell | Molecule の条件因子抽出 + Cell 形成ロジック | 高 |
| Phase 9 → Cell | Island + z-score profile → Cell 統合 | 高 |
| Cell → Organ | 上位条件因子によるグルーピング | 中 |
| Organ → Ecosystem | LLM による自然言語レポート生成 | 低 |

---

## 15. データファイル（data/）

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

### Observation C
| ファイル | 役割 |
|----------|------|
| `output/relations/{dataset}/{article}_edges.jsonl` | SVO トリプル→Atom接地エッジ |
| `output/relations/{dataset}/{article}_graph.json` | エンティティグラフ（UI用） |
| `output/relations/{dataset}/section_relation_profile.json` | Phase 9 Lens入力用プロファイル |
| `output/relations/{dataset}/diagnostic_report.json` | 機械可読診断レポート |
| `output/relations/{dataset}/diagnostic_report.md` | 人間可読診断レポート |

### Synapse Expansion
| ファイル | 役割 |
|----------|------|
| `patches/synapse_v3.1.json` | v3.1 パッチ（42 edges, 7 gap 解消） |
| `patches/synapse_v3.2.json` | v3.2 パッチ（27 edges, 4 gap 解消） |
| `proposals/{run_id}/diagnostic_before.json` | 提案時のベースライン診断（env_meta 含む） |
| `proposals/{run_id}/patch_candidate.json` | 自動生成された候補 Edge 一覧 |
| `proposals/{run_id}/proposal_report.md` | 人間可読提案レポート |
| `proposals/{run_id}/diagnostic_after.json` | 評価後の再診断結果 |
| `proposals/{run_id}/diagnostic_diff.json` | Before/After 差分（機械可読） |
| `proposals/{run_id}/diagnostic_diff.md` | Before/After 差分（人間可読） |

### Harvester
| ファイル | 役割 |
|----------|------|
| `data/artifacts/{article_id}.json` | Wikipedia生レスポンス（Layer A保存） |
| `data/datasets/{dataset}/{article_id}.txt` | 抽出済みテキスト（Layer B保存） |
| `data/datasets/{dataset}/manifest.json` | データセットメタデータ |

### Substrate
| ファイル | 役割 |
|----------|------|
| `data/substrate/context_records.jsonl` | 条件因子トレース（append-only） |

### Lexicon v2
| ファイル | 役割 |
|----------|------|
| `lexicon_wn/esde_dictionary.json` | 326 Atom 定義（Seed 自動生成の入力） |
| `lexicon_wn/seeds.json` | 各 Atom の WordNet seed synset 定義 |
| `lexicon_wn/expanded/*.json` | WordNet 展開結果（326 files） |
| `lexicon_wn/lexicon/*.json` | Core/Deviation 分離済み Lexicon Entry（326 files） |
| `lexicon_wn/lexicon/_summary.json` | 全 Lexicon Entry のサマリー |
| `lexicon_wn/report.csv` | 全体統計レポート（10 カラム） |
| `lexicon_wn/core_report.csv` | Core-only 統計レポート |
| `lexicon_wn/proposals.json` | Constitution v1.0 Proposal（17 件、Taka 承認済み） |

---

## 16. 統合処理フロー

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
  → Harvester: load_dataset() → ArticleRecord群
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

### C. Phase 7 フロー（未知解決）
```
unknown_queue.jsonl
  → esde_engine/resolver/aggregate_state.py (集約)
  → esde_engine/resolver/online.py (外部検索)
  → esde_engine/resolver/hypothesis.py (仮説評価)
  → esde_engine/resolver/ledger.py (監査証跡)
  → patch_*.jsonl (人間レビュー待ち)
```

### D. Observation C フロー（Relation Pipeline）
```
Harvester: {dataset} キャッシュ → Wikipedia sections
  → integration/relations/parser_adapter.py (spaCy SVO抽出)
  → integration/relations/relation_logger.py (Synapse Grounding)
    ├─ SynapseStore.load(base + patches)  ← v5.6.0: overlay 対応
    ├─ Light Verb → UNGROUNDED_LIGHTVERB
    └─ Normal Verb → WordNet → Synapse → POS Guard → Score Filter
  → {article}_edges.jsonl (生エッジ)
  → aggregate_entity_graph() → entity_graph.json
  → aggregate_section_profile() → section_profile.json (Phase 9 Lens input)
  → diagnostic_report.json / .md
```

### E. Harvester フロー（データ収集）
```
CLI: harvest --dataset {mixed|warlords}
  → harvester/fetcher.py (Wikipedia API)
  → harvester/distiller.py (raw → text + structure stats)
  → harvester/storage.py
    ├─ save_artifact() → data/artifacts/{article}.json (Layer A)
    └─ save_distilled() → data/datasets/{dataset}/ (Layer B)
  → save_manifest() → manifest.json (Substrate traces)
```

### F. Synapse Expansion フロー（Synapse 拡張）★v5.6.0-v5.6.1
```
[1] propose-synapse
    CLI: propose-synapse --dataset mixed --synapse esde_synapses_v3.json
         [--synapse-patches patches/synapse_v3.1.json]
      → run_relations.py (診断) → diagnostic_before.json
      → SynapseEdgeProposer
        ├─ coverage_gap 動詞抽出
        ├─ lemma → WordNet synsets
        ├─ synset → definition → embedding
        └─ embedding × 326 Atoms → scored candidates
      → patch_candidate.json + proposal_report.md
      → Run Directory に隔離保存

[2] 人間レビュー
    patch_candidate.json → score ≥ 0.55 でフィルタ → patch_candidate_055.json

[3] evaluate-synapse-patch
    CLI: evaluate-synapse-patch --run-dir {run_dir} --patch-file {patch}
         [--synapse-patches auto-inherit from diagnostic_before.json]
      → SynapseStore.load(base + baseline_patches + candidate_patch)
      → run_relations.py (再診断) → diagnostic_after.json
      → DiagnosticResult.diff(before, after)
      → Audit Gate: PASS / WARN / FAIL
      → diagnostic_diff.json + diagnostic_diff.md

[4] パッチ承認（PASS 時のみ）
    cp patch_candidate_055.json patches/synapse_v3.X.json
```

### G. Lexicon v2 フロー（WordNet 語彙供給）★v5.7.0
```
[1] Seed 生成
    python lexicon_wn/wn_auto_seed.py
      → esde_dictionary.json (326 atoms) → seeds.json

[2] WordNet 一括展開
    python lexicon_wn/wn_batch_expand.py
      → seeds.json → expanded/*.json (326 files, 12 relations)

[3] Core/Deviation 分離
    python lexicon_wn/wn_lexicon_entry.py
      → expanded/*.json → lexicon/*.json + _summary.json

[4a] 全体統計
    python lexicon_wn/wn_cross_stats.py
      → expanded/*.json → report.csv

[4b] Core-only 統計
    python lexicon_wn/wn_core_stats.py
      → lexicon/*.json → core_report.csv

[5] Proposal 生成
    python lexicon_wn/wn_proposal_gen.py
      → core_report.csv → proposals.json (Constitution v1.0)

[6] Taka 審査 → 承認/棄却
```

### H. Cell統合フロー（Phase 10: 未実装）
```
Phase 8 Output: Molecule群 + segment_id（条件因子）
     │
     ├─ binding_factor で引き合う
     │
Phase 9 Output: Island群 + z-score_profile
     │
     ▼
┌──────────────────────────────────┐
│  Cell Integrator（未実装）         │
│  ・条件因子で Molecule + Island   │
│  ・Lens情報 + k値を保持          │
│  ・混ぜない（並列保持）           │
└──────────────────────────────────┘
     │ 上位条件因子でグルーピング
     ▼
┌──────────────────────────────────┐
│  Organ Formation（未実装）        │
│  ・同一article_id の Cell群       │
│  ・記事レベルの統合               │
└──────────────────────────────────┘
     │ LLM統合
     ▼
┌──────────────────────────────────┐
│  Ecosystem Generation（未実装）   │
│  ・自然言語レポート生成           │
│  ・材料外の推測をしない           │
└──────────────────────────────────┘
```

---

## 17. 統合ツール設計のポイント

### 必要な統合ポイント

| 接続 | 現状 | 必要な作業 |
|------|------|-----------|
| **実装済み** |||
| Obs C → Phase 9 | ✓ section_profile | aggregate_section_profile() → Phase 9 Lens input として接続済み |
| Obs C → Phase 8 | ✓ Synapse共有 | SynapseGrounder が Phase 8 Sensor と同じ Synapse v3.0 を使用 |
| Obs C → Synapse Exp | ✓ 診断 → 提案 | Diagnostic Report の coverage gap が SynapseEdgeProposer への入力 |
| Synapse Exp → Obs C | ✓ パッチ適用 | SynapseStore overlay → run_relations --synapse-patches で評価 |
| Harvester → Phase 9 | ✓ データ供給 | run_full_pipeline.py が Harvester キャッシュを自動ロード |
| Harvester → Obs C | ✓ データ供給 | run_relations.py が Harvester キャッシュを読込 |
| Phase 9 → Substrate | ✓ trace記録 | threshold_trace, edge_policy_trace, chaining_metrics を記録 |
| **未実装（Phase 10）** |||
| Phase 8 → Cell | 独立 | Molecule + segment_id → 条件因子抽出 → Cell 形成ロジック |
| Phase 9 → Cell | 独立 | Island + z-score profile + lens + k → Cell 統合 |
| Cell → Organ | 設計段階 | 上位条件因子（article_id等）による Cell グルーピング |
| Organ → Ecosystem | 設計段階 | LLM による自然言語レポート生成（材料外推測禁止） |
| **Lexicon v2（v5.7.0 新設）** |||
| Lexicon v2 → Mapper | 設計段階 | Core Pool の語を 48 slot に配置（LLM Mapper 未実装） |
| Lexicon v2 → Phase 7 | 設計段階 | Deviation Pool を Unknown fuel として活用 |
| Lexicon v2 → Phase 9 | 設計段階 | Couple データ（Pattern B/C）を Phase 9 にバイパス |
| Lexicon v2 → Atom 再編 | 設計段階 | Constitution Merge/Subsume に基づく Atom ID 再編 |
| **移行中** |||
| Phase 7 → Phase 8 | 独立 | 解決済みトークン → Synapse 追加（手動パッチ適用） |
| Phase 9 legacy → v2.0 | 共存 | 旧パイプライン残存、将来的に整理の可能性 |

### Cell Architecture v2.0 の設計洞察

**物理学的アナロジー:**
- Phase 8 = 原子核（強い力で結合した Atom/Molecule）
- Phase 9 = 電子（統計的法則に従う Island）  
- 条件因子 = 電磁力（異なる系を引き合わせる）

**光学的アナロジー（Lens系統）:**
- k パラメータ = 焦点距離（k 小=望遠、k 大=広角）
- 相転移点 k=4（分解状態 ↔ 連鎖状態の臨界点）
- z-score = 偏差検出（「何を知っているか」→「何が偏っているか」）

### 統合CLIの候補機能

```bash
# 全フロー実行（将来的な Phase 10 統合CLI）
esde run --input articles/ --output results/ --lens hybrid

# Phase別実行
esde phase8 observe "I love you"
esde phase9 analyze --dataset mixed --lens hybrid --knn-k auto
esde phase7 resolve --limit 50

# データ収集
esde harvest --dataset mixed --force
esde relations --dataset mixed --synapse esde_synapses_v3.json

# モニタリング
esde monitor --live
esde status
esde cell-status  # Cell統合状況（Phase 10）
```

### 現在利用可能なCLI

```bash
# データ収集（Harvester）
python -m harvester.cli harvest --dataset mixed
python -m harvester.cli list

# Phase 9 分析（Lens統合パイプライン）
python -m statistics.pipeline.run_full_pipeline --dataset mixed --lens hybrid --knn-k auto

# Relation Pipeline（Observation C）
python -m integration.relations.run_relations --dataset mixed --synapse esde_synapses_v3.json

# Relation Pipeline + Synapse patches（Synapse Expansion 適用時）
python -m integration.relations.run_relations --dataset mixed --synapse esde_synapses_v3.json --synapse-patches patches/synapse_v3.1.json patches/synapse_v3.2.json

# Synapse Expansion: 候補 Edge 提案
python -m synapse.cli propose-synapse --dataset mixed --synapse esde_synapses_v3.json [--synapse-patches patches/synapse_v3.1.json]

# Synapse Expansion: パッチ評価（--synapse-patches 省略時は auto-inherit）
python -m synapse.cli evaluate-synapse-patch --run-dir proposals/{run_id} --patch-file {patch} --dataset mixed --synapse esde_synapses_v3.json

# Phase 8 観測（Live Mode）
python -m esde_cli_live.py observe

# Phase 7 解決
python -m esde_engine.resolver.resolve_unknown_queue_7bplus.py

```

---

*「記述せよ、しかし決定するな」*
