# ESDE Cell Architecture

**Version:** 2.3  
**Date:** 2026-02-08  
**Authors:** Taka (Human) + Claude (AI)  
**Status:** Synapse Expansion Phase 1-3 完了 + 実走 v3.2 まで完了  
**Previous:** v2.2 (2026-02-06) — SynapseStore + Overlay 実装完了

---

## 変更履歴

| 版 | 日付 | 概要 |
|----|------|------|
| 0.1 | 2026-01-27 | 初版（RFC、概念設計のみ） |
| 2.0 | 2026-02-02 | Phase 9 実装完了に基づく全面改訂。W層再定義、Lens導入、Threshold 3層化、Mutual-kNN + k-sweep 追加。旧0.1の未解決課題の大半が解決済み。 |
| 2.1 | 2026-02-05 | Observation C（Relation Pipeline）完了を反映。Synapse 動詞接地限界の発見を記録。Phase 7 → Synapse Expansion パスを追加。用語集拡充。 |
| 2.2 | 2026-02-06 | Synapse Expansion Phase 1 実装完了。SynapseStore（Overlay付き統合ストア）導入。GO条件（Phase 8 + Obs C 共有）テスト通過。Design Spec v2.1 準拠。 |
| 2.3 | 2026-02-08 | Synapse Expansion Phase 2-3 完了 + 実走 v3.2 まで完了。SynapseEdgeProposer（4-Pack Rewrite）、CLI（propose/evaluate）、Audit Gate、Baseline Patch Auto-Inherit（GPT §5）実装。v3.1（+5.8pt）・v3.2（+2.0pt）パッチ適用。逓減パターンの発見を記録 |

---

## 1. Executive Summary

本文書は、ESDE Phase 8（強い意味系）と Phase 9（弱い意味系）の統合アーキテクチャ「Cell」を定義する。

**核心的洞察（v0.1から不変）:**
- Phase 8 と Phase 9 は**別々の系**であり、混ぜてはならない
- **条件因子（Condition Factor）**が「引力」として機能し、両者を結合する

**v2.0 での主要変化:**
- 条件因子は外部メタデータ（source_type等）ではなく、**テキスト内部構造**（セクション名・ドキュメント名）から抽出される
- Phase 9 の分析単位は「記事」ではなく**「セクション」**
- **Lens（レンズ）**が導入され、同じデータを異なる観点（構造/意味/混合）で観測可能に
- **Island** は「書き方が統計的に類似したセクション群のクラスタ」
- 閾値・エッジ選択・クラスタリングの全工程が**動的・トレーサブル**

**v2.1 での追加:**
- **Observation C（Relation Pipeline）**が Phase 8 ↔ Phase 9 の橋渡し層として実装完了
- Synapse の**動詞接地限界**が診断的に特定され、3フィルタ（v0.2.0）で対策済み
- **Phase 7 → Synapse Expansion** パスが新しい統合経路として提案された

**v2.2 での追加:**
- **SynapseStore**（`synapse/store.py`）を単一データソースとして導入。Overlay patch 機構を実装
- **GO条件** 充足: Phase 8 Sensor と Observation C が同一 SynapseStore を参照し、patch 効果が両系に同時反映
- Design Spec v2.1（Gemini 設計 + GPT 監査）に基づく Phase 1 実装完了。監査チェックリスト 10/10 通過

**v2.3 での追加:**
- **Synapse Expansion Phase 2-3 完了**: SynapseEdgeProposer（4-Pack Rewrite による候補 Edge 自動生成）+ CLI（propose-synapse / evaluate-synapse-patch）+ Audit Gate（PASS/WARN/FAIL 機械判定）
- **実走 v3.1/v3.2**: v3.1 パッチ（42 edges, +5.8pt, 55.2%→61.0%）、v3.2 パッチ（27 edges, +2.0pt, 61.0%→63.0%）。回帰ゼロ
- **逓減パターン発見**: 高頻度かつ embedding 親和性の高い動詞が先に解消、残存動詞はスコア < 0.55 で自動提案の限界に接近
- **GPT §5 Baseline Patch Auto-Inherit**: evaluate 時に before 側のパッチを自動継承し、比較世界の一致を保証
- **CLI --synapse-patches**: propose/evaluate 両方で既採用パッチの overlay に対応

---

## 2. 物理学的アナロジー

### 2.1 原子構造との対応

```
物理学:
  原子核（陽子・中性子）  ←  強い力で結合
  電子                    ←  別の存在、別の法則
  
  これらは別々だが、電磁力で引き合って「原子」を形成


ESDE:
  Molecule（強い意味）    ←  Phase 8（326 Atoms + Synapse v3.0 + patches）
  Island（弱い意味）      ←  Phase 9（統計的パターン → セクション群のクラスタ）
  
  これらは別々だが、条件因子で引き合って「Cell」を形成
```

### 2.2 光学的アナロジー（v2.0 追加）

Phase 9 に**レンズ**の概念が導入された。同じデータを異なる「焦点」で観測する。

```
顕微鏡:
  対物レンズを変えると、同じ標本から異なる構造が見える
  4x → 組織全体の構造
  40x → 個々の細胞
  100x → 細胞内小器官

ESDE Phase 9:
  Lens を変えると、同じテキストから異なるパターンが見える
  Structure Lens → Wikipedia のテンプレート構造（Hub/Narrative/Institutional）
  Semantic Lens  → 主題の意味的類似性（戦争/哲学/都市）
  Hybrid Lens    → セクション内の意味的偏り（書き方の個性）
```

さらに Mutual-kNN の k パラメータは**レンズの焦点距離**として機能する:

```
k 大（広角） → 大域的テンプレート構造が見える（島が少ない、巨大成分が支配的）
k 小（望遠） → 局所的な主題クラスタが見える（島が多い、高結束の小集団）

k=2 : 望遠 → 108 islands（微細構造、noise 30%）
k=3 : 標準 → 61 islands（中粒度、noise 16%）
k≥4 : 臨界点超過 → 連鎖（gcr > 0.65、巨大成分が全体を飲み込む）
```

### 2.3 設計原則

| 原則 | 説明 |
|------|------|
| **非混合** | Phase 8 と Phase 9 は互いに侵食しない |
| **自然な結合** | 無理に粒度を合わせず、結合可能なものが自然に結合 |
| **引力としての条件因子** | 条件因子がなければ、ただの2つの独立した観測結果 |
| **記述せよ、決定するな** | 全ての判断は trace として記録され、後から検証可能 |
| **不確実性は結果** | 分類できない（noise）は正当な観測結果であり、排除しない |

---

## 3. 階層構造

### 3.1 完全な階層定義

```
Atom（326個）
    ↓ Phase 8: Synapse v3.0 + patches + LLM
Molecule（セグメント単位の意味構造）
    ↓
    │
    │   ←─── 条件因子（引力）───→   Island（統計的クラスタ）
    │                                     ↑
    │                              Phase 9: W1→W2→W3→W4→W5→W6
    │                              （Lens × k-sweep × Threshold）
    ↓
Cell（条件因子で結合された Molecule + Island）
    ↓ 条件因子の階層でグルーピング
Organ（同一上位条件因子の Cell 群）
    ↓ LLM 統合
Ecosystem（全体の意味構造 + 言語化レポート）
```

### 3.2 各層の定義

| 層 | 定義 | 生成元 | 粒度 |
|----|------|--------|------|
| **Atom** | 326個の最小意味単位（163対称ペア） | Foundation Layer | 固定 |
| **Molecule** | セグメント単位の意味構造（Atom + Formula） | Phase 8 | セグメント |
| **Island** | 書き方が統計的に類似した**セクション群**のクラスタ | Phase 9 (W5) | セクション |
| **Cell** | 条件因子で結合された Molecule + Island | 統合層 | 可変 |
| **Organ** | 上位条件因子でグループ化された Cell 群 | 統合層 | 可変 |
| **Ecosystem** | 全体の意味構造 + LLM による言語化 | 出力層 | 全体 |

**v0.1 → v2.0 の変化:** Island の定義が「共鳴ベクトルが類似した**記事群**」から「書き方が統計的に類似した**セクション群**」に変わった。Phase 9 の分析単位がセクション単位に確定したことによる。

---

## 4. Phase 9 パイプライン（実装済み）

### 4.1 W 層の定義（v2.0 確定版）

v0.1 では W0〜W6 が概念的に定義されていたが、実装を通じて以下に確定した:

| W層 | 名称 | 入力 | 出力 | 説明 |
|-----|------|------|------|------|
| **W1** | Feature Extraction | 生テキスト | 20次元トークン特徴 | spaCy による品詞/形態素解析。各トークンに20次元ベクトルを付与 |
| **W2** | Conditional Aggregation | W1 特徴 + 条件因子 | 条件別統計 | ConditionProvider が抽出した条件（セクション名等）ごとに特徴を集約 |
| **W3** | Profile Computation | W2 統計 | z-score プロファイル | 条件間の偏差を z-score で正規化。セクションの「個性」を数値化 |
| **W4** | Similarity Computation | W3 プロファイル | 全ペア類似度行列 | コサイン類似度（z-score ベース）。N 条件から N(N-1)/2 ペアを計算 |
| **W5** | Island Formation | W4 類似度 + Threshold + EdgeFilter | Island 構造 | 閾値フィルタ → エッジ選択 → 連結成分 → Island |
| **W6** | Export | W5 構造 | JSON/MD/CSV | 構造化データ + 人間可読レポート + k-sweep テーブル |

**W0（旧定義: データ正規化）は削除。** Harvester モジュールが担当する前処理であり、W 層の責務ではない。

### 4.2 条件因子（v2.0: 内部構造ベース）

v0.1 では条件因子を `source_type` / `language_profile` / `time_bucket` といった**外部メタデータ**で定義していた。v2.0 では**テキストの内部構造**から動的に抽出される:

```python
# v0.1（旧: 外部メタデータ）
{
    "source_type": "news",        # データ取得時に決定
    "language_profile": "en",     # 環境属性
    "time_bucket": "2026-01",     # 時間属性
}

# v2.0（現: 内部構造）
# SectionConditionProvider → セクション名が条件
"cao_cao__early_life"         # article_id__section_name
"san_francisco__demographics" # article_id__section_name
```

### 4.3 ConditionProvider 一覧

| ConditionProvider | 抽出条件 | 例 | 使用Lens |
|-------------------|----------|----|---------| 
| `SectionConditionProvider` | セクション名 | "cao_cao__early_life" | Structure, Hybrid |
| `DocumentConditionProvider` | ドキュメント名 | "cao_cao" | Semantic |
| `PassiveConditionProvider` | 受動態の有無 | "passive_1" / "passive_0" | Hybrid |
| `QuoteConditionProvider` | 引用文内の有無 | "quote_1" / "quote_0" | Structure |
| `ParenthesesConditionProvider` | 括弧内の有無 | "paren_1" / "paren_0" | Structure |
| `ProperNounConditionProvider` | 固有名詞の有無 | "propernoun_1" / "propernoun_0" | Semantic |

---

## 5. 統合アーキテクチャ（全体図）

### 5.1 パイプライン全体像

```
┌──────────────────────────┐   ┌──────────────────────────────┐
│  Phase 8（強い意味系）      │   │  Phase 9（弱い意味系）         │
│                          │   │                              │
│  1. テキスト入力 →         │   │  Harvester → キャッシュ       │
│     セグメント境界検出      │   │  Lens 選択                    │
│     （LLM）               │   │    ↓                         │
│  2. 原子化（WordNet/326） │   │  W1: 20次元トークン特徴抽出     │
│  3. 分子化（LLM + Synapse）│   │    ↓                         │
│                          │   │  W2: ConditionProvider で集約   │
│  出力: Molecule 群         │   │    ↓                         │
│       + segment_id       │   │  W3: z-score プロファイル       │
│       （動的条件因子）      │   │    ↓                         │
│                          │   │  W4: コサイン類似度行列         │
│                          │   │    ↓                         │
│                          │   │  W5: Threshold + Mutual-kNN    │
│                          │   │      + k-sweep → Island 形成   │
│                          │   │    ↓                         │
│                          │   │  W6: Export (JSON/MD/CSV)      │
│                          │   │                              │
│                          │   │  出力: Island 群               │
│                          │   │       + Chaining Metrics      │
│                          │   │       + Threshold Trace       │
│                          │   │       + k-sweep Table         │
└──────────────────────────┘   └──────────────────────────────┘
             │                               │
             │      互いに侵食しない            │
             │      別々の観測結果              │
             │                               │
             │   ┌─────────────────────┐      │
             │   │  Observation C      │      │
             │   │  (Relation Pipeline)│      │
             │   │                     │      │
             ├──→│  Phase 8 と共有:     │←─────┤
             │   │   SynapseStore      │      │
             │   │   (v3.0 + patches)  │      │
             │   │  Phase 9 へ供給:     │      │
             │   │   section_profile   │      │
             │   └─────────────────────┘      │
             │                               │
             └───────────────┬───────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                 統合層（条件因子による結合）                       │
│                                                              │
│  条件因子「section_name」で引く:                                │
│    Molecule(early_life) + Island(early_lifeが属す島)            │
│    → Cell 形成                                                │
│                                                              │
│  条件因子「article_id」で引く:                                  │
│    Cell群(cao_caoに属する) → Organ                             │
└──────────────────────────────────────────────────────────────┘
                             ↓
┌──────────────────────────────────────────────────────────────┐
│                 出力層（LLM 言語化）                             │
│                                                              │
│  ESDE の観測結果を自然言語レポートに変換                          │
│  ※ 自己流の解釈を加えない（材料外の推測をしない）                  │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 重要な制約

| 制約 | 説明 |
|------|------|
| **Phase 8 → Phase 9 への流入禁止** | Molecule の情報が W 層に影響しない |
| **Phase 9 → Phase 8 への流入禁止** | Island の情報が Molecule 生成に影響しない |
| **統合は条件因子のみで行う** | 結合ロジックに意味解釈を含めない |
| **全判断は trace として記録** | Threshold 決定、k 選択、エッジフィルタの全てが再現・検証可能 |
| **Obs C は橋渡しであり混合ではない** | Phase 8 と Synapse を共有し、Phase 9 に section_profile を供給するが、両系の独立性は保持 |

---

## 6. Cell の構造定義

### 6.1 Cell スキーマ（v2.0 更新案）

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

### 6.2 v0.1 → v2.0 の変化

| 項目 | v0.1 | v2.0 |
|------|------|------|
| binding_factor | `{"segment_id": "seg_0042"}` | `{"section_name": "early_life", "article_id": "cao_cao"}` |
| Phase 9 側 | `resonance_pattern` (dict) + `related_islands` (list) | `z_score_profile` (20-dim) + `island_membership` (single) + `cohesion_score` |
| Lens 情報 | なし | `lens_name` + `k_used` |

**重要な変化:** 1 セクション = 1 Island に帰属（多対多ではなく多対一）。セクションは高々1つの Island に属するか、noise（どこにも属さない）。これにより v0.1 で未解決だった「多対多の関係をどう扱うか」が解消された。

---

## 7. ESDE と LLM の分業

### 7.1 役割分担（v0.1 から不変）

```
┌──────────────────────────────────────────────────────────────┐
│  ESDE（観測層）                                                │
│                                                              │
│  責務:                                                        │
│    ・Phase 8: Molecule 生成（強い意味の構造化）                   │
│    ・Phase 9: Island/Evidence 抽出（弱い意味のパターン）           │
│    ・条件因子による結合（Cell/Organ 形成）                        │
│                                                              │
│  出力: 構造化されたデータ（JSON, スキーマ準拠, 検証可能）           │
│                                                              │
│  哲学: "記述せよ、しかし決定するな"                               │
└──────────────────────────────────────────────────────────────┘
                             ↓
                   解釈の材料を徹底的に提供
                             ↓
┌──────────────────────────────────────────────────────────────┐
│  LLM（言語化層）                                               │
│                                                              │
│  責務:                                                        │
│    ・ESDE の観測結果に基づいてレポート生成                        │
│    ・人間に分かりやすい自然言語で出力                              │
│                                                              │
│  制約:                                                        │
│    ・自己流の解釈を加えない                                      │
│    ・材料外の推測をしない                                        │
│    ・ESDE が提供した情報の範囲内で言語化                          │
│                                                              │
│  出力: 自然言語レポート（柔軟、読みやすさ重視）                    │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 厳密さのグラデーション

| 層 | 厳密さ | 理由 |
|----|--------|------|
| Atom / Molecule | **厳密** | 機械的に検証可能、再現性必須 |
| Island / Threshold / Edge Policy | **厳密** | 統計的根拠、全工程が trace 記録 |
| Cell / Organ | **厳密** | 条件因子による機械的結合、混ぜない |
| 最終レポート | **柔軟** | 人間が読むもの、自然言語の強みを活かす |

---

## 8. Substrate との関係

### 8.1 Substrate の役割

Substrate Layer は Phase 9 パイプラインの**横断的基盤**として機能する:

- **Context Fabric**: 全 trace の append-only 格納
- **決定論的 context_id 生成**: 入力データから一意に ID を計算
- **条件因子の管理**: ConditionProvider が生成した条件の記録

### 8.2 trace として記録されるもの

| trace 種別 | 内容 | 記録タイミング |
|-----------|------|-------------|
| **threshold_trace** | t_abs / t_rel / t_resolved / 分布統計 / abs_source | W5 実行時 |
| **edge_filter_trace** | selector 名 / k 値 / edges_before / edges_after | W5 エッジフィルタ時 |
| **edge_policy_trace** | k_chosen / sweep_summary / selection_reason / policy 制約 | k-sweep 実行時 |
| **chaining_metrics** | gcr / mean_intra / edge_sparsity / chaining_detected | W5 完了後 |
| **global_model** | lens/feature_mode 別の累積類似度分布 | 実行ごとに追記 |

---

## 9. Observation C: Phase 8 ↔ Phase 9 橋渡し（v2.1 追加）

### 9.1 位置づけ

Observation C（Relation Pipeline）は Phase 8 と Phase 9 の間に位置する**橋渡し層**である。テキストから SVO（Subject-Verb-Object）トリプルを決定論的に抽出し、動詞述語を Synapse 経由で Atom に接地する。LLM を使わない。

```
Text (Wikipedia sections)
  │
  ▼
ParserAdapter.extract()           ← spaCy dep parse → SVO triples
  │                                  受動態/否定/接続詞展開を検出
  ▼
RelationLogger.process_section()
  │
  ├─ Light Verb? → UNGROUNDED_LIGHTVERB (edge preserved, atom suppressed)
  │
  └─ SynapseGrounder.ground_verb()
       │ verb_lemma → WordNet synsets (POS=VERB)
       │ → Synapse lookup → raw candidates
       │ → POS Guard (block NAT/MAT/PRP/SPA)
       │ → Score Threshold (min_score=0.45)
       ▼
     edge dict → JSONL
  │
  ▼
Aggregation
  ├─ aggregate_entity_graph()      → entity_graph.json (UI facing)
  └─ aggregate_section_profile()   → section_profile.json (Phase 9 Lens input)
```

### 9.2 Phase 8 / Phase 9 との接続

| 接続 | 方向 | 内容 | 実装 |
|------|------|------|------|
| Obs C → Phase 8 | 共有 | 同じ SynapseStore（v3.0 + patches）を使用 | `SynapseGrounder.from_store()` |
| Obs C → Phase 9 | 供給 | section_profile を Lens input として提供 | `aggregate_section_profile()` |
| Phase 8 → Obs C | なし | Molecule は Obs C に流入しない | ― |
| Phase 9 → Obs C | なし | Island は Obs C に流入しない | ― |

**非混合原則は維持される:** Obs C は Phase 8 と同じ辞書（Synapse）を使い、Phase 9 にデータを供給するが、両系の間で意味情報が流入することはない。

### 9.3 Grounding Filters（v0.2.0）

Synapse の動詞接地で発見された3種の構造的問題に対するフィルタ:

| フィルタ | 対象 | 効果 |
|----------|------|------|
| **Light Verb Stoplist** | 13語（have, make, do, get, ...） | 軽動詞の Atom 付与を抑制。Edge は保持 |
| **POS Guard** | NAT/MAT/PRP/SPA カテゴリ | 動詞に不適切な名詞 Atom を除外 |
| **Score Threshold** | min_score=0.45 | 低スコア候補を UNGROUNDED に |

フィルタ前: grounding rate 89%（品質に問題あり）  
フィルタ後: grounding rate 55%（真のカバレッジを反映）  
v3.1 パッチ後: 61%（+5.8pt）  
v3.2 パッチ後: 63%（+2.0pt）

### 9.4 Synapse の動詞接地限界

Synapse v3.0 は名詞の概念空間に最適化されている。動詞を同じ辞書で引くと構造的ミスマッチが発生する:

**根本原因:** Glossary 定義が概念記述（名詞的）、WordNet 動詞定義がアクション記述（動詞的）であり、埋め込み空間での類似度が閾値を下回る。

```
Glossary:  EXS.death = "Cessation of biological life"      （状態記述）
WordNet:   kill.v.01  = "cause to die; put to death"        （動作記述）

→ 意味的には明らかに対応するが、cosine similarity < 0.3 で Synapse edge なし
```

**重要:** これは Synapse のバグではなく構造的特性。326 Atoms は変更不要。Synapse に動詞 edge を追加すれば解決する。

### 9.5 Phase 7 → Synapse Expansion パス（提案段階）

Observation C の診断結果と Phase 7 Route C のエビデンス蓄積を組み合わせた Synapse 拡張パス:

```
Phase 7 Route C: トークン単位のギャップ検出（遅い蓄積）
Relation Pipeline: コーパス規模のギャップ検出（即時）
     │
     ├─ 同じ出力: (verb, frequency) のランキングリスト
     ▼
候補 Edge 生成（自動: WordNet synset → Atom 類似度計算）
     ↓
エビデンス閾値（N回以上の出現 + 最低スコア）
     ↓
人間レビュー（Taka 承認）
     ↓
Synapse パッチ（append-only、バージョン管理）
```

**アーキテクチャ上の位置づけ:**

| 層 | アナロジー | 変更頻度 | ガバナンス |
|----|---------|----------|-----------|
| 326 Atoms | 周期表の元素 | 不変 | 創設時の設計判断 |
| Synapse edges | 化合物・反応のデータベース | 成長する | Phase 7 エビデンス + 人間レビュー |
| Code | 実験器具 | バージョン管理 | 3AI ワークフロー |

**Status:** Design Spec v2.1 確定（Gemini 設計 → GPT 監査 §1-§5 → Taka 承認）。Phase 1-3 全て実装完了。v3.1/v3.2 パッチ実走済み（Audit Gate PASS）。

**v2.2 実装済み（Phase 1: Data Model & Loader）:**

```
synapse/                        ← v5.6.0 新設、v5.6.1 拡充
├── __init__.py                 # パッケージ定義、SynapseStore / SynapsePatchEntry export
├── schema.py                   # SynapsePatchEntry（edge_key 付きパッチエントリ）
├── store.py                    # SynapseStore（Overlay 付き統合ストア）
├── proposer.py                 # SynapseEdgeProposer（4-Pack Rewrite → 候補 Edge 生成）
├── diagnostic.py               # DiagnosticResult（diff + Audit Gate 判定）
└── cli.py                      # CLI: propose-synapse / evaluate-synapse-patch

patches/                        ← パッチファイル格納
├── synapse_v3.1.json           # 42 edges, 7 gap 解消, Rate +5.8pt
└── synapse_v3.2.json           # 27 edges, 4 gap 解消, Rate +2.0pt
```

**Overlay ルール（Design Spec v2.1 §2）:**
- 適用順序: Base JSON → Patch v3.1 → Patch v3.2 ...
- edge_key = `{synset_id}::{atom_id}` で一意識別
- `disable_edge` が常に勝つ（tombstone: 永久除外、re-add 不可）
- `add_edge` 重複は後勝ち（スコア・メタデータ更新可能）
- 全衝突を `[OVERLAY_CONFLICT]` として DEBUG ログに記録

**GO 条件（GPT 監査ゲート）:**
- SynapseStore を Phase 8 Sensor と Observation C の両方で使用
- patch 効果は molecule 生成系と relation 抽出系に同時反映
- 片系がバイパスした場合はテスト Fail

---

## 10. 実験的発見

Phase 9 実装を通じて得られた知見（設計判断の根拠となるもの）:

### 10.1 Wikipedia のテンプレートトポロジー

Structure lens で発見された3層構造:
- **Hub 層**: 同心円的セクション配置（都市記事: demographics, economy, transport...）
- **Narrative 層**: 時系列セクション配置（戦国武将: early life, campaign, legacy...）
- **Institutional 層**: 制度的セクション配置（組織・法律系）

**含意:** Phase 9 が検出しているのは「ジャンル分類」ではなく「編集パターンの構造的類型」。

### 10.2 相転移点（k=3 → k=4）

Mutual-kNN の k-sweep で発見:
- k ≤ 3: 分解状態（意味のある島が形成される）
- k ≥ 4: 連鎖状態（巨大成分が全体を支配する）
- **臨界点は k=4**（gcr が 0.13 → 0.66 に急増）

**含意:** この臨界点はデータ固有の性質であり、パラメータチューニングとは無関係。「レンズの焦点距離」としての k の解釈を裏付ける。

### 10.3 z-score ベースの類似度

W4 でコサイン類似度の入力を `mean_vector`（生平均）から `z_score_vector`（標準化偏差）に変更:
- **生平均の問題:** 大数の法則により全条件の平均ベクトルが global baseline に収束 → 全ペアの類似度が 1.0 に
- **z-score の効果:** baseline からの**偏差**を比較 → 意味のある分布（-0.82〜0.77）が出現

**含意:** 「何を知っているか」ではなく「何が偏っているか」を比較するのが正しい。

### 10.4 ドメイン別接地特性（v2.1 追加, v2.3 更新）

Observation C の診断で発見されたドメイン別の Synapse 接地パターン（v3.0 + v3.1 + v3.2 時点）:

| ドメイン | 典型的な残存 Gap | 特徴 |
|----------|-----------------|------|
| 武将 (mil) | defeat, host, launch | 軍事動詞の不足。kill, attack は v3.1 で解消 |
| 学者 (sch) | operate, publish(misground) | 知的動詞は比較的良好。publish は misground 側の問題 |
| 都市 (city) | contain, host, employ, comprise | 軽動詞比率が高い（19-35%）。rate の分母が小さい |

**含意:** v3.1/v3.2 で高頻度動詞が優先解消され、残存 gap は embedding スコア < 0.55 の動詞。自動提案の限界に接近しており、今後は手動での定義拡張や Augmented definition が必要になる可能性がある。

### 10.5 Synapse Expansion 逓減パターン（v2.3 追加）

| ラウンド | 解消動詞 | Rate 上昇 | 候補 Edge 数 |
|---------|---------|----------|-------------|
| v3.1（1巡目） | write, serve, join, hold, attack, kill, occupy | +5.8pt | 42 |
| v3.2（2巡目） | cross, found, introduce, represent | +2.0pt | 27 |

高頻度かつ WordNet 定義が Atom と嚙み合う動詞が先に解消し、ラウンドごとに上昇幅が縮小する逓減パターンが観測された。これは4-Pack Rewrite の embedding 比較方式に起因する構造的傾向であり、パイプラインの不具合ではない。

---

## 11. 未解決の課題

### 11.1 解決済み（v0.1 からの移行）

| v0.1 課題 | 解決策 |
|-----------|--------|
| Phase 9 の入力単位（記事 vs セグメント） | **セクション単位に確定**。Lens の ConditionProvider が粒度を決定 |
| Island と Segment の多対多関係 | **多対一に確定**。各セクションは高々1つの Island に帰属、または noise |
| 条件因子の階層（どの階層で結合するか） | **Lens が決定**。Structure/Hybrid = section、Semantic = document |

### 11.2 解決済み（v2.0 → v2.1）

| v2.0 課題 | 解決策 |
|-----------|--------|
| Phase 8 ↔ Phase 9 の橋渡し手段 | **Observation C** が Synapse 共有 + section_profile 供給で実現 |
| Synapse の動詞接地品質 | **3フィルタ**（Light Verb / POS Guard / Score Threshold）で CATEGORY_MISMATCH を完全解消 |
| 動詞カバレッジギャップの特定 | **診断レポート**が Coverage Gap 動詞を頻度付きで自動検出 |

### 11.2b 解決済み（v2.1 → v2.2）

| v2.1 課題 | 解決策 |
|-----------|--------|
| Synapse パッチの適用基盤 | **SynapseStore**（`synapse/store.py`）が Overlay 付き統合ストアを提供 |
| Phase 8 / Obs C のデータ分断 | **GO 条件**として SynapseStore 共有を仕様化。監査テストで強制 |
| パッチの衝突解決 | **Design Spec v2.1 §2**: disable 優先 + Last One Wins + tombstone |
| パッチの監査追跡性 | **edge_key** による一意識別 + `[OVERLAY_CONFLICT]` ログ + audit trail API |

### 11.2c 解決済み（v2.2 → v2.3）

| v2.2 課題 | 解決策 |
|-----------|--------|
| Synapse Expansion Phase 2（Proposer） | **SynapseEdgeProposer**（`synapse/proposer.py`）が 4-Pack Rewrite で候補 Edge を自動生成 |
| Synapse Expansion Phase 3（CLI + 評価） | **synapse/cli.py** が propose-synapse / evaluate-synapse-patch を提供。Audit Gate（PASS/WARN/FAIL）で機械判定 |
| エビデンス閾値 T の値 | **MIN_SCORE=0.28**（広い候補生成）+ **Adoption=0.55**（採用閾値）の2段構え |
| Before/After 比較世界の一致 | **GPT §5 Baseline Patch Auto-Inherit**: evaluate 時に before のパッチを自動継承 |

### 11.3 現存する課題

**Cell 形成の実装:**
- Phase 8 (Molecule) と Phase 9 (Island) を条件因子で結合するコードは未実装
- Cell スキーマは設計段階。Phase 10 以降の課題

**Synapse Expansion の逓減:**
- v3.1（+5.8pt）→ v3.2（+2.0pt）と上昇幅が縮小
- 残存 gap 動詞（host, defeat, contain 等）は embedding スコア < 0.55 で自動提案の限界に接近
- Augmented definition（Gemini 設計質問）や手動定義拡張の検討が必要

**Edge Policy プリセット（profile）:**
- purity / balanced / overview 等の目的別プリセットは未実装
- 複数データセットの蓄積が前提。現状は k-sweep テーブルを人間が見て判断
- Phase 9 の範囲外（使いやすさの改善であり、設計思想の完成には不要）

**Lens 間の統合:**
- 3つの Lens が独立に Island を生成する。Lens 間の関係は未定義
- 同じセクションが Structure lens では island A、Hybrid lens では island B に属する場合の扱い

**global_model の成熟:**
- t_abs（絶対閾値）は累積データに基づく。現状は mixed dataset 1つ分のみ
- 十分なデータが蓄積されるまで fallback に頼る場面が多い

---

## 12. モジュール構成（実装リファレンス）

### 12.1 Phase 9 パイプライン

```
statistics/pipeline/
├── run_full_pipeline.py    # メインパイプライン（CLI）
├── lens.py                 # Lens 定義（Structure/Semantic/Hybrid）
├── condition_provider.py   # ConditionProvider 群（Section/Document/Passive/...）
├── w2_aggregator.py        # W2: 条件別集約
├── w3_calculator.py        # W3: S-Score（token mode）
├── w3_vector.py            # W3: z-score プロファイル（vector mode）
├── w4_projector.py         # W4: 共鳴ベクトル投影（token mode）
├── w4_vector.py            # W4: コサイン類似度（vector mode）
├── w5_w6_adapter.py        # W5: SimpleCondensator（島形成）+ W6 export
├── threshold.py            # ThresholdResolver（t_abs/t_rel/t_resolved）
├── global_model.py         # GlobalThresholdModel（累積分布管理）
├── edge_selector.py        # MutualKNNSelector / NoOpSelector
├── edge_policy.py          # EdgePolicyResolver（k-sweep + 自動選定）
└── chaining_metrics.py     # 連鎖診断指標（gcr, mean_intra, etc.）
```

### 12.2 Observation C（Relation Pipeline）

```
integration/relations/
├── __init__.py             # パッケージ定義（v0.1.0）
├── parser_adapter.py       # spaCy SVO 抽出（SVOTriple, ExtractionResult）
├── relation_logger.py      # Synapse Grounding v0.2.0（3フィルタ）
└── run_relations.py        # CLI + 診断レポート生成
```

### 12.2b Synapse パッケージ（v2.2 新設, v2.3 拡充）

```
synapse/                        # Synapse Expansion — Store + Proposer + CLI
├── __init__.py                 # SynapseStore, SynapsePatchEntry export
├── schema.py                   # SynapsePatchEntry（edge_key 付きデータモデル）
├── store.py                    # SynapseStore（Overlay, tombstone, conflict log）
├── proposer.py                 # SynapseEdgeProposer（4-Pack Rewrite → 候補 Edge）
├── diagnostic.py               # DiagnosticResult（diff + Audit Gate 判定）
└── cli.py                      # CLI: propose-synapse / evaluate-synapse-patch

patches/                        # パッチファイル格納
├── synapse_v3.1.json           # 42 edges, 7 gap 解消 (+5.8pt)
└── synapse_v3.2.json           # 27 edges, 4 gap 解消 (+2.0pt)

proposals/                      # Run Directory（実行成果物）
└── run_{timestamp}_{dataset}_{rand4}/
    ├── diagnostic_before.json  # ベースライン診断（env_meta 含む）
    ├── patch_candidate.json    # 自動生成候補 Edge
    ├── proposal_report.md      # 人間可読レポート
    ├── diagnostic_after.json   # 評価後再診断
    ├── diagnostic_diff.json    # Before/After 差分
    └── diagnostic_diff.md      # Before/After 差分（人間可読）

tests/
├── test_synapse_store.py       # 監査チェックリスト 10 テスト
└── test_phase3_cli.py          # CLI propose/evaluate 統合テスト
```

**消費者（GO 条件）:**
- Phase 8 Sensor: `sensor/loader_synapse.py` → SynapseStore にデリゲーション
- Observation C: `integration/relations/relation_logger.py` → `SynapseGrounder.from_store(store)` 
- Phase 7 Engine: `esde_engine/loaders.py` → SynapseStore にデリゲーション

**GPT Audit Amendments:**

| 条項 | 内容 |
|------|------|
| §1 | Run ID 衝突耐性（timestamp + PID + nanosecond hash） |
| §2 | Environment Metadata（8フィールド、Before/After 環境同一性保証） |
| §3 | 機械判定 FAIL 条件（CATEGORY_MISMATCH / 新規 CONSISTENT_MISGROUND） |
| §4 | patches/ への書き込み禁止（evaluate は run-dir 内のみ） |
| §5 | Baseline Patch Auto-Inherit（evaluate 時の比較世界一致保証） |

### 12.3 Harvester（データ収集）

```
harvester/
├── __init__.py
├── fetcher.py              # Wikipedia API アクセス
├── distiller.py            # Raw → テキスト + 構造統計
├── storage.py              # Artifact + dataset ファイル管理
└── cli.py                  # CLI（harvest, list）
```

### 12.4 Substrate（Layer 0）

```
substrate/
├── context_record.py       # ContextRecord（不変の観測単位）
├── registry.py             # SubstrateRegistry（Append-only JSONL）
└── trace.py                # Trace（namespace:name KV ペア）
```

---

## 13. 用語集（本文書で使用する主要概念）

| 用語 | 定義 |
|------|------|
| **Atom** | 326個の最小意味単位。163対称ペアで構成 |
| **Molecule** | Phase 8 が生成するセグメント単位の意味構造 |
| **Island** | Phase 9 (W5) が生成する、書き方が類似したセクション群のクラスタ |
| **Cell** | 条件因子で Molecule と Island を結合した統合単位 |
| **Lens** | (ConditionProvider, FeatureMode) のペア。観測の視点 |
| **Condition Factor** | テキスト内部構造から抽出される分類軸（セクション名等） |
| **z-score Profile** | global baseline からの偏差を標準化した20次元ベクトル |
| **Threshold (3層)** | t_abs（全履歴）+ t_rel（今回実行）→ t_resolved（合成） |
| **Mutual-kNN** | 双方向 k 近傍フィルタ。一方向の連鎖を防止 |
| **k-sweep** | 複数の k で W5 を試行し、最小適格 k を選定 |
| **Percolation Threshold** | k-sweep で観測される相転移点（島 → 巨大成分の臨界） |
| **Giant Component Ratio (gcr)** | 最大島サイズ / 総ノード数。連鎖の程度を示す |
| **Noise** | どの Island にも属さないセクション。正当な観測結果 |
| **Observation C** | Phase 8 ↔ 9 橋渡し層。SVO 抽出 + Synapse 動詞接地。LLM 不使用 |
| **SVO Triple** | Subject-Verb-Object の3項関係。spaCy 依存構造解析から抽出 |
| **Grounding Status** | GROUNDED / UNGROUNDED / UNGROUNDED_LIGHTVERB の3状態 |
| **Light Verb** | 意味が文脈依存の機能語的動詞（13語）。Atom 付与を抑制 |
| **POS Guard** | 動詞に不適切な名詞カテゴリ Atom を除外するフィルタ |
| **Score Threshold** | Synapse raw_score の最低閾値（default 0.45） |
| **Synapse Expansion** | Phase 7 エビデンス + 診断に基づく Synapse edge の追加（Phase 1-3 実装完了、v3.2 まで実走済み） |
| **SynapseStore** | `synapse/store.py`。Synapse データの単一ソース。Base JSON + Overlay patch を統合して提供。Phase 8 / Obs C / Phase 7 が共有 |
| **SynapsePatchEntry** | `synapse/schema.py`。パッチ1件のデータモデル。op（add_edge/disable_edge）+ edge_key で一意識別 |
| **SynapseEdgeProposer** | `synapse/proposer.py`。Coverage Gap 動詞から候補 Edge を 4-Pack Rewrite で自動生成 |
| **4-Pack Rewrite** | lemma → synsets → definition → embedding → scored candidates の4段階変換 |
| **DiagnosticResult** | `synapse/diagnostic.py`。診断レポートの型付きラッパー。diff + Audit Gate 判定 |
| **Audit Gate** | evaluate-synapse-patch の機械判定。PASS/WARN/FAIL の3状態。回帰検出で自動 FAIL |
| **Baseline Patch Auto-Inherit** | GPT §5。evaluate 時に before 側のパッチを自動継承し比較世界を一致させる仕組み |
| **edge_key** | `{synset_id}::{atom_id}` 形式の一意識別子。Synapse edge を衝突解決・監査で追跡する鍵 |
| **Tombstone** | `disable_edge` で無効化された edge_key。SynapseStore 内でメモリ保持されるが、get_edges() 結果には出現しない。re-add 不可 |
| **Overlay** | Base Synapse JSON の上にパッチを重ねて適用する仕組み。Design Spec v2.1 §2 で衝突解決ルールを定義 |
| **GO 条件** | GPT 監査ゲート。SynapseStore を Phase 8 と Obs C の両方で使用し、片系バイパス時はテスト Fail |
| **CONSISTENT_MISGROUND** | 同一動詞が同一 Atom に繰り返し接続される統計的異常パターン。修正には人間判断が必要 |

---

*Document generated from implementation record of ESDE Phase 9 (v1.0→v1.9) + Observation C (v0.2.0) + Synapse Expansion Phase 1-3 (v5.6.0-v5.6.1) + 実走 v3.1/v3.2*  
*Philosophy: Aruism — "Describe, but do not decide"*
