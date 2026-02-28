# ESDE 全貌 — 何を実現しようとしているか

**Existence Symmetry Dynamic Equilibrium**  
**存在対称動的均衡**

---

## 1. 一言で言うと

ESDE は **意味の透明な座標系** を作り、それを使って **テキストの意味構造を多層的に観測する** エンジン。

AI が「なぜそう判断したか」を説明できない問題に対して、人間が読める意味の地図を提供する。ベクトル空間のブラックボックスに対する、解釈可能なオルタナティブ。

---

## 2. 哲学的基盤: Aruism（アリズム）

全ての出発点は一つの認識:

> **「ある」は、ある。**（Aru wa, Aru）

存在そのものの肯定。ここから 3 つの原則が導かれる:

- **存在の対称性**: 全ての概念は対を持つ。好（EMO.like）↔ 嫌（EMO.dislike）。生（EXS.life）↔ 死（EXS.death）。163 組の対称ペア。
- **記述せよ、しかし決定するな**: システムは観測し記録するが、「正解」を決めない。不確実性は失敗ではなく、有効な観測結果。
- **十分な説明**: 完璧を目指さない。役に立つ程度に正確な地図を作る。

---

## 3. 意味の座標系（Foundation Layer）

### 3.1 Atom（原子）— 326 個の意味の基本単位

人間の言語で表現しうる概念を 326 個に分解。各 Atom は漢字一字で表される。

```
24 カテゴリ × 数個ずつ = 326 Atom（163 対称ペア）

ACT（行為）:  arrive（到）, abandon（棄）, build（造）...
EMO（感情）:  like（好）, anger（怒）, grief（悲）...
EXS（存在）:  life（生）, death（死）, being（存）...
COG（認知）:  think（思）, know（知）, forget（忘）...
...etc.
```

これは **周期表の元素** に相当する。変更されない。

### 3.2 10 軸 × 48 スロット — 意味の座標軸

各 Atom に属する単語が「意味空間のどこにいるか」を記述するための座標系:

| 軸 | 問い | レベル数 | 例 |
|----|------|---------|-----|
| temporal（時間的条件） | いつ・どのように変化するか | 7 | emergence → permanence |
| scale（空間/スケール条件） | どの規模で作用するか | 6 | individual → cosmic |
| epistemological（認識論的条件） | どう知られるか | 5 | perception → creation |
| ontological（存在論的条件） | 何で構成されているか | 5 | material → semantic |
| interconnection（相互連関条件） | 他とどう関わるか | 5 | independent → resonant |
| resonance（共鳴深度条件） | どれだけ本質に触れるか | 4 | superficial → existential |
| symmetry（対称性関係条件） | 対立とどう作用するか | 5 | destructive → cyclical |
| lawfulness（法則性条件） | どんな法則に従うか | 4 | predictable → necessary |
| experience（体験の質的条件） | どんな体験として現れるか | 3 | discovery → comprehension |
| value_generation（価値生成条件） | どんな価値を持つか | 4 | functional → sacred |

合計 48 スロット。各単語はこの 48 次元で **共鳴度（0-10）** のプロファイルを持つ。

### 3.3 共鳴度スコアリング — バイナリから連続値へ

当初の設計ではスロットへの所属はバイナリ（属する/属さない）だった。A1 パイプラインの実装過程で、LLM に「共鳴度を観測」させることで **連続値（0-10）** モデルが自然に出現した。

```
旧（バイナリ）: ACT_arrive × temporal.emergence = {arrive, land, reach, ...}
新（連続値）:   ACT_arrive × arrive × temporal.emergence = 8
                ACT_arrive × arrive × scale.cosmic      = 0
```

これにより可能になったこと:
- **意味の数学的演算**: コサイン類似度、クラスタリング、cross-Atom 比較
- **解釈可能な差異**: 「arrive と reach は epistemological.experience 軸で差がある」
- **不確実性の自然な表現**: スコア 2-3 = 弱い共鳴（バイナリ強制でない）

2026-02-15 Taka 承認。正式な設計方針として確定。

### 3.4 Synapse — 自然言語と Atom を繋ぐ橋

WordNet の synset を ESDE Atom にマッピングする辞書。テキスト中の単語を Atom に接地するための経路。

```
"kill.v.01" (WordNet) → EXS.death (ESDE Atom), score=0.85
```

現在 v3.2。動詞の接地率 63%。名詞はさらに高い。Synapse Expansion Phase 1-3 で SynapseStore + Overlay パッチシステムを構築済み。

### 3.5 Lexicon v2 — 48 次元共鳴度辞書（A1 パイプライン稼働中）

326 Atom × 約 18,000 語 × 48 スロットの共鳴度プロファイル。**2段パイプライン** で構築:

**Stage 1: 語彙供給** (`lexicon_wn/`)
- WordNet から 12 種のリレーションで語彙を自動展開
- Core Pool（座標決定用）と Deviation Pool（Phase 7/9 用）に分離
- Core 全体: 33,394 語 (avg 102/atom)。APW 4.7→2.0 に改善。汚染 66→0。
- Constitution v1.0 で 17 proposals（merge/subsume/couple/monitor）確定

**Stage 2: A1 観測** (`integration/lexicon/`)
- QwQ-32B を観測者として各語の 48 次元プロファイルを記述
- Mapper（生成）→ Auditor（構造的監査 C1-C5）→ Re-observe（制約付き再観測）
- softmax 正規化 → Shannon entropy → focus_rate で品質管理
- Score inflation 問題を発見・修正済み（nz_mean: 38.7→13.6、OK率: 78.4%→97.3%）
- Diffuse_Observation の REVISE 捕捉率: 0/13→12/12（完全捕捉）
- 326 atom 一括バッチ実行スクリプト（GPU 温度管理付き）

---

## 4. エンジン本体 — 多層観測アーキテクチャ

ESDE のエンジンは **一つの機能ではなく、複数の観測層の組み合わせ**。

```
入力: 任意の自然言語テキスト
  │
  ▼
┌─────────────────────────────────┐
│  Phase 7: Unknown Resolution    │  ← 未知の単語をどう扱うか
│  Phase 8: Introspective Engine  │  ← 強い意味（Atom → Molecule）
│  Obs C:   Relation Pipeline     │  ← 文の構造（誰が何をした）
│  Phase 9: Weak Axis Statistics  │  ← 弱い意味（書き方の統計的パターン）
│  Phase 10: Cell Architecture    │  ← 全てを統合（未実装）
└─────────────────────────────────┘
  │
  ▼
出力: テキストの意味構造の多角的な記述
```

### 4.1 Phase 7: Unknown Resolution（未知の解決）

テキスト中の単語が Synapse に載っていない場合の処理。4 つの仮説を並列評価:

- Route A: タイポ（typo）→ 修正候補を提示
- Route B: 固有名詞 → 特別扱い
- Route C: Synapse のカバレッジ不足 → 新 edge の提案
- Route D: 本当に未知の概念 → Unknown Queue に蓄積

「わからないもの」を排除するのではなく、分類して記録する。不確実性を管理するシステム。

7B+: 集約状態管理 + ボラティリティ測定、7C: 構造監査、7C': LLM 三重投票監査、7D: Meta-Auditor（ルール校正）まで実装済み。

### 4.2 Phase 8: Introspective Engine（内省エンジン）— 強い意味

テキストを **Atom と Molecule** で記述する。ESDE の心臓部。

```
テキスト "I love you"
  │
  ├─ トークン化: ["I", "love", "you"]
  ├─ Synset 抽出: love → love.v.01, love.n.01, ...
  ├─ Synapse 検索: love.v.01 → EMO.love (score=0.92)
  ├─ Molecule 生成: EMO.love (axis=ethical, level=3)
  │
  └─ 副産物:
     ├─ Ledger: 意味の記憶（減衰/強化/永続化）
     ├─ Semantic Index: 使用パターンの索引化
     ├─ Rigidity: 同じ解釈を繰り返していないかの検出
     └─ Feedback Loop: 硬直していたらパラメータを調整
```

重要なのは **Rigidity（硬直性）検出** 。同じ入力に対して常に同じ Molecule を返すシステムは、「考えずに答えを返している」状態。ESDE はこれを検出し、パラメータを揺さぶることで多様な解釈を試みる。

これが「内省」の意味。**自分の観測パターンを監視し、必要に応じて調整する。**

### 4.3 Observation C: Relation Pipeline（関係抽出）

テキストから **誰が・何を・した** の SVO トリプルを決定論的に（LLM を使わずに）抽出し、動詞を Atom に接地する。

```
"Caesar crossed the Rubicon"
  → Subject: Caesar
  → Verb: crossed → ACT.move (via Synapse)
  → Object: the Rubicon
```

v0.2.0 で品質ハードニング完了: Light Verb Stoplist、POS Guard、Score Threshold の 3 フィルター実装。grounding rate は 89%→55% に低下したが、CATEGORY_MISMATCH が完全消滅。

Phase 8（強い意味）と Phase 9（弱い意味）の橋渡し。文の構造的関係を捉える。

### 4.4 Phase 9: Weak Axis Statistics（弱い意味の統計）

テキストの **書き方のパターン** を統計的に分析する。Atom や Molecule のような「強い」意味ではなく、文体・構造・傾向のような「弱い」意味を捉える。

```
7 段階のパイプライン:

W1: 20 次元のトークン特徴抽出
W2: 条件因子で集約（セクション名、文書名、受動態の有無...）
W3: z-score プロファイル（「平均からどれだけ偏っているか」）
W4: コサイン類似度行列（セクション間の距離）
W5: Mutual-kNN + Island 形成（似たセクションのクラスタ）
W6: Export（JSON/Markdown/CSV）
```

**光学的アナロジー（Lens）:**

3 種類の「レンズ」で同じテキストを異なる角度から観測:

| Lens | 何を見るか | 例 |
|------|----------|-----|
| Structure | テンプレート構造 | 「Wikipedia の伝記は Early Life → Career → Legacy の形を持つ」 |
| Semantic | 意味的類似性 | 「戦争の記事と政治の記事は意味空間で近い」 |
| Hybrid | セクション内の意味的偏り | 「この段落は他と比べて EMO 系が異常に高い」 |

**k パラメータ = 焦点距離:**
- k 大（広角）→ 大きな構造が見える
- k 小（望遠）→ 局所的な特徴が見える
- k=4 が相転移点（分解状態 ↔ 連鎖状態の臨界点）

### 4.5 Phase 10: Cell Architecture（未実装）

Phase 8 と Phase 9 の観測結果を **混ぜずに** 統合する。

```
Phase 8（強い意味）= 原子核（Atom/Molecule）
Phase 9（弱い意味）= 電子（Island/統計パターン）
条件因子 = 電磁力（両者を結びつける引力）

Cell = Molecule + Island（同じセクションに属する）
Organ = Cell 群（同じ文書に属する）
Ecosystem = 全体（LLM による言語化レポート）
```

**非混合原則:** Phase 8 の出力と Phase 9 の出力は決して直接混合しない。独立した観測系として保持し、条件因子（「同じセクションに属する」等）だけで間接的に結合する。

---

## 5. インフラ層

### 5.1 Substrate Layer（Layer 0）

全フェーズの下に敷かれた観測記録層。意味の解釈は一切行わず、生の観測データを append-only で保存する。

```
ContextRecord:
  - context_id: 決定論的に生成
  - traces: [namespace:name = value] の列
  - 意味的判断なし、生データのみ
```

ArticleRecord/ContentGateway と統合済み（v5.4.7-SUB.1）。substrate_ref フィールドで Phase 9 データとリンク。

### 5.2 Harvester

外部データ（Wikipedia 等）を取得してローカルキャッシュに保存。全パイプラインのデータ供給源。「Fetch once, analyze many times」の原則。Artifact（生レスポンス）+ Distilled（テキスト）の 2 層保存。

### 5.3 Content Gateway

外部コンテンツを正規化して ESDE のフォーマットに変換するアダプタ。

---

## 6. 全体像

```
         Aruism（哲学）
         「ある」は、ある
              │
              ▼
     ┌── Foundation Layer ──┐
     │  326 Atoms (周期表)   │
     │  10 軸 × 48 スロット   │
     │  Synapse v3.2 (橋)   │
     │  Lexicon v2 (辞書)    │  ← A1 パイプライン稼働中
     └──────────┬───────────┘
                │
     ┌──────────▼───────────┐
     │   Substrate Layer 0   │  観測の記録（生データ、判断なし）
     └──────────┬───────────┘
                │
     ┌──────────▼───────────────────────────────────┐
     │              ESDE Engine                      │
     │                                               │
     │  Phase 7: 未知の管理（7B+/7C/7C'/7D 実装済み）  │
     │  Phase 8: 強い意味（Atom → Molecule → 内省）     │
     │  Obs C:   関係構造（SVO → Atom 接地, v0.2.0）    │
     │  Phase 9: 弱い意味（統計 → Island → Lens）完了   │
     │  Phase 10: 統合（Cell → Organ → Ecosystem）     │
     └──────────┬───────────────────────────────────┘
                │
                ▼
         あらゆるテキストの
         意味構造の多角的・透明な記述
```

---

## 7. 何が「できるようになる」のか

| 能力 | 必要なもの | 現状 |
|------|----------|------|
| 任意のテキストを意味座標に変換 | Synapse + Lexicon v2 | Synapse v3.2 稼働、Lexicon A1 パイプライン稼働中 |
| 単語の 48 次元共鳴度プロファイル | Lexicon v2 A1 Pipeline | Mapper + Auditor 実装済み。326 atom バッチ実行可能 |
| AI の出力を意味座標で監査 | Phase 8 + Lexicon | Phase 8 実装済み、Lexicon 構築進行中 |
| テキストの書き方パターンを可視化 | Phase 9 | **完了**（Lens 3 種 + k-sweep） |
| 文の構造的関係を Atom に接地 | Observation C | **稼働中**（grounding rate 55%, v0.2.0 品質ハードニング済み） |
| 未知の概念を分類・蓄積 | Phase 7 | **実装済み**（7B+/7C/7C'/7D） |
| 強い意味と弱い意味の統合 | Phase 10 (Cell) | 設計完了（Cell Architecture v2.2）、実装未着手 |
| 言語横断の意味比較 | Lexicon の多言語展開 | 英語のみ（漢字アンカーは言語中立） |
| AI の硬直性検出と自己修正 | Phase 8 Rigidity + Feedback | **実装済み** |
| 意味の数学的演算 | 共鳴度スコアリング | **可能**（連続値 0-10、コサイン類似度等） |

---

## 8. 現在の開発フロントライン

**Lexicon v2 の A1 パイプライン** — 326 Atom × ~18,000 語 × 48 スロットの共鳴度マッピング。

語彙供給基盤は完成（Core/Deviation 分離、Constitution v1.0 確定、全 17 proposals 承認済み）。A1 観測パイプライン（Mapper + Auditor + Batch Report + 326-atom バッチスクリプト）も実装完了し、本番観測が進行中。

score inflation 問題を発見・修正済み: nz_mean 38.7→13.6、Diffuse残存 18%→0%、OK率 78.4%→97.3%。共鳴度スコアリングモデルを正式に承認（バイナリ→連続値への設計転換、2026-02-15）。

これが完了すると Foundation Layer が「使える状態」になり、Phase 8/9/10 の全パイプラインが本来の精度で動き始める。今は Synapse だけで動いているため、接地精度に限界がある。Lexicon v2 は ESDE の解像度を根本的に上げる基盤工事。
