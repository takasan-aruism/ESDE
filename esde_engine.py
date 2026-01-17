"""
ESDE Core Engine v0.4.1
- Dictionary v2.0 Compatible (kanji removed, using name)
- Phase 3 Implementation: Ternary Emergence (三項創発)
- GPT監査済み仕様に基づく実装

Features:
  - Phase 1: Structure Weight (経験則)
  - Phase 2: Co-occurrence / Linkage (連動性)
  - Phase 3: Ternary Emergence (三項創発)
    - Saturation Detection (膠着検出)
    - Destructive Emergence (破壊的創発 / Axis Shift)
    - Creative Emergence (創造的創発)
"""
import json
import os
import math
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


class ESDEngine:
    def __init__(self, dictionary_path="esde_dictionary.json", state_path="esde_state.json"):
        self.dictionary_path = dictionary_path
        self.state_path = state_path
        
        # パラメータ設定 (Spec v0.2.1)
        self.TAU = 86400.0           # 連動性減衰定数 (秒) = 1日
        self.ALPHA_LINK = 0.7        # 連動性重みの頻度係数
        self.LAMBDA_SYM = 1.0        # 対称性誤差の重み
        self.LAMBDA_LINK = 0.5       # 連動性誤差の重み
        self.CONTEXT_WINDOW_SIZE = 5 # 共起判定ウィンドウサイズ
        self.GAMMA = 0.3             # 観測注入率
        
        # Phase 3 パラメータ
        self.SATURATION_VAL_THRESHOLD = 0.5    # 高活性閾値
        self.SATURATION_SYM_THRESHOLD = 0.3    # 高均衡閾値（ε_sym < この値）
        self.DISSOLUTION_FACTOR = 0.1          # 溶解係数
        self.BOOST_FACTOR = 1.5                # ブースト係数
        
        # 破壊的創発トリガーID (辞書に実在するもの)
        self.DESTRUCTIVE_CONCEPT_IDS = {
            "ACT.leave",      # leave
            "COG.separate",   # separate
            "SOC.refuse",     # refuse
            "ACT.abandon",    # abandon
        }
        
        self.dictionary = self._load_json(dictionary_path)
        self.state_data = self._load_state(state_path)
        self.level_scores = self._precompute_level_scores()
        
        dict_version = self.dictionary.get("meta", {}).get("version", "unknown")
        print(f"[INIT] ESDE Engine v0.4.1 initialized (Phase 3 Ready).")
        print(f"       Dictionary: v{dict_version}")
        print(f"       State File: {state_path}")
        print(f"       Phase: {self._get_current_phase()} (n={self.state_data['global_n']})")

    def _load_json(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_state(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # マイグレーション
                    if "links" not in data:
                        data["links"] = {}
                    if "recent_context" not in data:
                        data["recent_context"] = []
                    if "emergence_log" not in data:
                        data["emergence_log"] = []
                    return data
            except Exception as e:
                print(f"[WARN] State file corrupted ({e}), resetting.")
        return self._get_initial_state()

    def _get_initial_state(self):
        return {
            "global_n": 0,
            "concept_stats": {},
            "links": {},
            "recent_context": [],
            "emergence_log": [],
            "last_updated": datetime.now().isoformat()
        }

    def reset_state(self, force=False):
        if not force:
            print("[WARN] Reset skipped. Use reset_state(force=True).")
            return
        self.state_data = self._get_initial_state()
        self._save_state()
        print(f"[INFO] ESDE State has been reset: {self.state_path}")

    def _save_state(self):
        self.state_data["last_updated"] = datetime.now().isoformat()
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state_data, f, indent=2, ensure_ascii=False)

    def _precompute_level_scores(self):
        scores = {}
        for axis_id, axis_data in self.dictionary["axes"].items():
            levels = axis_data["levels"]
            total = len(levels)
            scores[axis_id] = {}
            for i, level_name in enumerate(levels):
                scores[axis_id][level_name] = round((i + 1) / total, 2)
        return scores

    def _get_current_phase(self):
        n = self.state_data["global_n"]
        if n < 100:
            return "Phase 1 (Structure Dominant)"
        if n < 1000:
            return "Phase 2 (Interconnectivity)"
        return "Phase 3 (Ternary Emergence)"

    def _calculate_alpha_t(self):
        n = self.state_data["global_n"]
        if n == 0:
            return 0.0
        return min(0.7, n * 0.007)

    def _get_concept_name(self, concept_id):
        """辞書から概念名を取得"""
        c = self.dictionary["concepts"].get(concept_id, {})
        return c.get("name", concept_id.split(".")[-1])

    # ========================================
    # Phase 2: 連動性ロジック
    # ========================================

    def _update_links(self, current_concept_id, timestamp):
        context = self.state_data["recent_context"]
        for item in context:
            prev_id = item["concept_id"]
            if prev_id == current_concept_id:
                continue
            link_key = ":".join(sorted([prev_id, current_concept_id]))
            link_data = self.state_data["links"].setdefault(
                link_key, {"count": 0, "last_seen": 0.0}
            )
            link_data["count"] += 1
            link_data["last_seen"] = timestamp

    def _calculate_wij(self, link_data, current_time):
        if not link_data:
            return 0.0
        count = link_data["count"]
        last_seen = link_data["last_seen"]
        delta_t = max(0, current_time - last_seen)
        term1 = self.ALPHA_LINK * math.log(1 + count)
        term2 = (1 - self.ALPHA_LINK) * math.exp(-delta_t / self.TAU)
        return term1 + term2

    def _calculate_epsilon_link(self, current_concept_id, current_val, current_time):
        total_error = 0.0
        linked_count = 0
        for key, data in self.state_data["links"].items():
            ids = key.split(":")
            if current_concept_id not in ids:
                continue
            pair_id = ids[1] if ids[0] == current_concept_id else ids[0]
            pair_stats = self.state_data["concept_stats"].get(pair_id, {"value": 0.0})
            pair_val = pair_stats["value"]
            w_ij = self._calculate_wij(data, current_time)
            error = w_ij * ((current_val - pair_val) ** 2)
            total_error += error
            linked_count += 1
        if linked_count > 0:
            avg_error = total_error / linked_count
        else:
            avg_error = 0.0
        return round(avg_error, 4), linked_count

    # ========================================
    # Phase 3: 三項創発ロジック
    # ========================================

    def _detect_saturation(self) -> Optional[Tuple[str, str, float, float, float]]:
        """膠着状態の検出"""
        context = self.state_data["recent_context"]
        if len(context) < 2:
            return None
        
        context_ids = [item["concept_id"] for item in context]
        candidates = []
        seen_pairs = set()
        
        for c_id in context_ids:
            if c_id not in self.dictionary["concepts"]:
                continue
            pair_id = self.dictionary["concepts"][c_id]["symmetric_pair"]
            
            if pair_id in context_ids:
                pair_key = ":".join(sorted([c_id, pair_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                val_a = self.state_data["concept_stats"].get(c_id, {}).get("value", 0.0)
                val_b = self.state_data["concept_stats"].get(pair_id, {}).get("value", 0.0)
                epsilon_sym = abs(val_a - val_b)
                total_energy = val_a + val_b
                candidates.append((c_id, pair_id, val_a, val_b, epsilon_sym, total_energy))
        
        if not candidates:
            return None
        
        best = max(candidates, key=lambda x: x[5])
        id_a, id_b, val_a, val_b, epsilon_sym, _ = best
        
        if val_a > self.SATURATION_VAL_THRESHOLD and \
           val_b > self.SATURATION_VAL_THRESHOLD and \
           epsilon_sym < self.SATURATION_SYM_THRESHOLD:
            return (id_a, id_b, val_a, val_b, epsilon_sym)
        
        return None

    def _evaluate_catalyst(self, concept_id: str, axis: str, level: str) -> str:
        """第三項の評価と分岐判定 (優先: Destructive > Creative)"""
        # 破壊的創発
        if axis == "interconnection" and level == "independent":
            return "destructive"
        if axis == "symmetry" and level == "destructive":
            return "destructive"
        if concept_id in self.DESTRUCTIVE_CONCEPT_IDS:
            return "destructive"
        
        # 創造的創発
        if axis == "interconnection" and level == "resonant":
            return "creative"
        if axis == "value_generation":
            return "creative"
        if axis == "symmetry" and level == "generative":
            return "creative"
        
        return "none"

    def _apply_destructive_emergence(self, dyad: Tuple[str, str]):
        """破壊的創発の適用（Axis Shift Protocol）"""
        id_a, id_b = dyad
        if id_a in self.state_data["concept_stats"]:
            self.state_data["concept_stats"][id_a]["value"] *= self.DISSOLUTION_FACTOR
        if id_b in self.state_data["concept_stats"]:
            self.state_data["concept_stats"][id_b]["value"] *= self.DISSOLUTION_FACTOR

    # ========================================
    # メイン処理
    # ========================================

    def observe(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        current_time = time.time()
        
        concept_id = observation.get("concept_id")
        axis_id = observation.get("axis")
        level_id = observation.get("level")
        sub_level = observation.get("sub_level", None)
        position_score = observation.get("position_score", 1.0)
        
        if not axis_id or not level_id:
            return {"error": "Missing axis or level"}
        if concept_id not in self.dictionary["concepts"]:
            return {"error": f"Unknown concept: {concept_id}"}
        
        # Phase 3: 膠着検出 & 創発判定
        emergence_info = {
            "type": "none",
            "trigger_concept": None,
            "target_dyad": None,
            "action": None
        }
        
        saturation = self._detect_saturation()
        input_intensity = 1.0
        
        if saturation:
            id_a, id_b, val_a, val_b, epsilon_sym = saturation
            dyad_key = f"{id_a}:{id_b}"
            catalyst_type = self._evaluate_catalyst(concept_id, axis_id, level_id)
            
            if catalyst_type == "destructive":
                self._apply_destructive_emergence((id_a, id_b))
                input_intensity = self.BOOST_FACTOR
                emergence_info = {
                    "type": "destructive",
                    "trigger_concept": concept_id,
                    "target_dyad": dyad_key,
                    "action": "dissolution_and_shift"
                }
                self.state_data["emergence_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "destructive",
                    "dyad": dyad_key,
                    "catalyst": concept_id,
                    "val_a_before": val_a,
                    "val_b_before": val_b
                })
                
            elif catalyst_type == "creative":
                emergence_info = {
                    "type": "creative",
                    "trigger_concept": concept_id,
                    "target_dyad": dyad_key,
                    "action": "reinforcement"
                }
                self.state_data["emergence_log"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "creative",
                    "dyad": dyad_key,
                    "catalyst": concept_id
                })
        
        # 学習更新
        self.state_data["global_n"] += 1
        n = self.state_data["global_n"]
        
        stats = self.state_data["concept_stats"].setdefault(
            concept_id, {"count": 0, "value": 0.0}
        )
        stats["count"] += 1
        stats["value"] = (1 - self.GAMMA) * stats["value"] + self.GAMMA * input_intensity
        
        self._update_links(concept_id, current_time)
        
        self.state_data["recent_context"].append({
            "concept_id": concept_id,
            "timestamp": current_time
        })
        if len(self.state_data["recent_context"]) > self.CONTEXT_WINDOW_SIZE:
            self.state_data["recent_context"].pop(0)

        if n % 10 == 0:
            self._save_state()

        # 重み計算
        base_level_score = self.level_scores.get(axis_id, {}).get(level_id, 0.5)
        w_structure = position_score * base_level_score
        w_data = stats["count"] / n if n > 0 else 0.0
        alpha_t = self._calculate_alpha_t()
        weight = alpha_t * w_data + (1 - alpha_t) * w_structure

        # 誤差分析
        pair_id = self.dictionary["concepts"][concept_id]["symmetric_pair"]
        pair_stats = self.state_data["concept_stats"].get(pair_id, {"value": 0.0})
        epsilon_sym = abs(stats["value"] - pair_stats["value"])
        
        epsilon_link, link_count = self._calculate_epsilon_link(
            concept_id, stats["value"], current_time
        )
        
        risk_sym = self.LAMBDA_SYM * epsilon_sym * weight
        risk_link = self.LAMBDA_LINK * epsilon_link * weight
        risk_score = risk_sym + risk_link

        concept_name = self._get_concept_name(concept_id)
        
        return {
            "meta": {
                "phase": self._get_current_phase(),
                "n": n,
                "alpha_t": round(alpha_t, 3)
            },
            "observation": {
                "concept": f"{concept_id} ({concept_name})",
                "axis": f"{axis_id}.{level_id}",
                "sub_level": sub_level,
                "weights": {
                    "structure": round(w_structure, 3),
                    "data": round(w_data, 3),
                    "final": round(weight, 3)
                }
            },
            "analysis": {
                "val_target": round(stats["value"], 3),
                "val_pair": round(pair_stats["value"], 3),
                "epsilon_sym": round(epsilon_sym, 3),
                "epsilon_link": epsilon_link,
                "link_count": link_count,
                "risk_score": round(risk_score, 3),
                "status": self._evaluate_status(risk_score),
                "emergence": emergence_info
            },
            "interpretation": self._generate_interpretation(
                risk_score, epsilon_link, concept_id, emergence_info
            )
        }

    def _evaluate_status(self, score):
        if score < 0.3:
            return "balanced"
        if score < 0.6:
            return "moderate"
        if score < 0.9:
            return "high"
        return "severe"

    def _generate_interpretation(self, score, e_link, c_id, emergence):
        c_def = self.dictionary["concepts"][c_id]
        name = c_def["name"]
        pair_name = self._get_concept_name(c_def["symmetric_pair"])
        
        if emergence["type"] == "destructive":
            return (
                f"【REBOOT】Axis shift occurred. "
                f"{emergence['target_dyad']} saturation dissolved by '{name}'. "
                f"[Axis Shift Protocol]"
            )
        
        if emergence["type"] == "creative":
            return (
                f"【GENERATIVE】Creative emergence occurred. "
                f"'{name}' integrated with {emergence['target_dyad']} equilibrium."
            )
        
        if e_link > 0.3:
            return (
                f"【WARNING】Context deviation detected (ε_link={e_link:.3f}). "
                f"'{name}' is deviating from usual linkage patterns."
            )
        
        if score >= 0.9:
            return (
                f"【DANGER】Extreme bias toward '{name}' (Score: {score:.2f}). "
                f"Equilibrium with '{pair_name}' is collapsing."
            )
        elif score >= 0.3:
            return (
                f"【CAUTION】'{name}' is activating (Score: {score:.2f}). "
                f"Be aware of '{pair_name}'."
            )
        else:
            return f"【SAFE】'{name}' is within stable range (Score: {score:.2f})."

    # ========================================
    # ユーティリティ
    # ========================================

    def get_link_stats(self, concept_id=None):
        links = self.state_data["links"]
        if concept_id:
            return {k: v for k, v in links.items() if concept_id in k.split(":")}
        return links

    def get_state_summary(self):
        return {
            "global_n": self.state_data["global_n"],
            "concept_count": len(self.state_data["concept_stats"]),
            "link_count": len(self.state_data["links"]),
            "emergence_count": len(self.state_data["emergence_log"]),
            "phase": self._get_current_phase()
        }

    def get_emergence_log(self, limit=10):
        return self.state_data["emergence_log"][-limit:]

    def get_saturation_status(self) -> Optional[Dict[str, Any]]:
        sat = self._detect_saturation()
        if sat:
            id_a, id_b, val_a, val_b, epsilon_sym = sat
            return {
                "saturated": True,
                "dyad": f"{id_a}:{id_b}",
                "val_a": val_a,
                "val_b": val_b,
                "epsilon_sym": epsilon_sym
            }
        return {"saturated": False}


# ==========================================
# 動作テスト
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ESDE Engine v0.4.1 - Dictionary v2.0 Compatibility Test")
    print("=" * 60)
    
    engine = ESDEngine(state_path="esde_state_test.json")
    engine.reset_state(force=True)
    
    print("\n[Test 1: Basic Observation]")
    print("-" * 60)
    
    report = engine.observe({
        "concept_id": "EMO.love",
        "axis": "resonance",
        "level": "existential"
    })
    
    print(f"  Concept: {report['observation']['concept']}")
    print(f"  Status: {report['analysis']['status']}")
    print(f"  Interpretation: {report['interpretation']}")
    
    print("\n[Test 2: Symmetric Pair]")
    print("-" * 60)
    
    report = engine.observe({
        "concept_id": "EMO.hate",
        "axis": "resonance",
        "level": "essential"
    })
    
    print(f"  Concept: {report['observation']['concept']}")
    print(f"  Val(target): {report['analysis']['val_target']}")
    print(f"  Val(pair): {report['analysis']['val_pair']}")
    print(f"  ε_sym: {report['analysis']['epsilon_sym']}")
    
    print("\n[State Summary]")
    print(engine.get_state_summary())