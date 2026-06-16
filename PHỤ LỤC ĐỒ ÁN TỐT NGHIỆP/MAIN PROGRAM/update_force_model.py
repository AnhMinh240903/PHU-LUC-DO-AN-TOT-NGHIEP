#!/home/anhminh/yolo/bin/python3


from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Sequence, List

try:
    import joblib
except Exception as exc:
    joblib = None
    JOBLIB_IMPORT_ERROR = exc
else:
    JOBLIB_IMPORT_ERROR = None

try:
    import pandas as pd
except Exception as exc:
    pd = None
    PANDAS_IMPORT_ERROR = exc
else:
    PANDAS_IMPORT_ERROR = None


DEFAULT_MODEL_DIR = "/home/anhminh/DATN/TEST/MAIN/MLP models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "grasp_mlp_models.joblib")
DEFAULT_METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, "metadata.json")

SENSOR_CONFIG = [
    {"key": "L_OUT", "rows": 3, "cols": 2},
    {"key": "L_IN", "rows": 3, "cols": 3},
    {"key": "R_IN", "rows": 3, "cols": 3},
    {"key": "R_OUT", "rows": 3, "cols": 2},
]

DEFAULT_INITIAL_FEATURES = [
    "object_class", "object_instance", "bbox_w_norm", "bbox_h_norm",
    "bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "conf",
]

DEFAULT_STATE_FEATURES = [
    "object_class", "object_instance", "bbox_w_norm", "bbox_h_norm",
    "bbox_area_norm", "aspect_ratio", "cx_norm", "cy_norm", "conf",
    "grip_ratio", "left_sum_avg", "right_sum_avg", "contact_area_avg",
    "L_OUT_avg", "L_IN_avg", "R_IN_avg", "R_OUT_avg",
    "L_OUT_max", "L_IN_max", "R_IN_max", "R_OUT_max",
    "total_avg", "total_max",
]

DEFAULT_POLICY = {
    "initial_clip_min": 0.0,
    "initial_clip_max": 1.0,
    "slip_increase_step": 0.02,
    "overforce_decrease_step": 0.01,
    "max_adjustments": 8,
    "success_prob_threshold": 0.55,
    "slip_prob_threshold": 0.45,
    "overforce_prob_threshold": 0.45,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _is_numeric_feature(name: str) -> bool:
    return name not in ("object_class", "object_instance", "label", "image")


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_pipeline_classes(model: Any) -> List[str]:
    if hasattr(model, "classes_"):
        return [str(x) for x in list(model.classes_)]
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "classes_"):
                return [str(x) for x in list(step.classes_)]
    return []


def make_single_row(values: Mapping[str, Any], feature_names: Sequence[str]) -> "pd.DataFrame":
    
    if pd is None:
        raise RuntimeError("pandas is required for model inference.") from PANDAS_IMPORT_ERROR

    row: Dict[str, Any] = {}
    for name in feature_names:
        raw = values.get(name, "unknown" if not _is_numeric_feature(name) else 0.0)
        if _is_numeric_feature(name):
            row[name] = _safe_float(raw, 0.0)
        else:
            row[name] = "unknown" if raw is None or str(raw).strip() == "" else str(raw)
    return pd.DataFrame([row], columns=list(feature_names))


def compute_tactile_features_from_sequence(
    sequence: Sequence[Mapping[str, Sequence[Sequence[float]]]],
) -> Dict[str, float]:
    
    features: Dict[str, float] = {}
    total_avg_accum = 0.0
    total_max_val = 0.0
    left_sum_total = 0.0
    right_sum_total = 0.0
    contact_sum = 0.0

    total_cells = sum(cfg["rows"] * cfg["cols"] for cfg in SENSOR_CONFIG)
    left_keys = {"L_OUT", "L_IN"}
    right_keys = {"R_OUT", "R_IN"}
    num_timesteps = len(sequence)

    if num_timesteps == 0:
        for cfg in SENSOR_CONFIG:
            key = cfg["key"]
            features[f"{key}_avg"] = 0.0
            features[f"{key}_max"] = 0.0
        features["total_avg"] = 0.0
        features["total_max"] = 0.0
        features["left_sum_avg"] = 0.0
        features["right_sum_avg"] = 0.0
        features["contact_area_avg"] = 0.0
        return features

    for cfg in SENSOR_CONFIG:
        key = cfg["key"]
        rows = int(cfg["rows"])
        cols = int(cfg["cols"])
        sum_val = 0.0
        count = 0
        max_val = 0.0

        for reading in sequence:
            mat = reading.get(key)
            if mat is None:
                continue
            for r in range(rows):
                for c in range(cols):
                    try:
                        v = float(mat[r][c])
                    except Exception:
                        v = 0.0
                    sum_val += v
                    count += 1
                    if v > max_val:
                        max_val = v

        avg_val = sum_val / float(count) if count > 0 else 0.0
        features[f"{key}_avg"] = float(avg_val)
        features[f"{key}_max"] = float(max_val)
        total_avg_accum += avg_val
        total_max_val = max(total_max_val, max_val)

    for reading in sequence:
        left_sum_step = 0.0
        right_sum_step = 0.0
        active_cells = 0

        for cfg in SENSOR_CONFIG:
            key = cfg["key"]
            rows = int(cfg["rows"])
            cols = int(cfg["cols"])
            mat = reading.get(key)
            if mat is None:
                continue
            for r in range(rows):
                for c in range(cols):
                    try:
                        v = float(mat[r][c])
                    except Exception:
                        v = 0.0
                    if v > 0.0:
                        active_cells += 1
                    if key in left_keys:
                        left_sum_step += v
                    elif key in right_keys:
                        right_sum_step += v

        left_sum_total += left_sum_step
        right_sum_total += right_sum_step
        if total_cells > 0:
            contact_sum += active_cells / float(total_cells)

    features["total_avg"] = float(total_avg_accum / float(len(SENSOR_CONFIG)))
    features["total_max"] = float(total_max_val)
    features["left_sum_avg"] = float(left_sum_total / float(num_timesteps))
    features["right_sum_avg"] = float(right_sum_total / float(num_timesteps))
    features["contact_area_avg"] = float(contact_sum / float(num_timesteps))
    return features


class GraspMLPModels:
    

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        verbose: bool = True,
    ):
        if joblib is None:
            raise RuntimeError("joblib is required to load the trained model.") from JOBLIB_IMPORT_ERROR

        self.model_path = str(model_path)
        self.metadata_path = str(metadata_path)
        self.verbose = bool(verbose)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model file not found: " + self.model_path)

        self.bundle = joblib.load(self.model_path)
        if os.path.exists(self.metadata_path):
            self.metadata = _load_json(self.metadata_path)
        else:
            self.metadata = {k: v for k, v in self.bundle.items() if k not in ("initial_model", "state_model")}

        self.initial_model = self.bundle.get("initial_model")
        self.state_model = self.bundle.get("state_model")
        if self.initial_model is None:
            raise KeyError("initial_model not found in model bundle.")
        if self.state_model is None:
            raise KeyError("state_model not found in model bundle.")

        self.initial_features = list(
            self.metadata.get("initial_features") or self.bundle.get("initial_features") or DEFAULT_INITIAL_FEATURES
        )
        self.state_features = list(
            self.metadata.get("state_features") or self.bundle.get("state_features") or DEFAULT_STATE_FEATURES
        )

        self.runtime_policy = dict(DEFAULT_POLICY)
        self.runtime_policy.update(
            self.metadata.get("runtime_policy") or self.bundle.get("runtime_policy") or {}
        )

        self.initial_clip_min = float(self.runtime_policy.get("initial_clip_min", 0.0))
        self.initial_clip_max = float(self.runtime_policy.get("initial_clip_max", 1.0))
        self.state_classes = _get_pipeline_classes(self.state_model)

        if self.verbose:
            print("[INFO] Loaded MLP model bundle")
            print("       model           :", self.model_path)
            print("       metadata        :", self.metadata_path if os.path.exists(self.metadata_path) else "from bundle")
            print("       initial_features:", len(self.initial_features))
            print("       state_features  :", len(self.state_features))
            print("       state_classes   :", self.state_classes)

    def predict_initial_grip_ratio(self, vision_features: Mapping[str, Any]) -> Dict[str, Any]:
        row = make_single_row(vision_features, self.initial_features)
        raw_pred = float(self.initial_model.predict(row)[0])
        clipped = clamp(raw_pred, self.initial_clip_min, self.initial_clip_max)
        return {
            "initial_grip_ratio": clipped,
            "raw_initial_grip_ratio": raw_pred,
            "features": row.iloc[0].to_dict(),
        }

    def build_state_input(
        self,
        vision_features: Mapping[str, Any],
        grip_ratio: float,
        tactile_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        merged.update(dict(vision_features))
        merged.update(dict(tactile_features))
        merged["grip_ratio"] = float(grip_ratio)
        return merged

    def predict_tactile_state(
        self,
        vision_features: Mapping[str, Any],
        grip_ratio: float,
        tactile_features: Mapping[str, Any],
    ) -> Dict[str, Any]:
        values = self.build_state_input(vision_features, grip_ratio, tactile_features)
        row = make_single_row(values, self.state_features)

        label = str(self.state_model.predict(row)[0])

        probabilities: Dict[str, float] = {}
        if hasattr(self.state_model, "predict_proba"):
            proba = self.state_model.predict_proba(row)[0]
            classes = self.state_classes
            if len(classes) == len(proba):
                probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}

        for key in ("success", "slip", "overforce"):
            probabilities.setdefault(key, 0.0)

        
        if "fail" in probabilities:
            probabilities["slip"] = float(probabilities.get("slip", 0.0) + probabilities.get("fail", 0.0))

        action = self.decide_action(probabilities)
        return {
            "state": label,
            "probabilities": probabilities,
            "action": action["action"],
            "delta_grip_ratio": action["delta_grip_ratio"],
            "features": row.iloc[0].to_dict(),
        }

    def decide_action(self, probabilities: Mapping[str, float]) -> Dict[str, Any]:
        p_success = float(probabilities.get("success", 0.0))
        p_slip = float(probabilities.get("slip", 0.0))
        p_overforce = float(probabilities.get("overforce", 0.0))

        success_th = float(self.runtime_policy.get("success_prob_threshold", 0.55))
        slip_th = float(self.runtime_policy.get("slip_prob_threshold", 0.45))
        overforce_th = float(self.runtime_policy.get("overforce_prob_threshold", 0.45))
        slip_step = float(self.runtime_policy.get("slip_increase_step", 0.02))
        overforce_step = float(self.runtime_policy.get("overforce_decrease_step", 0.01))

        
        if p_overforce >= overforce_th and p_overforce >= p_slip:
            return {"action": "decrease", "delta_grip_ratio": -abs(overforce_step)}
        if p_slip >= slip_th:
            return {"action": "increase", "delta_grip_ratio": abs(slip_step)}
        if p_success >= success_th:
            return {"action": "hold", "delta_grip_ratio": 0.0}

       
        return {"action": "observe", "delta_grip_ratio": 0.0}

    def apply_action_to_ratio(self, current_grip_ratio: float, action_result: Mapping[str, Any]) -> float:
        delta = float(action_result.get("delta_grip_ratio", 0.0))
        next_ratio = float(current_grip_ratio) + delta
        return clamp(next_ratio, self.initial_clip_min, self.initial_clip_max)


def demo() -> None:
    models = GraspMLPModels()
    vision_features = {
        "object_class": "ball",
        "object_instance": "ball_1",
        "bbox_w_norm": 0.30,
        "bbox_h_norm": 0.35,
        "bbox_area_norm": 0.105,
        "aspect_ratio": 1.16,
        "cx_norm": 0.50,
        "cy_norm": 0.55,
        "conf": 0.90,
    }
    initial = models.predict_initial_grip_ratio(vision_features)
    print("\nInitial prediction:")
    print(json.dumps(initial, indent=2, ensure_ascii=False))

    sequence = []
    for _ in range(20):
        sequence.append({
            "L_OUT": [[1000, 1200], [1500, 1400], [0, 0]],
            "L_IN": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "R_IN": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "R_OUT": [[1100, 1300], [1600, 1500], [0, 0]],
        })
    tactile_features = compute_tactile_features_from_sequence(sequence)
    state = models.predict_tactile_state(vision_features, initial["initial_grip_ratio"], tactile_features)
    print("\nTactile-state prediction:")
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo()


