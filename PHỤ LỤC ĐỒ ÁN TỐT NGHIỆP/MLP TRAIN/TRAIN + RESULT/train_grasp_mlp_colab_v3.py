# -*- coding: utf-8 -*-

# Cell 1: Install and import
!pip -q install joblib scikit-learn pandas numpy matplotlib

import os
import re
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

def make_ohe():
   
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



# Cell 2: Upload CSV files
from google.colab import files
uploaded = files.upload()

print("Uploaded files:")
for k in uploaded.keys():
    print(" -", k)

# Cell 3: Load CSV 

def load_csv_robust(path, expected_first_col=None):
    path = str(path)
    try:
        df_tmp = pd.read_csv(path)
        if expected_first_col is None or expected_first_col in df_tmp.columns:
            return df_tmp
    except Exception:
        pass

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines[:20]):
        if expected_first_col and line.strip().startswith(expected_first_col + ","):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Cannot find CSV header in {path}")

    return pd.read_csv(path, skiprows=header_idx)

files_list = list(uploaded.keys())
dataset_file = None
attempts_file = None
labels_file = None

for f in files_list:
    name = f.lower()
    if "grasp_dataset" in name or ("dataset" in name and name.endswith(".csv")):
        dataset_file = f
    elif "grasp_attempts" in name or ("attempt" in name and name.endswith(".csv")):
        attempts_file = f
    elif "labels" in name and name.endswith(".csv"):
        labels_file = f

print("dataset_file =", dataset_file)
print("attempts_file =", attempts_file)
print("labels_file   =", labels_file)

if dataset_file is None:
    raise ValueError("Missing grasp_dataset.csv")
if labels_file is None:
    print("WARNING: labels.csv not found. The notebook will parse object_instance from image filename.")

df = load_csv_robust(dataset_file, expected_first_col="attempt_id")
attempts = load_csv_robust(attempts_file, expected_first_col="attempt_id") if attempts_file else None
labels = load_csv_robust(labels_file, expected_first_col="image") if labels_file else None

print("grasp_dataset:", df.shape)
if attempts is not None:
    print("grasp_attempts:", attempts.shape)
if labels is not None:
    print("labels:", labels.shape)

display(df.head())
if labels is not None:
    display(labels.head())



# Cell 4: Clean and merge 

def parse_object_from_image(filename):
    name = os.path.splitext(os.path.basename(str(filename)))[0]
    parts = name.split("_")
    if len(parts) < 2:
        return "unknown", "unknown"

    obj_class = parts[0]

   
    ts_idx = None
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d{14}", p):
            ts_idx = i
            break

    if ts_idx is not None and ts_idx > 1:
        obj_instance = "_".join(parts[1:ts_idx])
    else:
        obj_instance = parts[1]

    return obj_class, obj_instance

bbox_features = [
    "bbox_w_norm", "bbox_h_norm", "bbox_area_norm",
    "aspect_ratio", "cx_norm", "cy_norm", "conf"
]


if labels is not None:
    keep_cols = ["image", "object_class", "object_instance"] + [c for c in bbox_features if c in labels.columns]
    labels_keep = labels[keep_cols].drop_duplicates("image").copy()

    drop_cols = [c for c in ["object_class", "object_instance"] + bbox_features if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore").merge(labels_keep, on="image", how="left")

else:
    parsed = df["image"].apply(parse_object_from_image)
    df["object_class"] = parsed.apply(lambda x: x[0])
    df["object_instance"] = parsed.apply(lambda x: x[1])


parsed = df["image"].apply(parse_object_from_image)
df["object_class"] = df["object_class"].fillna(parsed.apply(lambda x: x[0]))
df["object_instance"] = df["object_instance"].fillna(parsed.apply(lambda x: x[1]))


df["label"] = df["label"].astype(str).str.lower().str.strip()
df["success"] = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int)

for c in df.columns:
    if c not in ["image", "object_class", "object_instance", "label", "notes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

for c in bbox_features:
    if c not in df.columns:
        df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

print("Dataset after cleaning:", df.shape)
display(df[["image", "object_class", "object_instance", "grip_ratio", "success", "label"] + bbox_features].head())

print("\nSamples by object_instance:")
display(df.groupby(["object_class", "object_instance"]).size().reset_index(name="n"))

print("\nLabel distribution:")
display(df["label"].value_counts())



# Cell 5: Build  data for Model 1

vision_cat_features = ["object_class", "object_instance"]
initial_features = vision_cat_features + bbox_features

success_df = df[df["label"].eq("success") & df["grip_ratio"].notna()].copy()

if success_df.empty:
    raise ValueError("No success rows found. Cannot train initial grip model.")

idx = success_df.groupby("image")["grip_ratio"].idxmin()
initial_df = success_df.loc[idx].copy()
initial_df = initial_df.rename(columns={"grip_ratio": "best_grip_ratio"})
initial_df = initial_df.dropna(subset=initial_features + ["best_grip_ratio"])

print("Samples for Model 1:", initial_df.shape)
display(initial_df[["image", "object_class", "object_instance", "best_grip_ratio"] + bbox_features].head())

print("\nBest grip ratio by object_instance:")
display(initial_df.groupby(["object_class", "object_instance"])["best_grip_ratio"].agg(["count", "mean", "std", "min", "max"]))

# Cell 6: Train Model 1 - MLPRegressor

X_init = initial_df[initial_features]
y_init = initial_df["best_grip_ratio"].astype(float)
groups_init = initial_df["image"]

if len(initial_df) >= 20:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X_init, y_init, groups=groups_init))
    X_train, X_test = X_init.iloc[train_idx], X_init.iloc[test_idx]
    y_train, y_test = y_init.iloc[train_idx], y_init.iloc[test_idx]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_init, y_init, test_size=0.2, random_state=RANDOM_STATE
    )

pre_init = ColumnTransformer([
    ("cat", make_ohe(), vision_cat_features),
    ("num", StandardScaler(), bbox_features),
])

initial_model = Pipeline([
    ("preprocess", pre_init),
    ("model", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )),
])

initial_model.fit(X_train, y_train)

pred = np.clip(initial_model.predict(X_test), 0.0, 1.0)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"Model 1 initial grip predictor:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R2   = {r2:.4f}")

plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred)
plt.xlabel("True best_grip_ratio")
plt.ylabel("Predicted best_grip_ratio")
plt.title("Model 1 - Initial Grip Predictor")
plt.grid(True)
plt.show()

eval_init = X_test.copy()
eval_init["true"] = y_test.values
eval_init["pred"] = pred
eval_init["abs_error"] = np.abs(eval_init["true"] - eval_init["pred"])
display(eval_init.sort_values("abs_error", ascending=False).head(10))



# Cell 7: Build data for Model 2

sensor_features = [
    "left_sum_avg", "right_sum_avg", "contact_area_avg",
    "L_OUT_avg", "L_IN_avg", "R_IN_avg", "R_OUT_avg",
    "L_OUT_max", "L_IN_max", "R_IN_max", "R_OUT_max",
    "total_avg", "total_max",
]
control_features = ["grip_ratio"]
state_cat_features = ["object_class", "object_instance"]
state_num_features = bbox_features + control_features + sensor_features
state_features = state_cat_features + state_num_features

for c in state_num_features:
    if c not in df.columns:
        print("Missing numeric feature, filling with 0:", c)
        df[c] = 0.0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

valid_labels = ["success", "slip", "overforce", "fail"]
state_df = df[df["label"].isin(valid_labels)].copy()


state_df["label"] = state_df["label"].replace({"fail": "slip"})

state_df = state_df.dropna(subset=state_features + ["label"])

print("Samples for Model 2:", state_df.shape)
display(state_df["label"].value_counts())

print("\nMedian tactile features by label:")
display(state_df.groupby("label")[["grip_ratio", "total_avg", "total_max", "contact_area_avg"]].median())

# Cell 8: Train/test for Model 2

X_state = state_df[state_features]
y_state = state_df["label"]
groups_state = state_df["image"]

if state_df["label"].nunique() < 2:
    raise ValueError("Need at least 2 labels to train state classifier.")

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X_state, y_state, groups=groups_state))

train_df = state_df.iloc[train_idx].copy()
test_df = state_df.iloc[test_idx].copy()

print("Train label before oversampling:")
print(train_df["label"].value_counts())

max_count = train_df["label"].value_counts().max()
balanced_parts = []

for label, part in train_df.groupby("label"):
    if len(part) < max_count:
        part_up = resample(
            part,
            replace=True,
            n_samples=max_count,
            random_state=RANDOM_STATE,
        )
    else:
        part_up = part
    balanced_parts.append(part_up)

train_bal = pd.concat(balanced_parts).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nTrain label after oversampling:")
print(train_bal["label"].value_counts())

X_train_state = train_bal[state_features]
y_train_state = train_bal["label"]
X_test_state = test_df[state_features]
y_test_state = test_df["label"]

# Cell 9: Train Model 2 - MLPClassifier

pre_state = ColumnTransformer([
    ("cat", make_ohe(), state_cat_features),
    ("num", StandardScaler(), state_num_features),
])

state_model = Pipeline([
    ("preprocess", pre_state),
    ("model", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=RANDOM_STATE,
        early_stopping=False,
        validation_fraction=0.15,
        n_iter_no_change=50,
    )),
])

state_model.fit(X_train_state, y_train_state)
y_pred_state = state_model.predict(X_test_state)

print(classification_report(y_test_state, y_pred_state, digits=4))

labels_order = [x for x in ["success", "slip", "overforce"] if x in state_model.classes_]
cm = confusion_matrix(y_test_state, y_pred_state, labels=labels_order)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
disp.plot(xticks_rotation=45)
plt.title("Model 2 - Tactile State Classifier")
plt.show()


proba = state_model.predict_proba(X_test_state)
proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in state_model.classes_], index=X_test_state.index)
debug_state = test_df[["image", "object_class", "object_instance", "grip_ratio", "label"]].copy()
debug_state = pd.concat([debug_state, proba_df], axis=1)
display(debug_state.head(20))



# Cell 10: Save model bundle

out_dir = Path("/content/grasp_mlp_models")
out_dir.mkdir(exist_ok=True)


runtime_policy = {
    "initial_clip_min": 0.0,
    "initial_clip_max": 1.0,
    "slip_increase_step": 0.02,
    "overforce_decrease_step": 0.01,
    "max_adjustments": 8,
    "success_prob_threshold": 0.55,
    "slip_prob_threshold": 0.45,
    "overforce_prob_threshold": 0.45,
}

bundle = {
    "initial_model": initial_model,
    "state_model": state_model,
    "initial_features": initial_features,
    "state_features": state_features,
    "vision_cat_features": vision_cat_features,
    "bbox_features": bbox_features,
    "state_cat_features": state_cat_features,
    "state_num_features": state_num_features,
    "sensor_features": sensor_features,
    "runtime_policy": runtime_policy,
    "notes": {
        "initial_output": "best_grip_ratio",
        "state_output": "label: success/slip/overforce/fail",
        "important": "Current dataset does not contain calibrated physical force. Do not use target_force as output yet.",
    },
}

model_path = out_dir / "grasp_mlp_models.joblib"
joblib.dump(bundle, model_path)

meta = {
    k: v for k, v in bundle.items()
    if k not in ["initial_model", "state_model"]
}

with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

zip_path = "/content/grasp_mlp_models.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(model_path, arcname="grasp_mlp_models.joblib")
    z.write(out_dir / "metadata.json", arcname="metadata.json")

print("Saved model:", model_path)
print("Saved zip:", zip_path)

files.download(zip_path)



