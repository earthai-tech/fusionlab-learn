# export_full_city_payload.py
import json
import os
import numpy as np
import tensorflow as tf

from geoprior.models import GeoPriorSubsNet
from geoprior.utils import make_tf_dataset

def load_npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

stage1_manifest = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\manifest.json"
model_path = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_GeoPriorSubsNet_H3_final.keras"
full_inputs_npz = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\artifacts\full_inputs.npz"
out_payload = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\train_20260222-141331\nansha_phys_payload_fullcity.npz"

with open(stage1_manifest, "r", encoding="utf-8") as f:
    M = json.load(f)

X_full = load_npz_dict(full_inputs_npz)

# dummy targets only because make_tf_dataset expects X,y
n = X_full["coords"].shape[0]
h = X_full["coords"].shape[1]
y_dummy = {
    "subs_pred": np.zeros((n, h, 1), dtype=np.float32),
    "gwl_pred": np.zeros((n, h, 1), dtype=np.float32),
}

DYN_NAMES = M["config"]["features"]["dynamic"]
FUT_NAMES = M["config"]["features"]["future"]

ds_full = make_tf_dataset(
    X_full,
    y_dummy,
    batch_size=256,
    shuffle=False,
    mode=M["config"]["model"]["mode"],
    forecast_horizon=M["config"]["model"]["forecast_horizon_years"],
    check_npz_finite=True,
    check_finite=True,
    dynamic_feature_names=list(DYN_NAMES),
    future_feature_names=list(FUT_NAMES),
)

model = tf.keras.models.load_model(
    model_path,
    custom_objects={"GeoPriorSubsNet": GeoPriorSubsNet},
    compile=False,
)

payload = model.export_physics_payload(
    ds_full,
    max_batches=None,
    save_path=out_payload,
    format="npz",
    overwrite=True,
    metadata={
        "split": "full_city_union",
        "source_inputs_npz": full_inputs_npz,
    },
)

print("Saved:", out_payload)
print("Keys:", sorted(payload.keys()))
print("N rows:", len(payload["K"]))