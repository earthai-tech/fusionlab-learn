# make_full_inputs_npz.py
import json
import os
import numpy as np

def load_npz_dict(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def save_npz(path, data):
    np.savez_compressed(path, **data)

manifest_path = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\manifest.json"
out_npz = r"E:\nature\results\nansha_GeoPriorSubsNet_stage1\artifacts\full_inputs.npz"

with open(manifest_path, "r", encoding="utf-8") as f:
    M = json.load(f)

npz_art = M["artifacts"]["numpy"]
train_path = npz_art["train_inputs_npz"]
val_path   = npz_art["val_inputs_npz"]
test_path  = npz_art["test_inputs_npz"]

train = load_npz_dict(train_path)
val   = load_npz_dict(val_path)
test  = load_npz_dict(test_path)

keys = sorted(train.keys())
full = {}
for k in keys:
    full[k] = np.concatenate([train[k], val[k], test[k]], axis=0)

save_npz(out_npz, full)
print("Saved:", out_npz)
for k, v in full.items():
    print(k, v.shape)