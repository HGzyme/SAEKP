#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import lmdb
import pickle
import torch
import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm
import argparse

import io
import torch
import torch.storage

_ori_load_from_bytes = torch.storage._load_from_bytes

def _patched_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
torch.storage._load_from_bytes = _patched_load_from_bytes


def load_lmdb_tensor(env, key: str):
    with env.begin() as txn:
        raw = txn.get(key.encode("ascii"))
        if raw is None:
            raise KeyError(f" LMDB not found key: {key}")
        value = pickle.loads(raw)
        if isinstance(value, torch.Tensor):
            return value
        elif isinstance(value, np.ndarray):
            return torch.tensor(value)
        else:
            raise TypeError(f"LMDB not support: {type(value)}")


def load_model(model_path):
    model = xgboost.XGBRegressor(tree_method='hist', predictor='cpu_predictor')
    model.load_model(model_path)
    return model

def run_inference(input_csv, protein_lmdb, smi_lmdb, model_paths, output_csv):
    # open LMDB
    env_pro = lmdb.open(protein_lmdb, readonly=True, lock=False, readahead=False)
    env_smi = lmdb.open(smi_lmdb, readonly=True, lock=False, readahead=False)

    # loading
    df = pd.read_csv(input_csv)

    # loading
    models = {os.path.basename(p): load_model(p) for p in model_paths}
    print(f"loading {len(models)} ")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="infering"):
        pro_id = row["protein_id"]
        smi_id = row["smiles_id"]

        # ==========  key ==========
        if smi_id.startswith("SMI_"):
            smi_key = str(int(smi_id[4:]))
        else:
            smi_key = str(smi_id)

        try:
            pro_feat = load_lmdb_tensor(env_pro, pro_id)

            try:
                smi_feat = load_lmdb_tensor(env_smi, smi_id)
            except:
                smi_feat = load_lmdb_tensor(env_smi, smi_key)

        except Exception as e:
            print(f"skip {pro_id} / {smi_id}: {e}")
            continue

        x = torch.cat([pro_feat, smi_feat]).numpy().reshape(1, -1)
        result = row.to_dict()

        for name, model in models.items():
            y_log = model.predict(x)[0]
            name = name[:-13] if name.endswith(".json") else name
            result[f"y_pred_log_{name}"] = y_log
            result[f"y_pred_{name}"] = 10 ** y_log


        results.append(result)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

if __name__ == "__main__":

    run_inference(
        input_csv="/path/to/csv",
        protein_lmdb="/path/to/lmdb",
        smi_lmdb="/path/to/lmdb",
        model_paths=["/path/to/model",],
        output_csv="result.csv"
    )


