import os
import pandas as pd
import xgboost
import numpy as np
import torch
import pickle
import ast
import re
import lmdb
import esm
import argparse
import csv
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm



import io
import torch
import torch.storage

_ori_load_from_bytes = torch.storage._load_from_bytes
def _patched_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
torch.storage._load_from_bytes = _patched_load_from_bytes

cpu_device = torch.device("cpu")
cuda_device2 = torch.device(f"cuda:6")

client = ESMC.from_pretrained("esmc_600m").to(cuda_device2)


tokenizer_path = '/path/to/prot_t5_xl_uniref50'
model_path = '/path/to/prot_t5_xl_uniref50'
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
model_t5 = T5EncoderModel.from_pretrained(model_path).to(cuda_device2)
model_t5.half()

def load_model(model_path):
    model = xgboost.XGBRegressor(tree_method='hist', predictor='cpu_predictor')
    model.load_model(model_path)
    return model
import subprocess
import pickle

import subprocess, pickle, tempfile, os

def run_esm2_in_subprocess(seq_info, site_info):
    with tempfile.NamedTemporaryFile(delete=False) as fin, tempfile.NamedTemporaryFile(delete=False) as fout:
        input_path, output_path = fin.name, fout.name
    try:
        with open(input_path, "wb") as f:
            pickle.dump({"seq": seq_info, "site": site_info}, f)

        cmd = [
            "conda", "run", "-n", "SAEKP_env_2",
            "python", "esm2_extract.py", input_path, output_path
        ]
        subprocess.run(cmd, check=True)

        with open(output_path, "rb") as f:
            esm2_feat, esm2_con = pickle.load(f)

        return esm2_feat, esm2_con
    finally:
        os.remove(input_path)
        os.remove(output_path)

def extract_protein_feature(seq_info, site_info, protein_type="ESMC_T5_ESM2", feature_source="extract"):
    if feature_source == "lmdb":
        return None, None, None
    try:
        # ---------- ESM-C ----------
        with torch.no_grad():
            esmc_protein = ESMProtein(sequence=seq_info)
            protein_tensor = client.encode(esmc_protein)
            logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            esmc_embeddings = logits_output.embeddings[0, 1:-1]
            site_cons = torch.tensor(site_info).unsqueeze(1).to(cuda_device2)
            esmc_embeddings_fusion = esmc_embeddings * site_cons
            esmc_feat = torch.mean(esmc_embeddings, dim=0)       # 1152
            esmc_con  = torch.mean(esmc_embeddings_fusion, dim=0)# 1152

        # ---------- ProtT5 ----------
        with torch.no_grad():
            processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq_info)))
            processed_seq = "<AA2fold> " + processed_seq
            inputs = tokenizer(
                processed_seq,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt"
            ).to(cuda_device2)
            t5_output = model_t5(**inputs).last_hidden_state[0, 1:-1]
        site_cons_t5 = torch.tensor(site_info).unsqueeze(1).to(cuda_device2)
        t5_embeddings_fusion = t5_output * site_cons_t5
        t5_feat = torch.mean(t5_output, dim=0)        # 1024
        t5_con  = torch.mean(t5_embeddings_fusion, 0) # 1024

        if protein_type == "ESMC":
            protein_feature = esmc_feat
            con_feature = esmc_con
            combined_feature = torch.cat([esmc_feat, esmc_con], dim=0)
        elif protein_type == "T5":
            protein_feature = t5_feat
            con_feature = t5_con
            combined_feature = torch.cat([t5_feat, t5_con], dim=0)
        elif protein_type == "ESMC_abla":
            protein_feature = esmc_feat
            con_feature = None
            combined_feature = esmc_feat
        elif protein_type == "ESMC_T5_ESM2":
            # === esm2_extract ESM2 ===
            esm2_feat_np, esm2_con_np = run_esm2_in_subprocess(seq_info, site_info)
            esm2_feat = torch.tensor(esm2_feat_np, device=cuda_device2)
            esm2_con  = torch.tensor(esm2_con_np,  device=cuda_device2)

            protein_feature = torch.cat([esmc_feat, t5_feat, esm2_feat], dim=0)
            con_feature     = torch.cat([esmc_con, t5_con, esm2_con], dim=0)
            combined_feature = torch.cat(
                [esmc_feat, esmc_con,
                 t5_feat,  t5_con,
                 esm2_feat, esm2_con],
                dim=0
            )
        else:
            raise ValueError(f"not support protein_type: {protein_type}")

        return (combined_feature.cpu().numpy(),
                protein_feature.cpu().numpy(),
                con_feature.cpu().numpy() if con_feature is not None else None)

    except Exception as e:
        print(f"fail: {str(e)}")
        return None, None, None

def extract_protein_feature_from_lmdb(protein_id, lmdb_path="/path/to/Pro_esmc_t5_lmdb"):
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        pro_key = str(protein_id).encode("ascii")
        with env.begin() as txn:
            raw = txn.get(pro_key)
            if raw is None:
                raise ValueError(f"LMDB fail protein_id: {protein_id}")
            pro_feature = pickle.loads(raw)
            pro_feature = pro_feature[:1152 + 1152]
            pro_feature = pro_feature.detach().cpu().numpy()
        env.close()
        return pro_feature
    except Exception as e:
        print(f"LMDB fail: {str(e)}")
        return None

def extract_smile_feature(smile_type="MoleBERT", smiles_id=None, lmdb_path="/path/to/mb_u2_mo"):
    try:
        if smile_type == "MoleBERT":
            env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
            if smiles_id.startswith('SMI_'):
                smi_key = str(int(smiles_id[4:])).encode("ascii")
            else:
                smi_key = str(smiles_id).encode("ascii")
            with env.begin() as txn:
                raw = txn.get(smi_key)
                if raw is None:
                    raise ValueError(f"fail smiles_id: {smiles_id}")
                smi_feature = pickle.loads(raw)
                smi_feature = smi_feature.detach().cpu().numpy()
            env.close()
            return smi_feature
        else:
            raise ValueError(f"fail smile_type: {smile_type}")
    except Exception as e:
        print(f"fail: {str(e)}")
        return None

def predict(model, x):
    y_pred_log = model.predict(x)
    y_pred = 10 ** y_pred_log
    return y_pred_log, y_pred

def get_protein_type_by_model_path(model_path: str):
    if "T5_MoleBERT" in model_path:
        return "T5"
    else:
        return "ESMC"

def main(model_path, input_csv_path, output_csv_path, protein_type="ESMC", smile_type="MoleBERT", protein_feature_source="extract"):
    models = {path: load_model(path) for path in model_paths}
    input_data = pd.read_csv(input_csv_path)
    required_columns = ['sequence', 'isomeric_smiles', 'catalysis_site', 'bind_site', 'other_site']
    if protein_feature_source == "lmdb":
        required_columns.append('protein_id')
    if not all(col in input_data.columns for col in required_columns):
        raise ValueError(f"fail: {required_columns}")
    results = []
    failed_samples = []

    for _, row in tqdm(input_data.iterrows(), total=input_data.shape[0], desc="inference progress"):
        seq_info = row['sequence']
        isomeric_smiles = row['isomeric_smiles']
        smiles_id = row['smiles_id']
        seq_len = len(seq_info)
        site_list = [0] * seq_len

        def parse_positions(cell):
            if pd.isna(cell) or str(cell).strip() == '':
                return []
            s = str(cell).strip()
            if s.startswith('[') and s.endswith(']'):
                raw_list = ast.literal_eval(s)
            else:
                raw_list = re.split(r'[;,]\s*', s)
            idx_list = []
            for token in raw_list:
                try:
                    idx_list.append(int(float(token)))
                except (ValueError, TypeError):
                    continue
            return idx_list

        for pos in parse_positions(row['catalysis_site']):
            idx = int(pos) - 1
            if 0 <= idx < seq_len:
                site_list[idx] = 3
        for pos in parse_positions(row['bind_site']):
            idx = int(pos) - 1
            if 0 <= idx < seq_len:
                site_list[idx] = 2
        for pos in parse_positions(row['other_site']):
            idx = int(pos) - 1
            if 0 <= idx < seq_len:
                site_list[idx] = 1
        site_info = site_list
        row['all_important_sites_list'] = site_info

        # 6. feature
        # 6.1. protein_feature
        if protein_feature_source == "extract":
            combined_feature, protein_feature, con_feature = extract_protein_feature(seq_info, site_info, protein_type, feature_source="extract")
        else:
            protein_id = row['protein_id']
            combined_feature = extract_protein_feature_from_lmdb(protein_id)
            protein_feature = combined_feature
            con_feature = None
        if combined_feature is None:
            failed_samples.append(row.to_dict())
            continue
        # 6.2. smile_feature
        smi_feature = extract_smile_feature(smile_type, smiles_id, lmdb_path="/path/to/sub_lmdb/mb_u2_mo/")

        if smi_feature is None:
            failed_samples.append(row.to_dict())
            continue
        # 6.3. concatenate
        x = np.concatenate((combined_feature, smi_feature))

        # 7.
        result = row.to_dict()
        for model_path, model in models.items():
            suffix = model_suffix_map[model_path]
            y_pred_log, y_pred = predict(model, np.array([x]))
            result[f'y_pred_log_{suffix}'] = y_pred_log[0]
            result[f'y_pred_{suffix}'] = y_pred[0]
        result['protein_feature'] = protein_feature[:3].tolist()
        result['smile_feature'] = smi_feature[:3].tolist()
        if con_feature is not None:
            result['con_feature'] = con_feature[:3].tolist()
        else:
            result['con_feature'] = None
        if con_feature is not None:
            concat_feature = np.concatenate((protein_feature[:3], smi_feature[:3], con_feature[:3]))
        else:
            concat_feature = np.concatenate((protein_feature[:3], smi_feature[:3]))
        result['concat_feature'] = concat_feature.tolist()
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    failed_df = pd.DataFrame(failed_samples)
    failed_csv_path = os.path.join(os.path.dirname(output_csv_path), "failed_samples.csv")
    failed_df.to_csv(failed_csv_path, index=False)

import glob
import os

def expand_model_paths(model_suffix_map):
    expanded_map = {}
    for path, suffix in model_suffix_map.items():
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "**/*.json"), recursive=True) \
                  + glob.glob(os.path.join(path, "**/*.model"), recursive=True)
            for f in files:
                base = os.path.basename(f)
                name, _ = os.path.splitext(base)
                expanded_map[f] = name
        else:
            expanded_map[path] = suffix if suffix else os.path.splitext(os.path.basename(path))[0]
    return expanded_map


if __name__ == "__main__":
    model_suffix_map = {
        "/path/to/model/kcat_1.json": "saekp_1",
    }
    model_suffix_map = expand_model_paths(model_suffix_map)
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--model_paths", type=str, nargs='+', default=list(model_suffix_map.keys()),
                        )
    parser.add_argument("--input_csv_path",
                        type=str,
                        )
    parser.add_argument("--output_csv_path",
                        type=str,
                        )
    parser.add_argument("--protein_feature_source",
                        type=str,
                        default="extract",
                        choices=["extract", "lmdb"],
                        )
    parser.add_argument("--protein_type",
                        type=str,
                        default="ESMC",
                        )
    parser.add_argument("--smile_type",
                        type=str,
                        default="MoleBERT",
                        )
    args = parser.parse_args()

    batch_jobs = [
        (
            [path for path in model_suffix_map.keys()],
            "/path/to/input/input.csv",
            "/path/to/output/output.csv",
            "ESMC_T5_ESM2"
        ),
    ]
    for model_paths, input_csv, output_csv, protein_type in batch_jobs:
        main(model_paths, input_csv, output_csv, protein_type, args.smile_type, args.protein_feature_source)


