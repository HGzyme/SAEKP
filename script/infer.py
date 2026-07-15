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
import subprocess
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import numpy as np
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

import io
import torch
import torch.storage

_ori_load_from_bytes = torch.storage._load_from_bytes

def _patched_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
torch.storage._load_from_bytes = _patched_load_from_bytes

MAIN_CUDA_DEVICE = int(os.environ.get("MAIN_CUDA_DEVICE", "1"))

cpu_device = torch.device("cpu")
cuda_device2 = torch.device(
    f"cuda:{MAIN_CUDA_DEVICE}" if torch.cuda.is_available() else "cpu"
)

print(f"[MAIN] using cuda_device2 = {cuda_device2}")
client = ESMC.from_pretrained("esmc_600m").to(cuda_device2)
tokenizer_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
model_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
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

def run_esm2_in_subprocess(seq_info, site_info, esm2_cuda_device=0):
    with tempfile.NamedTemporaryFile(delete=False) as fin, tempfile.NamedTemporaryFile(delete=False) as fout:
        input_path, output_path = fin.name, fout.name
    try:
        # 写输入
        with open(input_path, "wb") as f:
            pickle.dump({"seq": seq_info, "site": site_info}, f)

        # 调子进程
        cmd = [
            "conda", "run", "-n", "sakpe_pytorch_251_cuda_118_esm2",
            "python", "esm2_extract.py", input_path, output_path
        ]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(esm2_cuda_device)

        print(f"[ESM2 subprocess] CUDA_VISIBLE_DEVICES={esm2_cuda_device}")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True, env=env)

        # 读输出
        with open(output_path, "rb") as f:
            esm2_feat, esm2_con = pickle.load(f)

        return esm2_feat, esm2_con

    finally:
        os.remove(input_path)
        os.remove(output_path)

# ====== 提取蛋白质特征 ======
def extract_protein_feature(
    seq_info,
    site_info,
    protein_type="ESMC_T5_ESM2",
    feature_source="extract",
    esm2_cuda_device=0,
):
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
        elif protein_type == "ESMC_T5_ESM2":
            # === 调子进程跑 ESM2 ===
            esm2_feat_np, esm2_con_np = run_esm2_in_subprocess(
                seq_info,
                site_info,
                esm2_cuda_device=esm2_cuda_device,
            )
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
            raise ValueError(f"不支持的 protein_type: {protein_type}")

        return (combined_feature.cpu().numpy(),
                protein_feature.cpu().numpy(),
                con_feature.cpu().numpy() if con_feature is not None else None)

    except Exception as e:
        print(f"蛋白质特征提取失败: {str(e)}")
        return None, None, None

# 新增：从 LMDB 提取蛋白质特征
def extract_protein_feature_from_lmdb(protein_id, lmdb_path=""):
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        # 构造 LMDB 键
        pro_key = str(protein_id).encode("ascii")
        with env.begin() as txn:
            raw = txn.get(pro_key)
            if raw is None:
                raise ValueError(f"未在 LMDB 中找到 protein_id: {protein_id}")
            pro_feature = pickle.loads(raw)
            pro_feature = pro_feature[:1152 + 1152]
            pro_feature = pro_feature.detach().cpu().numpy()  # 转换为 NumPy 数组
        env.close()
        return pro_feature
    except Exception as e:
        print(f"蛋白质特征从 LMDB 提取失败: {str(e)}")
        return None

# 提取小分子特征
def extract_smile_feature(
    smile_type="MoleBERT",
    smiles_id=None,
    isomeric_smiles=None,
    lmdb_path="",
    unimol_v2_model=None,
):
    """
    支持两种模式：

    1. MoleBERT
       从 MoleBERT LMDB 读取 300 维特征。

    2. MoleBERT_unimolv2_morgan
       在线提取：
           MoleBERT: 从当前脚本自动提取的 MoleBERT LMDB 读取
           UniMolV2: 当前行现场提取
           Morgan: 当前行现场提取
       最终拼接：
           MoleBERT(300) + UniMolV2(1024) + Morgan(1024) = 2348
    """
    try:
        # -------------------------
        # 1. 统一构造 MoleBERT key
        # -------------------------
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)

        if isinstance(smiles_id, str) and smiles_id.startswith("SMI_"):
            smi_key = str(int(smiles_id[4:])).encode("ascii")
        else:
            smi_key = str(int(smiles_id)).encode("ascii")

        with env.begin() as txn:
            raw = txn.get(smi_key)
            if raw is None:
                raise ValueError(
                    f"未在 MoleBERT LMDB 中找到 smiles_id: {smiles_id}, "
                    f"normalized_key={smi_key.decode()}, lmdb_path={lmdb_path}"
                )

            molebert_feature = pickle.loads(raw)

            if isinstance(molebert_feature, torch.Tensor):
                molebert_feature = molebert_feature.detach().cpu().numpy()
            else:
                molebert_feature = np.asarray(molebert_feature)

            molebert_feature = molebert_feature.astype(np.float32).reshape(-1)

        env.close()

        # -------------------------
        # 2. 只用 MoleBERT
        # -------------------------
        if smile_type == "MoleBERT":
            if molebert_feature.shape[0] != 300:
                print(
                    f"[WARNING] MoleBERT 维度异常: "
                    f"got={molebert_feature.shape[0]}, expected=300, smiles_id={smiles_id}"
                )
            return molebert_feature

        # -------------------------
        # 3. MoleBERT + UniMolV2 + Morgan
        # -------------------------
        elif smile_type == "MoleBERT_unimolv2_morgan":
            if isomeric_smiles is None:
                raise ValueError("MoleBERT_unimolv2_morgan 需要传入 isomeric_smiles")

            if unimol_v2_model is None:
                raise ValueError("MoleBERT_unimolv2_morgan 需要传入 unimol_v2_model")

            # -------------------------
            # 3.1 UniMolV2 在线提取
            # 尽量和旧 unimol_v2 LMDB 提取脚本保持一致
            # -------------------------
            batch_size = 1

            smile_batch = []
            id_batch = []

            # 注意：旧脚本没有 strip，所以这里也不 strip
            smile_batch.append(str(isomeric_smiles))
            id_batch.append(smiles_id)

            error_occurred = False

            try:
                unimol_v2_feature = np.array(
                    unimol_v2_model.get_repr(
                        smile_batch,
                        return_atomic_reprs=False
                    )["cls_repr"]
                )
                print(f"unimol_v2_feature shape: {unimol_v2_feature.shape}")

                feature = []
                feature = unimol_v2_feature

                print(f"Batch feature shape: {feature.shape}")

                # 旧脚本是 for i, id in enumerate(id_batch): feature[i]
                for i, id in enumerate(id_batch):
                    unimol_v2_feature = feature[i]

            except Exception as e:
                if not error_occurred:
                    print(f"Error processing UniMolV2 batch: {str(e)}\n")
                    error_occurred = True
                raise e

            finally:
                smile_batch, id_batch = [], []

            # 旧脚本保存 feature[i]，本质是一维 1024
            unimol_v2_feature = np.asarray(unimol_v2_feature).reshape(-1)

            # -------------------------
            # 3.2 Morgan 在线提取
            # -------------------------
            mol = Chem.MolFromSmiles(str(isomeric_smiles))
            if mol is None:
                raise ValueError(f"Invalid SMILES: {isomeric_smiles}")

            morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol,
                radius=4,
                nBits=1024
            )
            morgan_feature = np.array(morgan, dtype=np.float32).reshape(-1)

            # 3.3 拼接
            smi_feature = np.concatenate(
                [
                    molebert_feature,
                    unimol_v2_feature,
                    morgan_feature,
                ],
                axis=0
            ).astype(np.float32)

            # 3.4 维度检查
            if molebert_feature.shape[0] != 300:
                print(f"[WARNING] MoleBERT 维度异常: {molebert_feature.shape}")

            if unimol_v2_feature.shape[0] != 1024:
                print(f"[WARNING] UniMolV2 维度异常: {unimol_v2_feature.shape}")

            if morgan_feature.shape[0] != 1024:
                print(f"[WARNING] Morgan 维度异常: {morgan_feature.shape}")

            if smi_feature.shape[0] != 2348:
                print(
                    f"[WARNING] MoleBERT_unimolv2_morgan 总维度异常: "
                    f"got={smi_feature.shape[0]}, expected=2348, smiles_id={smiles_id}"
                )

            return smi_feature

        else:
            raise ValueError(f"不支持的 smile_type: {smile_type}")

    except Exception as e:
        print(f"小分子特征提取失败: {str(e)}")
        return None
# 预测
def predict(model, x):
    """使用模型进行预测"""
    y_pred_log = model.predict(x)
    y_pred = 10 ** y_pred_log
    return y_pred_log, y_pred

# 在 main() 前增加自动判断
def get_protein_type_by_model_path(model_path: str):
    if "T5_MoleBERT" in model_path:
        return "T5"
    else:
        return "ESMC"


def prepare_molebert_dataset_from_input_csv(
    input_data: pd.DataFrame,
    molebert_dataset_root: str,
    smiles_col: str = "isomeric_smiles",
    id_col: str = "smiles_id",
):
    """
    生成 MoleBERT 需要的:
        molebert_dataset_root/raw/smiles_ids.csv

    注意:
        1. 给 MoleBERT loader 用的文件列名必须是 id
        2. 给主推理脚本用的列名仍然是 smiles_id
    """
    if smiles_col not in input_data.columns:
        raise ValueError(f"输入 CSV 缺少必要列: {smiles_col}")

    df = input_data.copy()

    df[smiles_col] = df[smiles_col].astype(str).str.strip()

    valid_smiles = (
        df.loc[
            df[smiles_col].notna()
            & (df[smiles_col] != "")
            & (df[smiles_col].str.lower() != "nan"),
            smiles_col
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # 这里必须叫 id，因为 loader.py 里面写死读取 input_df['id']
    smiles_ids_df = pd.DataFrame({
        "id": [f"SMI_{i:03d}" for i in range(1, len(valid_smiles) + 1)],
        smiles_col: valid_smiles
    })

    raw_dir = Path(molebert_dataset_root) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    smiles_ids_csv = raw_dir / "smiles_ids.csv"
    smiles_ids_df.to_csv(smiles_ids_csv, index=False)

    # 主推理脚本里仍然保留 smiles_id 列
    smi_to_id = dict(zip(smiles_ids_df[smiles_col], smiles_ids_df["id"]))
    df[id_col] = df[smiles_col].map(smi_to_id)

    if df[id_col].isna().any():
        bad_n = df[id_col].isna().sum()
        raise ValueError(f"有 {bad_n} 行无法生成 smiles_id，请检查 isomeric_smiles 是否为空")

    print(f"[MoleBERT] 已生成去重文件: {smiles_ids_csv}")
    print(f"[MoleBERT] unique smiles 数量: {len(smiles_ids_df)}")

    return df, str(smiles_ids_csv)


def run_molebert_feature_extract(
    molebert_input_csv: str,
    molebert_dataset_root: str,
    molebert_lmdb_path: str,
    device: int = 2,
    molebert_script: str = "3_substrate_feature_molebert.py",
    force: bool = False,
):
    """
    调用新版 3_substrate_feature_mole_bert.py 生成 MoleBERT LMDB。

    新版调用方式:
        python 3_substrate_feature_mole_bert.py \
            --input_csv smiles_ids.csv \
            --dataset_root tmp_dataset \
            --output_lmdb_path molebert_lmdb \
            --device 5
    """
    molebert_lmdb_path = str(molebert_lmdb_path)

    # 如果 LMDB 已存在，默认不重复跑
    data_mdb = Path(molebert_lmdb_path) / "data.mdb"

    if (not force) and data_mdb.exists():
        print(f"[MoleBERT] LMDB 已存在，跳过重新提取: {molebert_lmdb_path}")
        return molebert_lmdb_path

    Path(molebert_lmdb_path).mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", molebert_script,
        "--input_csv", str(molebert_input_csv),
        "--dataset_root", str(molebert_dataset_root),
        "--output_lmdb_path", str(molebert_lmdb_path),
        "--device", str(device),
    ]

    print("[MoleBERT] 开始提取 MoleBERT 特征:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    if not data_mdb.exists():
        print(f"[WARNING] MoleBERT 脚本运行结束，但没有发现 {data_mdb}，请检查脚本输出。")

    return molebert_lmdb_path


def prepare_and_extract_molebert_for_inference(
    input_data: pd.DataFrame,
    molebert_dataset_root: str,
    molebert_lmdb_path: str,
    device: int = 2,
    force: bool = False,
):
    """
    一步完成:
    1. input_data[isomeric_smiles] 去重
    2. 生成 raw/smiles_ids.csv
    3. 给 input_data 添加 smiles_id
    4. 调用 MoleBERT 提取 LMDB
    """
    input_data, smiles_ids_csv = prepare_molebert_dataset_from_input_csv(
        input_data=input_data,
        molebert_dataset_root=molebert_dataset_root,
        smiles_col="isomeric_smiles",
        id_col="smiles_id",
    )

    run_molebert_feature_extract(
        molebert_input_csv=smiles_ids_csv,
        molebert_dataset_root=molebert_dataset_root,
        molebert_lmdb_path=molebert_lmdb_path,
        device=device,
        force=force,
    )

    return input_data, molebert_lmdb_path

# 主函数
def main(
    model_paths,
    input_csv_path,
    output_csv_path,
    protein_type="ESMC",
    smile_type="MoleBERT",
    protein_feature_source="extract",
    smile_feature_source="extract",
    molebert_dataset_root="...",
    molebert_lmdb_path="...",
    molebert_device=2,
    esm2_cuda_device=0,
    save_every=2,
    feature_pt_path=None,
    force_molebert=False,
):
    # 1. 加载多个模型
    models = {path: load_model(path) for path in model_paths}
    # 2. 加载数据
    input_data = pd.read_csv(input_csv_path)
    if feature_pt_path is None:
        feature_pt_path = os.path.splitext(output_csv_path)[0] + "_concat_features.pt"
    # 如果小分子特征来源是 extract，则自动:
    # 1. 基于 isomeric_smiles 生成 smiles_ids.csv
    # 2. 运行 3_substrate_feature_mole_bert.py
    # 3. 给 input_data 添加 smiles_id
    unimol_v2_model = None

    if smile_type in {"MoleBERT", "MoleBERT_unimolv2_morgan"} and smile_feature_source == "extract":
        input_data, molebert_lmdb_path = prepare_and_extract_molebert_for_inference(
            input_data=input_data,
            molebert_dataset_root=molebert_dataset_root,
            molebert_lmdb_path=molebert_lmdb_path,
            device=molebert_device,
            force=force_molebert,
        )

        if smile_type == "MoleBERT_unimolv2_morgan":
            from unimol_tools import UniMolRepr

            print("[UniMolV2] 初始化 UniMolV2，用于在线提取小分子特征")
            unimol_v2_model = UniMolRepr(
                data_type="molecule",
                model_name="unimolv2",
                model_size="310m",
                remove_hs=False,
                use_gpu=True
            )

    # 3. 检查必要列
    required_columns = ['id', 'sequence', 'isomeric_smiles', 'smiles_id', 'catalysis_site', 'bind_site', 'other_site']
    # 新增：如果使用 LMDB 特征，检查 protein_id 列
    if protein_feature_source == "lmdb":
        required_columns.append('protein_id')
    if not all(col in input_data.columns for col in required_columns):
        raise ValueError(f"输入 CSV 文件缺少必要列: {required_columns}")
    # 4. 准备结果列表和失败样本列表
    results = []
    failed_samples = []
    concat_feature_dict = {}

    def save_current_outputs():
        # 保存成功预测结果
        pd.DataFrame(results).to_csv(output_csv_path, index=False)

        # 保存失败预测结果
        failed_csv_path = os.path.join(os.path.dirname(output_csv_path), "failed_samples.csv")
        pd.DataFrame(failed_samples).to_csv(failed_csv_path, index=False)

        # 保存拼接后的完整特征向量
        torch.save(concat_feature_dict, feature_pt_path)

        print(
            f"[SAVE] 当前已保存: success={len(results)}, "
            f"failed={len(failed_samples)}, "
            f"feature_n={len(concat_feature_dict)}"
        )
        print(f"[SAVE] result_csv = {output_csv_path}")
        print(f"[SAVE] feature_pt = {feature_pt_path}")

    # 5. 遍历数据并处理
    for infer_i, (_, row) in enumerate(
            tqdm(input_data.iterrows(), total=input_data.shape[0], desc="推理进度"),
            start=1
    ):
        seq_info = row['sequence']
        isomeric_smiles = row['isomeric_smiles']
        smiles_id = row['smiles_id']
        # 根据三种 site 列自动构造权重列表
        seq_len = len(seq_info)
        site_list = [0] * seq_len

        def parse_positions(cell):
            """
            将单元格内容解析为整数索引列表。
            支持格式：
                • 空值 / 空字符串         → []
                • 逗号或分号分隔的串       → "5; 12.0, 20"
                • Python 列表字面量       → "[3, 7.0, 11]"
            返回：List[int]   （已去掉小数点并剔除非法项）
            """
            if pd.isna(cell) or str(cell).strip() == '':
                return []
            s = str(cell).strip()
            # ① Python 列表格式
            if s.startswith('[') and s.endswith(']'):
                raw_list = ast.literal_eval(s)
            else:
                # ② 普通分隔符
                raw_list = re.split(r'[;,]\s*', s)
            idx_list = []
            for token in raw_list:
                try:
                    # 先转 float 再转 int，容忍 "12.0" 这类写法
                    idx_list.append(int(float(token)))
                except (ValueError, TypeError):
                    # 跳过无法解析的条目，并可按需记录日志
                    continue
            return idx_list

        # ---- 生成位点权重 ----
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

        # 6. 构建特征
        # 6.1. 提取蛋白质特征
        # 修改：根据 protein_feature_source 选择特征提取方式
        if protein_feature_source == "extract":
            combined_feature, protein_feature, con_feature = extract_protein_feature(
                seq_info,
                site_info,
                protein_type,
                feature_source="extract",
                esm2_cuda_device=esm2_cuda_device,
            )
        else:  # lmdb
            # 新增：从 LMDB 提取蛋白质特征
            protein_id = row['protein_id']
            combined_feature = extract_protein_feature_from_lmdb(protein_id)
            protein_feature = combined_feature  # 假设 LMDB 特征已经是 combined_feature 格式
            con_feature = None  # LMDB 特征不区分 con_feature
        # combined_feature, protein_feature, con_feature = extract_protein_feature(seq_info, site_info, protein_type)
        if combined_feature is None:
            failed_samples.append(row.to_dict())
            continue
        # 6.2. 提取小分子特征
        smi_feature = extract_smile_feature(
            smile_type=smile_type,
            smiles_id=smiles_id,
            isomeric_smiles=isomeric_smiles,
            lmdb_path=molebert_lmdb_path,
            unimol_v2_model=unimol_v2_model,
        )
        if smi_feature is None:
            failed_samples.append(row.to_dict())
            continue
        # 6.3. 拼接特征
        x = np.concatenate((combined_feature, smi_feature))
        sample_id = row["id"]
        # 尽量保持 id 为原始整数，比如 1/2/3/4
        try:
            if float(sample_id).is_integer():
                sample_id = int(sample_id)
        except Exception:
            sample_id = str(sample_id)
        concat_feature_dict[sample_id] = torch.tensor(x, dtype=torch.float32)
        # 7. 使用多个模型预测
        result = row.to_dict()
        for col in ['protein_feature', 'smile_feature', 'con_feature', 'concat_feature']:
            result.pop(col, None)

        for model_path, model in models.items():
            suffix = model_suffix_map[model_path]
            y_pred_log, _ = predict(model, np.array([x]))
            result[f'y_pred_log_{suffix}'] = y_pred_log[0]

        log_fold_cols = [f'y_pred_log_fold_{i}' for i in range(1, 11)]
        result['y_pred_log_average'] = float(
            np.mean([result[col] for col in log_fold_cols])
        )

        results.append(result)
        if save_every > 0 and infer_i % save_every == 0:
            save_current_outputs()


    save_current_outputs()

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
        "KCAT/kcat_1": "fold_1",
        "KCAT/kcat_2": "fold_2",
        "KCAT/kcat_3": "fold_3",
        "KCAT/kcat_4": "fold_4",
        "KCAT/kcat_5": "fold_5",
        "KCAT/kcat_6": "fold_6",
        "KCAT/kcat_7": "fold_7",
        "KCAT/kcat_8": "fold_8",
        "KCAT/kcat_9": "fold_9",
        "KCAT/kcat_10": "fold_10",
    }
    model_suffix_map = expand_model_paths(model_suffix_map)
    parser = argparse.ArgumentParser(description="5_infer")
    parser.add_argument("--model_paths", type=str, nargs='+', default=list(model_suffix_map.keys()))
    parser.add_argument("--input_csv_path", type=str,default="/path/to/input.csv",)
    parser.add_argument("--output_csv_path", type=str, default="/path/to/output.csv",)
    parser.add_argument("--protein_feature_source", type=str, default="extract", choices=["extract", "lmdb"])
    parser.add_argument("--protein_type", type=str, default="ESMC_T5_ESM2",)
    parser.add_argument("--smile_type", type=str, default="MoleBERT_unimolv2_morgan",)
    parser.add_argument("--smile_feature_source", type=str, default="extract", choices=["extract", "lmdb"])
    parser.add_argument("--molebert_dataset_root", type=str, default="/share/home/qiujh/science/tools/DLenzyme/202606/5_infer/molebert/1_dataset/0_check/test")
    parser.add_argument("--molebert_lmdb_path", type=str, default="/share/home/qiujh/science/tools/DLenzyme/202606/5_infer/molebert/0_check/2_lmdb/test")
    parser.add_argument("--molebert_device", type=int, default=2)
    parser.add_argument("--force_molebert", action="store_true")
    args = parser.parse_args()
    batch_jobs = [
        {
            "model_paths": list(model_suffix_map.keys()),
            "input_csv": "/path/to/1_input_for_test.csv",
            "output_csv": "/path/to/2_output_for_test.csv",
            "protein_type": "ESMC_T5_ESM2",
            "protein_feature_source": "extract",
            "smile_type": "MoleBERT_unimolv2_morgan",
            "smile_feature_source": "extract",
            "molebert_device": 0,
            "esm2_cuda_device": 2,
            "save_every": 2,
            "feature_pt_path": "/path/to/1_input_for_test.pt",
            "force_molebert": True,
        },
    ]
    for job in batch_jobs:
        main(
            model_paths=job["model_paths"],
            input_csv_path=job["input_csv"],
            output_csv_path=job["output_csv"],
            protein_type=job["protein_type"],
            smile_type=job["smile_type"],
            protein_feature_source=job["protein_feature_source"],
            smile_feature_source=job["smile_feature_source"],
            # molebert_dataset_root=job["molebert_dataset_root"],
            # molebert_lmdb_path=job["molebert_lmdb_path"],
        # batch_jobs 中没有设置时，使用 argparse 的默认值
        molebert_dataset_root=job.get(
            "molebert_dataset_root",
            args.molebert_dataset_root
        ),
        molebert_lmdb_path=job.get(
            "molebert_lmdb_path",
            args.molebert_lmdb_path
        ),
            molebert_device=job["molebert_device"],
            esm2_cuda_device=job["esm2_cuda_device"],
            save_every=job["save_every"],
            feature_pt_path=job["feature_pt_path"],
            force_molebert=job["force_molebert"],
        )
