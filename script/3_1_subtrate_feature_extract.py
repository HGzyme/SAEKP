import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pandas as pd
import time
from tqdm import tqdm
import pickle
import lmdb
import torch
import csv

# —— 脚本最开头 ——
import os

# 限制 CPU 并行，避免多线程浮动
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# 强制同步 CUDA 调用，关闭 cuDNN
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import numpy as np
import torch

# 锁随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 关闭 cuDNN 非确定性算法
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False


# 2.3. 数据
import argparse
import os

# argparse 参数设置
parser = argparse.ArgumentParser(description="Extract ESMC + T5 + site features to LMDB")
parser.add_argument('--input_csv_path', type=str, required=True, help='Path to input protein ID + sequence file')
parser.add_argument('--output_lmdb_path', type=str, required=True, help='Path to output LMDB directory')
parser.add_argument('--fail_log_csv', type=str, required=True, help='Path to save failed entries')
parser.add_argument('--substrate_type', type=str, required=True, help='protein_type ESMC_T5')
parser.add_argument('--device', type=str, required=True, help='device')
args = parser.parse_args()
# 使用参数替代硬编码路径
site_data = pd.read_csv(args.input_csv_path)
id_list = site_data['id'].tolist()
smile_list = site_data['isomeric_smiles'].tolist()
# 配置参数
batch_size = 128
output_lmdb_path = args.output_lmdb_path
csv_file_path = args.fail_log_csv
os.makedirs(output_lmdb_path, exist_ok=True)

# === 新增: 通用保存函数 (将累计的特征与ID同时保存为 .pt 和 .pkl) ===
def _save_pt_pkl(all_features, all_ids, output_lmdb_path):
    """
    all_features: List[np.ndarray] 或 List[torch.Tensor]，每个元素为一条样本的一维向量
    all_ids: List[int]
    output_lmdb_path: LMDB 目录路径（函数会在同级生成 .pt 和 .pkl 文件）
    """
    if len(all_features) == 0:
        print("[save_pt_pkl] 警告：没有任何特征可保存，跳过生成 .pt/.pkl")
        return

    # 统一为 numpy，再堆叠为 (N, D)
    import numpy as _np
    import torch as _torch
    feats_np = []
    for v in all_features:
        if isinstance(v, _torch.Tensor):
            feats_np.append(v.detach().cpu().numpy())
        else:
            feats_np.append(_np.asarray(v))
    feats_np = _np.stack(feats_np, axis=0)  # (N, D)

    base = output_lmdb_path.rstrip("/\\")  # LMDB 是目录，我们直接在同级生成
    pt_output_path  = base + ".pt"
    pkl_output_path = base + ".pkl"

    # .pt
    _torch.save({"features": _torch.from_numpy(feats_np), "ids": all_ids}, pt_output_path)
    print(f"[save_pt_pkl] 所有特征已保存为：{pt_output_path} ；shape = {feats_np.shape}")

    # .pkl
    import pickle as _pickle
    with open(pkl_output_path, "wb") as f:
        _pickle.dump({"features": feats_np, "ids": all_ids}, f)
    print(f"[save_pt_pkl] 所有特征已保存为：{pkl_output_path} ；shape = {feats_np.shape}")

# morganfp_maccskeyfp morgan maccs
if args.substrate_type in {"morgan_maccs", "morgan", "maccs"}:
    # 1.1. 模型初始化
    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1073741824)
    # === 新增: 用于最终保存 .pt/.pkl 的累积容器 ===
    all_features_to_save, all_ids_to_save = [], []
    printed = False
    with env.begin(write=True) as txn:
        # 1.3. 数据提取(smile_batch 批存储 smiles) (id_batch 批存储 分子唯一标识符 id) (idx = [0, 1, 2, ...N-1])(N = smiles数)
        # 如N整除批次(完成一次批次收集)/已处理到最后一个分子 = 触发一次批处理 | 清空批处理
        smile_batch, id_batch = [], []
        for idx in tqdm(range(len(smile_list)), desc="Processing SMILES"):
            smile_batch.append(smile_list[idx])
            id_batch.append(id_list[idx])
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(smile_list):

                try:
                    # 2.1. 生成分子指纹
                    # 1. 用于存储每个分子的 Morgan 和 MACCS 指纹特征
                    morganfp_feature, maccskeyfp_feature = [], []
                    # 2. 将 SMILES 解析为 RDKit 对象
                    for smile in smile_batch:
                        mol = Chem.MolFromSmiles(smile)
                        if not mol:
                            raise ValueError(f"Invalid SMILES: {smile}")

                        # 3. 生成 Morgan 指纹特征(1024维)
                        # radius=环结构半径=4，控制原子环境的感知范围|nBits=1024=bit向量维度|输出类型= ExplicitBitVect(位向量)|转换为 float32 numpy 数组|
                        if args.substrate_type in {"morgan_maccs", "morgan"}:
                            morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
                            morganfp_feature.append(np.array(morgan, dtype=np.float32))

                        # 4. 生成 MACCS Keys 指纹(167维)(预定义的化学结构模式(如某些官能团)，表示为布尔值位向量
                        if args.substrate_type in {"morgan_maccs", "maccs"}:
                            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                            maccskeyfp_feature.append(np.array(maccs, dtype=np.float32))

                    # 5. np.stack 堆叠特征 (N, 1024) (N, 167) + np.concatenate 合并 (N, 1191)
                    feature = []
                    if args.substrate_type == "morgan":
                        feature = np.stack(morganfp_feature)
                    elif args.substrate_type == "maccs":
                        feature = np.stack(maccskeyfp_feature)
                    elif args.substrate_type == "morgan_maccs":
                        morganfp_feature = np.stack(morganfp_feature)
                        maccskeyfp_feature = np.stack(maccskeyfp_feature)
                        feature = np.concatenate([morganfp_feature, maccskeyfp_feature], axis=1)
                    if not printed:
                        print(f"Batch feature shape: {feature.shape}")
                        printed = True

                    # 6. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB(SMI_001))
                    for i, id in enumerate(id_batch):
                        txn.put(str(id).encode(), pickle.dumps(feature[i]))
                        # === 新增: 同步累计到内存，供收尾一次性保存 ===
                        all_features_to_save.append(feature[i])
                        all_ids_to_save.append(id)
                # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
                except Exception as e:
                    print(f"Error processing batch {idx // batch_size}: {str(e)}\n")
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        for id, smile in zip(id_batch, smile_batch):
                            writer.writerow([id, smile, f"exception_{str(e)}"])
                    continue
                smile_batch, id_batch = [], []
    env.close()
    # === 新增: 收尾一次性落盘 .pt 与 .pkl ===
    _save_pt_pkl(all_features_to_save, all_ids_to_save, output_lmdb_path)

# unimol_v1(512) unimol_v2(1024)
elif args.substrate_type in {"unimol_v1_v2", "unimol_v1", "unimol_v2"}:

    # 0. 导入环境
    import pandas as pd
    import lmdb
    from tqdm import tqdm
    import numpy as np
    import pickle
    import csv
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from unimol_tools import UniMolRepr

    # 1.1. 模型初始化 + 配置参数
    clf_v1 = UniMolRepr(data_type='molecule', remove_hs=False)
    clf_v2 = UniMolRepr(data_type='molecule', model_name='unimolv2', model_size='310m', remove_hs=False, use_gpu=True)  # 1.1B 310m 84m
    batch_size = 1

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1073741824)
    # === 新增: 累计容器 ===
    all_features_to_save, all_ids_to_save = [], []
    printed = False
    with env.begin(write=True) as txn:
        printed = False
        smile_batch, id_batch = [], []
        for idx in tqdm(range(len(smile_list)), desc="Processing SMILES"):
            smile_batch.append(smile_list[idx])
            id_batch.append(id_list[idx])
            if (idx + 1) % batch_size == 0 or (idx + 1) == len(smile_list):
                error_occurred = False
                try:
                    # 2.1. 生成UniMol特征
                    if args.substrate_type in {"unimol_v1_v2", "unimol_v1"}:
                        unimol_v1_feature = np.array(
                            clf_v1.get_repr(smile_batch, return_atomic_reprs=False)['cls_repr'])
                        print(f"unimol_v1_feature shape: {unimol_v1_feature.shape}")
                    if args.substrate_type in {"unimol_v1_v2", "unimol_v2"}:
                        unimol_v2_feature = np.array(
                            clf_v2.get_repr(smile_batch, return_atomic_reprs=False)['cls_repr'])
                        print(f"unimol_v2_feature shape: {unimol_v2_feature.shape}")

                    # 5. np.stack 堆叠特征 (N, 1024) (N, 167) + np.concatenate 合并 (N, 1191)
                    feature = []
                    if args.substrate_type == "unimol_v1":
                        feature = unimol_v1_feature
                    elif args.substrate_type == "unimol_v2":
                        feature = unimol_v2_feature
                    elif args.substrate_type == "unimol_v1_v2":
                        print(f"unimol_v1_feature shape before concat: {unimol_v1_feature.shape}")
                        print(f"unimol_v2_feature shape before concat: {unimol_v2_feature.shape}")
                        feature = np.concatenate([unimol_v1_feature, unimol_v2_feature], axis=1)
                    if not printed:
                        print(f"Batch feature shape: {feature.shape}")
                        printed = True

                    # 6. 序列化存储
                    for i, id in enumerate(id_batch):
                        txn.put(str(id).encode(), pickle.dumps(feature[i]))
                        # === 新增: 同步累计到内存
                        all_features_to_save.append(feature[i])
                        all_ids_to_save.append(id)
                except Exception as e:
                    if not error_occurred:
                        print(f"Error processing batch {idx // batch_size}: {str(e)}\n")
                        error_occurred = True
                    # 将失败的条目记录到 CSV 文件
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        for id, smile in zip(id_batch, smile_batch):
                            writer.writerow([id, smile, f"exception_{str(e)}"])

                finally:  # 新增 finally 块，确保无论成功/失败都清空 batch
                    smile_batch, id_batch = [], []
    env.close()
    _save_pt_pkl(all_features_to_save, all_ids_to_save, output_lmdb_path)

elif args.substrate_type in {"molt5"}:

    # ---- 1) 模型与分词器 ----
    args.molt5_model_path = "/share/home/qiujh/science/tools/weight/molt5-base-smiles2caption"
    # 额外导入（放在你的全局 import 区域）
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(args.molt5_model_path, model_max_length=512, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(args.molt5_model_path)
    device = f"cuda:{args.device}"
    model = model.to(device)
    model.eval()

    # ---- 2) LMDB 写入准备 ----
    env = lmdb.open(output_lmdb_path, map_size=1073741824)
    # === 新增: 累计容器 ===
    all_features_to_save, all_ids_to_save = [], []
    printed = False

    batch_size = 64

    with env.begin(write=True) as txn, torch.no_grad():
        smile_batch, id_batch = [], []
        for idx in tqdm(range(len(smile_list)), desc="Processing SMILES (MolT5)"):
            s = smile_list[idx]
            i = id_list[idx]

            # 允许 NaN/空字符串进入失败日志
            if not isinstance(s, str) or len(s.strip()) == 0:
                with open(csv_file_path, 'a') as f:
                    csv.writer(f).writerow([i, s, "exception_empty_or_nan_smiles"])
                continue

            smile_batch.append(s)
            id_batch.append(i)

            if (idx + 1) % batch_size == 0 or (idx + 1) == len(smile_list):
                try:
                    # ---- 3) 分词 ----
                    enc = tokenizer(
                        smile_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    input_ids = enc["input_ids"].to(device)
                    attention_mask = enc["attention_mask"].to(device)

                    # ---- 4) 仅跑 encoder ----
                    encoder = model.get_encoder()
                    enc_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                    last_hidden = enc_outputs.last_hidden_state  # (B, L, H)

                    # ---- 5) mask 平均池化 -> (B, H) ----
                    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
                    summed = (last_hidden * mask).sum(dim=1)                  # (B, H)
                    denom = mask.sum(dim=1).clamp(min=1e-6)                   # (B, 1)
                    pooled = (summed / denom).detach().cpu().numpy().astype(np.float32)  # (B, H)

                    if not printed:
                        print(f"[MolT5] batch embedding shape: {pooled.shape}")
                        printed = True

                    # ---- 6) LMDB 写入 ----
                    for j, mol_id in enumerate(id_batch):
                        txn.put(str(mol_id).encode(), pickle.dumps(pooled[j]))
                        # === 新增: 同步累计到内存
                        all_features_to_save.append(pooled[j])
                        all_ids_to_save.append(mol_id)
                except Exception as e:
                    # 整批失败 -> 全部记录
                    err = f"exception_{str(e)}"
                    print(f"[MolT5] Error at batch {idx // batch_size}: {err}")
                    with open(csv_file_path, 'a') as f:
                        w = csv.writer(f)
                        for mol_id, smi in zip(id_batch, smile_batch):
                            w.writerow([mol_id, smi, err])
                finally:
                    smile_batch, id_batch = [], []
    env.close()

    _save_pt_pkl(all_features_to_save, all_ids_to_save, output_lmdb_path)
