
import argparse
import numpy as np
from tqdm import tqdm
import pickle
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
from pathlib import Path
import pandas as pd
# ===== 导入 Mole-BERT 本地代码 =====
import sys

MOLEBERT_CODE_DIR = "/share/home/qiujh/science/tools/DLenzyme/202606/5_infer/molebert/model"
if MOLEBERT_CODE_DIR not in sys.path:
    sys.path.insert(0, MOLEBERT_CODE_DIR)

from model import GNN, GNN_graphpred
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

# ===== 导入自己的通用函数 =====
UNIVERSAL_FUNC_DIR = "/share/home/qiujh/science/tools/universal_functions"
if UNIVERSAL_FUNC_DIR not in sys.path:
    sys.path.insert(0, UNIVERSAL_FUNC_DIR)

from inspect_lmdb import inspect_lmdb


import io
import torch
import torch.storage

# 遇到保存在指定 device (7) 的特征向量 | 而目前 device 中无7 导致的报错
# 解决 = 修改 torch 内部加载函数 _load_from_bytes 直接加载到 cpu | (定义新函数 / 修改原函数)
# (RuntimeError: Attempting to deserialize object on CUDA device 7 but torch.cuda.device_count() is 1.)
# Please use torch.load with map_location to map your storages to an existing device.)

# 强制将 PyTorch 中所有使用 _load_from_bytes 加载的 tensor storage 映射到 CPU (无论其原始设备 = CUDA / CPU)
# 加载特征/模型/权重时强制避免使用 GPU / 在缺乏 GPU 资源/缺乏指定device名的环境中有用

# 1. 定义新函数
# 1.1. 备份 PyTorch 内部的 storage 加载函数
_ori_load_from_bytes = torch.storage._load_from_bytes
# 1.2. 定义一个新函数 _patched_load_from_bytes 替代默认的 _load_from_bytes
# b=bytes 是从模型文件中提取出来的 storage 的原始二进制数据 | torch.load(io.BytesIO(b), ...) = 把这个字节数据当作一个伪文件读取并反序列化
# map_location='cpu' = 强制将数据加载到 CPU，而不是原始保存位置(如 GPU) | weights_only=False/不只加载权重，而是加载完整对象(保持原始行为)
def _patched_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
# 1.2. 修改原函数
torch.storage._load_from_bytes = _patched_load_from_bytes
# ********************
def prepare_molebert_input_csv(
    input_csv,
    dataset_root,
    raw_filename="smiles_ids.csv",
    dummy_id="SMI_000",
    dummy_smiles="CC=O",
    force_reprocess=True,
):
    """
    将用户输入的 smiles csv 复制/整理到 Mole-BERT 需要的目录结构：

        dataset_root/
            raw/
                smiles_ids.csv

    同时自动在最前面增加一行：
        SMI_000,CC=O

    目的：
        避免 Mole-BERT / PyG 在只有 1 条分子时 self.slices=None 导致报错。

    注意：
        后续保存特征时会跳过 id=0，因此 SMI_000 不会进入最终 lmdb/pt/pkl。
    """

    input_csv = Path(input_csv)
    dataset_root = Path(dataset_root)
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    # sep=None 可以兼容逗号 CSV 和 tab 分隔文件
    df = pd.read_csv(input_csv, sep=None, engine="python")

    required_cols = ["id", "isomeric_smiles"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"输入文件必须包含列: {required_cols}，但当前列为: {list(df.columns)}"
            )

    df = df[required_cols].copy()
    df["id"] = df["id"].astype(str)
    df["isomeric_smiles"] = df["isomeric_smiles"].astype(str)

    # 避免用户输入里本身已经有 SMI_000，导致重复
    df = df[df["id"] != dummy_id].copy()

    dummy_df = pd.DataFrame(
        {
            "id": [dummy_id],
            "isomeric_smiles": [dummy_smiles],
        }
    )

    out_df = pd.concat([dummy_df, df], ignore_index=True)

    output_raw_csv = raw_dir / raw_filename
    out_df.to_csv(output_raw_csv, index=False)

    print(f"[INFO] 已生成 Mole-BERT raw 输入文件: {output_raw_csv}")
    print(f"[INFO] 原始分子数: {len(df)}")
    print(f"[INFO] 加入 dummy 后分子数: {len(out_df)}")
    print(f"[INFO] dummy 分子: {dummy_id} {dummy_smiles}")

    # 非常重要：如果 processed 目录已经存在，MoleculeDataset 可能直接读取旧缓存
    if force_reprocess:
        processed_dir = dataset_root / "processed"
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
            print(f"[INFO] 已删除旧 processed 缓存: {processed_dir}")

    return str(dataset_root)

def main():
    parser = argparse.ArgumentParser(description='(图神经网络预训练的Torch实现)')
    # 1. numpy/torch/torch.cuda全局随机数/设备
    parser.add_argument('--runseed', type=int, default=0, help="随机种子(批次选择和模型初始化)")
    parser.add_argument('--device', type=int, default=3, help='which gpu to use if any (default: 0)')
    # 2. 加载模型
    parser.add_argument('--num_layer', type=int, default=5, help='GNN 消息传递层数量(默认 = 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='嵌入维度(默认 = 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='Dropout率(默认 = 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean", help='图池化方式graph level pooling(sum/mean/max/set2set/attention)')
    parser.add_argument('--JK', type=str, default="last", help='跨层节点特征如何组合(last/sum/max/concat)')
    parser.add_argument('--gnn_type', type=str, default="gin", help='GNN 类型 = gin')
    parser.add_argument('--input_model_file', type=str, default='Mole-BERT', help='预训练模型文件名')
    parser.add_argument('--pre_trained_model_path', type=str, default='/share/home/qiujh/science/tools/DLenzyme/sakpe_20250529/3_substrate_feature_2507/mole_bert/data/model_gin/Mole-BERT.pth', help='预训练模型路径')

    # 3. 特征提取
    parser.add_argument('--batch_size', type=int, default=32, help='训练输入批次大小(默认 = 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据集加载的工作节点数量')
    parser.add_argument('--dataset', type=str, default='v3-2_all', help='数据集根目录。目前仅含分类')
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='输入 smiles_ids.csv，必须包含列: id,isomeric_smiles'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Mole-BERT 数据集根目录。脚本会在其中自动创建 raw/smiles_ids.csv'
    )
    parser.add_argument('--output_lmdb_path', type=str, default='/share/home/qiujh/science/tools/DLenzyme/sakpe_20250529/3_substrate_feature_2507/mole_bert/result/molebert_copy_1', help='数据集根路径')
    parser.add_argument('--seed', type=int, default=42, help="随机种子(数据集划分)")
    parser.add_argument('--epochs', type=int, default=100, help='训练周期数(默认 = 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率(默认 = 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1, help='特征提取层的相对学习率(默认 = 1)')
    parser.add_argument('--decay', type=float, default=0, help='权重衰减(默认 = 0)')
    parser.add_argument('--split', type=str, default="scaffold", help="划分方式 随机/骨架/随机骨架= random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=1, help='是否评估训练集')
    parser.add_argument('--filename', type=str, default='', help='输出文件名')
    args = parser.parse_args()
    import os
    os.makedirs(args.output_lmdb_path, exist_ok=True)
    # 1. numpy/torch/torch.cuda全局随机数/设备
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    import random
    random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
        torch.backends.cudnn.deterministic = True  # 确保 GPU 计算确定性
        torch.backends.cudnn.benchmark = False  # 禁用非确定性优化
    device = torch.device("cpu")
    if torch.__version__ >= '1.10':
        torch.use_deterministic_algorithms(True)

    # 2. 模型初始化 | 如指定预训练模型 = 加载 | 切换评估模式(不启用 Dropout) | num_tasks = 必须的自定义变量
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset in ["hiv", "v2", "v3-2_all"]:
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name/数据集名称无效")
    model = GNN_graphpred(args.num_layer,
                          args.emb_dim,
                          num_tasks,
                          JK=args.JK,
                          drop_ratio=args.dropout_ratio,
                          graph_pooling=args.graph_pooling,
                          gnn_type=args.gnn_type)
    if not args.input_model_file == "None":
        # print('Not from scratch = 并非从零开始 = 加载预训练模型')
        model.from_pretrained(args.pre_trained_model_path)
    model.to(device)
    model.to(torch.float64)  # 使用双精度
    model.eval()

    # 3. 特征提取
    # 3.1. ①. 从 SMILES 到分子图 (加载数据集 DataLoader(torch))
    prepared_dataset_root = prepare_molebert_input_csv(
        input_csv=args.input_csv,
        dataset_root=args.dataset_root,
        raw_filename="smiles_ids.csv",
        dummy_id="SMI_000",
        dummy_smiles="CC=O",
        force_reprocess=True,
    )
    dataset = MoleculeDataset(prepared_dataset_root, dataset=args.dataset)
    raw_id_csv = Path(prepared_dataset_root) / "raw" / "smiles_ids.csv"
    raw_id_df = pd.read_csv(raw_id_csv, dtype={"id": str})
    raw_ids = raw_id_df["id"].astype(str).tolist()

    if len(raw_ids) != len(dataset):
        raise ValueError(
            f"raw_ids 数量与 dataset 数量不一致: raw_ids={len(raw_ids)}, dataset={len(dataset)}"
        )

    print(f"[INFO] 原始 ID 示例: {raw_ids[:5]}")

    generate_embedding_loader = DataLoader(dataset,
                                           batch_size=1,  # args.batch_size,
                                           shuffle=False,
                                           num_workers=0  # args.num_workers
    )
    loader_bs = generate_embedding_loader.batch_size or 1

    all_features = []
    all_ids = []

    # 3.2. ②. 消息传递(Message Passing)得到节点表示
    # 1. lmdb 存储初始化 (map_size = 数据库最大字节 = 1 GB)
    env = lmdb.open(args.output_lmdb_path, map_size=1073741824)
    # 2. 遍历批次
    for step, batch in enumerate(tqdm(generate_embedding_loader, desc="进度 = Iteration")):
        print(f"Processing batch = 当前批次 = {step}")
        # 3. 将 Batch(节点特征/边索引/边特征/图索引) 移动到指定设备
        batch = batch.to(device)
        # 4. 禁用梯度计算加速前向推理并节省显存 | 输入模型得到每个节点表示 (shape = [total_nodes, emb_dim] = (3, 300,) )
        with torch.no_grad():
            node_representation = model(
                batch.x,  # 所有节点的特征矩阵
                batch.edge_index,  # 边连接关系，形状为 [2, num_edges]
                batch.edge_attr,  # 边特征
                batch.batch)  # 将节点映射到对应图的索引向量

        # 3.3. ③. 节点表示到图表示的池化(Pooling)
        # 1. 打开一个写事务，将本批次所有图表示打包写入 lmdb(键值均为bytes) | 遍历本批次中每个图(第 i 个图)
        with env.begin(write=True) as txn:
            for i in range(len(batch.ptr) - 1):
                # 2. 获得第 i 个图的id (batch.y)(转为 int)
                raw_idx = step * loader_bs + i
                mol_id = raw_ids[raw_idx]

                # 跳过自动补进去的 dummy 分子 SMI_000
                if mol_id == "SMI_000":
                    print("[INFO] 跳过 dummy 分子 SMI_000")
                    continue
                # 3. 获得第 i 个图所有节点表示的索引
                start_idx = batch.ptr[i]
                end_idx = batch.ptr[i + 1]
                # 4. 获得第 i 个图所有节点表示 (shape =  [num_nodes_i, emb_dim] = (5, 300,) ([4:9)为例))
                node_features = node_representation[start_idx:end_idx]
                # 5. 获得第 i 个图的图表示 (shape = [emb_dim] = (300,) ) (均值池化(Mean Pooling) = 对第 i 个图所有节点表示取平均)
                mean_feature = node_features.mean(dim=0)
                graph_feature = mean_feature
                if step == 0 and i == 0:
                    torch.save(node_representation, f"node_rep_batch_{step}.pt")
                    torch.save(graph_feature, f"graph_feature_{i}.pt")
                # 6. 表示序列化 + 写入 lmdb = torch.Tensor 转为二进制字节流 | {key = id 的 ascii 字节串, value = 序列化图表示} (同 key 写入会覆盖)
                serialized_data = pickle.dumps(graph_feature, protocol=4)
                txn.put(str(mol_id).encode('ascii'), serialized_data)
                all_features.append(graph_feature.cpu())
                all_ids.append(mol_id)

    # ===== 保存完整特征到 .pt 与 .pkl =====
    pt_output_path = os.path.splitext(args.output_lmdb_path)[0] + ".pt"
    torch.save({"features": all_features, "ids": all_ids}, pt_output_path)
    print(f"所有特征已保存为：{pt_output_path}")

    pkl_output_path = os.path.splitext(args.output_lmdb_path)[0] + ".pkl"
    with open(pkl_output_path, "wb") as f:
        pickle.dump({"features": all_features, "ids": all_ids}, f)
    print(f"所有特征已保存为：{pkl_output_path}")

    try:
        feature_matrix = torch.stack(all_features)
        print(f"最终特征矩阵形状: {feature_matrix.shape}")
    except Exception as e:
        print(f"无法堆叠特征: {str(e)}")


if __name__ == "__main__":
    main()
