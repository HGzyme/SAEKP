import os
import pandas as pd
import numpy as np
import torch
import pickle
import ast
import re

import argparse
import csv
from tqdm import tqdm

import torch
import re
import pickle
import ast
import lmdb
from tqdm import tqdm
import esm
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
parser.add_argument('--input_csv', type=str, required=True, help='Path to input protein ID + sequence file')
parser.add_argument('--output_lmdb', type=str, required=True, help='Path to output LMDB directory')
parser.add_argument('--fail_log_csv', type=str, required=True, help='Path to save failed entries')
parser.add_argument('--protein_type', type=str, required=True, help='protein_type ESMC_T5')
parser.add_argument('--esm2_type', type=str, default='3B', help='ESM-2 model scale: one of 650M, 3B, 15B (default: 3B)')
parser.add_argument('--cuda_device_1', type=str, required=True, help='cuda_device_1')
parser.add_argument('--site_column', type=str, default='all_important_sites_list', help='Column name containing site information')
args = parser.parse_args()
# 使用参数替代硬编码路径
site_data = pd.read_csv(args.input_csv)
csv_file_path = args.fail_log_csv
output_lmdb_path = args.output_lmdb
os.makedirs(output_lmdb_path, exist_ok=True)

# 设备配置
cpu_device = torch.device("cpu")
cuda_device_1 = torch.device(f"cuda:{args.cuda_device_1}")  # cuda:4

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# 在写入LMDB之前，初始化一个列表收集所有特征
all_features = []
all_ids = []
all_features_no_pooling = []
# esmc
if args.protein_type == "esmc":

    # 0. 导入环境
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    # 1. 模型初始化
    # 1.1. 初始化 ESMC (先把模型安全地加载到 CPU 再转移到cuda)
    client = ESMC.from_pretrained("esmc_600m").to(cuda_device_1)

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 创建 ESMProtein 对象 传递给他 sequence | 用 ESMC 模型对序列编码 得到tensor格式蛋白
                esmc_protein = ESMProtein(sequence=seq_info)
                protein_tensor = client.encode(esmc_protein)
                # 2. 用 ESMC 模型获取 sequence 的 logits 输出，并配置 LogitsConfig 返回emb
                logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                # 4. 提取 sequence emb (去掉特殊 token ([CLS] 和 [SEP]))
                protein_feature = logits_output.embeddings[0, 1:-1]

                # 2.2. 应用位点权重(位点信息(site_info)转换为 tensor|在维度1上添加一个维度以便广播(变为[batch_size, 1]) | 与嵌入相乘得到加权嵌入)
                site_weight = torch.tensor(site_info).unsqueeze(1).to(cuda_device_1)
                weighted_protein_feature = protein_feature * site_weight

                # 2.3. |mean pooling|mean pooling|特征合并|(1152 +1152 = 2304)
                mean_protein_feature = torch.mean(protein_feature, dim=0)
                weighted_mean_protein_feature = torch.mean(weighted_protein_feature, dim=0)
                concat_feature = torch.cat([
                    mean_protein_feature.to(cuda_device_1),
                    weighted_mean_protein_feature.to(cuda_device_1),
                ], dim=0)

                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(concat_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # ========== 新增：保存未pooling的protein_feature ==========
                # 创建_no_pooling路径
                no_pooling_lmdb_path = os.path.splitext(output_lmdb_path)[0] + "_no_pooling.lmdb"
                no_pooling_env = lmdb.open(no_pooling_lmdb_path, map_size=1099511627776)
                with no_pooling_env.begin(write=True) as np_txn:
                    serialized_np = pickle.dumps(protein_feature.cpu())
                    np_txn.put(str(id).encode(), serialized_np)
                no_pooling_env.close()

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(concat_feature.cpu())
                all_ids.append(id)
                all_features_no_pooling.append(protein_feature.cpu())  # 新增

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue

# esmc_no_pooling
if args.protein_type == "esmc_no_pooling":

    # 0. 导入环境
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    # 1. 模型初始化
    # 1.1. 初始化 ESMC (先把模型安全地加载到 CPU 再转移到cuda)
    client = ESMC.from_pretrained("esmc_600m").to(cuda_device_1)

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 创建 ESMProtein 对象 传递给他 sequence | 用 ESMC 模型对序列编码 得到tensor格式蛋白
                esmc_protein = ESMProtein(sequence=seq_info)
                protein_tensor = client.encode(esmc_protein)
                # 2. 用 ESMC 模型获取 sequence 的 logits 输出，并配置 LogitsConfig 返回emb
                logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
                # 4. 提取 sequence emb (去掉特殊 token ([CLS] 和 [SEP]))
                protein_feature = logits_output.embeddings[0, 1:-1]

                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(protein_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(protein_feature.cpu())
                all_ids.append(id)

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue

# t5
elif args.protein_type == "t5":

    # 0. 导入环境
    from transformers import T5Tokenizer, T5EncoderModel

    # 1. 模型初始化
    tokenizer_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
    model_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model_t5 = T5EncoderModel.from_pretrained(model_path).to(cuda_device_1)
    model_t5.half()
    model_t5.eval()

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 将序列中特定字符替换为 X，并在每个氨基酸间插入空格
                processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq_info)))
                # 2. 在开头添加特殊标记，方便模型识别输入类型
                processed_seq = "<AA2fold> " + processed_seq
                # 3. 用 tokenizer 对序列进行编码并返回张量 适配 GPU
                inputs = tokenizer(
                    processed_seq,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt"
                ).to(cuda_device_1)
                # 4. 在不计算梯度的上下文中获取模型输出
                with torch.no_grad():
                    protein_feature = model_t5(**inputs).last_hidden_state[0, 1:-1]

                # 2.2. 应用位点权重(位点信息(site_info)转换为 tensor|在维度1上添加一个维度以便广播(变为[batch_size, 1]) | 与嵌入相乘得到加权嵌入)
                site_weight = torch.tensor(site_info).unsqueeze(1).to(cuda_device_1)
                weighted_protein_feature = protein_feature * site_weight

                # 2.3. |mean pooling|mean pooling|特征合并|(1024 +1024 = 2408)
                mean_protein_feature = torch.mean(protein_feature, dim=0)
                weighted_mean_protein_feature = torch.mean(weighted_protein_feature, dim=0)
                concat_feature = torch.cat([
                    mean_protein_feature.to(cuda_device_1),
                    weighted_mean_protein_feature.to(cuda_device_1),
                ], dim=0)

                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(concat_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # ========== 新增：保存未pooling的protein_feature ==========
                # 创建_no_pooling路径
                no_pooling_lmdb_path = os.path.splitext(output_lmdb_path)[0] + "_no_pooling.lmdb"
                no_pooling_env = lmdb.open(no_pooling_lmdb_path, map_size=1099511627776)
                with no_pooling_env.begin(write=True) as np_txn:
                    serialized_np = pickle.dumps(protein_feature.cpu())
                    np_txn.put(str(id).encode(), serialized_np)
                no_pooling_env.close()

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(concat_feature.cpu())
                all_ids.append(id)
                all_features_no_pooling.append(protein_feature.cpu())  # 新增

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue

# t5_no_pooling
elif args.protein_type == "t5_no_pooling":

    # 0. 导入环境
    from transformers import T5Tokenizer, T5EncoderModel

    # 1. 模型初始化
    tokenizer_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
    model_path = '/share/home/qiujh/science/tools/weight/prot_t5_xl_uniref50'
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model_t5 = T5EncoderModel.from_pretrained(model_path).to(cuda_device_1)
    model_t5.half()
    model_t5.eval()

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 将序列中特定字符替换为 X，并在每个氨基酸间插入空格
                processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", seq_info)))
                # 2. 在开头添加特殊标记，方便模型识别输入类型
                processed_seq = "<AA2fold> " + processed_seq
                # 3. 用 tokenizer 对序列进行编码并返回张量 适配 GPU
                inputs = tokenizer(
                    processed_seq,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt"
                ).to(cuda_device_1)
                # 4. 在不计算梯度的上下文中获取模型输出
                with torch.no_grad():
                    protein_feature = model_t5(**inputs).last_hidden_state[0, 1:-1]

                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(protein_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(protein_feature.cpu())
                all_ids.append(id)

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue

# esm2
elif args.protein_type == "esm2":
    # 0. 导入环境
    import esm

    # 1. 模型初始化
    esm2_type = args.esm2_type.lower()
    if esm2_type == "3b":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm2_type == "15b":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    else:
        raise ValueError(f"Unsupported esm2_type '{args.esm2_type}'. Supported types: 650M, 3B, 15B")
    model = model.to(cuda_device_1).half()  # 使用半精度节省显存
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 构建输入列表(格式 = [(id, sequence)]|用于 batch_converter 处理)
                batch_inputs = [(id, seq_info)]
                # 2. 用 alphabet 提供的 batch_converter 将原始序列转为模型输入格式, 返回三个元素
                # (batch_tokens = 转换后的 token 张量|用于模型输入|形状 = [batch_size, seq_len]|(移动到指定设备)
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_inputs)
                batch_tokens = batch_tokens.to(cuda_device_1)
                # 3. 在禁用梯度计算的上下文中执行前向传播 = 节省显存+提升推理效率
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                # 4. 提取第 33 层表示 | 形状 = [batch, seq_len, hidden_dim=2560]
                token_embeddings = results["representations"][33]
                # 5. 去除特殊 token([CLS] 和 [EOS]) shape = [seq_len - 2, hidden_dim]，仅保留真实氨基酸对应位置的嵌入表示
                protein_feature = token_embeddings[0, 1:-1]

                # 2.2. 应用位点权重(位点信息(site_info)转换为 tensor|在维度1上添加一个维度以便广播(变为[batch_size, 1]) | 与嵌入相乘得到加权嵌入)
                site_weight = torch.tensor(site_info).unsqueeze(1).to(cuda_device_1)
                weighted_protein_feature = protein_feature * site_weight

                # 2.3. |mean pooling|mean pooling|特征合并|(2560 +2560 = 5120)
                mean_protein_feature = torch.mean(protein_feature, dim=0)
                weighted_mean_protein_feature = torch.mean(weighted_protein_feature, dim=0)
                concat_feature = torch.cat([
                    mean_protein_feature.to(cuda_device_1),
                    weighted_mean_protein_feature.to(cuda_device_1),
                ], dim=0)

                print(f"protein_feature.shape  = {tuple(protein_feature.shape)}")

                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(concat_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # ========== 新增：保存未pooling的protein_feature ==========
                # 创建_no_pooling路径
                no_pooling_lmdb_path = os.path.splitext(output_lmdb_path)[0] + "_no_pooling.lmdb"
                no_pooling_env = lmdb.open(no_pooling_lmdb_path, map_size=1099511627776)
                with no_pooling_env.begin(write=True) as np_txn:
                    serialized_np = pickle.dumps(protein_feature.cpu())
                    np_txn.put(str(id).encode(), serialized_np)
                no_pooling_env.close()

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(concat_feature.cpu())
                all_ids.append(id)
                all_features_no_pooling.append(protein_feature.cpu())  # 新增

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue


# esm2_no_pooling
elif args.protein_type == "esm2_no_pooling":
    # 0. 导入环境
    import esm

    # 1. 模型初始化
    esm2_type = args.esm2_type.lower()
    if esm2_type == "3b":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif esm2_type == "15b":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    else:
        raise ValueError(f"Unsupported esm2_type '{args.esm2_type}'. Supported types: 650M, 3B, 15B")
    model = model.to(cuda_device_1).half()  # 使用半精度节省显存
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # 1.2. 创建LMDB环境 | 开启一个写事务 批量写入 LMDB | 遍历所有行 tqdm 显示进度
    env = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for _, row in tqdm(site_data.iterrows(), total=site_data.shape[0], desc="Processing"):
            try:
                # 1.3. 数据提取(提取当前行id|sequence(转为字符串)|site_column(字符串转为列表)|如序列长度≠位点列表长，记录到失败文件+跳过)
                id = row['id']
                seq_info = str(row['sequence'])
                site_info = ast.literal_eval(row[args.site_column])
                if len(seq_info) != len(site_info):
                    with open(csv_file_path, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([id, seq_info, "length_mismatch"])
                    continue

                # 2.1. 特征提取
                # 1. 构建输入列表(格式 = [(id, sequence)]|用于 batch_converter 处理)
                batch_inputs = [(id, seq_info)]
                # 2. 用 alphabet 提供的 batch_converter 将原始序列转为模型输入格式, 返回三个元素
                # (batch_tokens = 转换后的 token 张量|用于模型输入|形状 = [batch_size, seq_len]|(移动到指定设备)
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_inputs)
                batch_tokens = batch_tokens.to(cuda_device_1)
                # 3. 在禁用梯度计算的上下文中执行前向传播 = 节省显存+提升推理效率
                # 用 ESM-2 模型获取指定层(33)的表示|返回一个包含各层表示的字典结构
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33])
                # 4. 提取第 33 层表示 | 形状 = [batch, seq_len, hidden_dim=2560]
                token_embeddings = results["representations"][33]
                # 5. 去除特殊 token([CLS] 和 [EOS]) shape = [seq_len - 2, hidden_dim]，仅保留真实氨基酸对应位置的嵌入表示
                protein_feature = token_embeddings[0, 1:-1]

                print(f"protein_feature.shape  = {tuple(protein_feature.shape)}")


                # 2.4. 序列化存储(将拼接后的特征移动到 CPU 并序列化为字节串 | 将序列化结果以 id 为 key 存入 LMDB)
                serialized = pickle.dumps(protein_feature.cpu())
                txn.put(str(id).encode(), serialized)

                # 2.5. 保存在pt 收集特征（CPU避免显存占用)
                all_features.append(protein_feature.cpu())
                all_ids.append(id)

            # 3. 捕获异常(出错条目记录到失败文件中 | 跳过当前条目继续下一条)
            except Exception as e:
                print(f"Error processing ID {id}: {str(e)}\n")
                with open(csv_file_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([id, seq_info, f"exception_{str(e)}"])
                continue


# 所有特征保存为.pt
pt_output_path = os.path.splitext(output_lmdb_path)[0] + ".pt"
torch.save({"features": all_features, "ids": all_ids}, pt_output_path)
print(f"所有特征处理完成，并已保存为：{pt_output_path}")

# 额外保存为.pkl
pkl_output_path = os.path.splitext(output_lmdb_path)[0] + ".pkl"
with open(pkl_output_path, "wb") as f:
    pickle.dump({"features": all_features, "ids": all_ids}, f)
print(f"所有特征处理完成，并已保存为：{pkl_output_path}")

pt_output_path_np = os.path.splitext(output_lmdb_path)[0] + "_no_pooling.pt"
torch.save({"features": all_features_no_pooling, "ids": all_ids}, pt_output_path_np)
print(f"未pooling特征已保存为：{pt_output_path_np}")

pkl_output_path_np = os.path.splitext(output_lmdb_path)[0] + "_no_pooling.pkl"
with open(pkl_output_path_np, "wb") as f:
    pickle.dump({"features": all_features_no_pooling, "ids": all_ids}, f)
print(f"未pooling特征已保存为：{pkl_output_path_np}")

try:
    feature_matrix = torch.stack(all_features)  # shape: [样本数, 2304]
    print(f"最终特征矩阵形状: {feature_matrix.shape}")
except Exception as e:
    print(f"无法堆叠特征: {str(e)}")


