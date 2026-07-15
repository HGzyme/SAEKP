#!/usr/bin/env python
import sys
import pickle
import torch
import re
import esm

# ====== 设备 ======
# 注意：这里用 cuda:0，因为外层会用 CUDA_VISIBLE_DEVICES 控制真实物理 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[ESM2] using device = {device}")

# ====== 加载 ESM2 模型 ======
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model = model.eval().to(device)
model = model.eval().to(device).half()
batch_converter = alphabet.get_batch_converter()

def extract_esm2(seq_info, site_info):
    # 输入格式化
    data = [("protein", seq_info)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33][0, 1:-1]  # 去掉 CLS/EOS

    # site 权重
    site_cons = torch.tensor(site_info).unsqueeze(1).to(token_reps.device)
    token_reps_fusion = token_reps * site_cons

    # 平均特征
    esm2_feat = torch.mean(token_reps, dim=0)
    esm2_con  = torch.mean(token_reps_fusion, dim=0)

    return esm2_feat.cpu().numpy(), esm2_con.cpu().numpy()

if __name__ == "__main__":
    input_path, output_path = sys.argv[1], sys.argv[2]

    # 读输入
    with open(input_path, "rb") as f:
        data = pickle.load(f)
    seq_info = data["seq"]
    site_info = data["site"]

    # 提取特征
    esm2_feat, esm2_con = extract_esm2(seq_info, site_info)

    # 写输出
    with open(output_path, "wb") as f:
        pickle.dump((esm2_feat, esm2_con), f)