#!/usr/bin/env python
import sys
import pickle
import torch
import re
import esm

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.eval().cuda()
batch_converter = alphabet.get_batch_converter()

def extract_esm2(seq_info, site_info):
    data = [("protein", seq_info)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_reps = results["representations"][33][0, 1:-1]

    site_cons = torch.tensor(site_info).unsqueeze(1).to(token_reps.device)
    token_reps_fusion = token_reps * site_cons

    esm2_feat = torch.mean(token_reps, dim=0)
    esm2_con  = torch.mean(token_reps_fusion, dim=0)

    return esm2_feat.cpu().numpy(), esm2_con.cpu().numpy()

if __name__ == "__main__":
    import sys, pickle
    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "rb") as f:
        data = pickle.load(f)
    seq_info = data["seq"]
    site_info = data["site"]

    esm2_feat, esm2_con = extract_esm2(seq_info, site_info)

    with open(output_path, "wb") as f:
        pickle.dump((esm2_feat, esm2_con), f)
