<!-- Add banner here -->

# SAEKP

**Introduction of SAEKP**


**Here is the framework of SAEKP**

        Download the required checkpoints from [Google Drive](https://drive.google.com/drive/folders/1qWRXV_uTdDhfmh3xznA78hjgcBSW12jo)

# Usage Recommendations

- **Who to Expect from This Project:**

  - (1). Predict *k*<sub>cat</sub> by sequences and substrate.
  - (2). Predict *K*<sub>m</sub> by sequences and substrate.
  - (3). Predict *K*<sub>i</sub> by sequences and substrate.

| Input_1 | Input_2 | Model | Output |
|--|--|--|--|
| QWER | C(=O)(C(F)(F)F)O | SAEKP for *k*<sub>cat</sub> | 0.01 s<sup>-1</sup> |
| QWER | C(=O)(C(F)(F)F)O | SAEKP for *K*<sub>m</sub> | 1.50 mM |
| QWER | C(=O)(C(F)(F)F)O | SAEKP for *K*<sub>i</sub> | 3.50 mM |

# Table of contents

- [SAEKP](#SAEKP)
- [Usage Recommendations](#usage-recommendations)
- [Table of contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contribute](#contribute)
- [Citation](#citation)

# Required Documents
[(Back to top)](#table-of-contents)

Recommended:

```text
SAEKP/
├── scripts/
├── model/
└── ...
```

# Usage
[(Back to top)](#table-of-contents)

- **To perform prediction using the deep learning model, run the following commands in the terminal:**

  - (1). Download documents

        git clone https://github.com/HGzyme/SAEKP
        cd SAEKP

  - (2). Create environment

        conda create -n SAEKP_env python=3.10
        conda activate SAEKP_env
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install -r requirements.txt
    
  - (3). Inference

        conda activate SAEKP_env
        python infer.py
    
- Example for predicting enzyme kinetic parameters from sequences and substrate by SAEKP:


# Contribute
[(Back to top)](#table-of-contents)

* <b>Peking University Shenzhen Graduate School, Shenzhen, 518055, China:</b><br/>

| Jiahe Qiu | Zongyin Lin | Ke-Wei Chen | Qiang Wang | Tian-Yu Sun | Li Yuan | Xian Zhang | Yonghong Tian | Yun-Dong Wu |
|:------:|:-------------:|:---------:|:---------------:|:------------:|:------------:|:------------:|:------------:|:------------:|


# Citation
[(Back to top)](#table-of-contents)

If you use this code or our models for your publication, please cite the original paper:

The preprint version:

 (https://www.biorxiv.org/content/10.1101/2025.04.30.651216v1)
