<!-- Add banner here -->

# SAEKP

**Introduction of SAEKP**

Prediction of enzyme kinetic parameters is essential for enzyme design, enzyme engineering, and metabolic pathway optimization. However, the limited generalization ability of existing prediction tools across diverse enzymatic systems restricts their practical applications. Here, we introduce SAEKP, a unified framework based on pretrained language models for predicting enzyme kinetic parameters, including enzyme turnover number (*k*<sub>cat</sub>), Michaelis constant (*K*<sub>m</sub>), and catalytic efficiency (*k*<sub>cat</sub> / *K*<sub>m</sub>), from protein sequences and substrate structures. A two-layer framework derived from SAEKP, namely EF-SAEKP, has also been proposed for robust *k*<sub>cat</sub> prediction by incorporating environmental factors, including pH and temperature. In addition, representative re-weighting methods have been systematically explored to reduce prediction errors in high-value prediction tasks. SAEKP and EF-SAEKP have been applied to enzyme discovery and directed evolution tasks, enabling the identification of new enzymes and enzyme mutants with improved activity. SAEKP provides a useful computational tool for enzyme kinetic analysis, enzyme mining, and enzyme engineering.

**Here is the framework of SAEKP**

<p align="center">
  <img src="Figures/SAEKP.png" width="800">
</p>

# Demo-Preview

- **For users who want to know what to expect in this project, as follows:**

  - (1). Predict *k*<sub>cat</sub> values from protein sequences and substrate structures.
  - (2). Predict *K*<sub>m</sub> values from protein sequences and substrate structures.
  - (3). Predict *k*<sub>cat</sub> / *K*<sub>m</sub> values from protein sequences and substrate structures.

| Input_v1 | Input_v2 | Model | Output |
|--|--|--|--|
| MSELMKLSAV...MAQR | CC(O)O | SAEKP for *k*<sub>cat</sub> | 2.75 s<sup>-1</sup> |
| MSELMKLSAV...MAQR | CC(O)O | SAEKP for *K*<sub>m</sub> | 0.36 mM |
| MSELMKLSAV...MAQR | CC(O)O | SAEKP for *k*<sub>cat</sub> / *K*<sub>m</sub> | 9.51 s<sup>-1</sup> mM<sup>-1</sup> |

# Table of contents

- [SAEKP](#SAEKP)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contribute](#contribute)
    - [Sponsor](#sponsor)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

# Prerequisites
[(Back to top)](#table-of-contents)

Notice:

- You need to download the pretrained protein language model *ProtT5-XL-UniRef50* to generate enzyme representations. The link is provided here: [ProtT5-XL-U50](https://zenodo.org/records/4644188).
- You also need to download the SAEKP models for ***k*<sub>cat</sub>**, ***K*<sub>m</sub>**, and ***k*<sub>cat</sub> / *K*<sub>m</sub>** prediction. The link is provided here: [SAEKP_model](https://huggingface.co/HanselYu/SAEKP/tree/main).
- Please place the downloaded models in the SAEKP directory.
- The pretrained molecular language model *SMILES Transformer* is included in this repository to generate substrate representations. The original implementation is available here: [SMILES Transformer](https://github.com/DSPsleeporg/smiles-transformer).

Recommended directory structure:

```text
SAEKP/
├── prot_t5_xl_uniref50/
├── SAEKP/
│   ├── SAEKP for kcat.pkl
│   ├── SAEKP for Km.pkl
│   └── SAEKP for kcat_Km.pkl
├── vocab.pkl
├── trfm_12_23000.pkl
└── ...
```

# Usage
[(Back to top)](#table-of-contents)

- **For users who want to use the deep learning model for prediction, please run these command lines at the terminal:**

  - (1). Download the SAEKP package

        git clone https://github.com/Luo-SynBioLab/SAEKP
        cd SAEKP

  - (2). Create and activate the conda environment

        conda create -n SAEKP python=3.7
        conda activate SAEKP

  - (3). Install required Python packages

        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
        pip install -r requirements.txt

- Example for predicting enzyme kinetic parameters from enzyme sequences and substrate structures by SAEKP:

**All predicted values are logarithmically transformed with base 10. Remember to apply the inverse transformation before interpreting the final predicted values.**

# Contribute
[(Back to top)](#table-of-contents)

* <b>Peking University Shenzhen Graduate School, Shenzhen, 518055, China:</b><br/>

| Jiahe Qiu | Zongyin Lin | Ke-Wei Chen | Yundong Wu |
|:------:|:-------------:|:---------:|:---------------:|:------------:|


# License
[(Back to top)](#table-of-contents)

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)

# Citation
[(Back to top)](#table-of-contents)

If you use this code or our models for your publication, please cite the original paper:

Yu, H., Deng, H., He, J. et al. SAEKP: a unified framework for the prediction of enzyme kinetic parameters. *Nature Communications* 14, 8211 (2023). [https://doi.org/10.1038/s41467-023-44113-1](https://doi.org/10.1038/s41467-023-44113-1)

The preprint version:

 (https://www.biorxiv.org/content/10.1101/2025.04.30.651216v1)

# Contact
[(Back to top)](#table-of-contents)
