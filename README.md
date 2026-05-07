<!-- Add banner here -->

# SAEKP

**Introduction of SAEKP**


**Here is the framework of SAEKP**

<p align="center">
  <img src="Figures/SAEKP.png" width="800">
</p>

# Demo-Preview

- **For users who want to know what to expect in this project, as follows:**

  - (1). Predict *k*<sub>cat</sub> values from protein sequences and substrate structures.
  - (2). Predict *K*<sub>m</sub> values from protein sequences and substrate structures.

| Input_v1 | Input_v2 | Model | Output |
|--|--|--|--|
| MSELMKLSAV...MAQR | CC(O)O | SAEKP for *k*<sub>cat</sub> | 2.75 s<sup>-1</sup> |
| MSELMKLSAV...MAQR | CC(O)O | SAEKP for *K*<sub>m</sub> | 0.36 mM |

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

Recommended directory structure:

```text
SAEKP/
├── scripts/
├── model/
└── ...
```

# Usage
[(Back to top)](#table-of-contents)

- **For users who want to use the deep learning model for prediction, please run these command lines at the terminal:**

  - (1). Download the SAEKP package

        git clone https://github.com/HGzyme/SAEKP
        cd SAEKP

  - (2). Create and activate the conda environment

        conda create -n SAEKP_env python=3.10
        conda activate SAEKP_env

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

The preprint version:

 (https://www.biorxiv.org/content/10.1101/2025.04.30.651216v1)

# Contact
[(Back to top)](#table-of-contents)
