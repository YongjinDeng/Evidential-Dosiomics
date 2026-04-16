# Evidential Dosiomics

[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxxx-blue)](https://doi.org/10.xxxx/xxxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Evidential Dosiomics: Decoupling and Propagating Registration Uncertainty for Robust Dosimetric Violation Prediction in Adaptive Radiotherapy.

## Overview

This repository contains the official implementation of the paper:

**"Evidential Dosiomics: Decoupling and Propagating Registration Uncertainty for Robust Dosimetric Violation Prediction in Adaptive Radiotherapy"**  
*Yongjin Deng, Shuang Wu, Yong Bao*  
*Medical Image Analysis (MedIA)*, 2026

## Key Features

- Physics-Informed Registration: PI-INR with Jacobian determinant penalty and dose-gradient-weighted regularization
- Uncertainty Decoupling: Separation of aleatoric (noise) and epistemic (anatomical volatility) uncertainties
- Penumbra-Targeted Extraction: Dual-pathway feature extraction within 20%-80% dose penumbra
- Leakage-Free Validation: Patient-level GroupKFold with worst-case pooling
- Interpretable Predictions: SHAP analysis revealing physical mechanisms

## Results

| Model | AUC | Brier Score |
|-------|-----|-------------|
| Traditional Dosiomics | 0.675 | 0.236 |
| Decoupled Dosiomics (Penumbra) | 0.741 | 0.217 |
| Global Decoupled (Whole Organ) | 0.529 | 0.260 |

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)

### Installation

git clone https://github.com/YongjinDeng/Evidential-Dosiomics.git
cd Evidential-Dosiomics
pip install -r requirements.txt

### Data Preparation

Download the Pancreatic-CT-CBCT-SEG dataset from TCIA (https://doi.org/10.7937/TCIA.ESHQ-4D90) and update the DATA_ROOT path in configuration files.

### Running the Pipeline

python code/01_PI_INR_Registration.py
python code/02_Extract_Clinical_Labels.py
python code/03_Extract_Longitudinal_Features.py
python code/04_Univariate_RISK_Analysis.py
python code/05_Ablation_Study.py
python code/06_GroupKFold_Modeling.py
python code/07_SHAP_Interpretability.py
python code/08_Temporal_Evolution.py
python code/09_Generate_Publication_Figures.py

## Repository Structure

Evidential-Dosiomics/
├── code/               # All Python scripts
├── config/             # Configuration files
├── results/            # Output directory
├── README.md
├── requirements.txt
└── LICENSE

## Citation

If you find this code useful for your research, please cite:

    @article{Deng2026Evidential,
      title={Evidential Dosiomics: Decoupling and Propagating Registration Uncertainty for Robust Dosimetric Violation Prediction in Adaptive Radiotherapy},
      author={Deng, Yongjin and Wu, Shuang and Bao, Yong},
      journal={Medical Image Analysis},
      year={2026},
      volume={xx},
      pages={xxx-xxx},
      doi={10.xxxx/xxxxxx}
    }

## Contact

- Shuang Wu: wush77@mail.sysu.edu.cn
- Yong Bao: baoyong@mail.sysu.edu.cn

## License

MIT License

## Acknowledgments

- National Natural Science Foundation of China (Grant No. 82101989)
- Guangdong Basic and Applied Basic Research Foundation (Grant No. 2019A151511117)
- The Cancer Imaging Archive (TCIA) for providing the Pancreatic-CT-CBCT-SEG dataset