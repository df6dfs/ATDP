# [Paper Title / Model Name]: ATDP: Adaptive Threshold Decoupled Prompting for Few-Shot Relation Extraction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper **"[ATDP: Adaptive Threshold Decoupled Prompting for Few-Shot Relation Extraction]"** (Accepted by / Submitted to [Conference/Journal Name]).

---

## 📌 Overview

This repository provides the official source code for our proposed few-shot relation extraction framework. Key highlights include:
- **Multi-Viewer Representation (MVRE)**: Gated fusion over multiple pseudo-token views for rich relation semantics.
- **Decoupled Multi-Task Curriculum Learning**: Joint training with subject/object entity reconstruction and binary relation existence detection via virtual tokens.
- **Class-Adaptive Threshold Selection**: Dynamic confidence boundaries to effectively mitigate the false-positive rejection challenge in open/NA categories.

---

## 🛠️ Environment Setup

We recommend using Conda to manage your local environment.

```bash
# 1. Clone this repository
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repo-Name].git
cd ATDP

# 2. Create and activate conda environment
conda create -n mvre python=3.8 -y
conda activate mvre

# 3. Install requirements (adjust CUDA version to match your hardware)
pip install -r requirements.txt


# 4. Install other dependencies
pip install -r requirements.txt
