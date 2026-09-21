# ATDP: Adaptive Threshold Decoupled Prompting for Few-Shot Relation Extraction

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

Official PyTorch implementation of the paper **"[ATDP: Adaptive Threshold Decoupled Prompting for Few-Shot Relation Extraction]"** (Accepted by / Submitted to [Conference/Journal Name]).

---

## 📌 Overview

This repository provides the official source code for our proposed few-shot relation extraction framework. Key highlights include:
- **Multi-Viewer Representation (MVRE)**: Gated fusion over multiple pseudo-token views for rich relation semantics.
- **Decoupled Multi-Task Curriculum Learning**: Joint training with subject/object entity reconstruction and binary relation existence detection via virtual tokens.
- **Class-Adaptive Threshold Selection**: Dynamic confidence boundaries to effectively mitigate the false-positive rejection challenge in open/NA categories.

---
## 📂 Repository Structure
```text
.
├── dataset/                    # Preprocessed K-shot benchmark splits
│   ├── semeval/
│   │   └── k-shot/             # Subfolders: 1-1, 5-1, 16-1, etc.
│   ├── tacred/
│   └── tacrev/
├── models/
│   ├── roberta-large/          # Hugging Face backbone weights and tokenizer configs
│   └── ...
├── lit_models/                 # PyTorch Lightning modules (BertLitModel, BaseLitModel)
├── data/                       # Dataset processing and pipeline loading
├── run_semeval.sh              # Execution script for SemEval-2010 Task 8
├── run_tacred.sh               # Execution script for TACRED
├── run_tacrev.sh               # Execution script for TACREV
├── main.py                     # Main entry script for training & evaluation
├── requirements.txt            # Python environment dependencies
└── README.md
```
---

## 🛠️ Environment Setup

We recommend using Conda to manage your local environment.

```bash
# 1. Clone this repository
git clone https://github.com/df6dfs/ATDP.git
cd ATDP

# 2. Create and activate conda environment
conda create -n mvre python=3.8 -y
conda activate mvre

# 3. Install other dependencies
pip install -r requirements.txt
```
---
## How to run

# Initialize the answer words

Use the comand below to get the answer words to use in the training.

```shell
python get_label_word.py --model_name_or_path bert-large-uncased  --dataset_name semeval
```

The `{answer_words}.pt`will be saved in the dataset, you need to assign the `model_name_or_path` and `dataset_name` in the `get_label_word.py`.

## Split few-shot dataset

Download the data first, and put it to `dataset` folder. Run the comand below, and get the few shot dataset.

```shell
python generate_k_shot.py --data_dir ./dataset --k 1 --dataset semeval
cd dataset
cd semeval
cp rel2id.json val.txt test.txt ./k-shot/1-1
```
You need to modify the `k` and `dataset` to assign k-shot and dataset. Here we default seed as 1,2,3,4,5 to split each k-shot, you can revise it in the `generate_k_shot.py`

## Run
```bash
bash scripts/run_semeval.sh 
bash scripts/run_tacred.sh
bash scripts/run_tacrev.sh
```
