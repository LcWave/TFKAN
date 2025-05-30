# TFKAN: Time-Frequency KAN for Long-Term Time Series Forecasting

This repository provides the official PyTorch implementation of our paper:  
**"TFKAN: Time-Frequency KAN for Long-Term Time Series Forecasting"**

TFKAN is a dual-branch architecture that integrates Kolmogorov-Arnold Networks (KANs) into both time and frequency domains, aiming to capture global periodicity and local trends more effectively for long-term forecasting tasks.

---

## Requirements

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```
---
## Getting Started

You can download the datasets as follows:

* **Air Quality** dataset: [UCI Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
* Other six benchmark datasets: [Google Drive Folder](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

After downloading, place them into the folder:

```
./dataset/
```

To train and evaluate TFKAN, simply run:

```bash
python run_longExp.py
```

Alternatively, you can execute a predefined script (e.g., for ETTm1) on a Linux server:

```bash
bash ./scripts/ettm1.sh
```

---

## Acknowledgements

We appreciate the following GitHub repositories for their valuable codebases and datasets:

1. [Informer](https://github.com/zhouhaoyi/Informer2020)
2. [Autoformer](https://github.com/thuml/Autoformer)
3. [FEDformer](https://github.com/MAZiqing/FEDformer)
4. [FreTS](https://github.com/aikunyi/FreTS)

---
