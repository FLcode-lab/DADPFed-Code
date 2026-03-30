# DADPFed-Code

This repository accompanies the manuscript submitted to *The Visual Computer*:

**Drift-Aware Dynamic Pruning: A Stabilising Heuristic for Heterogeneous Federated Action Recognition**

It provides the runnable implementation of the proposed methods, experiment scripts, and the dataset-construction / preprocessing pipeline needed to reproduce the reported federated learning experiments as closely as possible.

---
![Framework Diagram](./framework.png)
## Citation

If you use this code, scripts, or reconstruction pipeline in your research, please cite the corresponding manuscript:

**Zhihao Liu, Wei Guo, Jie Wu, Mengke Zhu, Jiamin Liang.**  
*Drift-Aware Dynamic Pruning: A Stabilising Heuristic for Heterogeneous Federated Action Recognition.*  
Submitted to *The Visual Computer*.

> This repository is directly related to the above manuscript.  
> Please cite the manuscript when using this code or derived artifacts.

---

## Permanent Archive

A permanent, citable archived snapshot of this repository is available on Zenodo:

**DOI:** https://doi.org/10.5281/zenodo.19325591

The Zenodo archive should be used as the stable reference version for long-term reproducibility and citation.

---

## Reproducibility Scope

This repository includes:

- the implementation of **DADPFed**
- the implementation of **DADPFed-SAM**
- federated training integration in the existing FL framework
- runnable experiment scripts
- output file conventions
- subset-construction and preprocessing support for the martial-arts benchmark subsets used in the manuscript

This repository is intended to support **transparent and reproducible evaluation** of the methods reported in the manuscript.

---

## Data Note

The study uses public benchmark datasets, including:

- **Kinetics**
- **UCF-101**
- **MNIST**
- **EMNIST**
- **CIFAR-10**
- **CIFAR-100**

The martial-arts subsets used in the manuscript (**Kinetics-MA** and **UCF101-MA**) are derived from the public Kinetics and UCF-101 benchmarks.

Due to copyright, licensing, and storage constraints, **raw video files are not redistributed** in this repository.  
To ensure reproducibility, we provide:

- subset construction logic
- category definitions
- train/test split files
- preprocessing scripts
- the federated training pipeline used by the implemented methods

These materials allow readers to reconstruct the experimental pipeline without redistributing copyrighted raw videos.

---

## Repository Structure

```text
.
├── client/
├── server/
├── scripts/
├── data/
├── out/
├── train.py
├── dataset.py
├── requirements.txt
└── framework.png
