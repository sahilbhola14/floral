<h1 align="center">FLORAL 🌸</h1>
<h3 align="center">Flow-Matching Operators for Residual-Augmented Learning</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2512.12749">📄 arXiv</a> |
  <a href="#overview">Overview</a> |
  <a href="#features">Features</a> |
  <a href="#features">Features</a> |
  <a href="#installation">Installation</a> |
  <a href="#citation">Citation</a> |
</p>

---

**FLORAL** is a framework for learning **neural operators** by combining
**flow matching** with **residual-augmented operator learning**.
It enables efficient and scalable surrogate modeling of
**high-dimensional physical fields**, with support for
**probabilistic inference** and **resolution-invariant evaluation**.

---

## Overview

Learning reliable surrogate models for partial differential equations (PDEs)
in data-scarce regimes remains challenging. Neural operators typically require
large amounts of high-fidelity (HF) data, while generative approaches often
sacrifice **resolution invariance**.

FLORAL formulates **flow matching directly in infinite-dimensional function
spaces**, learning a probabilistic transport from inexpensive
**low-fidelity (LF) approximations** to the manifold of HF PDE solutions via
**residual-augmented learning**. This enables uncertainty-aware inference at
arbitrary spatial resolutions *without retraining*.

---

## Features

- 🌊 **Operator-Valued Flow Matching**
  Learns probability flow ODEs in function space for stable and scalable training.

- ♻️ **Residual-Augmented Learning**
  Models probabilistic corrections from LF surrogates to HF solutions rather than
  learning the full solution operator from scratch.

- 🧠 **Neural Operator Framework**
  Designed for learning mappings between function spaces.

- 📐 **Resolution-Invariant Inference**
  Train once, evaluate at arbitrary spatial resolutions without retraining.

- 🧪 **Applications**
  PDE surrogate modeling, scientific design, inverse problems,
  and uncertainty quantification.

---

## Installation

```bash
git clone git@github.com:sahilbhola14/floral.git
cd floral
conda env create -f environment.yml
conda activate floral
./build.sh
```

---

## Citation

If you find this work useful, please cite:

```bash
@article{bhola2025floral,
  title   = {Flow matching Operators for Residual-Augmented Probabilistic Learning of Partial Differential Equations},
  author  = {Bhola, Sahil and Duraisamy, Karthik},
  journal = {arXiv preprint arXiv:2512.12749},
  year    = {2025}
}
```
