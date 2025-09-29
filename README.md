<h1 align="center">FLORAL 🌸</h1>
<h3 align="center">Flow-matching Operator for Residual-Augmented Learning </h3>


---

FLORAL is a novel method for learning neural operators by combining **flow matching** with **residual embeddings**, enabling efficient and scalable surrogate modeling of high-dimensional physical fields.
---

## 🚀 Features

- 🌊 **Flow-Matching Loss** — Leverages probability flow ODEsfor stable trainin
- ♻️ **Residual Embedding** — Captures discrepancy between high-and low-fidelity models
- 🧠 **Neural Operator Framework** — Built for learning mappings between function spaces
- 🧪 Suitable for PDE surrogate modeling, design problems, and uncertainty quantification

---

## 📦 Installation

```bash
git clone git@github.com:sahilbhola14/mfFlow.git
cd mfFlow
conda env create -f environment.yml
conda activate floral
./build.sh
