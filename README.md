<h1 align="center">FLORAL 🌸</h1>
<h3 align="center">Flow-matching Operator for Residual-Augmented Learning</h3>

---

FLORAL is a novel method for learning neural operators by combining **flow matching** with **residual embeddings**, enabling efficient and scalable surrogate modeling of high-dimensional physical fields.

---

## 🚀 Features

- 🌊 **Flow-Matching Loss** — Leverages probability flow ODEs for stable training
- ♻️ **Residual Embedding** — Captures discrepancy between base and target operator
- 🧠 **Neural Operator Framework** — Built for learning mappings between function spaces
- 🧪 Suitable for PDE surrogate modeling, design problems, and uncertainty quantification

---

## 📦 Installation

```bash
git clone git@github.com:sahilbhola14/floral.git
cd floral
conda env create -f environment.yml
conda activate floral
./build.sh
