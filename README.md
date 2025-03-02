# RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression

This repository implements "RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression" as presented in the
ICLR 2025 paper. RAG-SR is a novel framework that integrates evolutionary feature construction with neural networks for
symbolic regression without the need for pre-training.

## 🌟 Overview

RAG-SR addresses limitations in existing neural symbolic regression approaches by:

- 🔍 Using a pre-training-free paradigm that adaptively generates symbolic trees
- 📚 Implementing retrieval-augmented generation to mitigate language model hallucinations
- 🔄 Employing scale-invariant data augmentation for improved robustness
- 🧩 Using masked contrastive loss to better align semantics with symbolic expressions

## ✨ Key Features

- **Semantic Descent Algorithm** 📉: Optimizes symbolic models using online supervised learning
- **Retrieval-Augmented Generation** 🔎: Reduces hallucination by leveraging searched symbolic expressions
- **Scale-Invariant Data Augmentation** 📊: Exploits invariant properties of feature construction
- **Masked Contrastive Loss** 🎯: Aligns embeddings of desired semantics with retrieved expressions

## 🚀 Installation

**Note: This repository is provided for demonstration purposes only. The actual implementation relies on
the [EvolutionaryForest](https://github.com/hengzhe-zhang/EvolutionaryForest.git) package.**

To use this code:

```bash
# Install the required base package
pip install git+https://github.com/hengzhe-zhang/EvolutionaryForest.git

# Clone this repository
git clone https://github.com/yourusername/RAG-SR.git
cd RAG-SR
```

## 💻 Usage

Basic usage example:

```python
from rag_sr import RAGSRRegressor
from sklearn.datasets import load_boston

# Load dataset
data = load_boston()
X, y = data.data, data.target

# Initialize and run RAG-SR
model = RAGSRRegressor(
    n_gen=100,
    n_pop=200,
    gene_num=10
)

# Train the model
model.fit(X, y, categorical_features=np.zeros(X.shape[1]))

# Make predictions
predictions = model.predict(X)
```

## ⚙️ Parameters

- `n_pop`: Size of the population in the evolutionary algorithm
- `n_gen`: Maximum number of generations
- `neural_pool`: Probability of using neural generation vs retrieval
- `gene_num`: Number of trees (features) in each solution

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2025ragsr,
    title = {RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression},
    author = {Zhang, Hengzhe and Chen, Qi and Xue, Bing and Banzhaf, Wolfgang and Zhang, Mengjie},
    booktitle = {International Conference on Learning Representations},
    year = {2025}
}
```

## 📚 References

- [EvolutionaryForest](https://github.com/hengzhe-zhang/EvolutionaryForest)
- [Paper Link](https://openreview.net/forum?id=NdHka08uWn) (RAG-SR: Retrieval-Augmented Generation for Neural Symbolic
  Regression)

## 📄 License

This code is released under the MIT License.