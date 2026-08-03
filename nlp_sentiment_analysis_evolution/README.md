# The Evolution of NLP: Comparing Three Eras of Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-red)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 Project Overview

This project demonstrates the evolution of Natural Language Processing by implementing and comparing **three different approaches** to sentiment analysis on movie reviews:

1. **Traditional Machine Learning** - Statistical patterns using TF-IDF + Logistic Regression
2. **Deep Learning** - Sequential memory networks using PyTorch LSTM
3. **Pretrained Models** - Zero-shot inference using HuggingFace Transformers

**Assignment**: NLP Graded Assignment (20 Marks)  
**Course**: Natural Language Processing (MSc BITS)  
**Dataset**: NLTK Movie Reviews Corpus (2,000 reviews)  
**Task**: Binary Sentiment Classification (Positive/Negative)

---

## 🎯 Objectives

- Understand how NLP systems process human language
- Compare three different AI paradigms for sentiment analysis
- Evaluate trade-offs between accuracy, speed, and complexity
- Provide production deployment recommendation based on empirical findings

---

## 📂 Project Structure

```
nlp_sentiment_analysis_evolution/
├── README.md                          # This file
├── sentiment_analysis.ipynb           # Main Jupyter notebook (all tasks)
├── requirements.txt                   # Python dependencies
├── skills.md                          # NLP skills reference
├── PROJECT_PROMPT.md                  # Detailed project prompt
├── .gitignore                         # Git ignore rules
└── images/                            # Visualizations (generated)
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Upload `sentiment_analysis.ipynb` to Google Colab
2. Run the first cell to install dependencies:
   ```python
   !pip install transformers
   ```
3. Execute all cells sequentially

### Option 2: Local Environment

1. **Clone/Download** the repository
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```
4. **Execute all cells** sequentially

---

## 📊 Tasks Breakdown

### **Task 1: Data Sourcing & Preprocessing (3 Marks)**
- Load NLTK Movie Reviews dataset (2,000 reviews)
- Comprehensive text preprocessing:
  - Lowercasing, noise removal, stop-word removal
  - Punctuation and number removal
  - Negation preservation (critical for sentiment!)
- Train/test split (80/20) with stratification
- Data exploration and visualization

### **Task 2: Traditional Machine Learning (4 Marks)**
- **Vectorization**: TF-IDF with unigrams + bigrams
- **Model**: Logistic Regression
- **Evaluation**: Accuracy, F1-Score, Precision, Recall, ROC-AUC
- **Results**: ~80-85% accuracy, <3 seconds training

### **Task 3: Deep Learning Approach (5 Marks)**
- **Architecture**: PyTorch LSTM (2 layers, 128 hidden units)
- **Embedding**: Trainable nn.Embedding (100-dim)
- **Training**: 20 epochs with Adam optimizer
- **Evaluation**: Accuracy, F1-Score, training curves
- **Results**: ~82-88% accuracy, 2-10 minutes training

### **Task 4: Pretrained Model (3 Marks)**
- **Primary**: HuggingFace DistilBERT-SST-2 (transformer-based)
- **Secondary**: NLTK VADER (lexicon-based, for comparison)
- **Approach**: Zero-shot inference (no training!)
- **Evaluation**: Accuracy, F1-Score, inference time
- **Results**: HuggingFace ~85-90%, VADER ~65-75%

### **Task 5: Final Analysis (5 Marks)**
- Comprehensive comparison of all four approaches
- Domain adaptation analysis (why HuggingFace performed best)
- Evaluation metrics justification
- **Production Recommendation**: HuggingFace DistilBERT-SST-2
- Business insights and ROI analysis

---

## 📈 Results Summary

| Approach | Accuracy | F1-Score | Training Time | Inference Time | Recommendation |
|----------|----------|----------|---------------|----------------|----------------|
| **Traditional ML** | ~80-85% | ~0.80-0.85 | ~1-3s | ~1s | ⭐⭐⭐⭐ Good baseline |
| **LSTM** | ~82-88% | ~0.82-0.88 | ~2-10min | ~1-2s | ⭐⭐⭐⭐ Good accuracy |
| **HuggingFace** | ~85-90% | ~0.85-0.90 | **0s** | ~5-15s | ⭐⭐⭐⭐⭐ **BEST** |
| **VADER** | ~65-75% | ~0.65-0.75 | **0s** | <1s | ⭐⭐⭐ Fast but less accurate |

### **🏆 Winner: HuggingFace DistilBERT-SST-2**

**Why?**
- ✅ Highest accuracy (~85-90%)
- ✅ Zero training time (deploy immediately)
- ✅ No labeled data needed
- ✅ Perfect domain match (trained on movie reviews)
- ✅ Excellent generalization

---

## 🔧 Technical Stack

### **Core Libraries**
- **Python**: 3.8+
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### **NLP Libraries**
- **NLTK**: Dataset, preprocessing, VADER
- **Scikit-learn**: Traditional ML, metrics
- **PyTorch**: Deep learning framework
- **HuggingFace Transformers**: Pretrained models

### **Models Used**
- **Logistic Regression**: Scikit-learn
- **LSTM**: PyTorch (custom implementation)
- **DistilBERT-SST-2**: HuggingFace (`distilbert-base-uncased-finetuned-sst-2-english`)
- **VADER**: NLTK Sentiment Intensity Analyzer

---

## 📝 Key Findings

### **1. Domain Match is Critical**
- HuggingFace trained on movie reviews (SST-2) → Perfect match → Best performance
- VADER trained on social media → Domain mismatch → Lower performance
- **Lesson**: Always choose domain-aligned pretrained models

### **2. Pretrained Models Are Game-Changers**
- Zero training time vs hours/days for custom models
- No labeled data required
- State-of-the-art performance out-of-the-box
- **Lesson**: Don't always need to train from scratch

### **3. Trade-offs Are Inevitable**
- **Accuracy vs Speed**: HuggingFace most accurate, VADER fastest
- **Cost vs Performance**: Traditional ML cheapest, HuggingFace best ROI
- **Interpretability vs Accuracy**: Traditional ML most interpretable, HuggingFace most accurate
- **Lesson**: Choose based on business requirements

### **4. Multiple Metrics Matter**
- Accuracy alone is insufficient
- F1-Score provides balanced view
- Precision/Recall reveal error patterns
- **Lesson**: Comprehensive evaluation prevents blind spots

---

## 💼 Business Recommendation

### **Production Deployment: HuggingFace DistilBERT-SST-2**

**Rationale:**
- **Best accuracy** (5-10% improvement over alternatives)
- **Fastest deployment** (hours vs weeks)
- **No data requirements** (zero labeled samples needed)
- **Excellent ROI** (5-10x return on investment)
- **Scalable** (handles production workloads)

**When to Use Alternatives:**
- **Traditional ML**: Extremely high volume (>1M reviews/day) or real-time critical (<50ms)
- **LSTM**: Have abundant labeled data and need custom adaptation
- **VADER**: Resource-constrained environments (edge devices)

---

## 📚 Learning Outcomes

### **NLP Skills Demonstrated**
- ✅ Text preprocessing and cleaning
- ✅ Feature extraction (TF-IDF, embeddings)
- ✅ Traditional ML for NLP
- ✅ Deep learning (LSTM) implementation
- ✅ Pretrained model usage
- ✅ Domain adaptation analysis
- ✅ Model evaluation and comparison

### **Technical Skills**
- ✅ PyTorch deep learning
- ✅ HuggingFace Transformers
- ✅ Scikit-learn ML pipeline
- ✅ Data visualization
- ✅ Experiment design and analysis

### **Business Skills**
- ✅ Cost-benefit analysis
- ✅ Production deployment planning
- ✅ Risk assessment
- ✅ Stakeholder communication
- ✅ ROI calculation

---

## 🔬 Reproducibility

All experiments use **fixed random seed (42)** for reproducibility:

```python
import random
import numpy as np
import torch

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

**Same test set** used across all four approaches for fair comparison.

---

## 📖 References

### **Datasets**
- NLTK Movie Reviews Corpus
- Stanford Sentiment Treebank (SST-2) - for HuggingFace model

### **Models**
- [DistilBERT-SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

### **Libraries**
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 🎓 Course Context

**Assignment**: The Evolution of NLP – Comparing Three Eras of Sentiment Analysis  
**Course**: Natural Language Processing (MSc BITS)  
**Total Marks**: 20  
**Due Date**: August 8, 2026

**Evaluation Criteria:**
- Task 1 (3 Marks): Data preprocessing
- Task 2 (4 Marks): Traditional ML
- Task 3 (5 Marks): Deep Learning
- Task 4 (3 Marks): Pretrained models
- Task 5 (5 Marks): Analysis & recommendations

---

## 💡 Tips for Running

### **Google Colab**
1. Enable GPU: Runtime → Change runtime type → GPU
2. Install transformers: `!pip install transformers`
3. NLTK downloads are included in the notebook

### **Local Environment**
1. Use virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. For GPU support, install PyTorch with CUDA
3. Expect 2-10 minutes for LSTM training (CPU)

### **Memory Requirements**
- Minimum: 4GB RAM
- Recommended: 8GB RAM
- HuggingFace model: ~250MB download

---

## 🤝 Contributing

This is an academic assignment project. For educational purposes only.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**Student ID**: 2025EM1100102  
**Course**: Natural Language Processing  
**Institution**: BITS Pilani (MSc)  
**Semester**: 3

---

## 🙏 Acknowledgments

- NLTK for the movie reviews dataset
- HuggingFace for pretrained models
- PyTorch team for the deep learning framework
- Course instructors for the assignment design

---

## 📞 Contact

For questions or feedback about this project, please refer to the course discussion forum.

---

**Last Updated**: August 3, 2026  
**Version**: 1.0  
**Status**: ✅ Complete (All 5 tasks implemented)

---

## 🚀 Next Steps

1. **Run the notebook** end-to-end
2. **Review visualizations** and metrics
3. **Understand trade-offs** between approaches
4. **Apply learnings** to your own NLP projects
5. **Experiment** with different models and hyperparameters

---

**Happy Learning! 📚🤖**
