# 🤖 AI vs Human Content Detector 2025

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A lightweight yet powerful ML pipeline to automatically detect whether text is AI-generated or human-written.**

*Ideal for content moderation, academic integrity checks, and blog/article verification.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Getting Started](#-getting-started)
- [Model Performance](#-model-performance)
- [Usage](#-usage)
- [Author](#-author)

---

## 🧠 Overview

With the rapid rise of AI-generated content, distinguishing between human and machine-written text has become increasingly important. This project builds a **binary text classifier** using classical NLP and machine learning techniques to tackle this challenge.

**Key highlights:**
- Cleans and preprocesses raw text data
- Converts text to numerical features using **TF-IDF Vectorization**
- Trains a **Logistic Regression** classifier with high accuracy
- Saves the trained model and vectorizer as `.pkl` files for production reuse
- Includes an `app.py` script for real-time inference — no retraining needed

---

## 🎬 Demo

```
Enter text: The mitochondria is the powerhouse of the cell...

🔍 Prediction: HUMAN ✅
📊 Confidence: 91.4%
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **ML & NLP** | scikit-learn, TF-IDF, Logistic Regression |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib, seaborn |
| **Model Persistence** | joblib / pickle |
| **Environment** | Jupyter Notebook, VS Code |

---

## 📂 Dataset

| Field | Description |
|---|---|
| **File** | `ai_human_content_detection_dataset.csv` |
| **`text`** | Input text sample |
| **`label`** | Target class — `AI` or `Human` |
| **Split** | 80% Train / 20% Test |

---

## 🗂 Project Structure

```
ai-vs-human-content-detector-2025/
│
├── 📓 AI_vs_Human_Content_Detection.IPYNB   # Main notebook: EDA, preprocessing, training & evaluation
├── 📊 ai_human_content_detection_dataset.csv # Labeled dataset (AI & Human samples)
├── 🤖 logreg_model.pkl                       # Saved Logistic Regression model
├── 🔤 tfidf_vectorizer.pkl                   # Saved TF-IDF vectorizer
├── 🚀 app.py                                 # Inference script — load model & predict on new text
├── 📋 requirements.txt                       # Python dependencies
└── 📄 README.md                              # Project documentation
```

---

## ⚙️ How It Works

```
Raw Text
   │
   ▼
┌─────────────────────────────┐
│     Text Preprocessing      │
│  • Lowercasing              │
│  • Remove punctuation       │
│  • Strip extra whitespace   │
│  • (Optional) Stopwords     │
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│    TF-IDF Vectorization     │
│  Converts text → numbers    │
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│   Logistic Regression       │
│   Binary Classifier         │
│   AI  vs  Human             │
└─────────────────────────────┘
   │
   ▼
  Prediction + Confidence Score
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Musawir456/ai-vs-human-content-detector-2025.git
cd ai-vs-human-content-detector-2025
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook (Training & Analysis)

```bash
jupyter notebook "AI_vs_Human_Content_Detection.IPYNB"
```

### 5. Run the Inference App

```bash
python app.py
```

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | ~XX% |
| **Precision** | ~XX% |
| **Recall** | ~XX% |
| **F1-Score** | ~XX% |

> 📝 *Update this table with your actual evaluation results after training.*

---

## 💡 Usage

Once the model is trained, use `app.py` to predict on any new text:

```python
import joblib

model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

text = ["Your sample text goes here..."]
features = vectorizer.transform(text)
prediction = model.predict(features)

print(f"Prediction: {prediction[0]}")
```
## Screenshots:
![Project Screenshot](assets/Screenshot(1151).png)

---

## 👨‍💻 Author

<div align="center">

**Abdul Musawir**
*Machine Learning & Data Scientist*
📍 Lahore, Pakistan

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/musawir_4)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Musawir456)

</div>

---

<div align="center">

⭐ **If you found this project useful, please give it a star!** ⭐

*Made with ❤️ by Abdul Musawir*

</div>
