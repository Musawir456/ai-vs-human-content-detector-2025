# AI vs Human Content Detector 2025

This project aims to automatically detect whether a given piece of text is **AI‑generated** or **human‑written**. The goal is to build a lightweight machine learning pipeline that can be easily used in real‑world applications such as content moderation, academic integrity checks, and blog/article verification.

## Project Overview

- A labeled dataset of AI‑generated and human‑written text is used to train a classifier.
- Text is cleaned and converted into numerical features using TF‑IDF vectorization.
- A Logistic Regression model is trained on these features to distinguish AI vs Human content.
- The trained model and vectorizer are saved as `.pkl` files so they can be reused in production without retraining every time.

## Tech Stack

- **Language:** Python  
- **Libraries:** scikit‑learn, pandas, numpy, matplotlib / seaborn (for analysis and plots), joblib / pickle  
- **Environment:** Jupyter Notebook, VS Code  
- **Model:** Logistic Regression with TF‑IDF features  

## Dataset

- File: `ai_human_content_detection_dataset.csv`  
- Each row contains:
  - `text`: the input text sample.
  - `label`: the target class (e.g. `AI` or `Human`).  
- The dataset is split into training and testing sets (for example 80% train, 20% test) to evaluate the model performance.

## Preprocessing & Feature Engineering

The following preprocessing steps are applied:

- Lowercasing text.
- Removing extra spaces, punctuation or special characters (depending on your code).
- Optional: removing stopwords and applying tokenization.
- Converting text into TF‑IDF vectors using `TfidfVectorizer` from scikit‑learn.

These TF‑IDF vectors are then used as input features for the Logistic Regression model.

## Model Training

- A Logistic Regression classifier is trained on the TF‑IDF feature matrix.
- Evaluation metrics such as accuracy, precision, recall, and F1‑score are computed on the test set.
- The trained model is saved as `logreg_model.pkl`.
- The fitted TF‑IDF vectorizer is saved as `tfidf_vectorizer.pkl`.

You can load these artifacts later in `app.py` to perform predictions on new text without retraining.

## Files in this Repository

- `AI_vs_Human_Content_Detection.IPYNB` – main notebook for data exploration, preprocessing, model training, and evaluation.  
- `ai_human_content_detection_dataset.csv` – dataset containing AI and human text samples.  
- `logreg_model.pkl` – saved Logistic Regression model.  
- `tfidf_vectorizer.pkl` – saved TF‑IDF vectorizer.  
- `app.py` – Python script that loads the model and vectorizer to make predictions (e.g. command‑line or web interface).  
- `README.md` – project description and usage instructions.

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Musawir456/ai-vs-human-content-detector-2025.git
cd ai-vs-human-content-detector-2025

# (Optional) create virtual environment
# python -m venv venv
# venv\Scripts\activate  # on Windows

# Install dependencies (create requirements.txt with your libraries)
pip install -r requirements.txt

# Option 1: run the notebook for training/analysis
jupyter notebook "AI_vs_Human_Content_Detection.IPYNB"

# Option 2: run the app for inference
python app.py

## Author

**Name:** Abdul Musawir  
**Role:** Machine Learning & Data Science Enthusiast  
**Location:** Lahore, Pakistan  

Feel free to connect with me:

- LinkedIn: https://www.linkedin.com/in/<your-username>  
- GitHub:  https://github.com/Musawir456  
