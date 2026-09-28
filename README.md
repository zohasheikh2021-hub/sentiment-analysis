# AI-Powered Sentiment Analysis System

## Project overview
A traditional NLP + machine-learning system that reads a customer review and classifies it as **Positive**, **Negative** or **Neutral**. The trained model is packaged as one scikit-learn `Pipeline`, saved with `joblib`, and used by a Streamlit web app. No deep learning, transformers, LLMs or external APIs are used.

## Problem statement
Reading and labelling thousands of reviews by hand is slow. The goal is to build an end-to-end pipeline that takes raw review text, cleans it, converts it to numbers, trains and evaluates classifiers, selects a model and serves it in a simple web app.

## Objectives
- Inspect and clean the dataset; explore it with EDA
- Apply explainable text preprocessing
- Extract features with TF-IDF (Bag of Words as a baseline)
- Compare Logistic Regression, Multinomial Naive Bayes and Linear SVM
- Use a stratified train/test split, cross-validation and GridSearchCV
- Save the full pipeline and use it in a Streamlit app without retraining

## Dataset description
`data/reviews.csv` (provided file, 10,000 rows x 5 columns, no missing values, no duplicate rows or ids).

| Column | Description | Used for modelling? |
|---|---|---|
| `review_id` | Unique id | No (no predictive meaning) |
| `review_text` | Review text | **Yes - the input feature** |
| `sentiment` | Positive / Negative / Neutral | **Yes - the target** |
| `category` | Product category (8 values) | EDA only |
| `source` | Feedback source (4 values) | EDA only |

Classes are balanced (Positive 3,334, Negative 3,333, Neutral 3,333).

**Important finding:** the 10,000 rows contain only **300 unique review texts** (repeated with different ids/categories/sources; no text has conflicting labels). Splitting the raw rows randomly would put identical reviews in both train and test and inflate the score (the notebook demonstrates this: 100% of a naive test set were exact copies of training reviews). The project therefore keeps **one row per unique text (300 rows)** for modelling: 240 train / 60 test. The 10,000-row data is still used for EDA.

## Technologies used
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Streamlit, Jupyter.

## Project workflow
Load -> inspect -> clean -> EDA -> preprocess -> stratified split -> TF-IDF -> compare models (5-fold CV) -> GridSearchCV -> select final model (CV only) -> evaluate once on the test set -> save pipeline -> Streamlit app.

## Text preprocessing
`clean_text()` in `src/preprocessing.py` (runs inside the vectorizer): lowercase, remove URLs, replace punctuation/symbols with spaces (**apostrophes kept** so "don't" survives), collapse whitespace.
**Not used on purpose:** stop-word removal (standard lists contain *not/no/never*, which flip sentiment), stemming and lemmatization (little benefit on short simple reviews).

## Feature extraction
`TfidfVectorizer` with `sublinear_tf=True`; `ngram_range`, `min_df` and `max_df` are tuned. TF-IDF gives high weight to words that are frequent in one review but rare across all reviews. Bag of Words (raw counts) is used as a baseline. The vectorizer is inside the Pipeline, so it is fitted on training data only (no leakage).

## Models tested
Logistic Regression, Multinomial Naive Bayes, Linear SVM (LinearSVC), plus a BoW + Logistic Regression baseline.

## Model comparison
Baseline models with default settings, **5-fold stratified cross-validation on the training data** (macro-averaged precision/recall/F1):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| BoW + Logistic Regression | 0.9875 | 0.9882 | 0.9875 | 0.9875 |
| TF-IDF + Logistic Regression | 0.9875 | 0.9880 | 0.9875 | 0.9875 |
| TF-IDF + Naive Bayes | 0.9875 | 0.9880 | 0.9875 | 0.9875 |
| TF-IDF + Linear SVM | 0.9958 | 0.9961 | 0.9958 | 0.9958 |

After GridSearchCV (32 combinations per model, scoring = macro F1, 5-fold CV):

| Model | Best CV macro-F1 | Best parameters |
|---|---|---|
| Logistic Regression | 0.9958 | C=1, ngram_range=(1,2), min_df=1, max_df=0.9 |
| Naive Bayes | 1.0000 | alpha=0.01, ngram_range=(1,1), min_df=1, max_df=0.9 |
| Linear SVM | 0.9958 | C=0.1, ngram_range=(1,2), min_df=1, max_df=0.9 |

## Evaluation metrics
Accuracy (share correct), Precision (of predicted X, how many are X), Recall (of true X, how many found), F1 (balance of the two), Confusion matrix (true vs predicted per class), One-vs-Rest ROC/AUC. Precision/recall/F1 are **macro-averaged** (average of the three per-class values).

## Final model
**TF-IDF + Logistic Regression** (C=1, ngram_range=(1,2), min_df=1, max_df=0.9). Validation (CV) macro-F1 = 0.9958.

The three tuned models were effectively tied (Naive Bayes was 0.4 points higher, about one review of 240). By a rule fixed in advance and based on CV only, ties go to the model with native probabilities and easy interpretation, so Logistic Regression was chosen. Linear SVM has no `predict_proba`, and we never fake a confidence score.

**Final test performance (60 unseen reviews, evaluated once):** Accuracy 1.000, Precision 1.000, Recall 1.000, F1 1.000, macro OvR ROC-AUC 1.000; confusion matrix has 20/20/20 on the diagonal and no errors. **Read this with the limitation below: it does not mean 100% on real-world reviews.**

## How the model is saved
`joblib.dump(pipeline, "models/sentiment_pipeline.pkl")`. The file contains the cleaning function reference, the fitted TF-IDF vocabulary/IDF values and the trained classifier. The app loads it with `joblib.load` and never trains. Use the same scikit-learn version to load it; if you get a version error, run `python src/train.py` to regenerate it.

## Installation
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## How to run
```bash
# 1. (optional) retrain and re-save the model + reports
python src/train.py

# 2. notebook (already executed; run again with)
jupyter notebook notebooks/sentiment_analysis.ipynb

# 3. Streamlit app
streamlit run app.py
```
Run these commands from the project root folder.

## Example prediction
Real output of the saved pipeline (labels are what the model predicted, not forced):

| Review | Predicted | Confidence |
|---|---|---|
| The product is excellent. I love the quality and would definitely buy it again. | Positive | 54.3% |
| The product is terrible and stopped working after two days. | Negative | 44.7% |
| The package contains the product, charger, and instruction manual. | Neutral | 59.8% |

Confidences are modest because the regularised model spreads probability over three classes trained on only 240 reviews. Note that the model's confidence is not a guarantee of correctness.

## Project structure
```
sentiment-analysis/
├── data/reviews.csv
├── notebooks/sentiment_analysis.ipynb
├── models/sentiment_pipeline.pkl
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # loading, cleaning, clean_text()
│   ├── train.py             # split, CV, GridSearchCV, selection, save
│   └── evaluate.py          # metrics, confusion matrix, ROC
├── app.py                   # Streamlit app
├── requirements.txt
├── README.md
└── reports/
    ├── evaluation_report.md
    ├── confusion_matrix.png
    └── results.json         # numbers written by train.py
```

## Limitations
- Only 300 unique review texts exist, and every test sentence also occurs in the training reviews, so scores are optimistic and not evidence of real-world accuracy.
- The 60-review test set is small (one mistake = 1.7 percentage points).
- Reviews outside the training vocabulary are handled poorly. Example: "Not bad at all, actually quite nice!" was predicted **Negative** (bag-of-words models do not understand "not bad"). Text with no known words gets near-equal probabilities; the app shows a warning in that case.
- No sarcasm, mixed sentiment, misspellings or non-English text.

## Future improvements
Collect real, varied reviews and a separate test source; try character n-grams; probability calibration; error analysis on real user reviews; add screenshots of the app to this README.
