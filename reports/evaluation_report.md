<<<<<<< HEAD
# Evaluation Report - AI-Powered Sentiment Analysis

All numbers below were produced by running `src/train.py` and the notebook on the provided dataset (`random_state=42`). Nothing is estimated or invented. Full raw numbers: `reports/results.json`.

## 1. Dataset overview
- 10,000 rows, 5 columns (`review_id`, `review_text`, `sentiment`, `category`, `source`); no missing values, no duplicate rows, no duplicate ids, no invalid labels.
- Classes: Positive 3,334, Negative 3,333, Neutral 3,333 (balanced).
- Review length: 43-136 characters (mean about 95); Negative reviews are slightly shorter on average (about 90.5 vs 96-97 characters).
- **Only 300 unique review texts** (each repeated about 33 times); no text has conflicting labels.
- Modelling uses `review_text` (input) and `sentiment` (target). `review_id`, `category` and `source` are not used: they carry no signal (each category/source is roughly one-third of each sentiment) and are not available for a new review typed into the app.

## 2. Preprocessing decisions
Lowercasing, URL removal, punctuation-to-space (apostrophes kept), whitespace collapsing. Stop-words, stemming and lemmatization were **not** applied. Stop-word lists contain *not*, *no*, *never*, which carry sentiment. In a CV experiment (training data only), removing stop-words gave macro-F1 0.9917 vs 0.9875 with them kept: a gap of about one review out of 240, within noise, so it did not justify deleting negations.

## 3. Train/test split and leakage control
Because of the duplicate texts, modelling uses one row per unique text (300 rows): stratified 80/20 split = **240 train / 60 test** (20 per class in the test set), no text shared between them. A naive split of the raw 10,000 rows would have put exact copies of test reviews in the training set (100% of the naive test set in our check), giving an inflated score. The test set was used only once, after model selection.

## 4. Feature extraction
`TfidfVectorizer` (`sublinear_tf=True`, custom `clean_text` preprocessor, token pattern keeps apostrophes) inside a `Pipeline`, so vocabulary and IDF are learned only from the training part of each CV fold. Bag of Words was used as a baseline. TF-IDF is preferred because it down-weights words that appear everywhere without requiring stop-word removal.

## 5. Models and cross-validation
Logistic Regression, Multinomial Naive Bayes and Linear SVM. Evaluation used **5-fold stratified cross-validation on the training data** (shuffled, `random_state=42`). Metrics are macro-averaged.

| Model (default settings) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| BoW + Logistic Regression | 0.9875 | 0.9882 | 0.9875 | 0.9875 |
| TF-IDF + Logistic Regression | 0.9875 | 0.9880 | 0.9875 | 0.9875 |
| TF-IDF + Naive Bayes | 0.9875 | 0.9880 | 0.9875 | 0.9875 |
| TF-IDF + Linear SVM | 0.9958 | 0.9961 | 0.9958 | 0.9958 |

Per-fold macro-F1 for TF-IDF + Logistic Regression: 0.9791, 1.0, 1.0, 0.9583, 1.0 (mean 0.9875, std 0.0167).

## 6. Hyperparameter tuning
`GridSearchCV`, scoring = macro F1, 5-fold stratified CV, training data only, 32 combinations per model. TF-IDF grid: `ngram_range` in {(1,1),(1,2)}, `min_df` in {1,2}, `max_df` in {0.9,1.0}; plus `C` in {0.1,1,10,100} (LR, SVM) or `alpha` in {0.01,0.1,0.5,1.0} (NB).

| Model | Best CV macro-F1 | Best parameters |
|---|---|---|
| Logistic Regression | 0.9958 | C=1, ngram_range=(1,2), min_df=1, max_df=0.9 |
| Naive Bayes | 1.0000 | alpha=0.01, ngram_range=(1,1), min_df=1, max_df=0.9 |
| Linear SVM | 0.9958 | C=0.1, ngram_range=(1,2), min_df=1, max_df=0.9 |

## 7. Final model and justification
**TF-IDF + Logistic Regression** was selected. The selection rule was fixed in the code beforehand and uses cross-validation only: take the best CV F1; models within 0.005 of it (smaller than one review of 240) count as tied; ties are broken in favour of the model with native probabilities and easy interpretation (Logistic Regression > Naive Bayes > Linear SVM). All three were tied (Naive Bayes was 0.0042 higher, i.e. one review).
- Logistic Regression: real `predict_proba` for the app's confidence score; interpretable word weights.
- Linear SVM: no native probabilities; adding calibration was unnecessary for equal accuracy.
- Naive Bayes: equally accurate here, but its probabilities are usually less well-calibrated and it is less interpretable.
This is a practical choice, not a large performance gap; any of the three would be defensible on this data.

Validation (CV) macro-F1 of the final model: **0.9958**.

## 8. Final test-set results (60 unseen reviews, used once)
| Metric | Value |
|---|---|
| Accuracy | 1.0000 |
| Precision (macro) | 1.0000 |
| Recall (macro) | 1.0000 |
| F1 (macro) | 1.0000 |
| ROC-AUC (macro, one-vs-rest) | 1.0000 |

Confusion matrix (rows = true, columns = predicted; order Negative, Neutral, Positive), also in `reports/confusion_matrix.png`:

|  | Pred Negative | Pred Neutral | Pred Positive |
|---|---|---|---|
| **True Negative** | 20 | 0 | 0 |
| **True Neutral** | 0 | 20 | 0 |
| **True Positive** | 0 | 0 | 20 |

**Interpretation:** every test review was classified correctly; there is no confusion between classes. This validates that the pipeline is wired correctly and that identical reviews did not leak, but it should not be read as real-world accuracy: all 108 test *sentences* also appear inside training reviews (the data is built from a small pool of template sentences), and 60 reviews is a small test set.

## 9. Sanity checks on new reviews (real model output)
| Review | Predicted | Confidence |
|---|---|---|
| The product is excellent. I love the quality and would definitely buy it again. | Positive | 54.3% |
| The product is terrible and stopped working after two days. | Negative | 44.7% |
| The package contains the product, charger, and instruction manual. | Neutral | 59.8% |
| I would not recommend it, it wasn't good. | Negative | 67.8% |
| Not bad at all, actually quite nice! | Negative | 73.0% |
| zzz qqq xyz | Neutral | 36.1% |

The first three match the expected type. The fifth is wrong: it sounds positive, but the model has never seen "not bad" and reacts to the word "bad". The last has no known words, so probabilities are almost equal (the app shows a warning in that case). Confidence values are moderate even for correct predictions because of regularisation and the small training set.

## 10. Limitations
- Only 300 unique texts; test scores are optimistic and dataset-specific.
- Small test set (one error = 1.7 percentage points).
- Bag-of-words features ignore word order beyond bigrams: weak on sarcasm, unseen phrases, mixed sentiment, misspellings and non-English text.
- Reviews unlike the training data give unreliable predictions.

## 11. Conclusion
The project delivers a complete, reproducible traditional NLP + ML pipeline (clean, TF-IDF, model comparison, cross-validation, GridSearchCV, saved Pipeline, Streamlit app). Logistic Regression with TF-IDF unigrams+bigrams was chosen on cross-validation evidence and practical grounds, and reached a perfect score on the held-out test set. Because of the duplicated, template-like dataset, the honest conclusion is that the workflow is correct and the model fits this dataset; real-world performance is untested and needs more varied data.
=======

>>>>>>> 373438162095ac4bbb3a763fbe644399c6d7bdca
