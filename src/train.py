"""
End-to-end training script (same steps as the notebook, but runnable from the terminal):

    python src/train.py

Steps: load -> clean -> stratified split -> compare models with cross-validation
-> GridSearchCV tuning -> select final model (NO test data used) ->
evaluate ONCE on the test set -> save the complete pipeline with joblib.
"""
import json
import sys
from pathlib import Path

# Allow `python src/train.py` from any folder (the pickled pipeline refers to
# src.preprocessing.clean_text, so `src` must be importable as a package).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import evaluate_pipeline, multiclass_roc_auc, plot_confusion_matrix
from src.preprocessing import TARGET_COL, TEXT_COL, VALID_LABELS, clean_dataframe, clean_text, load_raw_data

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SCORING = {"Accuracy": "accuracy", "Precision (macro)": "precision_macro",
           "Recall (macro)": "recall_macro", "F1 (macro)": "f1_macro"}
MODEL_PATH = ROOT / "models" / "sentiment_pipeline.pkl"
REPORTS_DIR = ROOT / "reports"

# Tie-break order used ONLY when tuned models are (almost) equally good on
# cross-validation: prefer models with native probabilities + easy to explain.
PREFERENCE = ["Logistic Regression", "Naive Bayes", "Linear SVM"]
TIE_TOLERANCE = 0.005


def make_pipeline(classifier, vectorizer="tfidf") -> Pipeline:
    """Vectorizer + classifier in ONE Pipeline. Because the vectorizer lives inside
    the pipeline, cross-validation re-fits it on each training fold only (no leakage)."""
    if vectorizer == "bow":
        vec = CountVectorizer(preprocessor=clean_text, token_pattern=r"[a-z0-9']+")
    else:
        vec = TfidfVectorizer(preprocessor=clean_text, token_pattern=r"[a-z0-9']+", sublinear_tf=True)
    return Pipeline([("tfidf" if vectorizer != "bow" else "bow", vec), ("clf", classifier)])


def baseline_models() -> dict:
    """Untuned models (default settings) for the first comparison table."""
    return {
        "BoW + Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000), "bow"),
        "TF-IDF + Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
        "TF-IDF + Naive Bayes": make_pipeline(MultinomialNB()),
        "TF-IDF + Linear SVM": make_pipeline(LinearSVC(max_iter=10000)),
    }


# Small grids: TF-IDF settings + one classifier setting (practical for a laptop).
TFIDF_GRID = {"tfidf__ngram_range": [(1, 1), (1, 2)],
              "tfidf__min_df": [1, 2],
              "tfidf__max_df": [0.9, 1.0]}
TUNING = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"clf__C": [0.1, 1, 10, 100]}),
    "Naive Bayes": (MultinomialNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "Linear SVM": (LinearSVC(max_iter=10000), {"clf__C": [0.1, 1, 10, 100]}),
}


def cv_scores(pipeline, X, y, cv) -> dict:
    res = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)
    return {name: res[f"test_{name}"].mean() for name in SCORING}


def load_data():
    df, report = clean_dataframe(load_raw_data(), deduplicate_text=True)
    return df, report


def main():
    df, clean_report = load_data()
    X, y = df[TEXT_COL], df[TARGET_COL]

    # ---- Train / test split (stratified, reproducible). Test set is locked away. ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Unique reviews used: {len(df)} | train={len(X_train)} test={len(X_test)}")

    # ---- 1) Baseline comparison (cross-validation on TRAIN only) ----
    base_rows = []
    for name, pipe in baseline_models().items():
        base_rows.append({"Model": name, **cv_scores(pipe, X_train, y_train, cv)})
    baseline_df = pd.DataFrame(base_rows)
    print("\nBaseline models (5-fold CV on training data):\n", baseline_df.round(4).to_string(index=False))

    # ---- 2) Hyperparameter tuning with GridSearchCV (TRAIN only) ----
    tuned, tuned_rows = {}, []
    for name, (clf, clf_grid) in TUNING.items():
        pipe = make_pipeline(clf)
        grid = {**TFIDF_GRID, **clf_grid}
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        tuned[name] = gs
        tuned_rows.append({"Model": name, "Best CV F1 (macro)": gs.best_score_,
                           "Best parameters": str(gs.best_params_)})
        print(f"\n{name}: best CV F1={gs.best_score_:.4f}  params={gs.best_params_}")
    tuned_df = pd.DataFrame(tuned_rows)

    # ---- 3) Objective final-model selection (still NO test data) ----
    best_score = tuned_df["Best CV F1 (macro)"].max()
    candidates = tuned_df[tuned_df["Best CV F1 (macro)"] >= best_score - TIE_TOLERANCE]["Model"].tolist()
    final_name = next(m for m in PREFERENCE if m in candidates)
    final_pipeline = tuned[final_name].best_estimator_
    print(f"\nCandidates within {TIE_TOLERANCE} of best CV F1: {candidates}")
    print(f"Selected final model: {final_name}")

    # ---- 4) ONE-TIME final evaluation on the untouched test set ----
    result = evaluate_pipeline(final_pipeline, X_test, y_test, VALID_LABELS)
    print("\nTest metrics:", {k: round(v, 4) for k, v in result["metrics"].items()})
    print(result["report"])
    print("Confusion matrix (rows=true, cols=pred, order", VALID_LABELS, "):\n", result["confusion_matrix"])
    roc = multiclass_roc_auc(final_pipeline, X_test, y_test, VALID_LABELS)
    if roc:
        print("Macro one-vs-rest ROC AUC:", round(roc[0], 4))

    # ---- 5) Save the complete pipeline + figures + a small results file ----
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    REPORTS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(result["confusion_matrix"], VALID_LABELS, save_path=REPORTS_DIR / "confusion_matrix.png")
    results = {
        "cleaning_report": clean_report,
        "n_train": len(X_train), "n_test": len(X_test),
        "baseline_cv": baseline_df.round(4).to_dict(orient="records"),
        "tuned_cv": tuned_df.to_dict(orient="records"),
        "final_model": final_name,
        "final_params": {k: str(v) for k, v in tuned[final_name].best_params_.items()},
        "final_cv_f1": tuned[final_name].best_score_,
        "test_metrics": result["metrics"],
        "test_confusion_matrix": result["confusion_matrix"].tolist(),
        "roc_auc_macro_ovr": roc[0] if roc else None,
    }
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved pipeline -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
