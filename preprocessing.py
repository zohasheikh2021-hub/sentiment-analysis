"""
Data loading, cleaning and text preprocessing.

Design principle: keep preprocessing SMALL and EXPLAINABLE.
Every step below has a reason; steps without a clear reason are NOT applied
(see the notes in clean_text about stop-words, stemming and lemmatization).
"""
import re
from pathlib import Path

import pandas as pd

# Project root = the folder that contains src/, data/, models/ ...
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "reviews.csv"

TEXT_COL = "review_text"   # the only model input (feature)
TARGET_COL = "sentiment"   # the label we want to predict
VALID_LABELS = ["Negative", "Neutral", "Positive"]

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
# Keep letters, digits and apostrophes (so "don't" stays "don't").
# Everything else (punctuation, symbols, emojis) becomes a space.
_NON_TEXT_RE = re.compile(r"[^a-z0-9']+")
_SPACES_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Light-weight text normalisation used INSIDE the sklearn pipeline.

    1. lowercase          -> "Great" and "great" become the same feature.
    2. remove URLs        -> links carry no sentiment (none in this dataset,
                             but real reviews often contain them).
    3. punctuation/symbols -> spaces. Apostrophes are kept on purpose so that
                             negations such as "don't" / "wasn't" survive.
    4. collapse spaces    -> "a   b" becomes "a b".

    NOT done (on purpose): stop-word removal (would delete "not", "never"),
    stemming and lemmatization (little benefit on short, simple reviews).
    """
    if not isinstance(text, str):          # None / NaN -> empty string
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV exactly as provided."""
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame, deduplicate_text: bool = True):
    """Clean the dataset and return (clean_df, report_dict).

    report_dict records what every step removed, so the notebook / report can
    show real numbers instead of guessing.
    """
    report = {"rows_raw": len(df)}
    df = df.copy()

    # 1. Missing values in the two columns we need
    report["missing_review_text"] = int(df[TEXT_COL].isna().sum())
    report["missing_sentiment"] = int(df[TARGET_COL].isna().sum())
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])

    # 2. Consistent labels ("positive " / "POSITIVE" -> "Positive")
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.title()
    bad_labels = ~df[TARGET_COL].isin(VALID_LABELS)
    report["invalid_labels_removed"] = int(bad_labels.sum())
    df = df[~bad_labels]

    # 3. Tidy the text and drop empty reviews
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    empty = df[TEXT_COL].str.len() == 0
    report["empty_reviews_removed"] = int(empty.sum())
    df = df[~empty]

    # 4. Exact duplicate rows and duplicate review ids
    report["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    report["duplicate_review_ids"] = int(df["review_id"].duplicated().sum())
    df = df.drop_duplicates(subset="review_id", keep="first")
    report["rows_after_basic_cleaning"] = len(df)

    # 5. Duplicate REVIEW TEXTS (different ids, same wording).
    #    These are the data-leakage danger: if the same sentence lands in both
    #    train and test, the test score is inflated.
    key = df[TEXT_COL].map(clean_text)
    report["duplicate_texts"] = int(key.duplicated().sum())
    report["texts_with_conflicting_labels"] = int(
        (df.assign(_k=key).groupby("_k")[TARGET_COL].nunique() > 1).sum()
    )
    if deduplicate_text:
        df = df.assign(_k=key).drop_duplicates(subset=["_k", TARGET_COL]).drop(columns="_k")
    report["rows_final"] = len(df)
    return df.reset_index(drop=True), report
