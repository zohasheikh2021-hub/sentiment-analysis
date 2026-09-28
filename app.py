"""
Streamlit app: loads the SAVED pipeline (no training happens here).

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:      # the pickled pipeline needs `src.preprocessing`
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "sentiment_pipeline.pkl"
ICONS = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}


@st.cache_resource          # load once, reuse on every click
def load_pipeline():
    return joblib.load(MODEL_PATH)


def top_terms(pipeline, text, predicted_label, n=5):
    """Words/phrases in THIS review that pushed the model towards its prediction.
    Only possible for linear models (they expose coef_); otherwise returns []."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return []
    vec = pipeline.steps[0][1]
    row = vec.transform([text])
    class_idx = list(clf.classes_).index(predicted_label)
    coef = clf.coef_[class_idx]
    names = vec.get_feature_names_out()
    contrib = row.multiply(coef).tocoo()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: -p[1])[:n]
    return [(names[i], float(v)) for i, v in pairs if v > 0]


st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")
st.title("💬 AI-Powered Sentiment Analysis")
st.write("Type a customer review and the trained machine-learning model will classify it as "
         "**Positive**, **Negative** or **Neutral**. (TF-IDF features + a scikit-learn classifier.)")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python src/train.py` first to create "
             "`models/sentiment_pipeline.pkl`.")
    st.stop()

try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error("Could not load the saved model.")
    st.exception(exc)
    st.stop()

review = st.text_area("Enter a review:", height=140,
                      placeholder="The product quality is excellent and I really enjoyed using it.")

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            label = pipeline.predict([review])[0]
            st.subheader("Predicted Sentiment")
            st.markdown(f"## {ICONS.get(label, '')} {label}")

            if hasattr(pipeline, "predict_proba"):        # real probabilities only
                proba = pipeline.predict_proba([review])[0]
                st.metric("Confidence", f"{proba.max() * 100:.1f}%")
                st.bar_chart(pd.DataFrame({"Probability": proba}, index=pipeline.classes_))

            vec = pipeline.steps[0][1]
            if vec.transform([review]).nnz == 0:
                st.warning("None of the words in this review appeared in the training data, "
                           "so this prediction is not reliable.")

            with st.expander("How was this prediction made?"):
                st.write("**Cleaned text** (what the model actually sees):")
                st.code(vec.preprocessor(review) if vec.preprocessor else review)
                terms = top_terms(pipeline, review, label)
                if terms:
                    st.write(f"**Words/phrases that pushed towards {label}:**")
                    st.table(pd.DataFrame(terms, columns=["Term", "Contribution"]).round(3))
                st.caption("Steps: clean text → TF-IDF vector → classifier → probabilities → highest wins.")
        except Exception as exc:
            st.error(f"Something went wrong while analysing the review: {exc}")
