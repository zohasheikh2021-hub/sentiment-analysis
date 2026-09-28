import sys
from pathlib import Path
from html import escape

import joblib
from flask import Flask, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))          # the saved model needs src.preprocessing

app = Flask(__name__)                  # Vercel looks for this top-level "app"
pipeline = joblib.load(ROOT / "models" / "sentiment_pipeline.pkl")

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment Analysis</title>
<style>body{{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}}
textarea{{width:100%;height:120px;padding:8px}}button{{padding:8px 18px;margin-top:8px;cursor:pointer}}
.box{{margin-top:20px;padding:14px;border:1px solid #ddd;border-radius:8px}}.bar{{background:#eee;border-radius:4px;margin:4px 0}}
.bar div{{background:#4b8bf4;color:#fff;padding:2px 6px;border-radius:4px;font-size:13px}}.warn{{color:#b45309}}</style></head>
<body><h1>💬 AI-Powered Sentiment Analysis</h1>
<p>Enter a customer review. A TF-IDF + scikit-learn model classifies it as Positive, Negative or Neutral.</p>
<form method="post"><textarea name="review" placeholder="The product quality is excellent and I really enjoyed using it.">{review}</textarea><br>
<button type="submit">Analyze</button></form>{result}</body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    review, result = "", ""
    if request.method == "POST":
        review = request.form.get("review", "").strip()
        if not review:
            result = '<div class="box warn">Please enter a review first.</div>'
        else:
            try:
                label = pipeline.predict([review])[0]
                proba = pipeline.predict_proba([review])[0]
                bars = "".join(
                    f'<div class="bar"><div style="width:{p*100:.0f}%">{escape(c)} {p*100:.1f}%</div></div>'
                    for c, p in zip(pipeline.classes_, proba))
                result = (f'<div class="box"><h2>Predicted Sentiment: {escape(label)}</h2>'
                          f'<p>Confidence: <b>{proba.max()*100:.1f}%</b></p>{bars}</div>')
            except Exception as exc:
                result = f'<div class="box warn">Something went wrong: {escape(str(exc))}</div>'
<<<<<<< HEAD
    return PAGE.format(review=escape(review), result=result)
=======
    return PAGE.format(review=escape(review), result=result)
>>>>>>> 373438162095ac4bbb3a763fbe644399c6d7bdca
