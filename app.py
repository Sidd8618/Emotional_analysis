"""
Emotion Analysis — Streamlit App
Loads the model trained in Emotion_Analysis_NLP.ipynb and serves live predictions.

Run with:
    streamlit run app.py
"""

import re
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Emotion Analysis NLP", page_icon="🎭", layout="centered")

EMOJI_MAP = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

MODEL_PATH = "models/emotion_model.joblib"
VEC_PATH = "models/tfidf_vectorizer.joblib"


# ----------------------------------------------------------------------
# NLTK setup (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    stop = set(stopwords.words("english"))
    negations = {"no", "not", "nor", "never", "n't"}
    stop -= negations
    return stop, WordNetLemmatizer()


STOPWORDS, LEMMATIZER = setup_nltk()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")


def clean_text(text: str) -> str:
    """Must match the cleaning function used in the training notebook exactly."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


model, vectorizer = load_pipeline()

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🎭 Emotion Analysis NLP")
st.caption("TF-IDF + classical ML emotion classifier, trained on the open-source dair-ai/emotion dataset.")

if model is None:
    st.error(
        "No trained model found in `models/`. Run **Emotion_Analysis_NLP.ipynb** first — "
        "it saves `emotion_model.joblib` and `tfidf_vectorizer.joblib` into the `models/` folder."
    )
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies short text into one of six emotions: "
        "**joy, sadness, anger, fear, love, surprise**."
    )
    if os.path.exists("models/model_info.txt"):
        st.subheader("Model info")
        st.code(open("models/model_info.txt").read())
    st.markdown("---")
    st.caption(
        "Note: no text-emotion model is 100% accurate — language is ambiguous and this dataset is "
        "Twitter-style short text, so treat predictions as a strong estimate, not ground truth."
    )

text_input = st.text_area(
    "Enter a sentence to analyze",
    placeholder="e.g. I can't believe I finally finished the project, I'm so relieved and happy!",
    height=120,
)

examples = [
    "I just got promoted at work, I am beyond excited!",
    "I miss my grandmother so much, it still hurts.",
    "Why would you do that to me, I am so angry right now.",
    "I did not expect that at all, wow!",
    "I'm terrified of what might happen tomorrow.",
]
st.write("Try an example:")
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex.split(",")[0][:14] + "…", key=ex):
        text_input = ex
        st.session_state["_prefill"] = ex

if "_prefill" in st.session_state:
    text_input = st.session_state["_prefill"]

analyze = st.button("Analyze Emotion", type="primary")

if analyze and text_input.strip():
    cleaned = clean_text(text_input)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    st.markdown(f"## {EMOJI_MAP.get(pred, '🎭')} Predicted emotion: **{pred.upper()}**")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        prob_df = pd.DataFrame({"emotion": classes, "probability": proba}).sort_values(
            "probability", ascending=False
        )

        st.subheader("Confidence by emotion")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(prob_df["emotion"], prob_df["probability"], color="#6C63FF")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        for i, (emo, p) in enumerate(zip(prob_df["emotion"], prob_df["probability"])):
            ax.text(p + 0.01, i, f"{p:.1%}", va="center")
        st.pyplot(fig)

    with st.expander("See cleaned text fed to the model"):
        st.code(cleaned if cleaned else "(empty after cleaning)")

elif analyze:
    st.warning("Please enter some text first.")

st.markdown("---")
st.caption("Built with scikit-learn + Streamlit · Dataset: dair-ai/emotion (Saravia et al., CARER, EMNLP 2018)")
