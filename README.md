# Emotion Analysis 

A text emotion classifier (joy, sadness, anger, fear, love, surprise) trained on the open-source
**dair-ai/emotion** dataset (~20k English tweets, CARER paper, Saravia et al., EMNLP 2018), deployed
as a **Streamlit** web app.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run it

1. **Train the model** — open and run all cells in `Emotion_Analysis_notebook.ipynb` (Jupyter or VS Code).
   This downloads the dataset automatically (needs internet the first time), trains and saves the model
   into `models/`.
   - No internet / download blocked? The notebook prints fallback instructions to manually grab the same
     dataset from Kaggle: https://www.kaggle.com/datasets/parulpandey/emotion-dataset

2. **Launch the app**:
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. Type any sentence and see the predicted emotion + confidence chart.

## About the accuracy number

This dataset and task realistically top out around **88–94% accuracy** for well-tuned models — 100% is not
achievable on real, ambiguous language data without overfitting or data leakage, and claiming it is a red
flag to anyone technical reviewing your resume. The notebook reports a proper **held-out test accuracy and
macro-F1**, plus a confusion matrix and error analysis, which is what's actually worth putting on a resume:

> Built and deployed an NLP emotion-classification app (Streamlit) on a 20k-sample open-source Twitter
> dataset (6 classes); engineered TF-IDF n-gram features, benchmarked 4 ML models, and tuned the best one
> via GridSearchCV to ~90% test accuracy / ~0.85 macro-F1 on a held-out test set.

(Fill in your notebook's actual numbers — they'll vary slightly by run/tuning.)

## Ideas to extend it (good for interview talking points)
- Swap TF-IDF + classical ML for a fine-tuned transformer (e.g. DistilBERT) and compare accuracy/latency trade-offs
- Add SHAP/LIME explainability to show which words drove a prediction
- Deploy the Streamlit app publicly (Streamlit Community Cloud / Hugging Face Spaces) and link it on your resume
- Add a feedback button in the app so users can flag wrong predictions (shows product thinking)
