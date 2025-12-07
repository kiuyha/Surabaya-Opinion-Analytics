from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import pandas as pd

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    task="sentiment-analysis",  # type: ignore
    model=model,
    tokenizer=tokenizer,
)

LABEL_MAP = {
    "LABEL_0": "positive",
    "LABEL_1": "neutral",
    "LABEL_2": "negative"
}

def predict_sentiment_batch(texts: pd.Series, batch_size: int = 16) -> List[str]:
    """
    Predict sentiment for list of texts.
    """

    # Filter out non-strings (handle NaNs) to prevent pipeline crash
    valid_texts = [t if t and isinstance(t, str) else "" for t in texts]

    # The pipeline returns a generator/list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
    results = sentiment_pipeline(
        valid_texts,
        batch_size=batch_size,
        truncation=True,
        max_length=512
    )

    final_labels = [
        LABEL_MAP.get(res["label"], "neutral") 
        for res in results
    ]
    
    return final_labels