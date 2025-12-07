from .core import log
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import pandas as pd
from tqdm import tqdm
import torch

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
LABEL_MAP = {
    "LABEL_0": "positive",
    "LABEL_1": "neutral",
    "LABEL_2": "negative"
}

try:
    log.info(f"Loading sentiment model from {MODEL_NAME}...")
    sentiment_pipeline = pipeline(
        task="sentiment-analysis",  # type: ignore
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    log.info(f"sentiment model loaded successfully from {MODEL_NAME}")
except Exception as e:
    log.error(f"Failed to load sentiment model: {e}")
    sentiment_pipeline = None

def predict_sentiment_batch(texts: pd.Series, batch_size: int = 16) -> List[str]:
    if sentiment_pipeline is None:
        log.warning("sentiment pipeline not available, returning empty results")
        raise Exception("sentiment pipeline not available")
    try:
        # Filter out non-strings (handle NaNs) to prevent pipeline crash
        valid_texts = [t if t and isinstance(t, str) else "" for t in texts]
        generator = (t for t in valid_texts)

        pipeline_iterator = sentiment_pipeline(
            generator,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )

        results = list(tqdm(
            pipeline_iterator,
            total=len(valid_texts), 
            desc="Sentiment Extraction",
            unit="docs"
        ))

        # The pipeline returns a generator/list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
        final_labels = [
            LABEL_MAP.get(res["label"], "neutral") 
            for res in results
        ]
    except Exception as e:
        log.error(f"Error in sentiment extraction: {e}")
        raise e
    return final_labels