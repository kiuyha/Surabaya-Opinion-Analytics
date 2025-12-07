from email import generator
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import pandas as pd
from tqdm import tqdm

MODEL_NAME = "mdhugol/indonesia-bert-sentiment-classification"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    generator = (t for t in valid_texts)

    # The pipeline returns a generator/list of dicts: [{'label': 'LABEL_0', 'score': 0.9}, ...]
    pipeline_iterator = sentiment_pipeline(
        generator,
        batch_size=batch_size,
        truncation=True,
        max_length=512
    )

    results = list(tqdm(
        pipeline_iterator,
        total=len(valid_texts), 
        desc="NER Extraction",
        unit="docs"
    ))

    final_labels = [
        LABEL_MAP.get(res["label"], "neutral") 
        for res in results
    ]
    
    return final_labels