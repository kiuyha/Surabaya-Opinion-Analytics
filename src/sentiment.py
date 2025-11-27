from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

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

def predict_sentiment(cleaned_text: str) -> str|None:
    """Predict sentiment for a single text."""
    if not cleaned_text or not isinstance(cleaned_text, str):
        return None

    result = sentiment_pipeline(cleaned_text, truncation=True, max_length=512)[0]
    return LABEL_MAP.get(result["label"], "neutral")