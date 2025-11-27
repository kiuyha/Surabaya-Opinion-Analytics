import json
import joblib
from gensim.models import FastText
from huggingface_hub import hf_hub_download
from src.training.train_topics_model import text_pipeline
import pandas as pd
from typing import List, Optional, cast

REPO_ID = "Kiuyha/surabaya-opinion-tweet-clusters"
kmeans_path = hf_hub_download(repo_id=REPO_ID, filename="kmeans.joblib")
hf_hub_download(repo_id=REPO_ID, filename="fasttext.model.wv.vectors_ngrams.npy")
fasttext_path = hf_hub_download(repo_id=REPO_ID, filename="fasttext.model")
config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")

kmeans_model = joblib.load(kmeans_path)
fasttext_model = cast(FastText, FastText.load(fasttext_path))
with open(config_path, 'r') as f:
    config_data = json.load(f)
    TOPIC_MAP = config_data.get("topic_map", {})

print(f"K-Means model loaded with {kmeans_model.n_clusters} clusters.")
print(f"FastText model loaded with vector size {fasttext_model.vector_size}.")

def predict_topics(texts: pd.Series) -> List[Optional[int]]:
    _, vectors = text_pipeline(texts, model=fasttext_model)
    raw_predictions = kmeans_model.predict(vectors)
    
    # Filter predictions: Return ID if valid, None if junk
    final_topics = [
        TOPIC_MAP.get(label)
        for label in raw_predictions
    ]
    
    return final_topics