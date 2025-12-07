from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
from typing import List, Dict, cast
from src.training.train_topics_model import fetch_all_rows
from .core import log, supabase
import pandas as pd
from tqdm import tqdm

# Initialize model and tokenizer
MODEL_NAME = "Kiuyha/surabaya-opinion-indobert-ner"
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=512
)

try:
    log.info(f"Loading NER model from {MODEL_NAME}...")
    ner_pipeline = pipeline(
        task="ner", # type: ignore
        model=MODEL_NAME,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    log.info(f"NER model loaded successfully from {MODEL_NAME}")
except Exception as e:
    log.error(f"Failed to load NER model: {e}")
    ner_pipeline = None

def extract_entities_batch(texts: pd.Series, batch_size: int = 16) -> List[List[Dict]]:
    if ner_pipeline is None:
        log.warning("NER pipeline not available, returning empty results")
        return [[] for _ in texts]
    try:
        # Filter out non-strings (handle NaNs) to prevent pipeline crash
        valid_texts = [t if t and isinstance(t, str) else "" for t in texts]
        generator = (t for t in valid_texts)

        pipeline_iterator = ner_pipeline(
            generator,
            batch_size=batch_size
        )

        results = list(tqdm(
            pipeline_iterator,
            total=len(valid_texts), 
            desc="NER Extraction",
            unit="docs"
        ))
        
        return results
    except Exception as e:
        log.error(f"Error in NER extraction: {e}")
        return [[] for _ in texts]

def save_ner_to_supabase(df: pd.DataFrame):
    entities_to_save = [
        {
            'tweet_id': row.id if row.source_type == 'twitter' else None,
            'reddit_comment_id': row.id if row.source_type == 'reddit' else None,
            'entity_label': ent['entity_group'],
            'entity_text': ent['word'],
            'confidence_score': float(ent['score']),
            'start_position': ent['start'],
            'end_position': ent['end']
        }
        for row in df.itertuples()
        for ent in cast(List[Dict], row.entities)
    ]

    # Save entities to database
    if entities_to_save:
        log.info(f"Uploading {len(entities_to_save)} entities...")
        try:
            supabase.table('entities').upsert(entities_to_save).execute()
        except Exception as e:
            log.error(f"Error uploading entities: {e}")

if __name__ == "__main__":
    tweets_data = fetch_all_rows('tweets', ['id', 'processed_text_light'])
    reddit_data = fetch_all_rows('reddit_comments', ['id', 'processed_text_light'])

    if not tweets_data and not reddit_data:
        raise Exception("No data found in Supabase (tweets or reddit)")
    
    df_tweets = pd.DataFrame(tweets_data)
    df_reddit = pd.DataFrame(reddit_data)
    dfs_to_concat = []
    
    if not df_tweets.empty:
        df_tweets['source_type'] = 'tweets'
        dfs_to_concat.append(df_tweets)
    
    if not df_reddit.empty:
        df_reddit['source_type'] = 'reddit_comments'
        dfs_to_concat.append(df_reddit)
    
    df = pd.concat(dfs_to_concat, ignore_index=True)
    log.info(f"NER on {len(df)} records...")

    df['entities'] = extract_entities_batch(df['processed_text_light'])
    save_ner_to_supabase(df)