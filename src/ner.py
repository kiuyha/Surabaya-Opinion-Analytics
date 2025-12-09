from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import torch
from typing import List, Dict, cast
from src.training.train_topics_model import fetch_all_rows
from .core import log, config, supabase
import pandas as pd
from tqdm import tqdm
from .geocoding import run_geocoding

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
        task="ner",  # type: ignore
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
        
        return [
            [
                {
                    **ent,
                    'word' : normalize_entity(
                        ent['word'].replace("##", ""),
                        ent['entity_group']
                    ),
                }
                for ent in result
            ]
            for result in results
        ]
    
    except Exception as e:
        log.error(f"Error in NER extraction: {e}")
        return [[] for _ in texts]

def normalize_entity(text: str, label: str) -> str:
    """
    Normalize entities based on their type:
    - LOC: Location names, roads, kecamatan, kelurahan, city abbreviations
    - PERSON: Public figures, politicians, common name variations
    - ORG: Companies, brands, organizations
    """
    original = text
    t = text.lower().strip()
    
    #LOC
    if label == "LOC":
        # Road names
        road_prefixes = ["jl", "jl.", "jln", "jln."]
        for pref in road_prefixes:
            if t.startswith(pref + " "):
                return "jalan " + original[len(pref)+1:]
            if t == pref:
                return "jalan"

        # Kecamatan
        kec_prefixes = ["kec", "kec.", "kc", "kc."]
        for pref in kec_prefixes:
            if t.startswith(pref + " "):
                return "kecamatan " + original[len(pref)+1:]
            if t == pref:
                return "kecamatan"

        # Kelurahan
        kel_prefixes = ["kel", "kel."]
        for pref in kel_prefixes:
            if t.startswith(pref + " "):
                return "kelurahan " + original[len(pref)+1:]
            if t == pref:
                return "kelurahan"
        
        if t in config.loc_abbr:
            return config.loc_abbr[t]

        # Partial matches
        if "sby" in t:
            return "Surabaya"
        if "jak" in t or "jkt" in t:
            return "Jakarta"
        if "bdg" in t:
            return "Bandung"
        if "mlg" in t or "malang" in t:
            return "Malang"
        if "sda" in t:
            return "Sidoarjo"

    #PERSON
    elif label == "PERSON":
        if t in config.per_abbr:
            return config.per_abbr[t]

    #ORG
    elif label == "ORG":
        if t in config.org_abbr:
            return config.org_abbr[t]
    
    return original

def save_entities_to_supabase(df: pd.DataFrame):
    entities_to_save = [
        {
            'tweet_id': row.id if row.source_type == 'twitter' else None,
            'reddit_comment_id': row.id if row.source_type == 'reddit' else None,
            'label': ent.get('entity_group'),
            'text': ent.get('word'),
            'confidence_score': float(ent['score']) if ent.get('score') else None,
            'start': ent.get('start'),
            'end': ent.get('end'),
            'latitude': ent.get('latitude'),
            'longitude': ent.get('longitude'),
        }
        for row in df.itertuples()
        for ent in cast(List[Dict], row.entities)
    ]

    # Save entities to database
    if entities_to_save:
        log.info(f"Uploading {len(entities_to_save)} entities...")
        try:
            supabase.table('entities').upsert(
                entities_to_save,
                on_conflict='tweet_id, reddit_comment_id, text, start'
            ).execute()
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
        df_tweets['source_type'] = 'twitter'
        dfs_to_concat.append(df_tweets)
    
    if not df_reddit.empty:
        df_reddit['source_type'] = 'reddit'
        dfs_to_concat.append(df_reddit)
    
    df = pd.concat(dfs_to_concat, ignore_index=True)
    log.info(f"NER on {len(df)} records...")

    # NER
    log.info("Starting NER...")
    df['entities'] = extract_entities_batch(df['processed_text_light'])
    log.info("NER completed.")

    log.info("Starting geocoding LOC entities...")
    run_geocoding(df) 
    
    # Prepare entities for database
    save_entities_to_supabase(df)
    save_entities_to_supabase(df)