"""
This file only run for daily pipeline to get new tweets. For the monthly retrain, use training/train_kmeans.py
"""
from datetime import datetime, timezone

import numpy as np
from src.sentiment import predict_sentiment_batch
from src.topics import predict_topics
from .core import config, supabase, log
from .scraper import scrap_nitter
from .preprocess import processing_text
import pandas as pd

if __name__ == "__main__":
    # Check if training is in progress
    response = supabase.table('app_config').select('value').eq('key', 'training-in-progress').single().execute()
    if response.data['value']:
        log.info("Training in progress, skipping daily pipeline.")
        exit()

    # Update last-updated
    utc_now = datetime.now(timezone.utc)
    supabase.table('app_config').update({"value": utc_now.isoformat()}).eq('key', 'last-updated').execute()

    # Get new tweets (using list comprehension for efficiency)
    new_data = [
        tweet_data

        for search in config.scrape_config
        if search.get('query') is not None 
        for tweet_data in scrap_nitter(
            search_query=search['query'],
            depth=search.get('depth') or -1,
            time_budget=search.get('time_budget') or -1
        )
    ]
    
    log.info(f"Found {len(new_data)} new tweets.")

    if not new_data:
        log.info("No new tweets found, skipping daily pipeline.")
        exit()

    df = pd.DataFrame(new_data)
    log.info(f"Sample new tweet data:\n{df.head(5)}")

    # Doing preprocessing
    df['processed_text_light'] = df['text_content'].apply(lambda x: processing_text(x, level='light'))
    df['processed_text_hard'] =  df['text_content'].apply(lambda x: processing_text(x, level='hard'))

    # Add Sentiment Column
    df['sentiment'] = predict_sentiment_batch(df['processed_text_hard'])

    # Add topics column
    df['topic_id'] = predict_topics(df['processed_text_hard'])

    # Insert new tweets
    clean_df  = (
        df.drop(['processed_text_light', 'processed_text_hard'], axis=1)
        .convert_dtypes()
        .replace(np.nan, None)
    )
    supabase.table('tweets').upsert(
        clean_df.to_dict(orient='records')
    ).execute()