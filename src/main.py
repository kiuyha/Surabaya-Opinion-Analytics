"""
This file only run for daily pipeline to get new tweets. For the monthly retrain, use training/train_kmeans.py
"""
from datetime import datetime, timezone
import numpy as np
from src.sentiment import predict_sentiment_batch
from src.topics import predict_topics
from .core import config, supabase, log
from .scraper import scrap_nitter, scrape_reddit
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

    new_tweets = [
        {**tweet, 'source_type': 'twitter'}

        for search in config.scrape_config
        if search.get('query')
        for tweet in scrap_nitter(search['nitter']['query'])
    ]

    log.info(f"Found {len(new_tweets)} new tweets.")
    
    new_reddit = [
        {**comment, 'source_type': 'reddit'}
        
        for search in config.scrape_config
        if search.get('query')
        for comment in scrape_reddit(search['reddit']['query'])
    ]

    log.info(f"Found {len(new_reddit)} new comments.")

    if not new_tweets and not new_reddit:
        log.info("No new tweets or comments found")
        exit()

    df_tweets = pd.DataFrame(new_tweets)
    df_reddit = pd.DataFrame(new_reddit)

    dfs_to_concat = []
    if not df_tweets.empty:
        dfs_to_concat.append(df_tweets)
    if not df_reddit.empty:
        dfs_to_concat.append(df_reddit)

    full_df = pd.concat(dfs_to_concat, ignore_index=True)

    log.info(f"Processing {len(full_df)} combined records...")

    # Preprocessing
    full_df['processed_text_light'] = full_df['text_content'].apply(lambda x: processing_text(x, level='light'))
    full_df['processed_text_hard'] =  full_df['text_content'].apply(lambda x: processing_text(x, level='hard'))

    # Predictions
    full_df['sentiment'] = predict_sentiment_batch(full_df['processed_text_hard'])
    full_df['topic_id'] = predict_topics(full_df['processed_text_hard'])

    # Save to database
    if not df_tweets.empty:
        processed_tweets = full_df[full_df['source_type'] == 'twitter'].copy()
        tweets_to_upload = (
            processed_tweets
            .drop(['processed_text_light', 'processed_text_hard', 'source_type'], axis=1)
            .dropna(axis=1, how='all')
            .convert_dtypes()
            .replace(np.nan, None)
        )
        
        log.info(f"Uploading {len(tweets_to_upload)} tweets...")
        supabase.table('tweets').upsert(tweets_to_upload.to_dict(orient='records')).execute()

    if not df_reddit.empty:
        processed_reddit = full_df[full_df['source_type'] == 'reddit'].copy()
        reddit_to_upload = (
            processed_reddit
            .drop(['processed_text_light', 'processed_text_hard', 'source_type'], axis=1)
            .dropna(axis=1, how='all')
            .convert_dtypes()
            .replace(np.nan, None)
        )

        log.info(f"Uploading {len(reddit_to_upload)} reddit comments...")
        supabase.table('reddit_comments').upsert(reddit_to_upload.to_dict(orient='records')).execute()