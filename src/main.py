"""
This file only run for weekly pipeline to get new tweets. For the monthly retrain, use training/train_kmeans.py
"""
from datetime import datetime, timezone
import numpy as np
from src.geocoding import run_geocoding
from src.ner import extract_entities_batch, save_entities_to_supabase
from src.sentiment import predict_sentiment_batch
from src.topics import predict_topics
from .core import config, supabase, log
from .scraper import scrap_nitter, scrape_reddit
from .preprocess import processing_text
from .geocoding import run_geocoding
import pandas as pd

def upload_data(df: pd.DataFrame, table_name: str, source_type: str):
    processed_df = df[df['source_type'] == source_type].copy().dropna(axis=1, how='all')

    count_columns = [
        'like_count',
        'comment_count',
        'quote_count',
        'retweet_count',
        'upvote_count',
        'downvote_count'
    ]
    
    for col in count_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(0).astype(int)

    data_to_upload = (
        processed_df
        .drop(['source_type'], axis=1)
        .astype(object)
        .replace({
            pd.NA: None,
            np.nan: None
        })
    )

    log.info(f"Uploading {len(data_to_upload)} {source_type} records...")
    supabase.table(table_name).upsert(data_to_upload.to_dict(orient='records')).execute()

if __name__ == "__main__":
    # Check if training is in progress
    response = supabase.table('app_config').select('value').eq('key', 'training-in-progress').single().execute()
    
    if response.data['value']:
        log.info("Training in progress, skipping weekly pipeline.")
        exit()

    # Update last-updated
    utc_now = datetime.now(timezone.utc)
    supabase.table('app_config').update({"value": utc_now.isoformat()}).eq('key', 'last-updated').execute()

    new_tweets = [
        {**tweet, 'source_type': 'twitter'}
        
        for search in config.scrape_config['nitter']
        if search.get('query')
        for tweet in scrap_nitter(
            search_query=search['query'],
            depth=search.get('depth') or -1,
            time_budget=search.get('time_budget') or -1
        )
    ]

    log.info(f"Found {len(new_tweets)} new tweets.")
    
    new_reddit = [
        {**comment, 'source_type': 'reddit'}
        for search in config.scrape_config['reddit']
        if search.get('query')
        for comment in scrape_reddit(
            search_query=search['query'],
            depth=search.get('depth') or -1,
            time_budget=search.get('time_budget') or -1
        )
    ]

    log.info(f"Found {len(new_reddit)} new comments.")

    if not new_tweets and not new_reddit:
        log.info("No new tweets or comments found")
        exit()

    df_tweets = pd.DataFrame(new_tweets)
    df_reddit = pd.DataFrame(new_reddit)

    dfs_to_concat = []
    if not df_tweets.empty:
        dfs_to_concat.append(df_tweets.drop_duplicates(subset=['id'], keep='last'))
    if not df_reddit.empty:
        dfs_to_concat.append(df_reddit.drop_duplicates(subset=['id'], keep='last'))

    full_df = pd.concat(dfs_to_concat, ignore_index=True)

    log.info(f"Processing {len(full_df)} combined records...")

    # Preprocessing
    full_df['processed_text_light'] = full_df['text_content'].apply(lambda x: processing_text(x, level='light'))
    full_df['processed_text_hard'] =  full_df['text_content'].apply(lambda x: processing_text(x, level='hard'))

    # Remove empty rows
    target_cols = ['processed_text_light', 'processed_text_hard']
    full_df[target_cols] = full_df[target_cols].replace(r'^\s*$', np.nan, regex=True)
    full_df.dropna(subset=target_cols, inplace=True)
    log.info(f"Processed {len(full_df)} records.")

    # Predictions
    full_df['sentiment'] = predict_sentiment_batch(full_df['processed_text_hard'])
    full_df['topic_id'] = predict_topics(full_df['processed_text_hard'])

    # Save to database
    if not df_tweets.empty:
        upload_data(full_df, 'tweets', 'twitter')

    if not df_reddit.empty:
        upload_data(full_df, 'reddit_comments', 'reddit')
    
    # NER
    log.info("Starting NER...")
    full_df['entities'] = extract_entities_batch(full_df['processed_text_light'])
    log.info("NER completed.")

    log.info("Starting geocoding LOC entities...")
    run_geocoding(full_df) 
    
    # Prepare entities for database
    save_entities_to_supabase(full_df)

    log.info("Weekly pipeline run completed successfully.")