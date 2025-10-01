"""
This file only run for daily pipeline to get new tweets. For the monthly retrain, use training/train_kmeans.py
"""
from datetime import datetime, timezone
from src.core import config, supabase, log
from src.scraper import scrap_nitter
from src.preprocess import processing_text

if __name__ == "__main__":
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
            depth=search.get('depth', -1),
            time_budget=search.get('time_budget', -1)
        )
    ]

    log.info(f"Found {len(new_data)} new tweets.")

    if not new_data:
        log.info("No new tweets found, skipping daily pipeline.")
        exit()

    # Insert new tweets
    supabase.table('tweets').insert(new_data).execute()
    
    # Doing preprocessing
    # ner_data = processing_text(new_data, level='light')
    # kmeans_data = processing_text(new_data, level='hard') 