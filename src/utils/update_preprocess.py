from src.training.train_topics_model import fetch_all_rows
from src.core import supabase, log
from src.preprocess import processing_text

def update_preprocess_text(table_name):
    # fetch all data
    all_data = fetch_all_rows(table_name, ['id', 'text_content'])

    # update preprocess
    all_data = [
        {
            **data,
            'processed_text_light': processing_text(data['text_content'], level='light'),
            'processed_text_hard': processing_text(data['text_content'], level='hard')
        }
        for data in all_data
    ]

    # save to supabase
    supabase.table(table_name).upsert(all_data).execute()
    return

if __name__ == "__main__":
    log.info("Updating tweets...")
    update_preprocess_text('tweets')

    log.info("Updating reddit comments...")
    update_preprocess_text('reddit_comments')