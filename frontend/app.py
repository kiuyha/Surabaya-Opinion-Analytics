import streamlit as st
from streamlit_option_menu import option_menu
import dashboard
import diagnostics
import dataview
from dotenv import load_dotenv
import pandas as pd
import os
from supabase import create_client

load_dotenv()
# Initialize Supabase
url= os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if url is None or key is None:
    raise Exception("Missing Supabase URL or key")
supabase = create_client(url, key)

st.set_page_config(layout="wide", page_title="Surabaya Opinion Analysis")

def fetch_all_rows(table_name):
    all_data = []
    page_size = 1000
    current_page = 0

    while True:
        # Fetch one page of data
        response = supabase.table(table_name) \
            .select(
                '*', 'topics(label)', 
                'entities(text, label, start, end, confidence_score, latitude, longitude)'
            ) \
            .order('id', desc=False) \
            .limit(page_size) \
            .offset(current_page * page_size) \
            .execute()
        
        page_of_data = response.data        
        if not page_of_data:
            break
        
        # Add the fetched tweets to our main list and go to the next page
        all_data.extend(page_of_data)
        current_page += 1

    return all_data

@st.cache_data(ttl=None, show_spinner="Loading and processing datasets...")
def load_data():
    tweets_data = fetch_all_rows('tweets')
    reddit_data = fetch_all_rows('reddit_comments')

    if not tweets_data and not reddit_data:
        raise Exception("No data found in Supabase (tweets or reddit)")
    
    df_tweets = pd.DataFrame(tweets_data)
    df_reddit = pd.DataFrame(reddit_data)
    dfs_to_concat = []
    
    if not df_tweets.empty:
        df_tweets['source_type'] = 'Twitter'
        dfs_to_concat.append(df_tweets)
    
    if not df_reddit.empty:
        df_reddit['source_type'] = 'Reddit'
        dfs_to_concat.append(df_reddit)
    
    df = pd.concat(dfs_to_concat, ignore_index=True)
    columns = df.columns
    if 'posted_at' in columns:
        df['posted_at'] = pd.to_datetime(df['posted_at'])
    
    if 'topics' in columns:
        df['topic'] = df['topics'].str.get('label')
        df.drop(columns='topics', inplace=True)
    
    if 'entities' in df.columns:
        print("Halo")
        df = df.explode('entities') 
        entities_df = pd.json_normalize(df['entities'])
        entities_df.columns = [f"entity_{c}" for c in entities_df.columns]
        entities_df.index = df.index
        df = df.join(entities_df).drop(columns='entities')

    print(df.head())
    return df

@st.cache_data(ttl=600)
def load_config():
    response = supabase.table('app_config').select('key', 'value').execute()
    return {
        map['key']: map['value'] 
        for map in response.data
    }

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Model Performance", "Data View"],
    icons=["house", "bar-chart-line", "table"],
    orientation="horizontal",
    styles={
        "nav-link": {
            "font-size": "14px",
            "text-align": "center",
            "margin": "0px",
        },
        "nav-link-selected": {
            "background-color": "#00509E",
            "color": "white",
            "font-weight": "400"
        },
    }
)

config = load_config()
st.html(f"""
    <style>
    .top-right-corner {{
        position: fixed;
        top: 0.8rem;
        left: 5rem; 
        z-index: 99999999;
        font-size: 1rem;
        padding: 5px;
        border-radius: 5px;
        font-weight: 600;
    }}
    </style>
    
    <div class="top-right-corner">
        Last Updated: {
            pd.to_datetime(config['last-updated']).strftime('%Y-%m-%d %H:%M:%S')
        }
    </div>
    """)

if selected == 'Dashboard':
    df = load_data()
    dashboard.show(df)
elif selected == 'Model Performance':
    diagnostics.show()
elif selected == 'Data View':
    df = load_data()
    dataview.show(df)