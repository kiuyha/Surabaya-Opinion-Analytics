import streamlit as st
from streamlit_option_menu import option_menu
import dashboard
import diagnostics
import dataview
from dotenv import load_dotenv
import pandas as pd
import os
from supabase import create_client

st.set_page_config(layout="wide", page_title="Surabaya Opinion Analysis")

load_dotenv()
# Initialize Supabase
url= os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if url is None or key is None:
    raise Exception("Missing Supabase URL or key")
supabase = create_client(url, key)

@st.cache_data(ttl=600)
def load_config():
    response = supabase.table('app_config').select('key', 'value').execute()
    return {
        map['key']: map['value'] 
        for map in response.data
    }

last_updated = load_config().get('last-updated', None)
# last updated label
st.html(f"""
    <style>
    .top-right-corner {{
        position: fixed;
        top: 0.8rem;
        left: 5rem; 
        z-index: 999991;
        font-size: 1rem;
        padding: 5px;
        border-radius: 5px;
        font-weight: 600;
    }}
    </style>
    
    <div class="top-right-corner">
        Last Updated: {
            pd.to_datetime(last_updated).strftime('%Y-%m-%d %H:%M:%S')
        }
    </div>
    """)

def fetch_all_rows(table_name):
    all_data = []
    page_size = 1000
    current_page = 0

    while True:
        # Fetch one page of data
        response = supabase.table(table_name) \
            .select(
                '*', 'topics(label)', 
                'entities(text, label, start, end, confidence_score, normalized_address, latitude, longitude)'
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
def load_data(last_updated):
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
        df.drop(columns=['topics', 'topic_id'], inplace=True)
    
    if 'entities' in df.columns:
        df = df.explode('entities').reset_index(drop=True)
        entities_df = pd.json_normalize(df['entities'])
        entities_df.columns = [f"entity_{c}" for c in entities_df.columns]
        df.drop(columns=['entities'], inplace=True)
        df = pd.concat([df, entities_df], axis=1)
    return df

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

def show_data_filter(df: pd.DataFrame, header: str):
    st.header(header)

    df_unique = df.drop_duplicates(subset="id") if 'id' in df.columns else df.copy()

    min_date = df_unique["posted_at"].min().date()
    max_date = df_unique["posted_at"].max().date()
    unique_orgs = df[df['entity_label'] == 'ORG']['entity_text'].value_counts().index
    unique_pers = df[df['entity_label'] == 'PERSON']['entity_text'].value_counts().index

    with st.container(horizontal=True):
        selected_start = st.date_input(
            "Filter by Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        selected_end = st.date_input(
            "Filter by End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )
        selected_topics = st.multiselect(
            "Filter by Topic",
            options=df_unique["topic"].unique(),
            format_func=lambda x: "General" if x is None else x,
            placeholder="Choose topics (leave empty for all)",
        )

    with st.container(horizontal=True):
        selected_orgs = st.multiselect(
            "Filter by Organization",
            options=unique_orgs,
            placeholder="Choose organizations (leave empty for all)",
        )
        selected_pers = st.multiselect(
            "Filter by Person",
            options=unique_pers,
            placeholder="Choose people (leave empty for all)",
        )
        selected_sources = st.multiselect(
            "Filter by Source Type",
            options=df_unique["source_type"].unique(),
            placeholder="Choose source types (leave empty for all)",
        )
    
    if selected_start:
        start_ts = pd.to_datetime(selected_start).tz_localize("UTC")
        df_unique = df_unique[df_unique["posted_at"] >= start_ts]
        df = df[df["posted_at"] >= start_ts]

    if selected_end:
        end_ts = (pd.to_datetime(selected_end) + pd.Timedelta(days=1)).tz_localize("UTC")
        df_unique = df_unique[df_unique["posted_at"] < end_ts]
        df = df[df["posted_at"] < end_ts]
    
    if selected_topics:
        df_unique = df_unique[df_unique["topic"].isin(selected_topics)]
        df = df[df["topic"].isin(selected_topics)]
    
    if selected_orgs:
        df = df[
            (df['entity_label'] == 'ORG') & 
            (df['entity_text'].isin(selected_orgs))
        ]
        df_unique = df_unique[df_unique['id'].isin(df['id'].unique())]

    if selected_pers:
        df = df[
            (df['entity_label'] == 'PERSON') & 
            (df['entity_text'].isin(selected_pers))
        ]
        df_unique = df_unique[df_unique['id'].isin(df['id'].unique())]
    
    if selected_sources:
        df_unique = df_unique[df_unique["source_type"].isin(selected_sources)]
        df = df[df["source_type"].isin(selected_sources)]

    return df, df_unique

if selected == 'Dashboard':
    df = load_data(last_updated)
    df, df_unique = show_data_filter(df, "Dashboard Page")
    dashboard.show(df, df_unique)

elif selected == 'Model Performance':
    diagnostics.show()

elif selected == 'Data View':
    df = load_data(last_updated)
    df, df_unique = show_data_filter(df, "Data View Page")
    dataview.show(df, df_unique)