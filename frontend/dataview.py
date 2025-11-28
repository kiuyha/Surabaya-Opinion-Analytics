import streamlit as st
import pandas as pd

N_COLS = 3
ITEMS_TO_ADD = 15
METRICS_ICONS = {
    "Likes": "https://www.svgrepo.com/show/476608/like.svg",
    "Retweets": "https://www.svgrepo.com/show/447763/retweet.svg",
    "Quotes": "https://www.svgrepo.com/show/365674/quotes-thin.svg",
    "Comments": "https://www.svgrepo.com/show/522067/comment-1.svg",
    "Upvotes": "https://www.svgrepo.com/show/334337/upvote.svg",
    "Downvotes": "https://www.svgrepo.com/show/333916/downvote.svg",
}
SOURCE_STYLES = {
    "Twitter": {
        "icon": "https://www.svgrepo.com/show/475689/twitter-color.svg",
        "color": "#1DA1F2",
        "bg": "rgba(29, 161, 242, 0.1)",
        "label": "Twitter"
    },
    "Reddit": {
        "icon": "https://www.svgrepo.com/show/475675/reddit-color.svg",
        "color": "#FF4500",
        "bg": "rgba(255, 69, 0, 0.1)",
        "label": "Reddit"
    }
}

def show(df: pd.DataFrame):
    with st.container(horizontal=True):
        selected_sentiment = st.selectbox(
            "Filter by Sentiment",
            ["All", "positive", "negative", "neutral"],
            format_func=lambda x: x.title(),
            index=0,
        )
        selected_topic = st.selectbox(
            "Filter by Topic",
            ["All", *df["topic"].unique()],
            format_func=lambda x: "General" if x is None else x,
            index=0
        )
        selected_type = st.selectbox(
            "Filter by Source Type",
            ["All", *df["source_type"].unique()],
            index=0
        )

    if selected_sentiment != "All":
        df = df[df["sentiment"] == selected_sentiment]

    if selected_topic != "All":
        if pd.isna(selected_topic):
            df = df[df["topic"].isnull()]
        else:
            df = df[df["topic"] == selected_topic]
    
    if selected_type != "All":
        df = df[df["source_type"] == selected_type]

    if 'view_limit' not in st.session_state:
        st.session_state.view_limit = ITEMS_TO_ADD

    # Slice the dataframe to get only the rows we want to show right now
    visible_df = df.iloc[:st.session_state.view_limit]

    st.markdown("""
    <style>
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .fullname {
            font-weight: bold;
            font-size: 16px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 140px;
        }
        .username{
            font-size: 12px;
            color: gray;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 140px;
        }
        .platform-badge {
            font-size: 12px;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
        }
        .content-text {
            font-size: 14px;
            line-height: 1.4;
            display: -webkit-box;
            overflow: hidden;
            margin-bottom: 10px;
            color: var(--text-color);
            opacity: 0.9;
        }
        .meta-row {
            display: flex;
            gap: 8px;
            font-size: 12px;
            margin-top: auto;
        }
        .meta-item{
            display: flex;
            gap: 4px;
            align-items: center;
        }
        .toggle-checkbox {
            display: none;
        }
        
        .content-text {
            display: -webkit-box;
            -webkit-line-clamp: 3; /* Show only 3 lines */
            -webkit-box-orient: vertical;
            overflow: hidden;
            transition: all 0.3s ease;
            margin-bottom: 10px;
        }

        .toggle-label {
            font-size: 0.8rem;
            color: var(--primary-color);
            cursor: pointer;
            font-weight: bold;
            text-decoration: underline;
        }

        .toggle-checkbox:checked ~ .content-text {
            -webkit-line-clamp: unset;
            height: auto;
        }

        .toggle-checkbox:checked ~ .toggle-label::after {
            content: "Show Less";
        }
        
        .toggle-checkbox:not(:checked) ~ .toggle-label::after {
            content: "Show More";
        }
    </style>
    """, unsafe_allow_html=True)

    # Loop through the VISIBLE dataframe only
    for i in range(0, len(visible_df), N_COLS):
        # Create the layout columns
        cols = st.columns(N_COLS)
        
        # Get the batch of 3 rows
        batch = visible_df.iloc[i:i + N_COLS]
        
        # Iterate through columns and dataframe rows simultaneously
        for col, (_, row) in zip(cols, batch.iterrows()):
            with col:
                source_type = row.get('source_type', 'unknown')
                style = SOURCE_STYLES.get(source_type)
                
                # Container height ensures uniform grid
                with st.container(border=True, height=280):
                    fullname = row.get('fullname')
                    username = row.get('username', 'Unknown')
                    content = row.get('text_content', 'No content available.')
                    sentiment = row.get('sentiment', 'neutral')
                    topic = row.get('topic')
                    
                    metrics_html = ""
                    link_url = ""
                    link_label = ""
                    if source_type == "Twitter":
                        metrics = {
                            "Likes": row.get('like_count', 0) or 0,
                            "Retweets": row.get('retweet_count', 0) or 0,
                            "Quotes": row.get('quote_count', 0) or 0,
                            "Comments": row.get('comment_count', 0) or 0
                        }
                        link_url = f"https://twitter.com/{username}/status/{row['id']}"
                        link_label = "Open Twitter"

                    elif source_type == "Reddit":
                        metrics = {
                            "Upvotes": row.get('upvote_count', 0) or 0,
                            "Downvotes": row.get('downvote_count', 0) or 0
                        }
                        link_url = row.get('permalink', '#')
                        link_label = "Open Reddit"
                        
                    # Sentiment Color
                    sent_color = {
                        "positive": "#28a745", 
                        "negative": "#dc3545", 
                        "neutral": "#6c757d"
                    }.get(sentiment, "#6c757d")

                    metrics_html = "".join([
                        f"""
                            <div class="meta-item">
                                <img src={METRICS_ICONS.get(metric)} width="16" height="16"/> 
                                <span>{value:,.0f}</span>
                            </div>
                        """

                        for metric, value in metrics.items()
                    ])
                    
                    st.html(f"""
                        <div style="height: 100%; display: flex; flex-direction: column;">
                            <div class="card-header">
                                <div style="display: flex; flex-direction: column;">
                                    <span class="fullname" style="font-weight: bold; font-size: 18px;">
                                        {fullname if not pd.isna(fullname) else username}
                                    </span>

                                    <span class="username">{username if not pd.isna(fullname) else ""}</span>
                                </div>
                                
                                <span class="platform-badge" style="background-color: {style['bg']}; color: {style['color']};">
                                    <img src={style['icon']} width="12" height="12"/>
                                    {style['label']}
                                </span>
                            </div>

                            <div style="margin-top: auto;">
                                <div style="display: flex; gap: 8px; margin-bottom: 8px; flex-wrap: wrap;">
                                    <span style="background: {sent_color}20; color: {sent_color}; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                                        {sentiment.upper()}
                                    </span>
                                    <span style="color: gray; border: 1px solid #ced4da; padding: 1px 6px; border-radius: 4px; font-size: 11px;">
                                        #{topic or 'General'}
                                    </span>
                                </div>
                            </div>
                            
                            <div style="display: flex; flex-direction: column;">
                                <input type="checkbox" id="toggle-{row['id']}" class="toggle-checkbox">
                                
                                <div class="content-text">
                                    {content}
                                </div>

                                <label for="toggle-{row['id']}" class="toggle-label"></label>
                            </div>

                            <div class="meta-row">
                                {metrics_html}
                            </div>
                        </div>
                    """)

                    if link_url:
                        st.link_button(link_label, link_url, use_container_width=True)

    # The "Load More" Button
    if st.session_state.view_limit < len(df):
        st.divider()
        _, col_load_2, _ = st.columns([1, 2, 1])
        with col_load_2:
            if st.button(f"Load More ({len(df) - st.session_state.view_limit} remaining)", type="primary", use_container_width=True):
                st.session_state.view_limit += ITEMS_TO_ADD
                st.rerun()
                
    elif len(df) > 0:
        st.caption(
            "<span style='font-weight: bold; text-align: center'>You have reached the end of the list. </span>",
            unsafe_allow_html=True
        )
    else:
        st.info("No results found matching your filters.")
