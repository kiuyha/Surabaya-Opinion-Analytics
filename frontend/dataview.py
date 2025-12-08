import streamlit as st
import pandas as pd
import html

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
ENTITY_COLORS = {
    "PERSON": "#e86a56",
    "ORG": "#75f85b",
    "LOC": "#508ae9",
    "LOCATION": "#a0c4ff",
    "DEFAULT": "#abaeb1"
}

def highlight_entities(text, entities_df):
    if not text:
        return "No text available."
    
    # Sort entities by start position to process linearly
    entities = entities_df.dropna(subset=['entity_start', 'entity_end']).sort_values('entity_start')
    
    html_output = []
    current_pos = 0
    
    for _, ent in entities.iterrows():
        start = int(ent['entity_start'])
        end = int(ent['entity_end'])
        label = ent.get('entity_label', 'UNKNOWN')
        conf = ent.get('entity_confidence_score', 0.0)
        
        if start > current_pos:
            html_output.append(html.escape(text[current_pos:start]))
        
        color = ENTITY_COLORS.get(label, ENTITY_COLORS["DEFAULT"])
        tooltip = f"Label: {label} | Confidence: {conf:.2%}"
        
        entity_html = f"""
        <span style="display: inline-flex; flex-direction: column; align-items: center; vertical-align: text-top; margin: 0 2px; line-height: 1;" title="{tooltip}">
            <span style="border-bottom: 4px solid {color}; padding-bottom: 1px; font-weight: 500;">
                {html.escape(text[start:end])}
            </span>
            <span style="font-size: 0.8em; font-weight: bold; text-transform: uppercase; margin-top: 2px; letter-spacing: 0.5px;">
                {label}
            </span>
        </span>
        """
        html_output.append(entity_html)
        current_pos = end
        
    # 3. Append remaining text
    if current_pos < len(text):
        html_output.append(html.escape(text[current_pos:]))
        
    return "".join(html_output)

def close_dialog():
    st.session_state.ner_idx = 0
    st.session_state.dialog_open = False

@st.dialog("Entity Analysis (NER)", width="large", on_dismiss=close_dialog)
def ner_dialog(df_unique: pd.DataFrame, full_df: pd.DataFrame):
    current_idx = st.session_state.ner_idx

    # Boundary checks
    if current_idx < 0:
        current_idx = 0
    if current_idx >= len(df_unique):
        current_idx = len(df_unique) - 1
    
    # Get Data for current view
    row = df_unique.iloc[current_idx]
    tweet_id = row['id']
    text = row.get('processed_text_light', '')
    tweet_entities = full_df[full_df['id'] == tweet_id]

    col_prev, col_info, col_next = st.columns([1, 4, 1], vertical_alignment="center")
    
    with col_prev:
        if st.button("← Previous", disabled=(current_idx <= 0), width="stretch"):
            st.session_state.ner_idx -= 1
            st.rerun()
            
    with col_info:
        st.markdown(f"<div style='text-align: center; color: gray;'>Record {current_idx + 1} of {len(df_unique)}</div>", unsafe_allow_html=True)

    with col_next:
        if st.button("Next →", disabled=(current_idx >= len(df_unique) - 1), width="stretch"):
            st.session_state.ner_idx += 1
            st.rerun()

    styled_text = highlight_entities(text, tweet_entities)
    
    st.html(f"""
    <div style="font-family: sans-serif; font-size: 16px; line-height: 1.8; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: var(--secondary-background-color);">
        {styled_text}
    </div>
    """)
    
    st.caption("Hover over highlighted entities to see confidence scores.")

    # Show raw table of entities below
    with st.expander("View Entity Data Table"):
        if not tweet_entities.empty:
            cols_to_show = ['entity_text', 'entity_label', 'entity_confidence_score', 'entity_start', 'entity_end']
            # Filter columns that actually exist
            available_cols = [c for c in cols_to_show if c in tweet_entities.columns]
            st.dataframe(tweet_entities[available_cols].sort_values('entity_start'), hide_index=True, width='stretch')
        else:
            st.info("No entities detected in this record.")

def show(df: pd.DataFrame, df_unique: pd.DataFrame):
    if 'view_limit' not in st.session_state:
        st.session_state.view_limit = ITEMS_TO_ADD

    if 'ner_idx' not in st.session_state:
        st.session_state.ner_idx = 0
    
    if 'dialog_open' not in st.session_state:
        st.session_state.dialog_open = False
    
    # Slice the dataframe to get only the rows we want to show right now
    visible_df = df_unique.iloc[:st.session_state.view_limit]

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
        for col, (idx, row) in zip(cols, batch.iterrows()):
            absolute_index = i + list(batch.index).index(idx)

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

                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        if st.button("NER", key=f"ner_{row['id']}", width='stretch'):
                            st.session_state.ner_idx = absolute_index
                            st.session_state.dialog_open = True
                            st.rerun()
                    
                    with b_col2:
                         if link_url:
                            st.link_button(link_label, link_url, width='stretch')

    # The "Load More" Button
    if st.session_state.view_limit < len(df_unique):
        st.divider()
        _, col_load_2, _ = st.columns([1, 2, 1])
        with col_load_2:
            if st.button(f"Load More ({len(df_unique) - st.session_state.view_limit} remaining)", type="primary", width='stretch'):
                st.session_state.view_limit += ITEMS_TO_ADD
                st.rerun()
    elif len(df_unique) > 0:
        st.caption(
            "<span style='font-weight: bold; text-align: center'>You have reached the end of the list. </span>",
            unsafe_allow_html=True
        )
    else:
        st.info("No results found matching your filters.")

    if st.session_state.dialog_open:
        ner_dialog(df_unique, df)
