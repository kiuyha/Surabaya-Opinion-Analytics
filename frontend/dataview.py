import streamlit as st
import pandas as pd

N_COLS = 3
ITEMS_TO_ADD = 15

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
            index=0
        )

    if selected_sentiment != "All":
        df = df[df["sentiment"] == selected_sentiment]

    if selected_topic != "All":
        df = df[df["topic"] == selected_topic]

    if 'view_limit' not in st.session_state:
        st.session_state.view_limit = ITEMS_TO_ADD

    # Slice the dataframe to get only the rows we want to show right now
    visible_df = df.iloc[:st.session_state.view_limit]

    # Loop through the VISIBLE dataframe only
    for i in range(0, len(visible_df), N_COLS):
        # Create the layout columns
        cols = st.columns(N_COLS)
        
        # Get the batch of 3 rows
        batch = visible_df.iloc[i:i + N_COLS]
        
        # Iterate through columns and dataframe rows simultaneously
        for col, (_, row) in zip(cols, batch.iterrows()):
            with col:
                with st.container(border=True, height=250):
                    username = row.get('username', 'Unknown User')
                    content = row.get('text_content', 'No content')
                    sentiment = row.get('sentiment', 'neutral')
                    topic = row.get('topic')
                    
                    full_topic = f"# {topic}" if topic else ""
                    display_topic = f"# {topic[:30]}..." if len(topic) > 30 else full_topic
                    topic_color = "#CED3D8"

                    if sentiment == "positive":
                        sentiment_color = "#28a745"
                    elif sentiment == "negative":
                        sentiment_color = "#dc3545"
                    else:
                        sentiment_color = "#c3c3c4"

                    st.html(
                        f"""
<div style="margin-bottom: 3px;">
    <div style="font-weight: bold; font-size: 18px; line-height: 1.2; margin-bottom: 4px;">
        {username}
    </div>
    
    <div style="font-size: 14px; line-height: 1.2;">
        <span style="color: {sentiment_color}; font-weight: bold; margin-right: 8px;">
            {sentiment.title()}
        </span>
        
        <span title="{full_topic}" style="font-weight: bold; color: {topic_color};">
            {display_topic}
        </span>
    </div>
</div>
                        """
                    )

                    st.caption(content)
                    tweet_url = f"https://twitter.com/{username}/status/{row['id']}"
                    st.link_button("View on Twitter", tweet_url, width='stretch')
                    

    # The "Load More" Button
    if st.session_state.view_limit < len(df):
        st.write("") 
        if st.button("Load More Tweets", type="primary", width='stretch'):
            st.session_state.view_limit += ITEMS_TO_ADD
            st.rerun()
    elif len(df) > 0:
        st.caption(
            "<span style='font-weight: bold; text-align: center'>You have reached the end of the list. </span>",
            unsafe_allow_html=True
        )