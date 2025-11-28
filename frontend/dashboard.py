import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from matplotlib.figure import Figure

def show(df: pd.DataFrame):
    st.header("Dashboard Page")
    st.markdown("""
        <style>
            .metric-card {
                background-color: var(--secondary-background-color);
                border: 1px solid var(--secondary-background-color);
                border-radius: 10px;
                padding: 20px;
                text-align: center;
            }
            .metric-label {
                font-size: 1.2rem;
                margin-bottom: 5px;
                text-transform: uppercase;
                font-weight: 600;
            }
            .metric-value {
                font-size: 7rem;
                font-weight: 700;
                margin: 0;
                color: var(--text-color);
            }
            .metric-text {
                font-size: 2rem;
                font-weight: 700;
                margin: 0;
                color: var(--primary-color);
                overflow: hidden;
                text-overflow: ellipsis;
            }
        </style>
    """, unsafe_allow_html=True)

    min_date = df["posted_at"].min().date()
    max_date = df["posted_at"].max().date()
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
            max_value=max_date
        )
        selected_topic = st.selectbox(
            "Filter by Topic",
            ["All", *df["topic"].unique()],
            format_func=lambda x: "General" if x is None else x,
            index=0
        )
    
    if selected_start:
        start_ts = pd.to_datetime(selected_start).tz_localize("UTC")
        df = df[df["posted_at"] >= start_ts]

    if selected_end:
        end_ts = (pd.to_datetime(selected_end) + pd.Timedelta(days=1)).tz_localize("UTC")
        df = df[df["posted_at"] < end_ts]
    
    if selected_topic != "All":
        if pd.isna(selected_topic):
            df = df[df["topic"].isnull()]
        else:
            df = df[df["topic"] == selected_topic]

    col_pos, col_neg = st.columns(2, border=True)

    with col_pos:
        top_keywords(df[df["sentiment"] == "positive"], "Positive Sentiment")

    with col_neg:
        top_keywords(df[df["sentiment"] == "negative"], "Negative Sentiment")

    amount_data, top_topic, sentiment_pie = st.columns(3, border=True)

    with amount_data:
        total_data = len(df)
        st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Total Opinions</div>
                <div class="metric-value">{total_data:,}</div>
                <div class="metric-text">Opinions</div> 
            </div>
        """)

    with top_topic:
        most_common_topic = df["topic"].mode()[0] if df["topic"].mode().shape[0] > 0 else "General"
        st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Top Trending Topic</div>
                <div class="metric-text" title="{most_common_topic}">{most_common_topic}</div>
            </div>
        """)

    with sentiment_pie:
        st.subheader("Sentiment Distribution")
        make_pie_plot(df["sentiment"].value_counts(), ["positive", "negative", "neutral"], 'count')

@st.fragment
def top_keywords(df: pd.DataFrame, title: str):
    with st.container():
        st.subheader(title)
        
        view_type = st.pills(
            "Chart Type",
            options=["Bar", "Cloud", "Pie"],
            default="Bar",
            selection_mode="single",
            key=f"view_{title}",
            label_visibility="collapsed"
        )
        st.write("")

    if df.empty:
        st.info("No data available for this sentiment.")
        return

    all_keywords = ' '.join(df['processed_text_hard'].astype(str).fillna('')).split()
    word_counts = Counter(all_keywords)
    df_keywords = pd.DataFrame(word_counts.most_common(10), columns=['Keyword', 'Count'])

    if view_type == 'Bar':
        fig = px.bar(
            df_keywords, 
            x='Count', 
            y='Keyword', 
            orientation='h',
            text='Count',
            color='Count',
            color_continuous_scale='Blues' 
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_visible=False,
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=0, r=20, t=0, b=0),
            height=300,
            showlegend=False
        )
        fig.update_traces(textposition='outside', cliponaxis=False)
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    elif view_type == 'Cloud':
        wc = WordCloud(
            width=500, 
            height=300,
            background_color="white",
            prefer_horizontal=0.9
        ).generate_from_frequencies(word_counts)

        fig = Figure(figsize=(5, 3.5))
        ax = fig.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.tight_layout(pad=0)
        
        st.pyplot(fig, width='stretch')
    
    elif view_type == 'Pie':
        make_pie_plot(df_keywords, 'Keyword', 'Count')


def make_pie_plot(df: pd.DataFrame, names: str|list, values: str|list):
    fig = px.pie(df, values=values, names=names, hole=0.5)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )

    # Hide percentage labels if they are too small to avoid clutter
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})