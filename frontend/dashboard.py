import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from matplotlib.figure import Figure

def show(df: pd.DataFrame, df_unique=pd.DataFrame()):
    top_word_pos, top_word_neg = st.columns(2, border=True)

    with top_word_pos:
        pos_text = ' '.join(df_unique[df_unique["sentiment"] == "positive"]['processed_text_hard'].astype(str).fillna('')).split()
        if pos_text:
            pos_counts = Counter(pos_text)
            df_pos = pd.DataFrame(pos_counts.most_common(), columns=['Label', 'Count'])
            universal_chart(df_pos, 'Label', 'Count', "Positive Sentiment", enable_cloud=True)
        else:
            st.info("No positive data.")

    with top_word_neg:
        neg_text = ' '.join(df_unique[df_unique["sentiment"] == "negative"]['processed_text_hard'].astype(str).fillna('')).split()
        if neg_text:
            neg_counts = Counter(neg_text)
            df_neg = pd.DataFrame(neg_counts.most_common(), columns=['Label', 'Count'])
            universal_chart(df_neg, 'Label', 'Count', "Negative Sentiment", enable_cloud=True)
        else:
            st.info("No negative data.")

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

    amount_data, top_topic, sentiment_dist = st.columns(3, border=True)

    with amount_data:
        total_data = len(df_unique)
        st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Total Opinions</div>
                <div class="metric-value">{total_data:,}</div>
                <div class="metric-text">Opinions</div> 
            </div>
        """)

    with top_topic:
        most_common_topic = df_unique["topic"].mode()[0] if df_unique["topic"].mode().shape[0] > 0 else "General"
        st.html(f"""
            <div class="metric-card">
                <div class="metric-label">Top Trending Topic</div>
                <div class="metric-text" title="{most_common_topic}">{most_common_topic}</div>
            </div>
        """)

    with sentiment_dist:
        sent_counts = df_unique["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Label", "Count"]
        universal_chart(
            sent_counts,
            "Label",
            "Count",
            "Sentiment Distribution",
            enable_cloud=False,
            default_config={
                'view': "Pie",
                'is_horizontal': False
            }
        )

    top_org, top_per = st.columns(2, border=True)  
    
    with top_org:
        if 'entity_label' in df.columns:
            org_series = df[df['entity_label'] == 'ORG']['entity_text']
            if not org_series.empty:
                df_org = org_series.value_counts().reset_index()
                df_org.columns = ['Label', 'Count']
                universal_chart(df_org, 'Label', 'Count', "Top Organizations", enable_cloud=False)
            else:
                st.info("No organizations found.")
        else:
            st.warning("Entity data missing.")

    with top_per:
        if 'entity_label' in df.columns:
            per_series = df[df['entity_label'] == 'PERSON']['entity_text']
            if not per_series.empty:
                df_per = per_series.value_counts().reset_index()
                df_per.columns = ['Label', 'Count']
                universal_chart(df_per, 'Label', 'Count', "Top Persons", enable_cloud=False)
            else:
                st.info("No persons found.")
        else:
            st.warning("Entity data missing.")
@st.fragment
def universal_chart(df: pd.DataFrame, label_col: str, value_col: str, title: str, enable_cloud: bool = False, default_config: dict = {}):
    head_col, settings_col = st.columns([0.80, 0.15], vertical_alignment="center")
    
    with head_col:
        st.subheader(title)

    limit = default_config.get("limit", 10 if df.shape[0] > 10 else df.shape[0])
    view_type = default_config.get("view", "Bar")
    is_horizontal = default_config.get("is_horizontal", False)

    with settings_col:
        with st.popover("⚙️", width='stretch', help="Chart Settings"):
            st.markdown("##### Display Settings")
            
            options = ["Bar", "Pie"]
            if enable_cloud:
                options.append("Cloud")
            
            view_type = st.pills(
                "Chart Type",
                options=options,
                default=view_type,
                selection_mode="single",
                key=f"view_{title}"
            )

            if view_type == "Bar":
                bar_orientation = st.radio(
                    "Orientation",
                    ["Horizontal", "Vertical"],
                    horizontal=is_horizontal,
                    key=f"orient_{title}"
                )
                is_horizontal = bar_orientation == "Horizontal"
            
            if view_type != "Cloud":
                limit = st.number_input(
                    "Item Limit", 
                    min_value=1, 
                    max_value=df.shape[0], 
                    value=limit, 
                    step=1, 
                    key=f"limit_{title}"
                )

    if df.empty:
        st.info("No data available.")
        return

    df_show = df.head(limit).copy()

    if view_type == 'Bar':
        x_data = value_col if is_horizontal else label_col
        y_data = label_col if is_horizontal else value_col
        orientation_code = 'h' if is_horizontal else 'v'
        chart_margins = dict(l=0, r=50, t=0, b=0) if is_horizontal else dict(l=0, r=0, t=50, b=0)

        fig = px.bar(
            df_show, 
            x=x_data, 
            y=y_data, 
            orientation=orientation_code,
            text=value_col,
            color_discrete_sequence=["#0068C9"] 
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title=None,
            yaxis_title=None,
            yaxis={'categoryorder':'total ascending'} if is_horizontal else {},
            xaxis={'categoryorder':'total descending'} if not is_horizontal else {},
            margin=chart_margins,
            height=300,
            showlegend=False,
        )
        
        fig.update_traces(
            textposition='outside', 
            cliponaxis=False,
            marker_line_width=0
        )
        
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
    
    elif view_type == 'Pie':
        fig = px.pie(
            df_show, 
            values=value_col, 
            names=label_col, 
            hole=0.5,
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=300
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    elif view_type == 'Cloud' and enable_cloud:
        word_data = dict(zip(df[label_col], df[value_col]))
        
        wc = WordCloud(
            width=500, 
            height=300,
            background_color="white",
            colormap="viridis",
            prefer_horizontal=0.9
        ).generate_from_frequencies(word_data)

        fig = Figure(figsize=(5, 3.5))
        ax = fig.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.tight_layout(pad=0)
        
        st.pyplot(fig, width='stretch')