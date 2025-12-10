import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
from matplotlib.figure import Figure
from dataview import highlight_entities
from config import SENTIMENT_COLORS
import textwrap

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
            org_series = df[df['entity_label'] == 'ORG']['entity_text'].str.title()
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
            per_series = df[df['entity_label'] == 'PERSON']['entity_text'].str.title()
            if not per_series.empty:
                df_per = per_series.value_counts().reset_index()
                df_per.columns = ['Label', 'Count']
                universal_chart(df_per, 'Label', 'Count', "Top Persons", enable_cloud=False)
            else:
                st.info("No persons found.")
        else:
            st.warning("Entity data missing.")
    
    render_map(df)
    
@st.fragment
def render_map(df: pd.DataFrame):
    with st.container(border=True):
        st.subheader("Map View")
        df_map = df.dropna(subset=['entity_latitude', 'entity_longitude']).copy()

        # Filtering data
        confidence_filter, sentiment_filter = st.columns(2)

        with confidence_filter:
            min_confidence = st.slider(
                "Filter by NER Confidence Score", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.5, 
                step=0.01,
                help="Adjust to hide locations where the model was less sure about the extraction.",
            )

        with sentiment_filter:
            sentiment_filter = st.multiselect(
                "Filter by Sentiment",
                options=df_map["sentiment"].unique(),
                placeholder="Choose sentiments (leave empty for all)",
                default=['positive', 'negative']
            )
        
        df_map = df_map[df_map['entity_confidence_score'] >= min_confidence]
        if sentiment_filter:
            df_map = df_map[df_map['sentiment'].isin(sentiment_filter)]

        show_selected_record = st.checkbox(
            "Show selected record",
            value=True,
            help="Check to show the selected record section beside the map."
        )

        if df_map.empty:
            st.warning(f"No locations found with confidence score >= {min_confidence}")
            return

        def wrap_text(text, width=40):
            if not isinstance(text, str): return str(text)
            return "<br>".join(textwrap.wrap(text, width=width))

        df_map['wrapped_address'] = df_map['entity_normalized_address'].apply(lambda x: wrap_text(x))

        cols_to_sum = [
            'like_count', 'comment_count', 'retweet_count', 
            'quote_count', 'upvote_count', 'downvote_count'
        ]

        df_map[cols_to_sum] = df_map[cols_to_sum].fillna(0)
        
        df_map['total_engagement'] = (
            df_map['like_count'] + 
            df_map['comment_count'] + 
            df_map['retweet_count'] + 
            df_map['quote_count'] + 
            df_map['upvote_count'] + 
            df_map['downvote_count']
        )

        cap_value = df_map['total_engagement'].quantile(0.95)
        if cap_value == 0:
            cap_value = 1 

        df_map['size_scaled'] = (df_map['total_engagement'].clip(upper=cap_value) / cap_value) * 20

        col_map, col_details = st.columns(
            [2, 1],
            gap="medium",
        )

        if not show_selected_record:
            col_map = st.empty()

        with col_map:
            fig = px.scatter_mapbox(
                df_map,
                lat="entity_latitude",
                lon="entity_longitude",
                size="size_scaled",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                hover_name="wrapped_address", 
                hover_data={
                    "entity_confidence_score": ":.2f",
                    "entity_text": True,
                    "total_engagement": True,
                    "source_type": True,
                    "entity_latitude": False,
                    "entity_longitude": False,
                    "size_scaled": False,
                },
                zoom=11,
                center={"lat": -7.2575, "lon": 112.7521}, # Centered on Surabaya
                mapbox_style="carto-positron",
                opacity=0.6,
                height=600
            )

            fig.update_traces(marker=dict(sizemin=5, sizeref=0.1))
            fig.update_layout(
                legend=dict(
                    orientation="v",
                    xanchor="left",
                    x=0,
                    font=dict(weight=400)
                ),
                margin=dict(l=0, r=0, t=0, b=0)
            )

            event = st.plotly_chart(
                fig, width='stretch',
                on_select="rerun" if show_selected_record else 'ignore', 
                selection_mode="points"
            )
    
        if show_selected_record:
            with col_details:
                st.subheader("Selected Record")
                selected_points = event.selection.get("points", [])
                
                if not selected_points:
                    st.info("Click on a bubble in the map to view the full text and entities.")
                    return
                clicked_idx = selected_points[0]["point_index"]
                try:
                    selected_row = df_map.iloc[clicked_idx]
                    selected_id = selected_row['id']
                except IndexError:
                    st.error("Selection error. Please try again.")
                    return

                tweet_entities = df[df['id'] == selected_id]
                raw_text = selected_row.get('processed_text_light', '')

                styled_html = highlight_entities(raw_text, tweet_entities)

                st.html(f"""
                <div style="border:1px solid #ddd; padding:15px; border-radius:10px; background-color:var(--secondary-background-color);">
                    <div style="margin-bottom: 10px; font-weight: bold; color: gray;">
                        ID: {selected_id} <br>
                        Source: {selected_row['source_type'].title()}
                    </div>
                    {styled_html}
                </div>
                """)
        
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