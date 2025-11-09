from umap import UMAP
import plotly.express as px
import numpy as np
from typing import Dict, cast
import numpy as np
from src.core import log
import os
import pandas as pd

VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_cluster_plot(
    df: pd.DataFrame, 
    vectors: np.ndarray, 
    topic_labels: Dict[int, str]
):
    log.info("Generating interactive scatter plot with named topics...")
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        random_state=42,
        metric='cosine'
    )
    coords_2d = cast(np.ndarray, umap.fit_transform(vectors))

    df = df.copy()
    df['x'] = coords_2d[:, 0]
    df['y'] = coords_2d[:, 1]

    # Map the numeric labels to the generated names, labeling -1 as Junk/Noise
    df['topic_name'] = df['cluster_kmeans_label'].map(topic_labels).fillna('Junk/Noise')

    log.info("Creating Plotly figure...")
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='topic_name',  # Use the new name column for color
        hover_data=['text_content'],
        title="Tweet Clusters Visualization - Surabaya Topics",
        labels={'topic_name': 'Topic'},
        color_discrete_map={'Junk/Noise': "lightgrey"}
    )
    
    fig.update_layout(legend_title="Topics", title_x=0.5)
    fig.update_traces(marker=dict(opacity=0.6))

    output_path = f"{VISUALIZATION_DIR}/tweet_clusters.html"
    fig.write_html(output_path)
    log.info(f"Successfully exported interactive plot to '{output_path}'")

    return fig

def create_kmeans_eval_plot(df: pd.DataFrame, best_k: int):
    log.info("Generating interactive line plot for model evaluation...")
    df_long = df.melt(
        id_vars=['k'], 
        value_vars=['avg_coherence', 'separation_score', 'final_score'],
        value_name='Score'
    )
    
    fig = px.line(
        df_long,
        x='k',
        y='Score',
        color='variable',
        markers=True,
        title="Model Selection: Finding the Optimal K",
        labels={'k': 'Number of Clusters (K)'}
    )
    
    fig.add_vline(
        x=best_k, 
        line_width=2, 
        line_dash="dot", 
        line_color="limegreen",
        annotation_text=f"Best K = {best_k}", 
        annotation_position="top right"
    )
    
    fig.update_layout(hovermode="x unified")

    output_path = f"{VISUALIZATION_DIR}/kmeans_eval.html"
    fig.write_html(output_path)
    log.info(f"Successfully exported interactive plot to '{output_path}'")
    
    return fig