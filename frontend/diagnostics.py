import streamlit as st
from streamlit.components.v1 import html

def show():
    st.header("Model Performance Page")

    with st.expander("Topics Model Performance", expanded=True):
        with open("visualizations/plot_clusters.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)

        with open("visualizations/kmeans_eval.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)