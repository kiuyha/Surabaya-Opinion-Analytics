import streamlit as st
from streamlit.components.v1 import html
import pandas as pd

def show():
    st.header("Model Performance Page")

    with st.expander("Topics Model Performance", expanded=True):
        with open("visualizations/kmeans_eval.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)
            
        with open("visualizations/plot_clusters.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)
    
    with st.expander("NER Model Performance", expanded=True):
        with st.container(border=True):
            st.subheader("Training Logs")
            view_type = st.pills(
                "Chart Type",
                options=["Plot", "Table"],
                default="Plot",
                selection_mode="single",
                # key=f"view_{}",
                label_visibility="collapsed"
            )

            if view_type == 'Table':
                st.dataframe(
                    pd.read_csv('visualizations/training_logs.csv')
                )
            else:
                with open("visualizations/ner_train_curve.html", "r", encoding='utf-8') as f:
                    html(f.read(), height=500, scrolling=True)

        with open("visualizations/ner_classification_report.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)
        with open("visualizations/ner_confusion_matrix.html", "r", encoding='utf-8') as f:
            html(f.read(), height=500, scrolling=True)
