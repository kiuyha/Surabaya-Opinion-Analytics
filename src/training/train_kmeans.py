import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from kneed import KneeLocator
from src.core import log, supabase
from src.utils.gemini_api import labeling_cluster
from src.utils.hugging_face import get_hf_token, HF_REPO_KMEANS_ID
from typing import Tuple, Dict, List, Any
import polars as pl
import numpy as np
import os
from huggingface_hub import HfApi

def vectorize(texts: pl.Series) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Converts a series of text documents into a TF-IDF matrix.
    Returns:
        Tuple[TfidfVectorizer, np.ndarray]: The vectorizer and the TF-IDF matrix
    """
    vectorizer = TfidfVectorizer(max_features=2000)
    X_vectorized = vectorizer.fit_transform(texts)

    return vectorizer, X_vectorized

def find_optimal_k(vectors: np.ndarray, max_k: int=15)-> int:
    """
    Calculates and plots the inertia for a range of k values to find the optimal k using the elbow method.

    Args:
        vectors (np.ndarray): The vectorized text data.
        feature_name (str): Name of the vectorization method for plot title.
        max_k (int): The maximum number of clusters to test.

    Returns:
        int: The optimal number of clusters (k).
    """

    log.info(f"Finding optimal K...")
    k_range = range(2, max_k + 1) # at least it would have 2 clusters
    inertia_values = [
        KMeans(n_clusters=k, random_state=42).fit(vectors).inertia_
        for k in k_range
    ]

    # The KneeLocator finds the point of maximum curvature in the plot
    kn = KneeLocator(list(k_range), inertia_values, curve='convex', direction='decreasing')
    optimal_k = kn.elbow

    if optimal_k:
        log.info(f"Optimal K found: {optimal_k}")
    else:
        log.info("Could not automatically find an elbow. Please inspect the plot.")
        optimal_k = 5 # Fallback to a default value if no elbow is found

    return optimal_k

def train_cluster_model(vectors: np.ndarray, vectorizer: TfidfVectorizer, num_topics: int) -> Tuple[KMeans, Dict[int, List[str]]]:
    """Trains K-Means and extracts top keywords for each topic."""
    log.info(f"Fitting K-Means with K={num_topics}...")
    kmeans_model = KMeans(n_clusters=num_topics, random_state=42, n_init='auto')
    kmeans_model.fit(vectors)
    log.info("K-Means fitting complete.")

    # Get the actual terms (words) from the vectorizer
    log.info(f"Extracting top keywords for the {num_topics} discovered topics...")
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]

    keywords_by_topic = {}
    for i in range(num_topics):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        keywords_by_topic[i] = top_terms
        log.info(f"Topic #{i}: {', '.join(top_terms)}")
    
    return kmeans_model, keywords_by_topic

def save_to_supabase(keywords_by_topic: Dict[int, List[str]])-> List[str, Any]:
    """Saves the topic labels and keywords to the 'topics' table in Supabase."""
    log.info("updating topic labels and keywords to Supabase...")

    try:
        log.info("Clearing old topics from the database...")
        supabase.table('topics').delete().gte('id', 0).execute()

        cluster_labels = labeling_cluster(keywords_by_topic)
        data_to_insert = [
            {
                "keywords": keywords_by_topic[index],
                "label": cluster_labels[index]
            } 
            for index in keywords_by_topic.keys()
        ]

        if data_to_insert:
            response = supabase.table('topics').insert(data_to_insert).execute()
            log.info(f"Successfully inserted {len(data_to_insert)} topics into Supabase.")
            return response.data
        else:
            log.warning("No new topics were generated to save.")
            return []
    except Exception as e:
        log.error(f"Failed to update topics in Supabase: {e}", exc_info=True)
        return []

def push_models_to_hf(kmeans_model: KMeans, vectorizer: TfidfVectorizer):
    """Saves models locally, then pushes them to the Hugging Face Hub."""
    log.info(f"Preparing to push models to Hugging Face Hub repo: {HF_REPO_KMEANS_ID}")
    
    # Get token and create API client
    token = get_hf_token()
    api = HfApi()

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO_KMEANS_ID, token=token, exist_ok=True)
    
    # Save models locally first
    kmeans_filename = "kmeans_model.joblib"
    vectorizer_filename = "vectorizer.joblib"
    joblib.dump(kmeans_model, kmeans_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    # Upload files to the Hugging Face Hub
    try:
        log.info(f"Uploading {kmeans_filename}...")
        api.upload_file(
            path_or_fileobj=kmeans_filename,
            path_in_repo=kmeans_filename,
            repo_id=HF_REPO_KMEANS_ID,
            token=token
        )
        
        log.info(f"Uploading {vectorizer_filename}...")
        api.upload_file(
            path_or_fileobj=vectorizer_filename,
            path_in_repo=vectorizer_filename,
            repo_id=HF_REPO_KMEANS_ID,
            token=token
        )
        log.info("Models successfully pushed to Hugging Face Hub.")
    except Exception as e:
        log.error(f"An error occurred while uploading to Hugging Face: {e}")
    finally:
        # Clean up local files
        os.remove(kmeans_filename)
        os.remove(vectorizer_filename)

if __name__ == "__main__":    
    # Load the tweets from Supabase
    log.info('Loading tweets from Supabase...')
    tweets = supabase.table('tweets').select('text_content').execute().data
    if not tweets:
        raise Exception("No tweets found in Supabase")
    
    df = pl.DataFrame(tweets)
    
    # Mark training in progress
    supabase.table('app_config').uplate({"value": True}).eq('key', 'training-in-progress').execute()

    # Load the vectorized text data
    vectorizer, vectors = vectorize(df['text_content'])

    # Find the best K using the elbow method
    best_k = find_optimal_k(vectors, max_k=15)

    # Fit the K-Means model
    kmeans_model, keywords = train_cluster_model(vectors, vectorizer ,num_topics=best_k)

    # temporarily add the cluster label to the dataframe
    df_with_labels = df.with_columns(
        pl.Series(name="cluster_kmeans_label", values=kmeans_model.labels_)
    )

    # Save the discovered topic labels and keywords to your database
    newly_created_topics = save_to_supabase(keywords)
    if newly_created_topics:
        # Create the mapping from K-Means label to permanent DB ID
        kmeans_label_to_db_id = {i: topic['id'] for i, topic in enumerate(newly_created_topics)}
        df_with_topics = df_with_labels.with_columns(
            pl.col('cluster_kmeans_label').replace(kmeans_label_to_db_id).alias('topic_id')
        )

        log.info("Updating the 'topic_id' in 'tweets' table...")

        update_data = df_with_topics[['id', 'topic_id']].to_dicts()
        supabase.table('tweets').upsert(update_data).execute()
        
        log.info("Successfully updated tweet-topic relationships.")

    # Push models to Hugging Face Hub
    push_models_to_hf(kmeans_model, vectorizer)

    # Mark training complete
    supabase.table('app_config').uplate({"value": False}).eq('key', 'training-in-progress').execute()