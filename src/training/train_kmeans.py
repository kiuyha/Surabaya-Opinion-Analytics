import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import silhouette_score
from src.core import log, supabase
from src.utils.gemini_api import labeling_cluster
from src.utils.hugging_face import get_hf_token, HF_REPO_KMEANS_ID
from src.preprocess import processing_text
from typing import Tuple, Dict, List, Any
import polars as pl
import numpy as np
import os
from huggingface_hub import HfApi

def text_pipeline(texts: pl.Series, n_components: int) -> Tuple[Pipeline, np.ndarray]:
    """
    Converts a series of text documents into a vector representation using TF-IDF.
    Returns:
        Tuple[Pipeline, np.ndarray]: The lsa_pipeline and the vectorized text data.
    """
    vectorizer = TfidfVectorizer(max_features=2000)

    # Perform dimensionality reduction using SVD to lower the risk of smooth elbow plots
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    
    # Step 3: Create a pipeline that performs both steps in sequence
    lsa_pipeline = make_pipeline(vectorizer, svd)
    
    # Fit the entire pipeline to the data and transform it
    vectors_reduced = lsa_pipeline.fit_transform(texts.to_numpy())
    
    log.info(f"LSA complete. New matrix shape: {vectors_reduced.shape}")

    return lsa_pipeline, vectors_reduced

def find_optimal_k(vectors: np.ndarray, max_k: int = 15) -> int:
    """
    Finds the optimal k using the Silhouette Score instead of the elbow method, which is better for overlapping data.
    """
    log.info("Finding optimal K using Silhouette Score...")
    k_range = range(2, max_k + 1)
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        silhouette_scores.append(score)
        log.info(f"Silhouette Score for K={k}: {score:.4f}")

    # Find the k with the highest silhouette score
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        log.info(f"Optimal K found via Silhouette Score: {optimal_k}")
    else:
        log.warning("Could not calculate silhouette scores. Falling back to K=5.")
        optimal_k = 5

    return optimal_k

def train_cluster_model(vectors: np.ndarray, text_pipeline: Pipeline, num_topics: int) -> Tuple[KMeans, Dict[int, List[str]]]:
    """Trains K-Means and extracts top keywords for each topic."""
    log.info(f"Fitting K-Means with K={num_topics}...")
    kmeans_model = KMeans(n_clusters=num_topics, random_state=42, n_init='auto')
    kmeans_model.fit(vectors)
    log.info("K-Means fitting complete.")

    # Get the actual terms (words) from the lsa_pipeline
    log.info(f"Extracting top keywords for the {num_topics} discovered topics...")
    terms = text_pipeline.named_steps['tfidfvectorizer'].get_feature_names_out()

    # Get the SVD component matrix
    svd_components = text_pipeline.named_steps['truncatedsvd'].components_

    # Map cluster centers back to the original feature space
    original_space_centroids = kmeans_model.cluster_centers_.dot(svd_components)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    keywords_by_topic = {}
    for i in range(num_topics):
        top_terms = [terms[ind] for ind in order_centroids[i, :20]]
        keywords_by_topic[i] = top_terms
        log.info(f"Topic #{i}: {', '.join(top_terms)}")
    
    return kmeans_model, keywords_by_topic

def save_to_supabase(keywords_by_topic: Dict[int, List[str]])-> List[Dict[str, Any]]:
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

def push_models_to_hf(kmeans_model: KMeans, text_pipeline: Pipeline):
    """Saves models locally, then pushes them to the Hugging Face Hub."""
    log.info(f"Preparing to push models to Hugging Face Hub repo: {HF_REPO_KMEANS_ID}")
    
    # Get token and create API client
    token = get_hf_token()
    api = HfApi()

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO_KMEANS_ID, token=token, exist_ok=True)
    
    # Save models locally first
    kmeans_filename = "kmeans_model.joblib"
    text_pipeline_filename = "text_pipeline.joblib"
    joblib.dump(kmeans_model, kmeans_filename)
    joblib.dump(text_pipeline, text_pipeline_filename)
    
    # Upload files to the Hugging Face Hub
    try:
        log.info(f"Uploading {kmeans_filename}...")
        api.upload_file(
            path_or_fileobj=kmeans_filename,
            path_in_repo=kmeans_filename,
            repo_id=HF_REPO_KMEANS_ID,
            token=token
        )
        
        log.info(f"Uploading {text_pipeline_filename}...")
        api.upload_file(
            path_or_fileobj=text_pipeline_filename,
            path_in_repo=text_pipeline_filename,
            repo_id=HF_REPO_KMEANS_ID,
            token=token
        )
        log.info("Models successfully pushed to Hugging Face Hub.")
    except Exception as e:
        log.error(f"An error occurred while uploading to Hugging Face: {e}")
    finally:
        # Clean up local files
        os.remove(kmeans_filename)
        os.remove(text_pipeline_filename)

if __name__ == "__main__":    
    # Load the tweets from Supabase
    log.info('Loading tweets from Supabase...')
    tweets = supabase.table('tweets').select('id', 'text_content').execute().data
    if not tweets:
        raise Exception("No tweets found in Supabase")
    
    df = pl.DataFrame(tweets)
    
    # Mark training in progress
    supabase.table('app_config').update({"value": True}).eq('key', 'training-in-progress').execute()

    df = df.with_columns(
        pl.col('text_content').map_elements(processing_text)
    )

    # Load the vectorized text data
    lsa_pipeline, vectors = text_pipeline(df['text_content'], n_components=100)

    # Find the best K using the silhouette method
    best_k = find_optimal_k(vectors, max_k=15)

    # Fit the K-Means model
    kmeans_model, keywords = train_cluster_model(vectors, lsa_pipeline ,num_topics=best_k)

    # temporarily add the cluster label to the dataframe
    df = df.with_columns(
        pl.Series(name="cluster_kmeans_label", values=kmeans_model.labels_)
    )

    # Save the discovered topic labels and keywords to your database
    newly_created_topics = save_to_supabase(keywords)
    if newly_created_topics:
        # Create the mapping from K-Means label to permanent DB ID
        kmeans_label_to_db_id = {i: topic['id'] for i, topic in enumerate(newly_created_topics)}
        df = df.with_columns(
            pl.col('cluster_kmeans_label').replace(kmeans_label_to_db_id).alias('topic_id')
        )

        log.info("Updating the 'topic_id' in 'tweets' table...")

        update_data = df[['id', 'topic_id']].to_dicts()
        supabase.table('tweets').upsert(update_data).execute()
        
        log.info("Successfully updated tweet-topic.")

    # Push models to Hugging Face Hub
    push_models_to_hf(kmeans_model, lsa_pipeline)

    # Mark training complete
    supabase.table('app_config').update({"value": False}).eq('key', 'training-in-progress').execute()