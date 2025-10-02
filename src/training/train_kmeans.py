from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
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
from huggingface_hub import HfApi
import polars as pl
import numpy as np
import os
import joblib

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

def calculate_coherence_score(top_keywords_by_topic: List[List[str]], original_texts: List[str]) -> float:
    """
    Calculates the C_v coherence score for a set of topics.
    Measure how similar the original text is to the topics according to the keywords.
    Args:
        top_keywords_by_topic: A list of lists, where each inner list contains the top keywords for a topic.
        original_texts: The original list of tweet text strings, used to build the model.
    """
    if not original_texts:
        return 0.0
    
    tokenized_docs = [doc.split() for doc in original_texts]
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=top_keywords_by_topic,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )
    return coherence_model.get_coherence()

def calculate_separation_score(top_keywords_by_topic: List[List[str]]) -> float:
    """
    Calculates the average Jaccard similarity: (size of intersection) / (size of union).
    In the nutshell, this measures how much plagiarism there is between topics.
    A lower score means better topic separation.
    """
    if len(top_keywords_by_topic) <= 1:
        return 0.0

    similarity_scores = []
    for topic1_words, topic2_words in combinations(top_keywords_by_topic, 2):
        set1 = set(topic1_words)
        set2 = set(topic2_words)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            similarity = 0.0
        else:
            similarity = intersection / union
            
        similarity_scores.append(similarity)
    return np.mean(similarity_scores)

def find_top_k_candidates(vectors: np.ndarray, min_k: int = 2, max_k: int = 15, num_candidates: int = 3) -> Dict[int, Dict[str, KMeans]]:
    """
    Finds the top N candidate K values using the Silhouette Score and returns a dictionary containing the trained model and score for each candidate.
    """
    log.info(f"Finding top {num_candidates} candidate K values using Silhouette Score...")
    k_range = range(min_k, max_k + 1)
    
    all_results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(vectors)
        score = silhouette_score(vectors, kmeans.labels_)
        
        all_results.append({
            'k': k,
            'silhouette_score': score,
            'model': kmeans 
        })
        log.info(f"Silhouette Score for K={k}: {score:.4f}")

    if not all_results:
        log.warning("Could not calculate any silhouette scores.")
        return {}

    sorted_results = sorted(all_results, key=lambda x: x['silhouette_score'], reverse=True)
    top_n_candidates = sorted_results[:num_candidates]
    
    final_candidates_dict = {
        candidate['k']: {
            'silhouette_score': candidate['silhouette_score'],
            'model': candidate['model']
        }
        for candidate in top_n_candidates
    }
    
    log.info(f"Top candidates for K: {list(final_candidates_dict.keys())}")
    return final_candidates_dict

def get_keywoards_from_kmeans(kmeans_model: np.ndarray, text_pipeline: Pipeline, num_topics: int) -> Dict[int, List[str]]:
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
    
    return keywords_by_topic

def find_and_train_optimal_model(
        vectors: np.ndarray,
        text_pipeline: Pipeline,
        original_texts: List[str],
        min_k: int = 2,
        max_k: int = 15
    ) -> Tuple[KMeans, Dict[int, List[str]]]: 
    """
    Finds the optimal K using final score: (coherence score, silhouette score, jaccard similarity) and then trains the optimal K-Means model.
    """
    top_candidates = find_top_k_candidates(vectors, min_k, max_k)

    all_results = []
    for k, candidate in top_candidates.items():
        log.info(f"Check final score for K={k}...")
        keywords = get_keywoards_from_kmeans(candidate['model'], text_pipeline, k)
        coherence_score = calculate_coherence_score(keywords, original_texts)
        separation_score = calculate_separation_score(keywords)
        final_score = (coherence_score * 0.5) + (candidate['silhouette_score'] * 0.3) + (( 1 - separation_score) * 0.2)
        
        all_results.append({
            'k': k,
            'model': candidate['model'],
            'final_score': final_score,
            'keywords': keywords
        })
        log.info(f"Final score for K={k}: {final_score:.4f}")

    sorted_results = sorted(all_results, key=lambda x: x['final_score'], reverse=True)
    best_result = sorted_results[0]
    
    log.info(f"Best result for K={best_result['k']}: {best_result['final_score']:.4f}")
    return best_result['model'], best_result['keywords']

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

    # we use pagination method since the supabase only load 1000 tweets at a time in default (could be increased but using pagination is more reliable)
    all_tweets = []
    page_size = 1000
    current_page = 0

    while True:
        # Fetch one page of data
        response = supabase.table('tweets') \
            .select('id', 'text_content') \
            .limit(page_size) \
            .offset(current_page * page_size) \
            .execute()
        
        page_of_tweets = response.data        
        if not page_of_tweets:
            break
        
        # Add the fetched tweets to our main list and go to the next page
        all_tweets.extend(page_of_tweets)
        current_page += 1

    if not all_tweets:
        raise Exception("No tweets found in Supabase")
    
    log.info(f"Loaded {len(all_tweets)} tweets from Supabase.")
    df = pl.DataFrame(all_tweets)
    
    # Mark training in progress
    supabase.table('app_config').update({"value": True}).eq('key', 'training-in-progress').execute()

    df = df.with_columns(
        pl.col('text_content').map_elements(processing_text)
    )

    # Load the vectorized text data
    lsa_pipeline, vectors = text_pipeline(df['text_content'], n_components=100)

    # Find the best K using the silhouette method
    # min_k because if it too small the topic become too generic and max_k because if it too large the topic become too specific
    kmeans_model, keywords = find_and_train_optimal_model(vectors, lsa_pipeline, df['text_content'], min_k=4, max_k=10) 

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