from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from src.core import log, supabase, config
from src.utils.gemini_api import labeling_cluster
from src.preprocess import processing_text
from typing import Tuple, Dict, List, Any, cast
from huggingface_hub import HfApi
import polars as pl
import numpy as np
import os
import tempfile
import joblib
import plotly.express as px
from gensim.models import FastText
from umap import UMAP

def sentence_vector(doc: str, fastext_model: FastText) -> np.ndarray:
    vectors = [fastext_model.wv[word] for word in doc if word in fastext_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(fastext_model.vector_size)

def text_pipeline(
    texts: pl.Series,
    vector_size: int = 300,
) -> Tuple[FastText, np.ndarray]:
    """
    Converts a series of text documents into a vector representation using FastText
    by averaging word vectors for each document. Optionally applies UMAP dimensionality reduction.
    """
    log.info("Tokenizing text for FastText...")
    tokenized_texts = [doc.split() for doc in texts.to_list()]

    cpu_core = os.cpu_count()
    if cpu_core:
        log.info(f"Number of CPU cores: {cpu_core}")
    else:
        log.info("Number of CPU cores not detected. Defaulting to 4 cores...")
        cpu_core = 4

    log.info(f"Training FastText model with vector_size={vector_size}...")
    fasttext_model = FastText(
        tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=cpu_core,
        seed=42,
    )

    log.info("Creating document vectors by averaging...")
    vectors = [
        sentence_vector(doc, fasttext_model)
        for doc in tokenized_texts
    ]
    
    umap_model = None

    log.info("Normalizing document vectors...")
    vectors = normalize(np.array(vectors), norm='l2')

    log.info(f"Final matrix shape: {vectors.shape}")
    return fasttext_model, vectors

def calculate_coherence_score(
    keywords_per_topic: List[List[str]],
    original_texts: List[str]
) -> Tuple[float, List[float]]:
    """
    Calculates the c_v coherence (it like score to define whether the topic and its original text are making sense)
    for each topic and returns the average score and the list of individual scores.
    """
    tokenized_texts = [doc.split() for doc in original_texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_texts]
    
    individual_scores = []
    for topic_keywords in keywords_per_topic:
        coherence_model = CoherenceModel(
            topics=[topic_keywords],
            texts=tokenized_texts,
            dictionary=dictionary,
            corpus=corpus,
            coherence='c_v'
        )
        score = coherence_model.get_coherence()
        individual_scores.append(score)
        
    average_score = float(np.mean(individual_scores)) if individual_scores else 0.0
    
    return average_score, individual_scores

def calculate_separation_score(top_keywords_by_topic: List[List[str]], text_model: FastText, method: str = 'cosine') -> float:
    """
    Calculates separation using either cosine or Jaccard distance. In the nutshell cheks for plagiarism
    
    - cosine: Measures semantic distinctiveness (recommended for embeddings)
    - jaccard: Measures word overlap distinctiveness (simpler, faster)
    """
    if len(top_keywords_by_topic) <= 1:
        return 0.0

    separation_scores = []
    for topic1_words, topic2_words in combinations(top_keywords_by_topic, 2):
        if method == 'cosine':
            # Semantic separation
            topic1_vecs = [text_model.wv[word] for word in topic1_words if word in text_model.wv]
            topic2_vecs = [text_model.wv[word] for word in topic2_words if word in text_model.wv]

            if len(topic1_vecs) == 0 or len(topic2_vecs) == 0:
                separation_scores.append(1.0)
                continue
                
            vec1 = np.mean(topic1_vecs, axis=0).reshape(1, -1)
            vec2 = np.mean(topic2_vecs, axis=0).reshape(1, -1)
            cosine_sim = cosine_similarity(vec1, vec2)[0][0]
            separation_scores.append(1 - cosine_sim)
            
        elif method == 'jaccard':
            # Word overlap separation
            set1, set2 = set(topic1_words), set(topic2_words)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union > 0:
                jaccard_sim = intersection / union
                separation_scores.append(1 - jaccard_sim)
            else:
                separation_scores.append(0.0)
    return float(np.mean(separation_scores))

def get_keywords_from_kmeans(
    kmeans_model: KMeans,
    fasttext_model: FastText,
    num_topics: int,
    top_n_keywords: int = 50
) -> List[List[str]]:
    """
    Extracts keywords for each topic by finding the words in the vocabulary
    that are most similar to each cluster's centroid vector.

    Args:
        kmeans_model: The trained scikit-learn KMeans model.
        word2vec_model: The trained gensim Word2Vec model.
        num_topics: The number of clusters/topics.
        top_n_keywords: The number of top keywords to extract for each topic.

    Returns:
        A list of lists, where each inner list contains the top keywords for a topic.
    """
    log.info(f"Extracting top {top_n_keywords} keywords for the {num_topics} discovered topics...")
    keywords_by_topic = []

    for cluster_id in range(num_topics):

        # Get the centroid vector for the current cluster
        centroid_vector = kmeans_model.cluster_centers_[cluster_id]

        try:
            top_words = fasttext_model.wv.most_similar(positive=[centroid_vector], topn=top_n_keywords)
            topic_keywords = [word for word, _ in top_words]
            keywords_by_topic.append(topic_keywords)
            log.info(f"Topic #{cluster_id}: {', '.join(topic_keywords)}")
        except KeyError as e:
            log.warning(f"Could not find keywords for Topic #{cluster_id}: a word in the model might be missing. Error: {e}")
            keywords_by_topic.append([]) # Append empty list for this topic

    return keywords_by_topic

def find_and_train_optimal_model(
    vectors: np.ndarray,
    text_model: FastText,
    original_texts: List[str],
    min_k: int = 2,
    max_k: int = 15,
    abs_coherence_threshold: float = 0.45,  # The absolute quality floor
    rel_coherence_drop: float = 0.80      # Flag if score is less than 80% of the median
) -> Tuple[KMeans, List[List[str]], List[float] ,List[int]]: 
    """
    Finds the optimal K using final score: (coherence score, silhouette score, jaccard similarity) and then trains the optimal K-Means model.
    """

    all_results = []
    k_range = range(min_k, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vectors)
        keywords = get_keywords_from_kmeans(kmeans, text_model, k)
        avg_coherence, individual_coherences = calculate_coherence_score(keywords, original_texts)
        separation_score = calculate_separation_score(keywords, text_model)
        log.info(f"K={k}: Coherence Score: {avg_coherence:.4f}, Separation Score: {separation_score:.4f}")

        final_score = (avg_coherence * 0.6) + (separation_score * 0.4)
        all_results.append({
            'k': k,
            'model': kmeans,
            'labels': labels,
            'keywords': keywords,
            'final_score': final_score,
            'individual_scores': individual_coherences
        })
        log.info(f"Final score for K={k}: {final_score:.4f}")

    best_result = max(all_results, key=lambda x: x['final_score'])
    log.info(f"Best result for K={best_result['k']}: {best_result['final_score']:.4f}")

    best_individual_scores = best_result['individual_scores']
    median_score = np.median(best_individual_scores)
    relative_threshold = median_score * rel_coherence_drop
    
    log.info(f"Individual Scores: {[f'{s:.2f}' for s in best_individual_scores]}")
    log.info(f"Median score is {median_score:.4f}. Relative threshold at {rel_coherence_drop*100}% is {relative_threshold:.4f}.")
    log.info(f"Absolute threshold is {abs_coherence_threshold:.4f}.")

    # Get the junk topic based on absolute or relative coherence threshold
    junk_topic_ids = [
        i for i, score in enumerate(best_individual_scores) 
        if score < abs_coherence_threshold or score < relative_threshold
    ]
    
    if junk_topic_ids:
        log.info(f"Identified {len(junk_topic_ids)} junk topic(s): {junk_topic_ids}")
    else:
        log.info("All topics are above the quality threshold. No topics will be removed.")
    return best_result['model'], best_result['keywords'], best_result['labels'], junk_topic_ids

def create_interactive_plot(
    df: pl.DataFrame, 
    vectors: np.ndarray, 
    topic_labels: Dict[int, str]
):
    """
    Creates an interactive plot using human-readable topic labels.
    """
    log.info("Generating interactive scatter plot with named topics...")
    umap = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.5,
        random_state=42,
        metric='cosine'
    )
    coords_2d = cast(np.ndarray, umap.fit_transform(vectors))

    df_pd = df.to_pandas()
    df_pd['x'] = coords_2d[:, 0]
    df_pd['y'] = coords_2d[:, 1]

    # Map the numeric labels to the generated names, labeling -1 as Junk/Noise
    df_pd['topic_name'] = df_pd['cluster_kmeans_label'].map(topic_labels).fillna('Junk/Noise')

    log.info("Creating Plotly figure...")
    fig = px.scatter(
        df_pd,
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
    
    output_filename = "tweet_clusters_named.html"
    fig.write_html(output_filename)
    log.info(f"Successfully exported interactive plot to '{output_filename}'")

def save_to_supabase(
    keywords_by_topic: List[List[str]], 
    cluster_labels: List[str]
) -> List[Dict[str, Any]]:
    """Saves the topic keywords and their generated labels to Supabase."""
    log.info("Updating topic labels and keywords in Supabase...")
    try:
        log.info("Clearing old topics from the database...")
        supabase.table('topics').delete().gte('id', 0).execute()

        data_to_insert = [
            {"keywords": keywords, "label": label}
            for keywords, label in zip(keywords_by_topic, cluster_labels)
        ]

        if data_to_insert:
            response = supabase.table('topics').insert(data_to_insert).execute()
            log.info(f"Successfully inserted {len(data_to_insert)} topics.")
            return response.data
        else:
            log.warning("No new topics were generated to save.")
            return []
    except Exception as e:
        log.error(f"Failed to update topics in Supabase: {e}", exc_info=True)
        return []

def push_models_to_hf(kmeans_model: KMeans, text_model: FastText):
    """Saves models locally, then pushes them to the Hugging Face Hub."""
    HF_REPO_KMEANS_ID = config.env.get("HF_REPO_KMEANS_ID")

    if not HF_REPO_KMEANS_ID:
        raise ValueError("Environment variable 'HF_REPO_KMEANS_ID' is not set.")
    
    log.info(f"Preparing to push models to Hugging Face Hub repo: {HF_REPO_KMEANS_ID}")
    
    # Get token and create API client
    token = config.hf_token
    api = HfApi()

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO_KMEANS_ID, token=token, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Define file paths inside the temp directory
        kmeans_path = os.path.join(temp_dir, "kmeans.joblib")
        fasttext_path = os.path.join(temp_dir, "fasttext.model")
        readme_path = os.path.join(temp_dir, "README.md")

        # Save models using their recommended, native methods
        log.info(f"Saving K-Means model to {kmeans_path}...")
        joblib.dump(kmeans_model, kmeans_path)

        log.info(f"Saving FastText model to {fasttext_path}...")
        text_model.save(fasttext_path) # <-- Correct, native method for gensim

        # Create a README.md file (Model Card) to document the models
        log.info("Generating README.md (Model Card)...")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(config.readme_train_kmeans)
        
        try:
            log.info(f"Uploading all model files from {temp_dir} to {HF_REPO_KMEANS_ID}...")
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=HF_REPO_KMEANS_ID,
                repo_type="model",
                token=token,
            )
            log.info("Models successfully pushed to Hugging Face Hub.")
        except Exception as e:
            log.error(f"An error occurred while uploading to Hugging Face: {e}")

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
            .order('id', desc=False) \
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
        pl.col('text_content').map_elements(
            processing_text,
            return_dtype=pl.String
        ).alias('processed_text')
    )

    # Load the vectorized text data
    text_model, vectors = text_pipeline(df['processed_text'])

    # Find the best K using the silhouette method
    # min_k because if it too small the topic become too generic and max_k because if it too large the topic become too specific
    kmeans_model, keywords, labels, junk_topics_id = find_and_train_optimal_model(
        vectors,
        text_model,
        df['processed_text'].to_list(),
        min_k=3,
        max_k=8
    )

    sil_score = silhouette_score(vectors, labels)
    log.info(f"Silhouette score for optimal model: {sil_score:.2f}")

    # temporarily add the cluster label to the dataframe
    df = df.with_columns(
        pl.Series(name="cluster_kmeans_label", values=labels)
    )

    if junk_topics_id:
        log.info(f"Re-labeling junk topics {junk_topics_id} to -1...")
        df = df.with_columns(
            pl.when(pl.col("cluster_kmeans_label").is_in(junk_topics_id))
              .then(-1) # Assign -1 to junk topics
              .otherwise(pl.col("cluster_kmeans_label")) # Keep the original label for good topics
              .alias("cluster_kmeans_label")
        )

    clean_keywords = [
        keyword_list for i, keyword_list in enumerate(keywords) 
        if i not in junk_topics_id
    ]
    generated_labels = labeling_cluster(clean_keywords)

    # Create a mapping from the good numeric labels to the generated names for plotting
    good_kmeans_labels = sorted([i for i in df['cluster_kmeans_label'].unique() if i != -1])
    topic_labels_map = {label: name for label, name in zip(good_kmeans_labels, generated_labels)}
    
    # Create the interactive plot using the generated names
    create_interactive_plot(df, vectors, topic_labels_map)
    
    # Save the clean topics and their generated labels to the database
    newly_created_topics = save_to_supabase(clean_keywords, generated_labels)

    # Update the 'tweets' table with the correct topic IDs
    if newly_created_topics:
        # Create the final mapping from numeric label to permanent database ID
        kmeans_label_to_db_id = {label: topic['id'] for label, topic in zip(good_kmeans_labels, newly_created_topics)}
        
        df = df.with_columns(
            pl.col('cluster_kmeans_label').replace(kmeans_label_to_db_id, default=None).alias('topic_id')
        )

        log.info("Updating 'topic_id' in 'tweets' table (setting junk topics to NULL)...")
        update_data = df[['id', 'topic_id']].to_dicts()
        supabase.table('tweets').upsert(update_data).execute()
        
        log.info("Successfully updated tweet topics.")

    # Push models to Hugging Face Hub
    push_models_to_hf(kmeans_model, text_model)

    # Mark training complete
    supabase.table('app_config').update({"value": False}).eq('key', 'training-in-progress').execute()