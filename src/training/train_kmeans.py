from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from src.core import log, supabase, config
from src.utils.gemini_api import labeling_cluster
from src.preprocess import processing_text
from typing import Tuple, Dict, List, Any
from huggingface_hub import HfApi
import polars as pl
import numpy as np
import os
import tempfile
import joblib
import plotly.express as px
from sklearn.manifold import TSNE
import gensim.models

def text_pipeline(texts: pl.Series, vector_size: int = 300) -> Tuple[gensim.models.FastText, np.ndarray]:
    """
    Converts a series of text documents into a vector representation using FastText
    by averaging word vectors for each document.
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
    fasttext_model = gensim.models.FastText(
        tokenized_texts,
        vector_size=vector_size,
        window=5,
        min_count=2,
        workers=cpu_core,
        seed=42
    )

    log.info("Creating document vectors by averaging...")
    document_vectors = []
    for doc in tokenized_texts:
        word_vectors = [fasttext_model.wv[word] for word in doc if word in fasttext_model.wv]
        if word_vectors:
            document_vectors.append(np.mean(word_vectors, axis=0))
        else:
            document_vectors.append(np.zeros(vector_size)) # Use the specified vector_size

    vectors = np.array(document_vectors)

    log.info("Normalizing document vectors...")
    vectors = normalize(vectors, norm='l2')
    
    log.info(f"FastText pipeline complete. Final matrix shape: {vectors.shape}")

    return fasttext_model, vectors

def calculate_coherence_score(keywords_per_topic: List[List[str]], original_texts: List[str]) -> Tuple[float, List[float]]:
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
    return float(np.mean(similarity_scores))

def get_keywords_from_kmeans(
    kmeans_model: KMeans,
    fasttext_model: gensim.models.FastText,
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

    for i in range(num_topics):
        # Get the centroid vector for the current cluster
        centroid_vector = kmeans_model.cluster_centers_[i]

        # Find the top N most similar words in the Word2Vec model's vocabulary
        # The 'positive' argument takes a list of vectors to find similarities for
        try:
            top_words = fasttext_model.wv.most_similar(positive=[centroid_vector], topn=top_n_keywords)
            topic_keywords = [word for word, similarity in top_words]
            keywords_by_topic.append(topic_keywords)
            log.info(f"Topic #{i}: {', '.join(topic_keywords)}")
        except KeyError as e:
            log.warning(f"Could not find keywords for Topic #{i}: a word in the model might be missing. Error: {e}")
            keywords_by_topic.append([]) # Append empty list for this topic

    return keywords_by_topic

def find_and_train_optimal_model(
        vectors: np.ndarray,
        text_model: gensim.models.FastText,
        original_texts: List[str],
        min_k: int = 2,
        max_k: int = 15,
        abs_coherence_threshold: float = 0.45,  # The absolute quality floor
        rel_coherence_drop: float = 0.80      # Flag if score is less than 80% of the median
    ) -> Tuple[KMeans, List[List[str]], List[int]]: 
    """
    Finds the optimal K using final score: (coherence score, silhouette score, jaccard similarity) and then trains the optimal K-Means model.
    """

    all_results = []
    k_range = range(min_k, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(vectors)
        sil_score = silhouette_score(vectors, kmeans.labels_)
        keywords = get_keywords_from_kmeans(kmeans, text_model, k)
        avg_coherence, individual_coherences = calculate_coherence_score(keywords, original_texts)
        separation_score = calculate_separation_score(keywords)
        log.info(f"K={k}: Silhouette Score: {sil_score:.4f}, Coherence Score: {avg_coherence:.4f}, Separation Score: {separation_score:.4f}")

        final_score = (avg_coherence * 0.3) + (sil_score * 0.2) + (( 1 - separation_score) * 0.5)
        all_results.append({
            'k': k,
            'model': kmeans,
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
    
    return best_result['model'], best_result['keywords'], junk_topic_ids

def create_interactive_plot(vectors: np.ndarray):
    """
    Creates an interactive plot using Plotly Express.
    """
    log.info("Generating interactive scatter plot...")
    log.info("Reducing vector dimensions with t-SNE for visualization...")
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        max_iter=1000,    
        random_state=42
    )
    coords_2d = tsne.fit_transform(vectors)

    df_pd = df.to_pandas()
    df_pd['x'] = coords_2d[:, 0]
    df_pd['y'] = coords_2d[:, 1]

    log.info("Creating Plotly figure...")
    fig = px.scatter(
        df_pd,
        x='x',
        y='y',
        color='cluster_kmeans_label',
        hover_data=['text_content'],
        title="Tweet Clusters Visualization",
        labels={'cluster_kmeans_label': 'Topic Cluster'},
        color_discrete_map={
            -1: "lightgrey" 
        },
        category_orders={"cluster_kmeans_label": sorted(df_pd['cluster_kmeans_label'].unique())}
    )
    
    fig.for_each_trace(
        lambda t: t.update(hovertemplate=t.hovertemplate.replace("cluster_kmeans_label=-1", "Topic Cluster=Junk/Noise"))
    )

    fig.update_layout(legend_title="Clusters", title_x=0.5)
    
    output_filename = "tweet_clusters.html"
    fig.write_html(output_filename)
    log.info(f"Successfully exported interactive plot to '{output_filename}'")

def save_to_supabase(keywords_by_topic: List[List[str]])-> List[Dict[str, Any]]:
    """Saves the topic labels and keywords to the 'topics' table in Supabase."""
    log.info("updating topic labels and keywords to Supabase...")

    try:
        log.info("Clearing old topics from the database...")
        supabase.table('topics').delete().gte('id', 0).execute()

        cluster_labels = labeling_cluster(keywords_by_topic)
        data_to_insert = [
            {
                "keywords": keywords,
                "label": cluster_label
            } 
            for keywords, cluster_label in zip(keywords_by_topic, cluster_labels)
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

def push_models_to_hf(kmeans_model: KMeans, text_model: gensim.models.FastText):
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
    kmeans_model, keywords, junk_topics_id = find_and_train_optimal_model(
        vectors,
        text_model,
        df['processed_text'].to_list(),
        min_k=3,
        max_k=8
    ) 

    # temporarily add the cluster label to the dataframe
    df = df.with_columns(
        pl.Series(name="cluster_kmeans_label", values=kmeans_model.labels_)
    )

    if junk_topics_id:
        log.info(f"Re-labeling junk topics {junk_topics_id} to -1...")
        df = df.with_columns(
            pl.when(pl.col("cluster_kmeans_label").is_in(junk_topics_id))
              .then(-1) # Assign -1 to junk topics
              .otherwise(pl.col("cluster_kmeans_label")) # Keep the original label for good topics
              .alias("cluster_kmeans_label")
        )

    create_interactive_plot(vectors)

    # Save the discovered topic labels and keywords to your database
    clean_keywords = [
        keyword_list for i, keyword_list in enumerate(keywords) 
        if i not in junk_topics_id
    ]
    newly_created_topics = save_to_supabase(clean_keywords)

    if newly_created_topics:
        # Create the mapping from K-Means label to permanent DB ID
        good_kmeans_labels = sorted([i for i in df['cluster_kmeans_label'].unique() if i != -1])
        kmeans_label_to_db_id = {label: topic['id'] for label, topic in zip(good_kmeans_labels, newly_created_topics)}
        df = df.with_columns(
            pl.col('cluster_kmeans_label').replace(kmeans_label_to_db_id).alias('topic_id')
        )

        # set the junk topics (-1) to None, which will become NULL in the database
        df = df.with_columns(
            pl.when(pl.col('cluster_kmeans_label') == -1)
              .then(None)
              .otherwise(pl.col('topic_id'))
              .alias('topic_id')
        )

        log.info("Updating the 'topic_id' in 'tweets' table...")

        update_data = df[['id', 'topic_id']].to_dicts()
        supabase.table('tweets').upsert(update_data).execute()
        
        log.info("Successfully updated tweet-topic.")

    # Push models to Hugging Face Hub
    push_models_to_hf(kmeans_model, text_model)

    # Mark training complete
    supabase.table('app_config').update({"value": False}).eq('key', 'training-in-progress').execute()