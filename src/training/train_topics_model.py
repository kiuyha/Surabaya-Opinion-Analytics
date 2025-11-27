from itertools import combinations
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from src.core import log, supabase, config
from src.utils.gemini_api import labeling_cluster
from src.preprocess import processing_text
from typing import Tuple, Dict, List, Any, cast
from huggingface_hub import HfApi
import numpy as np
import os
import tempfile
import joblib
from gensim.models import FastText
import pandas as pd
from src.utils.visualize import create_cluster_plot, create_kmeans_eval_plot
import json

def sentence_vector(doc: str, fastext_model: FastText) -> np.ndarray:
    vectors = [fastext_model.wv[word] for word in doc if word in fastext_model.wv]
    if vectors:
        return np.mean(vectors, axis=0, dtype=np.float32)
    return np.zeros(fastext_model.vector_size, dtype=np.float32)

def text_pipeline(
    texts: pd.Series,
    vector_size: int = 300,
    model: FastText|None = None
) -> Tuple[FastText, np.ndarray]:
    """
    Converts a series of text documents into a vector representation using FastText by averaging word vectors for each document.
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
    if model:
        fasttext_model = model
    else:
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

def get_keywords_from_labels(
    labels: np.ndarray,
    original_texts: List[str],
    top_n: int = 10,
    ignore_label: int|None = None
) -> List[List[str]]:
    """
    Extracts top keywords for each cluster using TF-IDF. Each cluster's documents are aggregated into one "document" for TF-IDF analysis.
    """
    # Aggregate text for each cluster
    cluster_texts = {}
    
    unique_labels = set(labels)
    if ignore_label is not None and ignore_label in unique_labels:
        unique_labels.remove(ignore_label)
        
    for label in unique_labels:
        cluster_texts[label] = ""

    for i, text in enumerate(original_texts):
        label = labels[i]
        if label == ignore_label:
            continue
        cluster_texts[label] += " " + text

    # Sort keys to ensure topic 0 is index 0, topic 1 is index 1, etc.
    sorted_labels = sorted(cluster_texts.keys())
    corpus = [cluster_texts[l] for l in sorted_labels]
    
    # Handle edge case where all clusters are empty or only noise exists
    if not corpus:
        return []

    try:
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = cast(np.ndarray, vectorizer.fit_transform(corpus))
        feature_names = vectorizer.get_feature_names_out()
        
        keywords_by_topic = []
        for i in range(len(sorted_labels)):
            row = np.squeeze(tfidf_matrix[i].toarray())
            top_indices = row.argsort()[-top_n:][::-1]
            topic_keywords = [feature_names[j] for j in top_indices]
            keywords_by_topic.append(topic_keywords)
        
        return keywords_by_topic
        
    except ValueError as e:
        log.error(f"TF-IDF failed: {e}")
        return [[] for _ in range(len(sorted_labels))]

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
    Finds the optimal K using final score: (coherence score, separation score) and then trains the optimal K-Means model.
    """

    all_results = []
    k_range = range(min_k, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(vectors)
        keywords = get_keywords_from_labels(labels, original_texts, top_n=50, ignore_label=-1)

        # Only use the first 10 keywords since the rest are likely noise
        top_10_keywords = [kw[:10] for kw in keywords]
        avg_coherence, individual_coherences = calculate_coherence_score(top_10_keywords, original_texts)
        separation_score = calculate_separation_score(keywords, text_model)
        log.info(f"K={k}: Coherence Score: {avg_coherence:.4f}, Separation Score: {separation_score:.4f}")

        final_score = (avg_coherence * 0.6) + (separation_score * 0.4)
        all_results.append({
            'k': k,
            'model': kmeans,
            'labels': labels,
            'keywords': keywords,
            'final_score': final_score,
            'avg_coherence': avg_coherence,
            'separation_score': separation_score,
            'individual_scores': individual_coherences
        })
        log.info(f"Final score for K={k}: {final_score:.4f}")

    best_result = max(all_results, key=lambda x: x['final_score'])
    log.info(f"Best result for K={best_result['k']}: \n Coherence Score: {best_result['avg_coherence']:.4f}, Separation Score: {best_result['separation_score']:.4f}, Final Score: {best_result['final_score']:.4f}")
    
    # Export interactive plot for model evaluation
    create_kmeans_eval_plot(pd.DataFrame(all_results), best_result['k'])

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

def save_to_supabase(
    keywords_by_topic: List[List[str]], 
    cluster_labels: List[str]
) -> List[Dict[str, Any]]:
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

def push_models_to_hf(kmeans_model: KMeans, text_model: FastText, topic_map: Dict[int, int]):
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
        config_path = os.path.join(temp_dir, "config.json")
        readme_path = os.path.join(temp_dir, "README.md")

        # Save models using their recommended, native methods
        log.info(f"Saving K-Means model to {kmeans_path}...")
        joblib.dump(kmeans_model, kmeans_path)

        log.info(f"Saving FastText model to {fasttext_path}...")
        text_model.save(fasttext_path)

        # Save Junk IDs to JSON
        with open(config_path, "w") as f:
            json.dump({"topic_map": topic_map}, f)

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

def fetch_all_rows(table_name, columns):
    """Helper to fetch all rows from a table using pagination."""
    log.info(f'Loading {table_name} from Supabase...')
    all_data = []
    page_size = 1000
    current_page = 0

    while True:
        response = supabase.table(table_name) \
            .select(*columns) \
            .order('id', desc=False) \
            .limit(page_size) \
            .offset(current_page * page_size) \
            .execute()
        
        page_of_data = response.data        
        if not page_of_data:
            break
        
        all_data.extend(page_of_data)
        current_page += 1
    
    log.info(f"Loaded {len(all_data)} rows from {table_name}.")
    return all_data

if __name__ == "__main__":
    try:
        # Mark training in progress
        supabase.table('app_config').update({"value": True}).eq('key', 'training-in-progress').execute()

        log.info('Loading data from Supabase...')

        tweets_data = fetch_all_rows('tweets', ['id', 'text_content'])
        reddit_data = fetch_all_rows('reddit_comments', ['id', 'text_content'])

        if not tweets_data and not reddit_data:
            raise Exception("No data found in Supabase (tweets or reddit)")
        
        df_tweets = pd.DataFrame(tweets_data)
        df_reddit = pd.DataFrame(reddit_data)
        dfs_to_concat = []
        
        if not df_tweets.empty:
            df_tweets['source_type'] = 'tweets'
            dfs_to_concat.append(df_tweets)
        
        if not df_reddit.empty:
            df_reddit['source_type'] = 'reddit_comments'
            dfs_to_concat.append(df_reddit)
        
        df = pd.concat(dfs_to_concat, ignore_index=True)
        log.info(f"Training on {len(df)} records...")
        
        df['processed_text'] = df['text_content'].apply(processing_text)

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
        df['cluster_label'] = labels

        if junk_topics_id:
            log.info(f"Re-labeling junk topics {junk_topics_id} to -1...")
            df['cluster_label'] = df['cluster_label'].replace(junk_topics_id, -1)

        clean_keywords = [
            keyword_list for i, keyword_list in enumerate(keywords) 
            if i not in junk_topics_id
        ]

        USE_GEMINI_API = config.env.get('USE_GEMINI_API', False)
        if USE_GEMINI_API:
            # only use 10 top keywords for API
            keywords_for_api = [
                keyword_list[:10] for keyword_list in clean_keywords
            ]
            generated_labels = labeling_cluster(keywords_for_api)
        else:
            log.info("Generating labels from top 10 keywords (no API)...")
            generated_labels = [
                ", ".join(keyword[:10])
                for keyword in clean_keywords
            ]
            for i, label in enumerate(generated_labels):
                log.info(f"Generated label for Topic #{i}: {label}")

        # Create a mapping from the good numeric labels to the generated names for plotting
        good_kmeans_labels = sorted([
            i 
            for i in df['cluster_label'].unique()
            if i != -1
        ])
        topic_labels_map = {
            label: name
            for label, name in zip(good_kmeans_labels, generated_labels)
        }
        
        # Create the interactive plot using the generated names
        create_cluster_plot(df, vectors, topic_labels_map)

        # Save the clean topics and their generated labels to the database
        newly_created_topics = save_to_supabase(clean_keywords, generated_labels)
        kmeans_label_to_db_id = {}

        # Update the 'tweets' table with the correct topic IDs
        if newly_created_topics:
            # Create the final mapping from numeric label to permanent database ID
            kmeans_label_to_db_id = {
                int(label): topic['id']
                for label, topic in zip(good_kmeans_labels, newly_created_topics)
            }
            
            df['topic_id'] = (
                df['cluster_label']
                .map(kmeans_label_to_db_id)
                .replace(np.nan, None)
                .astype('Int64')
            )

            for table in ['tweets', 'reddit_comments']:
                subset = df[df['source_type'] == table]
                
                if not subset.empty:
                    log.info(f"Updating 'topic_id' in '{table}' table...")
                    update_data = subset[['id', 'topic_id']].to_dict(orient='records')
                    supabase.table(table).upsert(update_data).execute()
            
            log.info("Successfully updated all topics.")
        
        # Push models to Hugging Face Hub
        push_models_to_hf(kmeans_model, text_model, kmeans_label_to_db_id)

    except Exception as e:
        log.error(f"Error training topics model: {e}")
        raise e

    finally:
        # Mark training complete
        supabase.table('app_config').update({"value": False}).eq('key', 'training-in-progress').execute()