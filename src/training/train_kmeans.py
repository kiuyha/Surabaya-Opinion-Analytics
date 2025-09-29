import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from kneed import KneeLocator
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def find_optimal_k_elbow(vector_feature: pd.DataFrame, max_k: int=15)-> int:
    """
    Calculates and plots the inertia for a range of k values to find the optimal k using the elbow method.

    Args:
        vector_feature (pd.DataFrame): The vectorized text data.
        feature_name (str): Name of the vectorization method for plot title.
        max_k (int): The maximum number of clusters to test.

    Returns:
        int: The optimal number of clusters (k).
    """
    inertia_values = []
    k_range = range(2, max_k + 1) # Start from 2 clusters

    logging.info(f"Finding optimal K...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(vector_feature)
        inertia_values.append(kmeans.inertia_)

    # The KneeLocator finds the point of maximum curvature in the plot
    kn = KneeLocator(list(k_range), inertia_values, curve='convex', direction='decreasing')
    optimal_k = kn.elbow

    if optimal_k:
        logging.info(f"Optimal K found: {optimal_k}")
    else:
        logging.info("Could not automatically find an elbow. Please inspect the plot.")
        optimal_k = 5 # Fallback to a default value if no elbow is found

    return optimal_k


def train_kmeans(vector_feature: pd.DataFrame, num_topics: int)-> None:
    """ 
    Trains a K-Means clustering model on the vectorized text data.

    Args:
        vector_feature (pd.DataFrame): The vectorized text data.
        num_topics (int): The number of topics to cluster.
    """
    logging.info(f"\nFitting K-Means with K={num_topics}...")
    
    kmeans = KMeans(n_clusters=num_topics, random_state=42, n_init='auto')
    kmeans.fit(vector_feature)

    logging.info(f"Top keywords for the {num_topics} discovered topics:")
    terms = vector_feature.columns
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_topics):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        logging.info(f"Topic #{i}: {', '.join(top_terms)}")
    
    # Save the trained model
    joblib.dump(kmeans, 'kmeans_model.joblib')
    

if __name__ == "__main__":
    # Load the vectorized text data

    # Find the best K using the elbow method
    best_k = find_optimal_k_elbow(vector_feature, max_k=15)

    # Fit the K-Means model
    train_kmeans(vector_feature, num_topics=best_k)