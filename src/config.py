from supabase import Client
import logging
from typing import Any, List, Mapping
from src.utils.types import SearchConfigDict

# Not using log from src/config.py because it will make a circular import
log = logging.getLogger(__name__)

class Config:
    def __init__(self, supabase_client: Client, env: Mapping[str, str]) -> None:
        self.supabase = supabase_client
        self.env = env
        self._hf_token = None
        self._scrape_config = None
        self._unnecessary_hashtags = None
        self._slang_mapping = None
        self._curse_words = None
        self._readme_train_kmeans = None

    def _get_app_config(self, key: str, default_value: Any = None):
        """
        Fetches a specific configuration value from the 'app_config' table.
        """
        try:
            # Use .eq() to filter by key and .single() to get one record
            response = self.supabase.table('app_config').select('value').eq('key', key).single().execute()
            
            # .single() returns data directly, not in a list
            if response.data:
                # Return the actual value from the 'value' column
                return response.data['value']
            else:
                log.warning(f"App config key '{key}' not found. Using default value.")
                return default_value
                
        except Exception as e:
            log.error(f"Error getting app config for key '{key}': {e}")
            return default_value
    
    @property
    def scrape_config(self) -> List[SearchConfigDict]:
        """
        Fetches the search configuration from Supabase.
        """
        if self._scrape_config is None:
            log.info("Fetching search config from Supabase...")
            response = self._get_app_config(key='scrape-config')
            if response is None:
                raise Exception("Search config not found in Supabase")
            else:
                self._scrape_config = response
        return self._scrape_config
        
    @property
    def unnecessary_hashtags(self) -> set:
        """
        Fetches the list of unnecessary hashtags to remove in preprocessing from Supabase.
        """
        if self._unnecessary_hashtags is None:
            log.info("Fetching unnecessary hashtags from Supabase...")
            default_value = [
                "#fyp",
                "#viral",
                "#like4like",
                "#viralbanget",
                "#trending",
                "#fypage",
                "#foryoupage",
                "#explore",
                "#instagood",
                "#followforfollow",
                "#f4f",
                "#likeforlike",
                "#l4l",
                "#commentforcomment",
                "#instadaily",
                "#photooftheday",
                "#viralvideo",
                "#xyzbca",
                "#beritaterkini",
                "#terkini",
            ]
            self._unnecessary_hashtags = set(
                self._get_app_config(
                    key='unnecessary-hashtags',
                    default_value=default_value
                )
            )
        return self._unnecessary_hashtags
    
    @property
    def slang_mapping(self) -> dict:
        """
        Fetches the slang mapping dictionary for preprocessing from Supabase.
        """
        if self._slang_mapping is None:
            log.info("Fetching slang mapping from Supabase...")
            default_value = {
                "gimik": "gimmick",
                "emg": "memang",
                "remeh temeh": "remeh",
                "liyu": "pusing",
                "wdym": "what do you mean",
                "mangnya": "memangnya",
                "sisok": "besok",
                "lfg": "let's go",
                "gede": "besar",
                "aja": "saja",
                "kalo": "kalau",
                "yg": "yang",
                "jg": "juga",
                "klo": "kalau",
                "trs" : "terus",
                "lbh": "lebih",
                "gk": "gak",
                "lol": "laugh out loud",
                "lmao": "laugh my ass off",
                "rofl": "rolling on floor laughing",
                "lmfao": "laughing my fucking ass off",
                "roflmao": "rolling on floor laughing my ass off",
            }
            self._slang_mapping = self._get_app_config(
                key='slang-mapping',
                default_value=default_value
            )
            
        return self._slang_mapping
    
    @property
    def curse_words(self) -> set:
        """
        Fetches the curse words for preprocessing from Supabase.
        """
        if self._curse_words is None:
            log.info("Fetching curse words from Supabase...")
            default_value = [
                "anjing",
                "monyet",
                "anjir",
                "bjir",
            ]
            self._curse_words = set(self._get_app_config(
                key='curse-words',
                default_value=default_value
            ))
            
        return self._curse_words
    
    @property
    def hf_token(self) -> str:
        """
        Gets the Hugging Face token from an environment variable (for Actions)
        or the local folder (for development).
        """
        from huggingface_hub import HfFolder

        if self._hf_token is None:
            # Prioritize the environment variable for CI/CD environments
            token = self.env.get("HF_TOKEN")
            if token:
                log.info("Found Hugging Face token in environment variable.")
                return token
            
            # Fallback to local token for development
            log.info("No HF_TOKEN env var found, checking local hf folder.")
            try:
                token = HfFolder.get_token()
                if token:
                    log.info("Found Hugging Face token in local folder.")
                    return token
            except Exception:
                pass # Ignore errors if folder doesn't exist
            
            raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable or run `huggingface-cli login`.")
        
        return self._hf_token

    @property
    def readme_train_kmeans(self) -> str:
        """
        Fetches the readme train kmeans model from Supabase.
        """
        if self._readme_train_kmeans is None:
            log.info("Fetching readme train kmeans model from Supabase...")

            self._readme_train_kmeans = str(self._get_app_config(
                key='readme-train-kmeans',
                default_value=f"""
---
license: mit
language:
- id
tags:
- topic-modeling
- kmeans
- fasttext
- surabaya
---

# Topic Models for Surabaya Tweet Analysis

This repository contains a set of models for performing topic modeling on tweets from Surabaya.

## Models Included

This repository contains two key components:

1.  **`fasttext.model`**: A `gensim` FastText model trained on processed tweet text. It is used to generate semantic vector embeddings for documents.
2.  **`kmeans.joblib`**: A `scikit-learn` K-Means model trained on the vectors produced by the FastText model. It contains the final topic cluster centroids.

## How to Use

Load the models using `gensim`, `joblib`, and `huggingface_hub`.

```python
import joblib
from gensim.models import FastText
from huggingface_hub import hf_hub_download

# Download and load the models
REPO_ID = "{self.env.get("HF_REPO_KMEANS_ID")}"
kmeans_path = hf_hub_download(repo_id=REPO_ID, filename="kmeans.joblib")
hf_hub_download(repo_id=REPO_ID, filename="fasttext.model.wv.vectors_ngrams.npy")
fasttext_path = hf_hub_download(repo_id=REPO_ID, filename="fasttext.model")

kmeans_model = joblib.load(kmeans_path)
fasttext_model = FastText.load(fasttext_path)

print(f"K-Means model loaded with {{kmeans_model.n_clusters}} clusters.")
print(f"FastText model loaded with vector size {{fasttext_model.vector_size}}.")

# You can now use these models for inference.
"""
            ))
        return self._readme_train_kmeans