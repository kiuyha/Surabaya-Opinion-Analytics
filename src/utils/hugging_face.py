from src.core import log
import os
from dotenv import load_dotenv
from huggingface_hub import HfFolder

load_dotenv()

HF_REPO_KMEANS_ID = os.getenv("HF_REPO_KMEANS_ID")
HF_REPO_NER_ID = os.getenv("HF_REPO_NER_ID")

def get_hf_token() -> str:
    """
    Gets the Hugging Face token from an environment variable (for Actions)
    or the local folder (for development).
    """
    # Prioritize the environment variable for CI/CD environments
    token = os.environ.get("HF_TOKEN")
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