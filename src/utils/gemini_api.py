from src.core import log
from typing import List
import requests
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY= os.environ.get("GEMINI_API_KEY")

def labeling_cluster(keywords_by_topic: List[str]) -> List[str]:
    """
    Uses the Gemini API to generate a descriptive, max-two-word label for each topic cluster.

    Args:
        keywords_by_topic (List[str]): A dictionary mapping a cluster ID to its list of keywords.

    Returns:
        List[str]: A dictionary mapping each cluster ID to its generated label.
    """
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY environment variable not set. Cannot generate labels.")
        # Return a default label for each topic
        return {
            cluster_id: ", ".join(keywords)
            for cluster_id, keywords in enumerate(keywords_by_topic)
        }

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
    topic_labels = []

    log.info(f"Generating labels for {len(keywords_by_topic)} topics using Gemini...")

    for cluster_id, keywords in enumerate(keywords_by_topic):
        # Join the list of keywords into a single comma-separated string
        keyword_string = ", ".join(keywords)

        prompt = (
            "You are an expert topic modeler. Your task is to create a short, descriptive topic label from a list of keywords. "
            "The label must be a maximum of two words. Do not add any extra explanation or formatting. "
            "Here are the keywords from a cluster of Indonesian tweets:\n\n"
            f"Keywords: {keyword_string}\n"
            "Label:"
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)
            
            # Safely extract the text from the response
            result = response.json()
            label = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', f"Topic {cluster_id}")
            
            # Clean up the label (remove newlines, quotes, etc.)
            cleaned_label = label.strip().replace('"', '').replace("'", "")
            
            topic_labels.append(cleaned_label)
            log.info(f"Generated label for Topic #{cluster_id}: '{cleaned_label}'")

        except requests.exceptions.RequestException as e:
            log.error(f"API request failed for cluster {cluster_id}: {e}")
            topic_labels.append(keyword_string)
        except (KeyError, IndexError) as e:
            log.error(f"Could not parse Gemini response for cluster {cluster_id}: {e}")
            topic_labels.append(keyword_string)

    return topic_labels