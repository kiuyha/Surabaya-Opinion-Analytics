from src.core import log, config
from typing import List
import requests

def labeling_cluster(keywords_by_topic: List[List[str]]) -> List[str]:
    """
    Uses the Gemini API to generate a descriptive, max-two-word label for each topic cluster.

    Args:
        keywords_by_topic (List[List[str]): A list of lists, where each inner list contains the keywords for a topic.

    Returns:
        List[str]: A dictionary mapping each cluster ID to its generated label.
    """
    GEMINI_API_KEY = config.env.get("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY environment variable not set. Cannot generate labels.")
        # Return a default label for each topic
        return [
            ", ".join(keywords)
            for keywords in keywords_by_topic[:6]
        ]

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
    topic_labels = []

    log.info(f"Generating labels for {len(keywords_by_topic)} topics using Gemini...")

    for cluster_id, keywords in enumerate(keywords_by_topic):
        # Join the list of keywords into a single comma-separated string
        keyword_string = ", ".join(keywords)

        prompt = (
            "You are an expert social media analyst who is great at writing simple, clear headlines.\n"
            "The following keywords are from a cluster of tweets from Surabaya, Indonesia. Your task is to create a simple and easy-to-understand topic label in English that captures the main idea.\n\n"
            "GUIDELINES:\n"
            "1. The label should be between 2 and 4 words.\n"
            "2. The label should be clear and concise.\n"
            "3. The label must be for a **general public audience**. Avoid technical or academic jargon. Use term that even 10 year olds can understand.\n"
            "4. The label must not contains word 'topics' or 'cluster'.\n"
            "4. **Output only the label itself, with no extra explanation or formatting**.\n\n"
            "--- EXAMPLE ---\n"
            "Keywords: km, tol, lajur, gempol, sidoarjo, waru, arah, rusak, banjir, truk\n"
            "Label: Traffic Jams\n"
            "--- END EXAMPLE ---\n\n"
            "--- TASK ---\n"
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
            topic_labels.append(keyword_string[:6])
        except (KeyError, IndexError) as e:
            log.error(f"Could not parse Gemini response for cluster {cluster_id}: {e}")
            topic_labels.append(keyword_string[:6])

    return topic_labels