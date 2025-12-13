import requests
import unicodedata
import re
import string
import demoji
import nltk
import html
from nltk.corpus import stopwords
from .core import log, config

nltk.download('stopwords')

def get_tld_list(retry_count: int = 3)-> list:
    """
    Fetches the latest list of TOP LEVEL DOMAIN such as .com, .net etc from IANA.
    """
    url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
    while retry_count > 0:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            tlds = response.text.strip().split('\n')

            log.info("Fetched TLD list successfully.")
            # The list may contain comments, so we filter them out.
            return [tld.lower() for tld in tlds if not tld.startswith('#')]
        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching TLD list: {e} (Retrying...)")
            retry_count -= 1
    return []


def split_hashtag(tag: str)-> str:
    tag = tag.lstrip("#")
    tag = tag.replace("_", " ")
    return re.sub(r'([a-z])([0-9])', r'\1 \2', tag, flags=re.I)

def normalize_text(text: str, level: str = 'hard')-> str:
    # Normalize slang words
    normalized_words = [config.slang_mapping.get(word, word) for word in text.split()]
    text = " ".join(normalized_words)

    if level == 'hard':
        # remove curse words since they not useful for topic modeling
        text = " ".join([word for word in text.split() if word not in config.curse_words])

    return text

tld_list = get_tld_list()
def processing_text(text: str, level: str = 'hard') -> str:
    """ Performs preprocessing on the given text for different NLP tasks.

    Args:
        text (str): The text to preprocess.
        level (str, optional): The level of preprocessing. Defaults to 'hard'.
            - 'hard': Aggressive cleaning for tasks like clustering (K-Means).
                      Includes lowercasing, stopword removal, and lemmatization.
            - 'light': Minimal cleaning for tasks like Named Entity Recognition (NER).
                       Preserves case, stopwords, and original word forms.

    Returns:
        str: The preprocessed text, or None if the input is invalid.
    """
    if not isinstance(text, str):
        return None
    
    # Remove HTML tags
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith('S'))
    text = re.sub(r'[<>\[\]]', '', text)

    # Normalize unicode characters (e.g., '𝖒𝖚𝖉𝖆𝖍' -> 'mudah')
    text = unicodedata.normalize('NFKC', text)

    # Remove URLs and email addresses as they are noise for both tasks
    tld_pattern_group = '|'.join(re.escape(tld) for tld in tld_list)
    url_pattern = re.compile(
        'https?://\\S+|www\\.\\S+|[\\w\\.-]+\\.(?:' + tld_pattern_group + ')[\\S]*'
    )
    text = url_pattern.sub(r'', text)
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    text = email_pattern.sub(r'', text)
    
    # Remove unnecessary hashtags and split meaningful ones into words
    hashtags = re.findall(r"#\w+", text)
    for tag in hashtags:
        if tag.lower() not in config.unnecessary_hashtags:
            text = text.replace(tag, split_hashtag(tag))
        else:
            text = text.replace(tag, "")

    if level == 'hard':
        # Make the text lower case
        text = text.lower()

        # Replace mentions with a empty string (e.g., '@prabowo' -> '')
        text = re.sub(r'@\w+', '', text)

        # Remove all punctuation
        punctuation_pattern = f"[{re.escape(string.punctuation)}]"
        text = re.sub(punctuation_pattern, ' ', text)

        # Expand word repetitions (e.g., 'anak2' -> 'anak-anak')
        text = re.sub(r'\b([a-zA-Z]{2,})2\b', r'\1 \1', text)

        # Replaces characters repeated more than once with a single character. e.g., 'monyettt' -> 'monyet'
        text = re.sub(r'(\w)\1+', r'\1', text)

        # Perform lemmatization to get the root form of words
        text = normalize_text(text, level=level)

        # Remove stopwords for both Indonesian and English
        stop_words_id = set(stopwords.words('indonesian'))
        stop_words_en = set(stopwords.words('english'))
        all_stopwords = stop_words_id.union(stop_words_en)
        text = " ".join([word for word in text.split() if word not in all_stopwords])

        # Remove word that is actually a number, because it usually noise in topic modeling
        text = " ".join([word for word in text.split() if not word.isdigit()])

        # remove surabaya since it in the query search
        text = text.replace("surabaya", "")

    elif level == 'light':
        # Remove the '@' symbol from mentions but keep the name, as it's an entity (e.g., '@prabowo' -> 'prabowo')
        mention_pattern = re.compile(r'@(\w+)')
        text = mention_pattern.sub(r'\1', text)
        text = normalize_text(text, level=level)

    # Replace emoji with its text description (e.g., '😊' -> '')
    text = demoji.replace(text, '')

    # remove extra whitespace created during processing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text