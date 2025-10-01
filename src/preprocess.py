import requests
import unicodedata
import re
import string
import demoji
import nltk
from nltk.corpus import stopwords
from mpstemmer import MPStemmer
from nltk.stem import WordNetLemmatizer
from .core import log, config

nltk.download('stopwords')
nltk.download('wordnet')

def get_tld_list()-> list:
    """
    Fetches the latest list of TOP LEVEL DOMAIN such as .com, .net etc from IANA.
    """
    url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        tlds = response.text.strip().split('\n')

        # The list may contain comments, so we filter them out.
        return [tld.lower() for tld in tlds if not tld.startswith('#')]
    except requests.exceptions.RequestException as e:
        log.error(f"Error fetching TLD list: {e}")
        return []


def split_hashtag(tag: str)-> str:
    tag = tag.lstrip("#")
    return re.sub(r'([a-z])([0-9])', r'\1 \2', tag, flags=re.I)

indonesian_stemmer = MPStemmer()
english_lemmatizer = WordNetLemmatizer()
def lemmatization(text: str, level: str = 'hard')-> str:
    # Normalize slang words
    words = text.split()
    normalized_words = [config.slang_mapping.get(word, word) for word in words]
    text = " ".join(normalized_words)

    if level == 'hard':
        # Lemmatize English words
        english_lemmatized_words = [english_lemmatizer.lemmatize(word) for word in text.split()]
        text = " ".join(english_lemmatized_words)

        # Stem Indonesian words
        text = indonesian_stemmer.stem(text)

    return text

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

    # Normalize unicode characters (e.g., '𝖒𝖚𝖉𝖆𝖍' -> 'mudah')
    text = unicodedata.normalize('NFKC', text)

    # Remove URLs and email addresses as they are noise for both tasks
    tld_pattern_group = '|'.join(re.escape(tld) for tld in get_tld_list())
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

    # Replace emoji with its text description (e.g., '😊' -> 'smiling face')
    text = demoji.replace_with_desc(text, ' ')

    if level == 'hard':
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation.replace('@', '')))

        # Replace mentions with a generic token (e.g., '@prabowo' -> ':user')
        mention_pattern = re.compile(r'@\w+')
        text = mention_pattern.sub(':user', text)

        # Make the text lower case
        text = text.lower()

        # Expand word repetitions (e.g., 'anak2' -> 'anak-anak')
        text = re.sub(r'(\w+?)2', r'\1-\1', text)

        # Remove stopwords for both Indonesian and English
        stop_words_id = set(stopwords.words('indonesian'))
        stop_words_en = set(stopwords.words('english'))
        all_stopwords = stop_words_id.union(stop_words_en)
        text = " ".join([word for word in text.split() if word not in all_stopwords])

        # Perform lemmatization to get the root form of words
        text = lemmatization(text, level=level)

    elif level == 'light':
        # Remove the '@' symbol from mentions but keep the name, as it's an entity (e.g., '@prabowo' -> 'prabowo')
        mention_pattern = re.compile(r'@(\w+)')
        text = mention_pattern.sub(r'\1', text)
        text = lemmatization(text, level=level)

    # remove extra whitespace created during processing
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text