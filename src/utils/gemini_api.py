from src.core import log, supabase
from typing import Dict, List

def labeling_cluster(keywords: Dict[int, List[str]]) -> Dict[int, str]:
    """
    Using Gemini API to get descriptive labels for each topic based on keywords
    """
    pass