from supabase import Client
import logging
from typing import List, TypedDict, Optional

# Not using log from src/config.py because it will make a circular import
log = logging.getLogger(__name__)

class SearchConfigDict(TypedDict):
    """A dictionary representing a single search configuration."""
    query: str
    depth: Optional[int]
    time_budget: Optional[int]

class Config:
    def __init__(self, supabase_client: Client) -> None:
        self.supabase = supabase_client
        self._scrape_config = None
        self._unnecessary_hashtags = None
        self._slang_mapping = None

    def _get_app_config(self, key: str, default_value=None):
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
                # The key was not found in the database, return the default
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
    def slang_mapping(self) -> set:
        """
        Fetches the slang mapping dictionary for preprocessing from Supabase.
        """
        if self._unnecessary_hashtags is None:
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
            self._unnecessary_hashtags = self._get_app_config(
                key='slang-mapping',
                default_value=default_value
            )
            
        return self._unnecessary_hashtags