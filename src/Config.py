from logging import basicConfig, INFO, getLogger, Logger
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Configure logging
basicConfig(
    level=INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

log: Logger = getLogger(__name__)

def get_app_config(supabase_client: Client, key: str, default_value=None):
    """
    Fetches a specific configuration value from the 'app_config' table.
    """
    try:
        # Use .eq() to filter by key and .single() to get one record
        response = supabase_client.table('app_config').select('value').eq('key', key).single().execute()
        
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

# Set of unnecessary hashtags use to remove from tweets
unnecessary_hashtags = set(get_app_config(
    supabase_client=Client,
    key='unnecessary-hashtags',
    default_value= [
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
    ])
)

# Dictionary to normalize slang words to their correct form
slang_mapping = get_app_config(
    supabase_client=Client,
    key='slang-mapping',
    default_value= {
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
)