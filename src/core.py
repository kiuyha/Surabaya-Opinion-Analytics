from dotenv import load_dotenv
import os
from supabase import create_client, Client
from logging import basicConfig, INFO, getLogger, Logger
from .config import Config

load_dotenv()

# Configure logging
basicConfig(
    level=INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log: Logger = getLogger(__name__)

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Create the single, shared config instance
config = Config(supabase_client=supabase)