from dotenv import load_dotenv
import os
from supabase import create_client
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
url= os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if url is None or key is None:
    raise Exception("Missing Supabase URL or key")
supabase = create_client(url, key)

# Create the single, shared config instance
config = Config(supabase_client=supabase, env=os.environ)