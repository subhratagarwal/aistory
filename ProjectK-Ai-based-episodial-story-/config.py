import os
from dotenv import load_dotenv
import logging
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config file path
CONFIG_FILE = "user_config.json"

def load_user_config():
    """Load user configuration from JSON file."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {
            "model_provider": "ollama",  # Default to offline mode
            "openai_api_key": "",
            "ollama_base_url": "http://localhost:11434",
            "small_model_map": {
                "openai": "gpt-3.5-turbo",
                "ollama": "mistral:7b-text-q4_0"
            },
            "big_model_map": {
                "openai": "gpt-4-turbo",
                "ollama": "mistral:7b-text-q4_0"
            }
        }
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def save_user_config(config):
    """Save user configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

# Initial load of user configuration
USER_CONFIG = load_user_config()

# Function to get current model provider (dynamic)
def get_model_provider():
    """Get current model provider dynamically from config."""
    return USER_CONFIG.get("model_provider", "ollama")

# Function to get OpenAI API Key (dynamic)
def get_openai_api_key():
    """Get OpenAI API key dynamically from config."""
    return USER_CONFIG.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))

# Function to get model names (dynamic)
def get_model_names(model_type="small"):
    """Get model names dynamically for the current provider."""
    provider = get_model_provider()
    if model_type == "small":
        return USER_CONFIG.get("small_model_map", {}).get(provider, "mistral:7b-text-q4_0")
    else:
        return USER_CONFIG.get("big_model_map", {}).get(provider, "mistral:7b-text-q4_0")

# Function to get Ollama base URL (dynamic)
def get_ollama_base_url():
    """Get Ollama base URL dynamically from config."""
    return USER_CONFIG.get("ollama_base_url", "http://localhost:11434")

# For backwards compatibility, keep these variables but make them functions
MODEL_PROVIDER = get_model_provider
OPENAI_API_KEY = get_openai_api_key
SMALL_MODEL = lambda: get_model_names("small")
BIG_MODEL = lambda: get_model_names("big")
OLLAMA_BASE_URL = get_ollama_base_url

# Memory Settings
MAX_MEMORY_EPISODES = int(os.getenv("MAX_MEMORY_EPISODES", "3"))
ENABLE_SUMMARIZATION = os.getenv("ENABLE_SUMMARIZATION", "True").lower() == "true"

# Content Settings
DEFAULT_EPISODE_LENGTHS = {
    "Short": 500,
    "Medium": 1000,
    "Long": 1500
}

# Cache Settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "True").lower() == "true"
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "3600"))  # 1 hour default

# Log current configuration
logger.info(f"Initial Model Provider: {get_model_provider()}")
logger.info(f"Initial Small Model: {get_model_names('small')}")
logger.info(f"Initial Big Model: {get_model_names('big')}")
logger.info(f"Memory Settings: Max Episodes={MAX_MEMORY_EPISODES}, Summarization={ENABLE_SUMMARIZATION}")

def reload_config():
    """Reload configuration and refresh global CONFIG."""
    global USER_CONFIG
    USER_CONFIG = load_user_config()
    logger.info(f"Configuration reloaded, using provider: {get_model_provider()}")
    logger.info(f"Small Model: {get_model_names('small')}")
    logger.info(f"Big Model: {get_model_names('big')}")
    return USER_CONFIG
