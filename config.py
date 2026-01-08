"""Configuration settings for Reddit scraper and categorizer."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Always load .env from this directory (works regardless of current working directory)
_ENV_PATH = Path(__file__).parent / ".env"
try:
    load_dotenv(dotenv_path=_ENV_PATH)
except (OSError, PermissionError):
    # If the environment restricts file access (e.g., sandboxed execution), fall back to default behavior.
    try:
        load_dotenv()
    except Exception:
        pass

# OpenAI API Configuration (only needed for categorizer)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CLASSIFIER_MODEL = os.getenv("OPENAI_CLASSIFIER_MODEL")

# Reddit API Configuration
USER_AGENT = "Mozilla/5.0 (compatible; RedditScraper/1.0; +http://yoursite.com)"
REDDIT_BASE_URL = "https://www.reddit.com"
RATE_LIMIT_DELAY_SECONDS = 2  # Delay between requests (30 req/min, well under 60/min limit)

# Download Filtering Settings
MIN_SELFTEXT_CHARS = 400  # Minimum meaningful length for selftext
MAX_POST_AGE_YEARS = 3  # Discard posts older than this

BAIT_PHRASES = [
    "hot take",
    "unpopular opinion",
    "change my mind",
    "prove me wrong",
    "fight me",
    "rage bait",
    "bait",
    "triggered",
    "woke",
]

MAX_UPPERCASE_RATIO = 0.35  # Maximum ratio of uppercase letters
MAX_EXCLAMATIONS = 8  # Maximum number of exclamation marks
MAX_QUESTION_MARKS = 8  # Maximum number of question marks

# Pagination / Target Settings
TARGET_KEPT_POSTS = 500  # Number of posts to keep per subreddit after filtering

# Curation Settings
TOP_N_PER_TYPE = 5  # Number of top posts to select per category type

# LLM Classification Settings
# Default model if OPENAI_CLASSIFIER_MODEL is not set.
LLM_MODEL = OPENAI_CLASSIFIER_MODEL or "gpt-4o-mini"
LLM_TEMPERATURE = 0.3  # Some models may not support non-default temperature; code will fall back if needed.
BATCH_SIZE = 10  # Number of posts to classify per API call
MAX_RETRIES = 3  # Maximum retry attempts for API calls

# File Paths
DOWNLOADED_DIR = os.path.join(os.path.dirname(__file__), "downloaded")
CATEGORISED_DIR = os.path.join(os.path.dirname(__file__), "categorised")
CURATED_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "curated.json")
CLASSIFICATION_RUBRIC_PATH = os.path.join(os.path.dirname(__file__), "classification_rubric.md")

