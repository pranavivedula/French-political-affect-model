"""Configuration settings for the French Political Affect Analysis system."""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/data/political_affect.db")

# Scraping Configuration
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "3.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

# NLP Model Configuration
VALENCE_MODEL = os.getenv("VALENCE_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
AROUSAL_MODEL = os.getenv("AROUSAL_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(PROJECT_ROOT / "models"))

# Text Processing
MAX_SEQUENCE_LENGTH = 512
MIN_TEXT_LENGTH = 50
SENTENCE_MIN_WORDS = 5

# Scheduler Configuration
ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
SCRAPING_SCHEDULE_HOUR = int(os.getenv("SCRAPING_SCHEDULE_HOUR", "2"))
ANALYSIS_SCHEDULE_HOUR = int(os.getenv("ANALYSIS_SCHEDULE_HOUR", "2"))
ANALYSIS_SCHEDULE_MINUTE = int(os.getenv("ANALYSIS_SCHEDULE_MINUTE", "30"))

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
DASHBOARD_TITLE = os.getenv("DASHBOARD_TITLE", "French Political Affect Analysis")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(PROJECT_ROOT / "logs" / "app.log"))

# Aggregation Configuration
RECENCY_HALFLIFE_DAYS = 60
MAX_WORD_COUNT_FOR_WEIGHTING = 1000
RECENCY_WEIGHT = 0.7
LENGTH_WEIGHT = 0.3
MIN_DOCUMENTS_FOR_AGGREGATION = 10

# Confidence Intervals
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval


def load_party_config():
    """Load party configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "party_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['parties']


def get_party_by_code(code: str):
    """Get party configuration by party code."""
    parties = load_party_config()
    for party in parties:
        if party['code'] == code:
            return party
    return None


# Create necessary directories
os.makedirs(PROJECT_ROOT / "data" / "raw", exist_ok=True)
os.makedirs(PROJECT_ROOT / "data" / "processed", exist_ok=True)
os.makedirs(PROJECT_ROOT / "logs", exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
