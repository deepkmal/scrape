#!/usr/bin/env python3
"""Reddit scraper that fetches subreddit about.json data and saves to a single JSON file."""
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

import config
from subreddits import subreddits

# Setup logging to file and console
LOG_DIR = Path(__file__).parent / "scrape_logs"
LOG_DIR.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"about_scraper_{timestamp}.txt"

# Configure logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove existing handlers
logger.handlers.clear()

# File handler
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging to {log_file}")

# Output file for about data
OUTPUT_FILE = Path(__file__).parent / "subreddits_about.json"


def load_existing_data() -> Dict[str, Dict]:
    """Load existing about data from output file. Returns dict keyed by lowercase subreddit name."""
    if not OUTPUT_FILE.exists():
        return {}
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            if isinstance(data_list, list):
                return {item.get('name', '').lower(): item for item in data_list if item.get('name')}
            return {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load existing data from {OUTPUT_FILE}: {e}")
        return {}


def save_data(data: Dict[str, Dict]) -> None:
    """Save about data to output file (sorted by subscriber count)."""
    data_list = sorted(data.values(), key=lambda x: x.get('subscribers', 0), reverse=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)


def dedupe_subreddits(subreddit_list: List[str]) -> List[str]:
    """Deduplicate subreddits case-insensitively."""
    seen = set()
    deduped = []
    for sub in subreddit_list:
        sub_lower = sub.lower()
        if sub_lower not in seen:
            seen.add(sub_lower)
            deduped.append(sub)
    return deduped


def fetch_subreddit_about(subreddit: str, headers: Dict[str, str]) -> Optional[Dict]:
    """Fetch about.json for a subreddit with retry logic."""
    url = f"{config.REDDIT_BASE_URL}/r/{subreddit}/about.json"

    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                continue

            # Handle other errors
            if response.status_code == 403 or response.status_code == 404:
                logger.warning(f"Subreddit {subreddit} returned {response.status_code} (private/banned/missing)")
                return None

            if response.status_code != 200:
                logger.error(f"Unexpected status code {response.status_code} for {subreddit}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None

            data = response.json()
            if 'data' not in data:
                logger.error(f"Unexpected response shape for {subreddit}")
                return None

            return data['data']

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error (attempt {attempt + 1}/{config.MAX_RETRIES}): {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {subreddit}: {e}")
            return None

    return None


def transform_about_data(about_data: Dict) -> Dict:
    """Transform about data to keep relevant fields."""
    return {
        'name': about_data.get('display_name', ''),
        'display_name_prefixed': about_data.get('display_name_prefixed', ''),
        'title': about_data.get('title', ''),
        'public_description': about_data.get('public_description', ''),
        'description': about_data.get('description', ''),
        'subscribers': about_data.get('subscribers', 0),
        'active_user_count': about_data.get('active_user_count'),
        'accounts_active': about_data.get('accounts_active'),
        'created_utc': about_data.get('created_utc', 0),
        'subreddit_type': about_data.get('subreddit_type', ''),
        'over18': about_data.get('over18', False),
        'lang': about_data.get('lang', ''),
        'url': about_data.get('url', ''),
        'icon_img': about_data.get('icon_img', ''),
        'banner_img': about_data.get('banner_img', ''),
        'community_icon': about_data.get('community_icon', ''),
        'header_img': about_data.get('header_img', ''),
        'primary_color': about_data.get('primary_color', ''),
        'key_color': about_data.get('key_color', ''),
        'allow_images': about_data.get('allow_images', True),
        'allow_videos': about_data.get('allow_videos', True),
        'submission_type': about_data.get('submission_type', ''),
        'wiki_enabled': about_data.get('wiki_enabled', False),
        'spoilers_enabled': about_data.get('spoilers_enabled', False),
        'allow_polls': about_data.get('allow_polls', False),
        'fetched_at': datetime.now().isoformat(),
    }


def main():
    """Main scraper function."""
    # Check for CLI arguments
    cli_subreddits = sys.argv[1:]
    cli_mode = bool(cli_subreddits)

    if cli_mode:
        target_subreddits = cli_subreddits
        logger.info(f"CLI mode: Scraping about for {len(target_subreddits)} specified subreddit(s): {', '.join(target_subreddits)}")
    else:
        target_subreddits = subreddits
        logger.info("Batch mode: Scraping about for all subreddits from subreddits.py")

    logger.info("Starting Reddit about scraper...")
    logger.info(f"Output file: {OUTPUT_FILE}")

    # Load existing data
    existing_data = load_existing_data()
    logger.info(f"Loaded {len(existing_data)} existing subreddit records")

    # Dedupe subreddits
    deduped_subreddits = dedupe_subreddits(target_subreddits)
    logger.info(f"Processing {len(deduped_subreddits)} subreddits (deduped from {len(target_subreddits)})")

    # Setup headers
    headers = {'User-Agent': config.USER_AGENT}

    # Stats
    stats = {
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'updated': 0,
    }

    # Process each subreddit
    for i, subreddit in enumerate(tqdm(deduped_subreddits, desc="Scraping about data")):
        try:
            subreddit_lower = subreddit.lower()

            # Skip if already exists (in batch mode), update in CLI mode
            if subreddit_lower in existing_data and not cli_mode:
                logger.info(f"[{i+1}/{len(deduped_subreddits)}] Skipping r/{subreddit} (already in output file)")
                stats['skipped'] += 1
                continue

            is_update = subreddit_lower in existing_data
            action = "Updating" if is_update else "Fetching"
            logger.info(f"[{i+1}/{len(deduped_subreddits)}] {action} about for r/{subreddit}...")

            about_data = fetch_subreddit_about(subreddit, headers)

            if about_data is None:
                logger.error(f"[{i+1}/{len(deduped_subreddits)}] Failed to fetch about for r/{subreddit}")
                stats['failed'] += 1
                continue

            # Transform and add to data
            transformed = transform_about_data(about_data)
            existing_data[subreddit_lower] = transformed

            # Incremental save after each successful fetch
            save_data(existing_data)

            subscribers = transformed.get('subscribers', 'N/A')
            if is_update:
                logger.info(f"[{i+1}/{len(deduped_subreddits)}] r/{subreddit}: Updated (subscribers: {subscribers})")
                stats['updated'] += 1
            else:
                logger.info(f"[{i+1}/{len(deduped_subreddits)}] r/{subreddit}: Added (subscribers: {subscribers})")
                stats['success'] += 1

            # Rate limiting
            time.sleep(config.RATE_LIMIT_DELAY_SECONDS)

        except Exception as e:
            logger.error(f"[{i+1}/{len(deduped_subreddits)}] Error processing r/{subreddit}: {e}", exc_info=True)
            stats['failed'] += 1
            continue

    logger.info(f"Scraping complete! Added: {stats['success']}, Updated: {stats['updated']}, Failed: {stats['failed']}, Skipped: {stats['skipped']}")
    logger.info(f"Total subreddits in output file: {len(existing_data)}")


if __name__ == "__main__":
    main()
