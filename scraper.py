#!/usr/bin/env python3
"""Reddit scraper that fetches top posts, filters them, and saves to JSON files."""
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm

import config
from subreddits import subreddits

# Setup logging to file and console
LOG_DIR = Path(__file__).parent / "scrape_logs"
LOG_DIR.mkdir(exist_ok=True)

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"scraper_{timestamp}.txt"

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


def sanitize_filename(subreddit: str) -> str:
    """Sanitize subreddit name for use as filename."""
    # Convert to lowercase and replace invalid characters
    sanitized = re.sub(r'[^\w\-_]', '_', subreddit.lower())
    return sanitized


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


def load_existing_posts(file_path: Path) -> List[Dict]:
    """Load existing posts from JSON file."""
    if not file_path.exists():
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
            return posts if isinstance(posts, list) else []
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load existing posts from {file_path}: {e}")
        return []


def merge_posts(existing_posts: List[Dict], new_posts: List[Dict]) -> Tuple[List[Dict], int]:
    """Merge new posts with existing, deduplicating by (subreddit, title, created_utc)."""
    def post_key(post: Dict) -> Tuple:
        return (post.get('subreddit', '').lower(), post.get('title', ''), post.get('created_utc', 0))

    seen_keys = set()
    merged = []

    for post in existing_posts:
        key = post_key(post)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(post)

    new_added = 0
    for post in new_posts:
        key = post_key(post)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(post)
            new_added += 1

    merged.sort(key=lambda p: p.get('composite_score', 0), reverse=True)
    return merged, new_added


def compute_composite_score(score: int, num_comments: int, upvote_ratio: float) -> float:
    """Compute composite score for ranking posts."""
    return (
        1.0 * math.log10(score + 1) +
        0.5 * math.log10(num_comments + 1) +
        0.75 * upvote_ratio
    )


def is_bait_post(title: str, selftext: str) -> bool:
    """Check if post is likely bait/rage bait."""
    text_lower = (title + " " + selftext).lower()
    
    # Check for bait phrases
    for phrase in config.BAIT_PHRASES:
        if phrase.lower() in text_lower:
            return True
    
    # Check uppercase ratio
    total_chars = len(title + selftext)
    if total_chars > 0:
        uppercase_chars = sum(1 for c in (title + selftext) if c.isupper())
        uppercase_ratio = uppercase_chars / total_chars
        if uppercase_ratio > config.MAX_UPPERCASE_RATIO:
            return True
    
    # Check excessive punctuation
    exclamations = title.count('!') + selftext.count('!')
    questions = title.count('?') + selftext.count('?')
    if exclamations > config.MAX_EXCLAMATIONS or questions > config.MAX_QUESTION_MARKS:
        return True
    
    return False


def should_keep_post(post_data: Dict, now_utc: float) -> Tuple[bool, Optional[str]]:
    """Determine if post should be kept after filtering. Returns (keep, reason_if_filtered)."""
    # Check age
    created_utc = post_data.get('created_utc', 0)
    age_seconds = now_utc - created_utc
    age_years = age_seconds / (365 * 24 * 60 * 60)
    if age_years > config.MAX_POST_AGE_YEARS:
        return False, f"too_old ({age_years:.1f} years)"
    
    # Check selftext
    selftext = post_data.get('selftext', '')
    if not selftext or selftext in ['[removed]', '[deleted]', '']:
        return False, "missing_selftext"
    
    # Check length
    if len(selftext.strip()) < config.MIN_SELFTEXT_CHARS:
        return False, f"too_short ({len(selftext.strip())} chars)"
    
    # Check bait
    title = post_data.get('title', '')
    if is_bait_post(title, selftext):
        return False, "bait_detected"
    
    return True, None


def transform_post(post_data: Dict, now_utc: float) -> Optional[Dict]:
    """Transform Reddit post data into reduced schema."""
    try:
        # Extract required fields
        transformed = {
            'score': post_data.get('score', 0),
            'title': post_data.get('title', ''),
            'selftext': post_data.get('selftext', ''),
            'subreddit': post_data.get('subreddit', ''),
            'upvote_ratio': post_data.get('upvote_ratio', 0.0),
            'subreddit_type': post_data.get('subreddit_type', 'public'),
            'ups': post_data.get('ups', 0),
            'downs': post_data.get('downs', 0),
            'created_utc': post_data.get('created_utc', 0),
            'media': post_data.get('media'),
            'is_video': post_data.get('is_video', False),
            'num_comments': post_data.get('num_comments', 0),
            'num_reports': post_data.get('num_reports'),
            'over_18': post_data.get('over_18', False),
            'category': '',
            'category_confidence': None,
            'category_rationale': '',
            'composite_score': 0.0,
        }
        
        # Compute composite score
        transformed['composite_score'] = compute_composite_score(
            transformed['score'],
            transformed['num_comments'],
            transformed['upvote_ratio']
        )
        
        return transformed
    except Exception as e:
        logger.error(f"Error transforming post: {e}")
        return None


def fetch_reddit_posts(subreddit: str, headers: Dict[str, str], after: Optional[str] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """Fetch a page of posts from Reddit API with retry logic."""
    url = f"{config.REDDIT_BASE_URL}/r/{subreddit}/top.json"
    params = {'t': 'all', 'limit': 100}
    if after:
        params['after'] = after
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                continue
            
            # Handle other errors
            if response.status_code == 403 or response.status_code == 404:
                logger.warning(f"Subreddit {subreddit} returned {response.status_code} (private/banned/missing)")
                return None, None
            
            if response.status_code != 200:
                logger.error(f"Unexpected status code {response.status_code} for {subreddit}")
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return None, None
            
            data = response.json()
            if 'data' not in data or 'children' not in data['data']:
                logger.error(f"Unexpected response shape for {subreddit}")
                return None, None
            
            next_after = data['data'].get('after')
            return data, next_after
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error (attempt {attempt + 1}/{config.MAX_RETRIES}): {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            return None, None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {subreddit}: {e}")
            return None, None
    
    return None, None


def scrape_subreddit(subreddit: str, headers: Dict[str, str], now_utc: float) -> Tuple[List[Dict], Dict[str, int]]:
    """Scrape a single subreddit, returning kept posts and filter stats."""
    kept_posts = []
    seen_ids: Set[str] = set()
    filter_stats = {
        'total_fetched': 0,
        'kept': 0,
        'filtered_age': 0,
        'filtered_selftext': 0,
        'filtered_length': 0,
        'filtered_bait': 0,
        'duplicates': 0,
    }
    
    after = None
    
    while len(kept_posts) < config.TARGET_KEPT_POSTS:
        # Fetch page
        data, next_after = fetch_reddit_posts(subreddit, headers, after)
        if data is None:
            break
        
        posts = data['data']['children']
        if not posts:
            break
        
        filter_stats['total_fetched'] += len(posts)
        
        # Process posts
        for post_wrapper in posts:
            if 'data' not in post_wrapper:
                continue
            
            post_data = post_wrapper['data']
            post_id = post_data.get('name') or post_data.get('id')
            
            # Check for duplicates
            if post_id in seen_ids:
                filter_stats['duplicates'] += 1
                continue
            seen_ids.add(post_id)
            
            # Filter
            keep, filter_reason = should_keep_post(post_data, now_utc)
            if not keep:
                if filter_reason == 'too_old':
                    filter_stats['filtered_age'] += 1
                elif filter_reason == 'missing_selftext':
                    filter_stats['filtered_selftext'] += 1
                elif filter_reason == 'too_short':
                    filter_stats['filtered_length'] += 1
                elif filter_reason == 'bait_detected':
                    filter_stats['filtered_bait'] += 1
                continue
            
            # Transform
            transformed = transform_post(post_data, now_utc)
            if transformed:
                kept_posts.append(transformed)
                filter_stats['kept'] += 1
                
                if len(kept_posts) >= config.TARGET_KEPT_POSTS:
                    break
        
        # Check if we have more pages
        if not next_after:
            break
        after = next_after
        
        # Rate limiting
        time.sleep(config.RATE_LIMIT_DELAY_SECONDS)
    
    return kept_posts, filter_stats


def main():
    """Main scraper function."""
    # Check for CLI arguments
    cli_subreddits = sys.argv[1:]
    cli_mode = bool(cli_subreddits)

    if cli_mode:
        target_subreddits = cli_subreddits
        logger.info(f"CLI mode: Scraping {len(target_subreddits)} specified subreddit(s): {', '.join(target_subreddits)}")
    else:
        target_subreddits = subreddits
        logger.info("Batch mode: Scraping all subreddits from subreddits.py")

    logger.info("Starting Reddit scraper...")

    # Ensure downloaded directory exists
    Path(config.DOWNLOADED_DIR).mkdir(parents=True, exist_ok=True)

    # Dedupe subreddits
    deduped_subreddits = dedupe_subreddits(target_subreddits)
    logger.info(f"Processing {len(deduped_subreddits)} subreddits (deduped from {len(target_subreddits)})")

    # Setup headers
    headers = {'User-Agent': config.USER_AGENT}

    # Current time for age filtering
    now_utc = datetime.now(timezone.utc).timestamp()

    # Process each subreddit
    for subreddit in tqdm(deduped_subreddits, desc="Scraping subreddits"):
        try:
            sanitized_name = sanitize_filename(subreddit)
            output_path = Path(config.DOWNLOADED_DIR) / f"{sanitized_name}.json"

            # Determine behavior based on mode and file existence
            if output_path.exists():
                if cli_mode:
                    # CLI mode: Update existing file
                    logger.info(f"Updating r/{subreddit} (file exists, will merge)...")
                    existing_posts = load_existing_posts(output_path)
                else:
                    # Batch mode: Skip existing files (original behavior)
                    logger.info(f"Skipping {subreddit} (file already exists)")
                    continue
            else:
                existing_posts = []
                logger.info(f"Scraping r/{subreddit}...")

            kept_posts, filter_stats = scrape_subreddit(subreddit, headers, now_utc)

            # Merge with existing posts if applicable
            if existing_posts:
                final_posts, new_added = merge_posts(existing_posts, kept_posts)
                logger.info(f"Merged {new_added} new posts with {len(existing_posts)} existing posts (total: {len(final_posts)})")
            else:
                final_posts = kept_posts

            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_posts, f, indent=2, ensure_ascii=False)

            logger.info(
                f"r/{subreddit}: Kept {filter_stats['kept']}/{filter_stats['total_fetched']} posts. "
                f"Filtered: age={filter_stats['filtered_age']}, "
                f"selftext={filter_stats['filtered_selftext']}, "
                f"length={filter_stats['filtered_length']}, "
                f"bait={filter_stats['filtered_bait']}, "
                f"duplicates={filter_stats['duplicates']}"
            )

        except Exception as e:
            logger.error(f"Error processing {subreddit}: {e}", exc_info=True)
            continue

    logger.info("Scraping complete!")


if __name__ == "__main__":
    main()

