#!/usr/bin/env python3
"""Categorize posts using LLM and create curated output."""
import json
import logging
import re
import sys
import time
import atexit
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

import config

# Setup logging to file and console
LOG_DIR = Path(__file__).parent / "scrape_logs"
LOG_DIR.mkdir(exist_ok=True)

# Create timestamped log file (include model name for easier debugging)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
safe_model = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in config.LLM_MODEL)
log_file = LOG_DIR / f"categorizer_{safe_model}_{timestamp}.txt"

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

# OpenAI client will be initialized in main() after API key validation

# Allowed categories
ALLOWED_CATEGORIES = {'listicle', 'value-add', 'playbook', 'rant', 'other', None}


class StreamingJSONArrayWriter:
    """Incrementally writes a JSON array to disk.

    Produces valid JSON only if `close()` is called (writes the final ']').
    We still flush after each append so partial progress is persisted even if interrupted.
    """

    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "w", encoding="utf-8")
        self._first = True
        self._closed = False
        self._f.write("[\n")
        self._f.flush()

    def append(self, obj: Dict) -> None:
        if self._closed:
            raise RuntimeError("Writer is closed")
        if not self._first:
            self._f.write(",\n")
        json.dump(obj, self._f, ensure_ascii=False, indent=2)
        self._f.flush()
        self._first = False

    def close(self) -> None:
        if self._closed:
            return
        self._f.write("\n]\n")
        self._f.flush()
        self._f.close()
        self._closed = True


def load_rubric() -> str:
    """Load classification rubric from file."""
    try:
        with open(config.CLASSIFICATION_RUBRIC_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Rubric file not found: {config.CLASSIFICATION_RUBRIC_PATH}")
        raise


def create_classification_prompt(rubric: str, posts: List[Dict]) -> str:
    """Create prompt for classifying posts."""
    posts_text = []
    for i, post in enumerate(posts):
        title = post.get('title', '')
        selftext = post.get('selftext', '')[:2000]  # Truncate if too long
        posts_text.append(f"Post {i + 1}:\nTitle: {title}\nContent: {selftext}\n")
    
    prompt = f"""You are a classifier for Reddit posts. Classify each post according to the rubric below.

{rubric}

Now classify these posts. Return ONLY valid JSON (no markdown, no code blocks, no prose). Return an array of objects, one per post, in the same order. Each object must have exactly these fields:
- category: one of "listicle", "value-add", "playbook", "rant", "other", or null
- category_confidence: float between 0.0 and 1.0
- category_rationale: one sentence explaining the classification

Posts to classify:
{''.join(posts_text)}

Return ONLY the JSON array, nothing else."""
    
    return prompt


def classify_posts_batch(posts: List[Dict], rubric: str, client: OpenAI) -> List[Dict]:
    """Classify a batch of posts using OpenAI API."""
    prompt = create_classification_prompt(rubric, posts)
    batch_id = f"batch_{len(posts)}_posts"
    
    logger.info(f"Starting LLM classification for {batch_id}")

    attempt = 0
    while attempt < config.MAX_RETRIES:
        try:
            logger.debug(f"LLM API call attempt {attempt + 1}/{config.MAX_RETRIES} for {batch_id}")

            kwargs = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a JSON-only classifier. Return only valid JSON arrays, no markdown, no code blocks."},
                    {"role": "user", "content": prompt},
                ],
            }

            response = client.chat.completions.create(**kwargs)

            # Log successful API call with token usage if available
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                logger.info(
                    f"LLM API call successful for {batch_id} - "
                    f"tokens: {usage.total_tokens} (prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens})"
                )
            else:
                logger.info(f"LLM API call successful for {batch_id}")

            content = response.choices[0].message.content.strip()

            # Handle case where response might be wrapped in markdown code blocks
            if content.startswith('```'):
                lines = content.split('\n')
                json_lines = [l for l in lines if not l.strip().startswith('```')]
                content = '\n'.join(json_lines)

            content = content.strip()

            # Parse response
            result = json.loads(content)

            # Handle if wrapped in object
            if isinstance(result, dict):
                for key in ['results', 'classifications', 'posts', 'data', 'items']:
                    if key in result and isinstance(result[key], list):
                        result = result[key]
                        break
                if isinstance(result, dict):
                    for v in result.values():
                        if isinstance(v, list):
                            result = v
                            break

            if not isinstance(result, list):
                raise ValueError(f"Response is not a list, got {type(result)}")

            if len(result) != len(posts):
                raise ValueError(f"Expected {len(posts)} classifications, got {len(result)}")

            logger.info(f"Successfully classified {batch_id} - got {len(result)} valid classifications")
            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in LLM response for {batch_id} (attempt {attempt + 1}/{config.MAX_RETRIES}): {e}"
            )
            logger.error(f"Response content (first 500 chars): {content[:500] if 'content' in locals() else ''}")
            attempt += 1
            if attempt < config.MAX_RETRIES:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retrying {batch_id} after {wait_time}s...")
                time.sleep(wait_time)
            continue
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(
                f"LLM API call failed for {batch_id} (attempt {attempt + 1}/{config.MAX_RETRIES}): "
                f"{error_type}: {error_msg}"
            )
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP status code: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Response: {e.response}")

            attempt += 1
            if attempt < config.MAX_RETRIES:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retrying {batch_id} after {wait_time}s...")
                time.sleep(wait_time)
            continue
    
    # If all retries failed, return fallback classifications
    logger.error(
        f"All {config.MAX_RETRIES} retry attempts failed for {batch_id}. "
        f"Using fallback classifications (category='other', confidence=0.0) for {len(posts)} posts"
    )
    return [
        {
            'category': 'other',
            'category_confidence': 0.0,
            'category_rationale': 'classification_failed'
        }
        for _ in posts
    ]


def validate_classification(classifications: List[Dict]) -> List[Dict]:
    """Validate and fix classifications."""
    validated = []
    for cls in classifications:
        category = cls.get('category')
        
        # Normalize category
        if category not in ALLOWED_CATEGORIES:
            # Try to fix common issues
            if category == 'value_add':
                category = 'value-add'
            elif category == 'Value-add':
                category = 'value-add'
            else:
                logger.warning(f"Invalid category '{category}', defaulting to 'other'")
                category = 'other'
        
        # Ensure confidence is float
        confidence = cls.get('category_confidence', 0.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except (ValueError, TypeError):
            confidence = 0.0
        
        # Ensure rationale is string
        rationale = str(cls.get('category_rationale', ''))[:200]  # Limit length
        
        validated.append({
            'category': category,
            'category_confidence': confidence,
            'category_rationale': rationale
        })
    
    return validated


def categorize_subreddit_posts(posts: List[Dict], rubric: str, client: OpenAI) -> List[Dict]:
    """Categorize all posts for a subreddit."""
    # Process in batches
    categorized_posts = []
    
    total_batches = (len(posts) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    logger.info(f"Processing {len(posts)} posts in {total_batches} batches of up to {config.BATCH_SIZE} posts each")
    
    for i in range(0, len(posts), config.BATCH_SIZE):
        batch = posts[i:i + config.BATCH_SIZE]
        batch_num = i // config.BATCH_SIZE + 1
        logger.info(f"Classifying batch {batch_num}/{total_batches} ({len(batch)} posts)")
        
        try:
            classifications = classify_posts_batch(batch, rubric, client)
            classifications = validate_classification(classifications)
            
            # Count classifications by category
            category_counts = {}
            for cls in classifications:
                cat = cls.get('category', 'unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            logger.info(
                f"Batch {batch_num} complete - classifications: {category_counts}, "
                f"avg confidence: {sum(c.get('category_confidence', 0) for c in classifications) / len(classifications):.2f}"
            )
            
            # Update posts with classifications
            for post, cls in zip(batch, classifications):
                post['category'] = cls['category']
                post['category_confidence'] = cls['category_confidence']
                post['category_rationale'] = cls['category_rationale']
                categorized_posts.append(post)
        
        except Exception as e:
            logger.error(f"Error classifying batch {batch_num}: {type(e).__name__}: {e}", exc_info=True)
            logger.error(f"Applying fallback classifications to batch {batch_num} ({len(batch)} posts)")
            # Apply fallback to batch
            for post in batch:
                post['category'] = 'other'
                post['category_confidence'] = 0.0
                post['category_rationale'] = 'classification_failed'
                categorized_posts.append(post)
    
    return categorized_posts


def select_top_posts_by_category(posts: List[Dict], top_n: int) -> Dict[str, List[Dict]]:
    """Select top N posts per category by composite_score."""
    # Group by category
    by_category = defaultdict(list)
    for post in posts:
        category = post.get('category')
        if category and category != 'null' and category != 'other':
            by_category[category].append(post)
    
    # Sort each category by composite_score (descending), then tie-breakers
    selected = {}
    for category, category_posts in by_category.items():
        sorted_posts = sorted(
            category_posts,
            key=lambda p: (
                -p.get('composite_score', 0),
                -p.get('score', 0),
                -p.get('upvote_ratio', 0),
                -p.get('num_comments', 0)
            )
        )
        selected[category] = sorted_posts[:top_n]
    
    return selected


def get_subreddit_json_path(subreddit: str, downloaded_dir: Path) -> Path:
    """Get the JSON file path for a subreddit. Raises FileNotFoundError if not found."""
    sanitized = re.sub(r'[^\w\-_]', '_', subreddit.lower())
    json_path = downloaded_dir / f"{sanitized}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Subreddit file not found: {json_path}. Run 'python scraper.py {subreddit}' first.")

    return json_path


def main():
    """Main categorizer function."""
    # Check for CLI arguments
    cli_subreddits = sys.argv[1:]
    cli_mode = bool(cli_subreddits)

    if cli_mode:
        logger.info(f"CLI mode: Processing {len(cli_subreddits)} specified subreddit(s): {', '.join(cli_subreddits)}")
    else:
        logger.info("Batch mode: Processing all subreddits in downloaded/")

    logger.info("Starting categorization...")

    # Check for API key
    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY environment variable is required for categorization")
        logger.error("Please set it in a .env file or as an environment variable")
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Initialize OpenAI client
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Load rubric
    rubric = load_rubric()
    logger.info("Loaded classification rubric")

    # Ensure categorised directory exists (for incremental outputs)
    categorised_dir = Path(config.CATEGORISED_DIR)
    categorised_dir.mkdir(parents=True, exist_ok=True)

    # Prepare incremental curated output file (model + timestamp)
    safe_model = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in config.LLM_MODEL)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    incremental_curated_path = categorised_dir / f"curated_{safe_model}_{ts}.json"
    writer = StreamingJSONArrayWriter(incremental_curated_path)
    atexit.register(writer.close)
    logger.info(f"Incremental curated output will be written to {incremental_curated_path}")

    # Find JSON files to process
    downloaded_dir = Path(config.DOWNLOADED_DIR)

    if cli_mode:
        # CLI mode: get specific subreddit files (fails if not found)
        json_files = []
        for subreddit in cli_subreddits:
            json_path = get_subreddit_json_path(subreddit, downloaded_dir)
            json_files.append(json_path)
    else:
        # Batch mode: process all JSON files in downloaded/
        json_files = list(downloaded_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON files found to process")
        return

    logger.info(f"Found {len(json_files)} subreddit files to process")
    
    # Process each subreddit file
    all_curated = []
    
    for json_file in tqdm(json_files, desc="Categorizing subreddits"):
        try:
            # Load posts
            with open(json_file, 'r', encoding='utf-8') as f:
                posts = json.load(f)
            
            if not posts:
                logger.warning(f"No posts in {json_file.name}")
                continue
            
            subreddit_name = json_file.stem
            
            logger.info(f"Processing {subreddit_name} ({len(posts)} posts)...")
            
            # Categorize posts
            categorized_posts = categorize_subreddit_posts(posts, rubric, client)
            
            # Save updated posts back to file (in-place update)
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(categorized_posts, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Updated {json_file.name} with classifications")
            
            # Select top posts per category
            top_posts = select_top_posts_by_category(categorized_posts, config.TOP_N_PER_TYPE)
            
            # Build curated entry
            curated_entry = {
                'subreddit': subreddit_name,
                'listicle': top_posts.get('listicle', []),
                'value-add': top_posts.get('value-add', []),
                'playbook': top_posts.get('playbook', []),
                'rant': top_posts.get('rant', []),
            }
            
            all_curated.append(curated_entry)
            # Incrementally append to curated output file
            writer.append(curated_entry)
            
            logger.info(
                f"{subreddit_name}: "
                f"listicle={len(curated_entry['listicle'])}, "
                f"value-add={len(curated_entry['value-add'])}, "
                f"playbook={len(curated_entry['playbook'])}, "
                f"rant={len(curated_entry['rant'])}"
            )
        
        except Exception as e:
            logger.error(f"Error processing {json_file.name}: {e}", exc_info=True)
            continue
    
    # Finalize incremental curated file (make it valid JSON)
    writer.close()

    # Also write a complete copy to the repo root (model + timestamp in filename)
    curated_dir = Path(config.CURATED_JSON_PATH).parent
    curated_dir.mkdir(parents=True, exist_ok=True)
    curated_path = curated_dir / f"curated_{safe_model}_{ts}.json"
    with open(curated_path, "w", encoding="utf-8") as f:
        json.dump(all_curated, f, indent=2, ensure_ascii=False)
    
    # Summary statistics
    total_curated = sum(
        len(entry['listicle']) + len(entry['value-add']) + 
        len(entry['playbook']) + len(entry['rant'])
        for entry in all_curated
    )
    
    logger.info(f"Wrote curated output to {curated_path}")
    logger.info(f"Wrote incremental curated output to {incremental_curated_path}")
    logger.info("=" * 60)
    logger.info("Categorization Summary:")
    logger.info(f"  - Processed {len(all_curated)} subreddits")
    logger.info(f"  - Total curated posts: {total_curated}")
    logger.info(f"  - Average per subreddit: {total_curated / len(all_curated) if all_curated else 0:.1f}")
    logger.info("=" * 60)
    logger.info("Categorization complete!")


if __name__ == "__main__":
    main()

