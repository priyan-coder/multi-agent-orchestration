#!/usr/bin/env python3
"""
Simplified Core Functions for Portfolio Enrichment Multi-Agent System

Focused on:
- Tiktoken-based semantic similarity
- Prioritized API calls
- Robust state management with proper file locking
- Minimal redundant logic
"""

import os
import re
import json
import time
import random
import logging
from datetime import datetime
from functools import wraps
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, NamedTuple

import pandas as pd
import requests
from dotenv import load_dotenv
from filelock import FileLock

# Load environment variables
load_dotenv()

# Try to import tiktoken for semantic similarity
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

# =========================
# Configuration
# =========================

class Config:
    """Configuration constants for the portfolio enrichment system."""
    
    # API Configuration
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    DEFAULT_TIMEOUT = 10
    DEFAULT_MIN_SCORE = 0.6
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE = 60
    RATE_LIMIT_BUFFER = 0.9
    
    # Retry Configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 1.0
    MAX_BACKOFF = 30.0
    BACKOFF_MULTIPLIER = 2.0
    
    # File Configuration
    DEFAULT_STATE_FILE = "portfolio_state.json"
    DEFAULT_OUTPUT_FILE = "portfolio_output.csv"
    DEFAULT_LOG_FILE = "monitoring.log"
    DEFAULT_MD_FILE = "monitoring.md"


class ResultStatus(Enum):
    """Status codes for API operation results."""
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    LOW_CONFIDENCE = "low_confidence"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


class ApiResult(NamedTuple):
    """Structured result for API operations."""
    status: ResultStatus
    value: Optional[str] = None
    confidence: float = 0.0
    message: str = ""
    attempts: int = 1


# =========================
# Rate Limiting & Retry
# =========================

class RateLimiter:
    """Simple in-memory rate limiter for API calls."""
    
    def __init__(self, max_requests: int = Config.MAX_REQUESTS_PER_MINUTE):
        self.max_requests = int(max_requests * Config.RATE_LIMIT_BUFFER)
        self.requests = []
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request) + 0.1
            if wait_time > 0:
                logging.getLogger(__name__).warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self.requests.append(now)


_rate_limiter = RateLimiter()


def retry_with_backoff(max_retries: int = Config.MAX_RETRIES):
    """Decorator for API calls with exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    _rate_limiter.wait_if_needed()
                    return func(*args, **kwargs)
                
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        backoff = min(
                            Config.INITIAL_BACKOFF * (Config.BACKOFF_MULTIPLIER ** attempt),
                            Config.MAX_BACKOFF
                        )
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = backoff * jitter
                        logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.warning(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"API call failed after {max_retries} attempts: {e}")
                
                except Exception as e:
                    logger.error(f"Unexpected error in API call: {e}")
                    raise
            
            raise last_exception or Exception("All retry attempts failed")
        return wrapper
    return decorator


# =========================
# Logging & Markdown Log
# =========================

def setup_logging(name: str = __name__) -> logging.Logger:
    """Configure namespaced logging to avoid conflicts."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(Config.DEFAULT_LOG_FILE)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def append_markdown(md_path: str, text: str) -> None:
    """Thread-safe markdown append with file locking."""
    lock_path = md_path + ".lock"
    try:
        with FileLock(lock_path, timeout=5):
            with open(md_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to write to markdown: {e}")


def log_event(task_name: str, task_id: str, event: str, message: str, md_path: str, **extras) -> None:
    """Log event with markdown output."""
    timestamp = datetime.now().isoformat()
    logging.info(f"[{task_name}:{task_id}] {event}: {message}")
    
    if event == 'start':
        append_markdown(md_path, f"\n## [{timestamp}] Task: {task_name} (id={task_id})\n")
        append_markdown(md_path, f"- Input: {message}")
    elif event == 'progress':
        append_markdown(md_path, f"- Progress: {message}")
    elif event == 'end':
        append_markdown(md_path, f"- End: {message}")
    elif event in ['warning', 'error']:
        append_markdown(md_path, f"- {event.upper()}: {message}")


# =========================
# State Management
# =========================

def save_state(portfolio_data: Dict[int, Dict[str, str]], state_file: str) -> bool:
    """Serialize portfolio_data to JSON file with robust file locking."""
    lock_path = state_file + ".lock"
    logger = logging.getLogger(__name__)
    
    try:
        # Use longer timeout for state operations and ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)
        
        with FileLock(lock_path, timeout=30):
            # Convert int keys to strings for JSON serialization
            json_data = {str(k): v for k, v in portfolio_data.items()}
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = state_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            
            # Atomic rename
            os.rename(temp_file, state_file)
        
        logger.info(f"State saved to {state_file} ({len(portfolio_data)} rows)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save state to {state_file}: {e}")
        # Clean up temp file if it exists
        temp_file = state_file + ".tmp"
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False


def load_state(state_file: str) -> Optional[Dict[int, Dict[str, str]]]:
    """Load portfolio_data from JSON file with robust file locking and retry logic."""
    if not os.path.exists(state_file):
        logging.getLogger(__name__).warning(f"State file {state_file} does not exist")
        return None
    
    lock_path = state_file + ".lock"
    logger = logging.getLogger(__name__)
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            with FileLock(lock_path, timeout=30):
                with open(state_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
            
            # Convert string keys back to integers and validate data
            portfolio_data = {}
            for k, v in loaded_data.items():
                try:
                    row_idx = int(k)
                    if isinstance(v, dict):
                        portfolio_data[row_idx] = v
                    else:
                        logger.warning(f"Invalid data format for row {k}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse row {k}: {e}")
            
            if portfolio_data:
                logger.info(f"Loaded {len(portfolio_data)} rows from {state_file}")
                return portfolio_data
            else:
                logger.warning(f"State file {state_file} contains no valid data")
                return None
        
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed to load state from {state_file}: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
    
    logger.error(f"Failed to load state after {max_retries} attempts")
    return None


def update_state_batch(portfolio_data: Dict[int, Dict[str, str]], state_file: str,
                       updates: Dict[int, Dict[str, str]]) -> bool:
    """Update multiple rows in state file atomically with improved locking."""
    lock_path = state_file + ".lock"
    logger = logging.getLogger(__name__)
    
    try:
        with FileLock(lock_path, timeout=30):
            # Load current state from file to ensure we have latest data
            current_portfolio = {}
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        loaded_data = json.load(f)
                    current_portfolio = {int(k): v for k, v in loaded_data.items()}
                except Exception as e:
                    logger.warning(f"Failed to load existing state: {e}")
            
            # Apply updates to both in-memory and file data
            for row_idx, row_data in updates.items():
                current_portfolio[row_idx] = row_data
                portfolio_data[row_idx] = row_data
            
            # Write updated data atomically
            json_data = {str(k): v for k, v in current_portfolio.items()}
            temp_file = state_file + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
            os.rename(temp_file, state_file)
        
        logger.info(f"Batch updated {len(updates)} rows in {state_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to batch update state in {state_file}: {e}")
        return False


# =========================
# Tiktoken-based Similarity
# =========================

class SemanticMatcher:
    """Tiktoken-based semantic similarity for company name matching."""
    
    def __init__(self):
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        if _HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.logger.info("Loaded tiktoken tokenizer for semantic matching")
            except Exception as e:
                self.logger.warning(f"Failed to load tiktoken: {e}")
    
    def calculate_semantic_similarity(self, query: str, candidate: str) -> float:
        """Calculate semantic similarity using tiktoken embeddings."""
        if not self.tokenizer or not query.strip() or not candidate.strip():
            return 0.0
        
        try:
            # Normalize and tokenize
            query_clean = self._normalize_text(query)
            candidate_clean = self._normalize_text(candidate)
            
            query_tokens = set(self.tokenizer.encode(query_clean))
            candidate_tokens = set(self.tokenizer.encode(candidate_clean))
            
            if not query_tokens or not candidate_tokens:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(query_tokens.intersection(candidate_tokens))
            union = len(query_tokens.union(candidate_tokens))
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Add word-level similarity bonus
            query_words = set(query_clean.split())
            candidate_words = set(candidate_clean.split())
            
            word_intersection = len(query_words.intersection(candidate_words))
            word_union = len(query_words.union(candidate_words))
            word_score = word_intersection / word_union if word_union > 0 else 0.0
            
            # Combine scores with weighting
            final_score = (0.7 * jaccard_score) + (0.3 * word_score)
            
            return min(1.0, final_score)
            
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Remove common corporate suffixes
        suffixes = ['inc', 'corp', 'corporation', 'company', 'co', 'ltd', 'limited']
        words = normalized.split()
        filtered_words = [w for w in words if w not in suffixes]
        
        return ' '.join(filtered_words).strip()


# =========================
# Simplified API Calls
# =========================

@retry_with_backoff()
def _make_api_request(url: str, params: Dict[str, Any], timeout: int = Config.DEFAULT_TIMEOUT) -> requests.Response:
    """Make a single API request with comprehensive error handling."""
    logger = logging.getLogger(__name__)
    
    if not url or not isinstance(params, dict):
        raise ValueError("Invalid URL or parameters for API request")
    
    try:
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code == 429:
            logger.warning("Rate limit exceeded")
            raise requests.exceptions.HTTPError("Rate limit exceeded", response=response)
        elif response.status_code >= 400:
            logger.warning(f"HTTP error {response.status_code}")
            raise requests.exceptions.HTTPError(f"HTTP error: {response.status_code}", response=response)
        
        if not response.content:
            raise requests.exceptions.HTTPError("Empty response", response=response)
        
        response.raise_for_status()
        return response
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        raise


def resolve_symbol(company_name: str, api_key: str, 
                   timeout: int = Config.DEFAULT_TIMEOUT,
                   min_score: float = Config.DEFAULT_MIN_SCORE) -> Optional[str]:
    """
    Main symbol resolution entry point with smart caching and tiktoken.
    
    Priority: 1) Smart resolver (API + tiktoken caching), 2) Fallback API search
    """
    if not company_name or not company_name.strip():
        return None
    
    logger = logging.getLogger(__name__)
    original = company_name.strip()
    
    # Priority 1: Use smart resolver (API + tiktoken + caching)
    smart_result = smart_resolve_symbol(original, api_key, min_score)
    if smart_result:
        return smart_result
    
    # Priority 2: Fallback to direct API search as last resort
    logger.info(f"Smart resolver returned no match for '{original}', trying fallback API search")
    api_result = _search_api_direct(original, api_key, timeout)
    if api_result.status == ResultStatus.SUCCESS and api_result.confidence >= min_score * 0.5:
        logger.info(f"Fallback API match: '{original}' → {api_result.value} (confidence: {api_result.confidence:.3f})")
        return api_result.value
    
    logger.info(f"No suitable match found for '{original}'")
    return None


def _search_api_direct(query: str, api_key: str, timeout: int) -> ApiResult:
    """Direct API search with tiktoken-enhanced scoring."""
    logger = logging.getLogger(__name__)
    
    try:
        url = f"{Config.FINNHUB_BASE_URL}/search"
        params = {'q': query.strip(), 'token': api_key}
        
        response = _make_api_request(url, params, timeout)
        data = response.json()
        
        if 'result' not in data or not data['result']:
            return ApiResult(
                status=ResultStatus.NOT_FOUND,
                message=f"No results for '{query}'"
            )
        
        # Score results using tiktoken if available
        best_symbol, best_score = _score_api_results(query, data['result'])
        
        if best_symbol and best_score > 0:
            status = ResultStatus.SUCCESS if best_score >= 0.6 else ResultStatus.LOW_CONFIDENCE
            return ApiResult(
                status=status,
                value=best_symbol,
                confidence=best_score,
                message=f"Found {best_symbol} with score {best_score:.3f}"
            )
        
        return ApiResult(
            status=ResultStatus.NOT_FOUND,
            message=f"No good matches for '{query}'"
        )
        
    except Exception as e:
        logger.error(f"API search failed for '{query}': {e}")
        return ApiResult(
            status=ResultStatus.ERROR,
            message=f"API error: {e}"
        )


def _score_api_results(query: str, results: List[Dict]) -> Tuple[Optional[str], float]:
    """Score API results using improved semantic similarity and heuristics."""
    if not results:
        return None, 0.0
    
    matcher = SemanticMatcher()
    best_symbol = None
    best_score = 0.0
    
    print(f"    📊 Scoring {len(results)} API results:")
    
    query_clean = query.lower().strip()
    query_words = [w.strip() for w in query_clean.split() if w.strip()]
    
    for i, result in enumerate(results[:10]):  # Limit to top 10
        symbol = result.get('symbol', '').upper()
        description = result.get('description', '')
        
        if not symbol or not description:
            continue
        
        desc_clean = description.lower().strip()
        desc_words = [w.strip() for w in desc_clean.split() if w.strip()]
        
        # Start with semantic similarity
        semantic_score = matcher.calculate_semantic_similarity(query, description)
        
        # Exact company name match bonus (high priority)
        if query_clean in desc_clean or desc_clean in query_clean:
            semantic_score += 0.4
        
        # Word overlap scoring (improved)
        if query_words and desc_words:
            query_set = set(query_words)
            desc_set = set(desc_words)
            overlap = len(query_set.intersection(desc_set))
            overlap_score = overlap / len(query_set) if query_set else 0.0
            semantic_score += overlap_score * 0.3
        
        # Acronym bonus (strong indicator)
        acronym_bonus = 0.0
        if len(query_words) >= 2:
            # Full acronym match
            acronym = ''.join([w[0].upper() for w in query_words[:4]])  # Limit to 4 chars
            if acronym == symbol:
                acronym_bonus = 0.5
            # Partial acronym match
            elif len(acronym) >= 2 and symbol.startswith(acronym[:2]):
                acronym_bonus = 0.2
        
        # Company suffix matching
        suffix_bonus = 0.0
        query_has_co = any(word in query_clean for word in ['company', 'corp', 'inc', 'co.'])
        desc_has_co = any(word in desc_clean for word in ['company', 'corp', 'inc', 'co.'])
        if query_has_co and desc_has_co:
            suffix_bonus = 0.1
        
        # Symbol quality bonus
        symbol_bonus = 0.0
        if len(symbol) <= 4 and symbol.isalpha():
            symbol_bonus = 0.05
        elif '.' in symbol and len(symbol) <= 6:  # Class shares like BRK.A
            symbol_bonus = 0.03
        
        # Exchange preference (US markets)
        exchange_bonus = 0.0
        exchange = result.get('mic', '')
        if exchange in ['XNAS', 'XNYS', 'ARCX']:
            exchange_bonus = 0.05
        
        # Combined score
        final_score = semantic_score + acronym_bonus + suffix_bonus + symbol_bonus + exchange_bonus
        final_score = min(1.0, max(0.0, final_score))
        
        print(f"      {i+1}. {symbol} ({description[:60]}...) → Score: {final_score:.3f}")
        
        if final_score > best_score:
            best_score = final_score
            best_symbol = symbol
    
    return best_symbol, best_score


def _generate_search_variations(company_name: str) -> List[str]:
    """Generate simplified search variations."""
    variations = []
    
    # Remove corporate suffixes
    cleaned = company_name
    suffixes = [' Inc.', ' Corporation', ' Corp.', ' Co.', ' Company', ' Ltd.', ' Limited']
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
            break
    
    if cleaned != company_name:
        variations.append(cleaned.strip())
    
    # Remove "The" prefix
    if company_name.startswith('The '):
        variations.append(company_name[4:].strip())
    
    # Handle class designations
    if ' Class ' in company_name:
        without_class = company_name.split(' Class ')[0].strip()
        variations.append(without_class)
    
    # First two words only
    words = company_name.split()
    if len(words) >= 2:
        variations.append(' '.join(words[:2]))
    
    # Remove duplicates and empty strings
    unique_variations = []
    for var in variations:
        if var.strip() and var not in unique_variations and var != company_name:
            unique_variations.append(var.strip())
    
    return unique_variations


def resolve_name(symbol: str, api_key: str, timeout: int = Config.DEFAULT_TIMEOUT) -> Optional[str]:
    """Get company name for a stock symbol using Finnhub profile API."""
    logger = logging.getLogger(__name__)
    
    if not symbol or not symbol.strip():
        return None
    
    try:
        symbol = symbol.upper().strip()
        
        url = f"{Config.FINNHUB_BASE_URL}/stock/profile2"
        params = {'symbol': symbol, 'token': api_key}
        
        print(f"🏢 Looking up name for symbol: '{symbol}'")
        response = _make_api_request(url, params, timeout)
        
        data = response.json()
        
        if not isinstance(data, dict):
            logger.warning(f"Unexpected response format for symbol '{symbol}'")
            return None
        
        name = data.get('name', '').strip()
        if name and len(name) > 1:
            message = f"Found name '{name}' for '{symbol}'"
            print(f"    ✅ {message}")
            logger.info(message)
            return name
        
        message = f"No name found for '{symbol}'"
        print(f"    ❌ {message}")
        logger.info(message)
        return None
        
    except Exception as e:
        error_msg = f"Error resolving name for '{symbol}': {e}"
        print(f"    💥 {error_msg}")
        logger.error(error_msg)
        return None


# =========================
# Utilities
# =========================

def create_shards(items: List[int], shard_size: int) -> List[List[int]]:
    """Create shards of items for parallel processing."""
    shards = []
    for i in range(0, len(items), shard_size):
        shards.append(items[i:i + shard_size])
    return shards


def get_api_key() -> str:
    """Get API key from environment."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not found in environment variables")
    return api_key


# =========================
# Smart Symbol Resolution with Caching
# =========================

class SmartSymbolResolver:
    """Smart symbol resolver with caching and tiktoken-based similarity."""
    
    def __init__(self, api_key: str, cache_file: str = "company_cache.json"):
        self.api_key = api_key
        self.cache_file = cache_file
        self.company_data = None
        self.semantic_matcher = SemanticMatcher()
        self.logger = logging.getLogger(__name__)
    
    def load_or_fetch_companies(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load companies from cache or fetch from API."""
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                if (datetime.now() - cache_time).days < 7:
                    self.logger.info(f"Using cached company data ({len(data['companies'])} companies)")
                    return pd.DataFrame(data['companies'])
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        return self.fetch_companies()
    
    def fetch_companies(self) -> pd.DataFrame:
        """Fetch all US companies from Finnhub and cache them."""
        self.logger.info("Fetching all US companies from Finnhub...")
        
        try:
            response = requests.get(
                f"{Config.FINNHUB_BASE_URL}/stock/symbol",
                params={'exchange': 'US', 'token': self.api_key},
                timeout=30
            )
            
            if response.status_code == 429:
                self.logger.warning("Rate limited, waiting...")
                time.sleep(2)
                response = requests.get(
                    f"{Config.FINNHUB_BASE_URL}/stock/symbol",
                    params={'exchange': 'US', 'token': self.api_key},
                    timeout=30
                )
            
            response.raise_for_status()
            data = response.json()
            
            # Process and filter data
            companies = [
                {
                    'symbol': item.get('symbol', '').strip(),
                    'name': item.get('description', '').strip(),
                    'type': item.get('type', ''),
                    'currency': item.get('currency', 'USD')
                }
                for item in data
                if (item.get('symbol', '').strip() and 
                    item.get('description', '').strip() and 
                    len(item.get('symbol', '')) <= 6 and
                    'Common Stock' in item.get('type', ''))
            ]
            
            df = pd.DataFrame(companies).drop_duplicates(subset=['symbol'], keep='first')
            self.logger.info(f"Fetched {len(df)} companies")
            
            # Cache the data
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'companies': df.to_dict('records')
                }, f, indent=2)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch companies: {e}")
            return pd.DataFrame(columns=['symbol', 'name', 'type', 'currency'])
    
    def resolve_symbol(self, company_name: str, min_score: float = 0.7) -> Optional[str]:
        """
        Resolve company name to symbol using three-tier approach:
        1. Primary API search (most accurate)
        2. Original API search with variations  
        3. String matching against cached data (tiktoken-based)
        """
        if not company_name or not company_name.strip():
            return None
        
        # Initialize cached data if needed
        if self.company_data is None:
            self.company_data = self.load_or_fetch_companies()
            if self.company_data.empty:
                self.logger.error("No company data available")
                return None
        
        # Method 1: Primary API search (most accurate)
        api_result = self.try_api_search(company_name)
        if api_result:
            return api_result
        
        # Method 2: Original API search with variations  
        original_result = self.try_original_api_search(company_name, min_score)
        if original_result:
            return original_result
        
        # Method 3: String matching against cached data
        string_result = self.try_tiktoken_matching(company_name, min_score)
        if string_result:
            return string_result
        
        self.logger.info(f"No matches found for '{company_name}' in any method")
        return None
    
    def try_api_search(self, company_name: str) -> Optional[str]:
        """Try direct API search first."""
        try:
            result = _search_api_direct(company_name, self.api_key, Config.DEFAULT_TIMEOUT)
            if result.status == ResultStatus.SUCCESS and result.value and result.confidence >= 0.6:
                self.logger.info(f"API match: '{company_name}' → {result.value}")
                return result.value
        except Exception as e:
            self.logger.warning(f"Primary API search failed: {e}")
        return None
    
    def try_original_api_search(self, company_name: str, min_score: float) -> Optional[str]:
        """Try original API search with variations."""
        try:
            variations = _generate_search_variations(company_name)
            for variation in variations:
                result = _search_api_direct(variation, self.api_key, Config.DEFAULT_TIMEOUT)
                if result.status == ResultStatus.SUCCESS and result.value and result.confidence >= min_score:
                    self.logger.info(f"Original API match: '{company_name}' (via '{variation}') → {result.value}")
                    return result.value
        except Exception as e:
            self.logger.warning(f"Original API search failed: {e}")
        return None
    
    def try_tiktoken_matching(self, company_name: str, min_score: float) -> Optional[str]:
        """Try tiktoken-based string matching against cached data."""
        matches = self.find_tiktoken_matches(company_name, top_k=5)
        if matches:
            symbol, name, score = matches[0]
            if score >= min_score:
                self.logger.info(f"Tiktoken match: '{company_name}' → {symbol} ({name}) [score: {score:.3f}]")
                return symbol
        return None
    
    def find_tiktoken_matches(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Find best matches using tiktoken-based similarity."""
        if self.company_data is None or self.company_data.empty:
            return []
        
        results = []
        
        for _, row in self.company_data.iterrows():
            name = row['name']
            symbol = row['symbol']
            
            # Calculate tiktoken-based similarity
            similarity = self.semantic_matcher.calculate_semantic_similarity(query, name)
            
            # Add simple heuristics bonuses
            query_lower = query.lower()
            name_lower = name.lower()
            
            # Exact match bonus
            if query_lower == name_lower:
                similarity = 1.0
            # Acronym match bonus
            elif len(query.split()) >= 2:
                acronym = ''.join([w[0] for w in query.split() if w]).upper()
                if acronym == symbol:
                    similarity = max(similarity, 0.9)
            
            results.append((symbol, name, similarity))
        
        # Sort and return top matches
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# Global resolver instance
_symbol_resolver = None

def get_symbol_resolver(api_key: str) -> SmartSymbolResolver:
    """Get or create the global symbol resolver instance."""
    global _symbol_resolver
    if _symbol_resolver is None:
        _symbol_resolver = SmartSymbolResolver(api_key)
    return _symbol_resolver

def smart_resolve_symbol(company_name: str, api_key: str, min_score: float = 0.7) -> Optional[str]:
    """
    Smart symbol resolution with API priority and tiktoken.
    Priority: 1) API → 2) Original API → 3) Tiktoken matching
    """
    if not company_name.strip():
        return None
    
    resolver = get_symbol_resolver(api_key)
    return resolver.resolve_symbol(company_name, min_score)
