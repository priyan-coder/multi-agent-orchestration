#!/usr/bin/env python3
"""
Core Functions for Portfolio Enrichment Multi-Agent System

Two-Step Symbol Resolution:
1. Try FinnHub API with full company name
2. If fails, use OpenAI GPT-4 to directly query for ticker symbol

Features:
- Direct OpenAI integration for challenging cases
- Robust state management with proper file locking
- Simplified logic with high accuracy
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

# OpenAI integration for direct symbol resolution
try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

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
    MAX_RETRIES = 2
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
    """
    Thread-safe in-memory rate limiter for API calls.

    Tracks request timestamps and enforces rate limits by blocking
    when the maximum requests per minute threshold is exceeded.
    """

    def __init__(self, max_requests: int = Config.MAX_REQUESTS_PER_MINUTE):
        """
        Initialize rate limiter with request limit.

        Args:
            max_requests: Maximum requests allowed per minute
        """
        self.max_requests = int(max_requests * Config.RATE_LIMIT_BUFFER)
        self.requests = []

    def wait_if_needed(self) -> None:
        """
        Block execution if rate limit would be exceeded.

        Automatically cleans up old request timestamps and calculates
        wait time to respect the configured rate limit.
        """
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        if len(self.requests) >= self.max_requests:
            oldest_request = min(self.requests)
            wait_time = 60 - (now - oldest_request) + 0.1
            if wait_time > 0:
                logging.getLogger(__name__).warning(
                    f"Rate limit reached, waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)

        self.requests.append(now)


_rate_limiter = RateLimiter()


def retry_with_backoff(max_retries: int = Config.MAX_RETRIES):
    """
    Decorator implementing exponential backoff retry logic for API calls.

    Automatically retries failed requests with increasing delays,
    respecting rate limits and handling common API failures gracefully.

    Args:
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic

    Raises:
        Original exception after all retries exhausted
    """

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
                            Config.INITIAL_BACKOFF
                            * (Config.BACKOFF_MULTIPLIER**attempt),
                            Config.MAX_BACKOFF,
                        )
                        jitter = random.uniform(0.5, 1.5)
                        wait_time = backoff * jitter
                        logger.warning(
                            f"API call failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        logger.warning(f"Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"API call failed after {max_retries} attempts: {e}"
                        )

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
    """
    Configure namespaced logging with file and console handlers.

    Creates a logger with both file output (monitoring.log) and console
    output, avoiding duplicate handlers on repeated calls.

    Args:
        name: Logger namespace (defaults to current module)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(Config.DEFAULT_LOG_FILE)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def append_markdown(md_path: str, text: str) -> None:
    """
    Thread-safe markdown file append with file locking.

    Safely appends text to markdown files using file locks to prevent
    corruption in multi-process environments.

    Args:
        md_path: Path to markdown file
        text: Text content to append
    """
    lock_path = md_path + ".lock"
    try:
        with FileLock(lock_path, timeout=5):
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to write to markdown: {e}")


def log_event(
    task_name: str, task_id: str, event: str, message: str, md_path: str, **extras
) -> None:
    """
    Log structured events with both console and markdown output.

    Creates timestamped log entries for tracking task execution across
    parallel processes, with markdown formatting for reporting.

    Args:
        task_name: Name of the executing task
        task_id: Unique identifier for this task instance
        event: Event type (start, progress, end, error)
        message: Human-readable event description
        md_path: Path to markdown log file
        **extras: Additional metadata to include
    """
    timestamp = datetime.now().isoformat()
    logging.info(f"[{task_name}:{task_id}] {event}: {message}")

    if event == "start":
        append_markdown(
            md_path, f"\n## [{timestamp}] Task: {task_name} (id={task_id})\n"
        )
        append_markdown(md_path, f"- Input: {message}")
    elif event == "progress":
        append_markdown(md_path, f"- Progress: {message}")
    elif event == "end":
        append_markdown(md_path, f"- End: {message}")
    elif event in ["warning", "error"]:
        append_markdown(md_path, f"- {event.upper()}: {message}")


# =========================
# State Management
# =========================


def save_state(portfolio_data: Dict[int, Dict[str, str]], state_file: str) -> bool:
    """
    Atomically serialize portfolio data to JSON with file locking.

    Writes portfolio state to disk using atomic operations and file locking
    to prevent corruption in multi-process environments.

    Args:
        portfolio_data: Dictionary mapping row indices to portfolio entries
        state_file: Path to state file

    Returns:
        True if save successful, False otherwise
    """
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
            with open(temp_file, "w", encoding="utf-8") as f:
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
    """
    Load portfolio data from JSON state file with retry logic.

    Safely reads serialized portfolio state from disk with file locking
    and automatic retry on transient failures.

    Args:
        state_file: Path to state file

    Returns:
        Portfolio data dictionary or None if load fails
    """
    if not os.path.exists(state_file):
        logging.getLogger(__name__).warning(f"State file {state_file} does not exist")
        return None

    lock_path = state_file + ".lock"
    logger = logging.getLogger(__name__)
    max_retries = 5

    for attempt in range(max_retries):
        try:
            with FileLock(lock_path, timeout=30):
                with open(state_file, "r", encoding="utf-8") as f:
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
            logger.error(
                f"Attempt {attempt + 1} failed to load state from {state_file}: {e}"
            )
            if attempt < max_retries - 1:
                wait_time = (2**attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

    logger.error(f"Failed to load state after {max_retries} attempts")
    return None


def update_state_batch(
    portfolio_data: Dict[int, Dict[str, str]],
    state_file: str,
    updates: Dict[int, Dict[str, str]],
) -> bool:
    """
    Atomically update multiple portfolio rows in state file.

    Applies batch updates to portfolio state while maintaining data
    consistency using file locking and atomic operations.

    Args:
        portfolio_data: Current in-memory portfolio data
        state_file: Path to state file
        updates: Dictionary of row updates to apply

    Returns:
        True if all updates applied successfully, False otherwise
    """
    lock_path = state_file + ".lock"
    logger = logging.getLogger(__name__)

    try:
        with FileLock(lock_path, timeout=30):
            # Load current state from file to ensure we have latest data
            current_portfolio = {}
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
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
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            os.rename(temp_file, state_file)

        logger.info(f"Batch updated {len(updates)} rows in {state_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to batch update state in {state_file}: {e}")
        return False


# =========================
# Simplified API Calls
# =========================


@retry_with_backoff()
def _make_api_request(
    url: str, params: Dict[str, Any], timeout: int = Config.DEFAULT_TIMEOUT
) -> requests.Response:
    """
    Execute HTTP GET request with comprehensive error handling.

    Makes API requests with automatic retry logic, rate limiting,
    and detailed error handling for various HTTP failure modes.

    Args:
        url: Target API endpoint URL
        params: Query parameters dictionary
        timeout: Request timeout in seconds

    Returns:
        HTTP response object

    Raises:
        HTTPError: For HTTP errors (4xx, 5xx)
        RequestException: For network/connection errors
    """
    logger = logging.getLogger(__name__)

    if not url or not isinstance(params, dict):
        raise ValueError("Invalid URL or parameters for API request")

    try:
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 429:
            logger.warning("Rate limit exceeded")
            raise requests.exceptions.HTTPError(
                "Rate limit exceeded", response=response
            )
        elif response.status_code >= 400:
            logger.warning(f"HTTP error {response.status_code}")
            raise requests.exceptions.HTTPError(
                f"HTTP error: {response.status_code}", response=response
            )

        if not response.content:
            raise requests.exceptions.HTTPError("Empty response", response=response)

        response.raise_for_status()
        return response

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        raise


def resolve_symbol(
    company_name: str,
    api_key: str,
    timeout: int = Config.DEFAULT_TIMEOUT,
    min_score: float = Config.DEFAULT_MIN_SCORE,
    resolver: Optional["SmartSymbolResolver"] = None,
) -> Optional[str]:
    """
    Simplified two-step symbol resolution:
    1. Try FinnHub API with full company name
    2. If fails, use OpenAI GPT-4 to directly query for ticker symbol

    Args:
        company_name: The company name to resolve
        api_key: FinnHub API key
        timeout: Request timeout in seconds
        min_score: Minimum match score for fuzzy matching
        resolver: Optional resolver instance (creates new one if None)
    """
    if not company_name or not company_name.strip():
        return None

    logger = logging.getLogger(__name__)
    original = company_name.strip()

    # Create resolver instance if not provided (no global state)
    if resolver is None:
        resolver = SmartSymbolResolver(api_key)

    result = resolver.resolve_symbol(original, min_score)

    if result:
        logger.info(f"Symbol resolved: '{original}' → {result}")
    else:
        logger.info(f"No suitable match found for '{original}'")

    return result


def resolve_name(
    symbol: str, api_key: str, timeout: int = Config.DEFAULT_TIMEOUT
) -> Optional[str]:
    """
    Resolve company name from stock symbol using Finnhub API.

    Looks up company profile information to retrieve the official
    company name for a given stock ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        api_key: Finnhub API key
        timeout: Request timeout in seconds

    Returns:
        Company name or None if not found
    """
    logger = logging.getLogger(__name__)

    if not symbol or not symbol.strip():
        return None

    try:
        symbol = symbol.upper().strip()

        url = f"{Config.FINNHUB_BASE_URL}/stock/profile2"
        params = {"symbol": symbol, "token": api_key}

        print(f"🏢 Looking up name for symbol: '{symbol}'")
        response = _make_api_request(url, params, timeout)

        data = response.json()

        if not isinstance(data, dict):
            logger.warning(f"Unexpected response format for symbol '{symbol}'")
            return None

        name = data.get("name", "").strip()
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
    """
    Split a list of items into chunks for parallel processing.

    Divides a list into smaller sublists of approximately equal size
    to enable efficient parallel processing across multiple workers.

    Args:
        items: List of items to shard
        shard_size: Maximum size of each shard

    Returns:
        List of sublists (shards)
    """
    shards = []
    for i in range(0, len(items), shard_size):
        shards.append(items[i : i + shard_size])
    return shards


def get_api_key() -> str:
    """
    Retrieve Finnhub API key from environment variables.

    Reads the API key from the FINNHUB_API_KEY environment variable
    with validation to ensure it's properly configured.

    Returns:
        Finnhub API key string

    Raises:
        ValueError: If API key not found in environment
    """
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY not found in environment variables")
    return api_key


# =========================
# Smart Symbol Resolution with Caching
# =========================


class SmartSymbolResolver:
    """
    Intelligent symbol resolver with two-step fallback strategy.

    Implements a robust symbol resolution system that first attempts
    direct API lookup, then falls back to AI-powered resolution for
    challenging cases. Includes company data caching for efficiency.
    """

    def __init__(self, api_key: str, cache_file: str = "company_cache.json"):
        """
        Initialize symbol resolver with API credentials and caching.

        Args:
            api_key: Finnhub API key for data access
            cache_file: Path to company data cache file
        """
        self.api_key = api_key
        self.cache_file = cache_file
        self.company_data = None
        self.logger = logging.getLogger(__name__)

    def load_or_fetch_companies(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load company data from cache or fetch fresh data from API.

        Attempts to use cached company data if recent (< 7 days),
        otherwise fetches fresh data from Finnhub API.

        Args:
            force_refresh: If True, skip cache and fetch fresh data

        Returns:
            DataFrame containing company information
        """
        if not force_refresh and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                cache_time = datetime.fromisoformat(data.get("timestamp", "2000-01-01"))
                if (datetime.now() - cache_time).days < 7:
                    self.logger.info(
                        f"Using cached company data ({len(data['companies'])} companies)"
                    )
                    return pd.DataFrame(data["companies"])
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")

        return self.fetch_companies()

    def fetch_companies(self) -> pd.DataFrame:
        """
        Fetch comprehensive US company data from Finnhub API.

        Downloads all US exchange company listings with filtering
        and deduplication, then caches the results locally.

        Returns:
            DataFrame with company symbols, names, and metadata

        Raises:
            RequestException: If API call fails
        """
        self.logger.info("Fetching all US companies from Finnhub...")

        try:
            response = requests.get(
                f"{Config.FINNHUB_BASE_URL}/stock/symbol",
                params={"exchange": "US", "token": self.api_key},
                timeout=30,
            )

            if response.status_code == 429:
                self.logger.warning("Rate limited, waiting...")
                time.sleep(2)
                response = requests.get(
                    f"{Config.FINNHUB_BASE_URL}/stock/symbol",
                    params={"exchange": "US", "token": self.api_key},
                    timeout=30,
                )

            response.raise_for_status()
            data = response.json()

            # Process and filter data
            companies = [
                {
                    "symbol": item.get("symbol", "").strip(),
                    "name": item.get("description", "").strip(),
                    "type": item.get("type", ""),
                    "currency": item.get("currency", "USD"),
                }
                for item in data
                if (
                    item.get("symbol", "").strip()
                    and item.get("description", "").strip()
                    and len(item.get("symbol", "")) <= 6
                    and "Common Stock" in item.get("type", "")
                )
            ]

            df = pd.DataFrame(companies).drop_duplicates(
                subset=["symbol"], keep="first"
            )
            self.logger.info(f"Fetched {len(df)} companies")

            # Cache the data
            with open(self.cache_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "companies": df.to_dict("records"),
                    },
                    f,
                    indent=2,
                )

            return df

        except Exception as e:
            self.logger.error(f"Failed to fetch companies: {e}")
            return pd.DataFrame(columns=["symbol", "name", "type", "currency"])

    def resolve_symbol(
        self, company_name: str, min_score: float = 0.7
    ) -> Optional[str]:
        """
        Resolve stock symbol using intelligent two-step strategy.

        First attempts direct API lookup, then falls back to AI-powered
        resolution for challenging cases that require interpretation.

        Args:
            company_name: Company name to resolve
            min_score: Minimum fuzzy match confidence threshold

        Returns:
            Stock symbol or None if resolution fails
        """
        if not company_name or not company_name.strip():
            return None

        company_clean = company_name.strip()

        # Step 1: Try FinnHub API with full company name
        self.logger.info(f"Trying FinnHub API for '{company_clean}'")
        api_result = self.try_simple_api_search(company_clean)
        if api_result:
            self.logger.info(f"FinnHub API success: '{company_clean}' → {api_result}")
            return api_result

        # Step 2: Ask OpenAI directly for the ticker symbol
        self.logger.info(
            f"FinnHub API failed for '{company_clean}', asking OpenAI directly"
        )
        openai_result = self.ask_openai_for_ticker(company_clean)
        if openai_result:
            self.logger.info(f"OpenAI success: '{company_clean}' → {openai_result}")
            return openai_result

        self.logger.info(f"Both methods failed for '{company_clean}'")
        return None

    def try_simple_api_search(self, company_name: str) -> Optional[str]:
        """
        Perform direct API search with intelligent filtering.

        Searches company database for name matches, prioritizing
        simple US stock symbols without complex suffixes or modifiers.

        Args:
            company_name: Company name to search for

        Returns:
            Best matching stock symbol or None
        """
        try:
            url = f"{Config.FINNHUB_BASE_URL}/search"
            params = {"q": company_name, "token": self.api_key}

            response = _make_api_request(url, params, Config.DEFAULT_TIMEOUT)
            data = response.json()

            if "result" not in data or not data["result"]:
                return None

            # Look for the simplest, most likely US symbol
            for result in data["result"][:5]:  # Check first 5 results
                symbol = result.get("symbol", "").upper().strip()
                description = result.get("description", "").lower()

                # Prefer simple symbols without dots or complex suffixes
                if (
                    len(symbol) <= 5
                    and symbol.isalpha()
                    and "." not in symbol
                    and company_name.lower() in description
                ):
                    self.logger.info(f"Simple API match: {symbol}")
                    return symbol

            # If no simple match, return the first result
            first_result = data["result"][0]
            symbol = first_result.get("symbol", "").upper().strip()
            if symbol and len(symbol) <= 6:
                self.logger.info(f"First API result: {symbol}")
                return symbol

            return None

        except Exception as e:
            self.logger.warning(f"Simple API search failed: {e}")
            return None

    def ask_openai_for_ticker(self, company_name: str) -> Optional[str]:
        """
        Use AI to resolve challenging symbol lookups.

        Leverages GPT-4 for intelligent symbol resolution when direct
        API searches fail. Handles edge cases like share classes,
        complex corporate names, and ambiguous company references.

        Args:
            company_name: Company name to resolve

        Returns:
            Stock ticker symbol or None if resolution fails
        """
        try:
            import openai
            import os
            from dotenv import load_dotenv

            load_dotenv()
            openai_api_key = os.getenv("OPEN_AI_API_KEY")

            if not openai_api_key:
                self.logger.warning("OpenAI API key not found")
                return None

            client = openai.OpenAI(api_key=openai_api_key)

            prompt = f"""What is the exact stock ticker symbol for "{company_name}"?

Requirements:
- Provide ONLY the ticker symbol (like "PG" or "BRK.B")
- Use the symbol traded on major US exchanges (NYSE, NASDAQ)
- For companies with multiple share classes, match the class mentioned in the name
- If the company doesn't exist or isn't publicly traded, respond with "NONE"
- No explanation, just the symbol

Company: {company_name}
Ticker:"""

            response = client.chat.completions.create(
                model="gpt-4",  # Use latest available model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data expert. Provide exact stock ticker symbols for US-listed companies.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            result = response.choices[0].message.content.strip().upper()

            # Validate the result
            if result and result != "NONE" and len(result) <= 6:
                self.logger.info(f"OpenAI suggested ticker: {result}")
                return result

            return None

        except Exception as e:
            self.logger.warning(f"OpenAI ticker query failed: {e}")
            return None


def create_symbol_resolver(api_key: str) -> "SmartSymbolResolver":
    """
    Create a new symbol resolver instance.

    Use this when you need to resolve multiple symbols and want to reuse
    the same resolver instance for efficiency (shared caching).

    Args:
        api_key: FinnHub API key

    Returns:
        A new SmartSymbolResolver instance

    Example:
        # For single resolution (simple)
        symbol = resolve_symbol("Apple Inc", api_key)

        # For batch resolution (efficient)
        resolver = create_symbol_resolver(api_key)
        symbols = [resolve_symbol(name, api_key, resolver=resolver) for name in names]
    """
    return SmartSymbolResolver(api_key)


def resolve_symbols_batch(
    company_names: List[str],
    api_key: str,
    timeout: int = Config.DEFAULT_TIMEOUT,
    min_score: float = Config.DEFAULT_MIN_SCORE,
) -> List[Optional[str]]:
    """
    Resolve multiple symbols efficiently using a shared resolver instance.

    Args:
        company_names: List of company names to resolve
        api_key: FinnHub API key
        timeout: Request timeout in seconds
        min_score: Minimum match score for fuzzy matching

    Returns:
        List of resolved symbols (None for failed resolutions)
    """
    if not company_names:
        return []

    # Create a single resolver instance for all resolutions
    resolver = create_symbol_resolver(api_key)

    return [
        resolve_symbol(name, api_key, timeout, min_score, resolver)
        for name in company_names
    ]
