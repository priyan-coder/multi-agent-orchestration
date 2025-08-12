# Portfolio Enrichment System - Logic Documentation

## 🏗️ System Architecture

The Portfolio Enrichment System is a **distributed multi-agent processing system** that enriches portfolio CSV data by resolving missing stock symbols and company names using the Finnhub API with intelligent sharding, parallel execution, and advanced semantic matching.

## 🔄 Core API Logic & Resolution Engine (`core_functions.py`)

### Symbol Resolution Three-Tier Strategy

The system implements a sophisticated **three-tier resolution strategy** with priority fallback:

#### **Main Entry Point: `resolve_symbol()`**

```text
Priority Flow: Smart Resolver → Direct API Fallback
Confidence Threshold: 0.6 (configurable)
Fallback Threshold: 0.3 (reduced for last resort)
```

#### **Smart Symbol Resolver: `SmartSymbolResolver`**

##### **Tier 1: Primary API Search**

- Direct Finnhub `/search` API call
- Enhanced tiktoken-based result scoring
- Returns immediately if confidence ≥ 0.6
- Most accurate, real-time results

##### **Tier 2: API Search with Intelligent Variations**

- Generates semantic search variations:
  - Corporate suffix removal (Inc., Corp., Ltd., Co.)
  - "The" prefix handling
  - Class designation stripping (Class A/B)
  - Two-word truncation for focus
- Each variation tested with full scoring
- Returns first match above minimum threshold

##### **Tier 3: Tiktoken-Based Cached Matching**

- Uses pre-fetched US company universe (7-day TTL)
- **Advanced semantic similarity**:
  - Token-level Jaccard similarity (cl100k_base encoding)
  - Word-level overlap bonus weighting
  - Exact match detection (1.0 score)
  - Acronym matching heuristics
- Offline fallback when API unavailable

### Enhanced API Result Scoring Algorithm

```python
Final Score = (0.7 × Semantic Score) + Bonuses

Bonuses:
- Exact name match: +0.4
- Acronym match (full): +0.5
- Acronym match (partial): +0.2
- Corporate suffix alignment: +0.1
- Preferred exchange (NASDAQ/NYSE): +0.05
- Symbol quality (≤4 chars, alpha): +0.05
```

### Company Name Resolution: `resolve_name()`

- **API Endpoint**: Finnhub `/stock/profile2`
- **Input**: Stock symbol (normalized uppercase)
- **Output**: Company name with validation
- **Error Handling**: Specific handling for invalid symbols

## 🚦 Rate Limiting & Error Handling

### Intelligent Rate Limiter

```python
Configuration:
- Base Rate: 60 requests/minute
- Safety Buffer: 90% (54 effective requests/minute)
- Sliding Window: In-memory timestamp tracking
- Auto-throttling: Dynamic wait calculation
```

### Exponential Backoff Retry

```python
Strategy: @retry_with_backoff decorator
- Max Retries: 3 attempts
- Base Delay: 1.0 seconds
- Backoff Multiplier: 2.0x
- Max Delay: 30 seconds
- Jitter: 0.5-1.5x randomization
```

### Comprehensive Error Classification

```python
Status Codes:
- SUCCESS: Successful resolution
- NOT_FOUND: No matches found
- LOW_CONFIDENCE: Match below threshold
- RATE_LIMITED: 429 HTTP response
- ERROR: Network/API failures
```

## 🗄️ Advanced State Management

### Atomic Operations with FileLock

```python
Pattern: Load → Modify → Write to Temp → Atomic Rename
Locking: FileLock with 30-second timeout
Concurrency: Multiple shards can safely update state
Recovery: Automatic temp file cleanup on failure
```

### State File Structure

```json
{
  "0": {
    "Name": "Apple Inc.",
    "Symbol": "AAPL",
    "Price": "150.00",
    "# of Shares": "100",
    "Market Value": "15000"
  }
}
```

### Batch Update Strategy

```python
Process:
1. Load current state under lock
2. Apply all shard updates to memory
3. Write updated state atomically
4. Update in-memory portfolio_data
5. Release lock
```

## 📊 Intelligent Sharding System - Dividing Work for Parallel Processing

Sharding is like dividing a big pile of homework among several students so everyone can work simultaneously instead of one person doing everything sequentially. Our system automatically divides the portfolio data into optimal chunks for parallel processing.

#### The Problem: Sequential Processing is Slow

Imagine you have a portfolio with 10,000 stocks, and 2,000 of them are missing symbols. If you process them one by one:

- Each API call takes ~1 second (including rate limiting)
- Total time: 2,000 seconds = 33 minutes
- Only one CPU core is working while others sit idle
- If one stock lookup fails, it doesn't affect others, but you're still going slowly

#### The Solution: Intelligent Work Division

Instead of processing stocks one by one, we divide them into "shards" (chunks) that can be processed simultaneously by different workers:

```python
# Example: 1,000 missing symbols divided into 5 shards
missing_symbols = [1, 5, 7, 12, 15, 18, 23, 25, 28, 30, ...1000 more...]

# Create shards of size 200 each
shard_0 = [1, 5, 7, 12, 15, 18, 23, 25, 28, 30, ...] # 200 symbols
shard_1 = [45, 67, 89, 92, 105, 134, 156, 178, ...] # 200 symbols
shard_2 = [234, 267, 289, 312, 345, 367, 389, ...]  # 200 symbols
shard_3 = [445, 467, 489, 512, 534, 556, 578, ...]  # 200 symbols
shard_4 = [645, 667, 689, 712, 734, 756, 778, ...]  # 200 symbols
```

Now instead of 33 minutes sequentially, we get:

- 5 workers processing simultaneously
- Each handles 200 symbols = ~200 seconds = 3.3 minutes
- Total time: 3.3 minutes (10x faster!)

#### Our Sharding Algorithm: `create_shards()`

Our sharding function is deceptively simple but powerful:

```python
def create_shards(items: List[int], shard_size: int) -> List[List[int]]:
    """Create balanced shards for parallel processing."""
    shards = []
    for i in range(0, len(items), shard_size):
        shards.append(items[i:i + shard_size])
    return shards
```

Let's break down what this does step by step:

**Step 1: Input Analysis**

- `items`: List of row indices that need processing [1, 5, 7, 12, 15, 18, 23, 25, 28, 30]
- `shard_size`: How many items per shard (configurable: 2 for demo, 50-100 for production)

**Step 2: Iteration Logic**

- `range(0, len(items), shard_size)` creates positions: [0, shard_size, 2*shard_size, ...]
- For 10 items with shard_size=3: positions are [0, 3, 6, 9]

**Step 3: Slice Creation**

- `items[0:3]` = first shard = [1, 5, 7]
- `items[3:6]` = second shard = [12, 15, 18]
- `items[6:9]` = third shard = [23, 25, 28]
- `items[9:12]` = fourth shard = [30] (last shard may be smaller)

#### Two Types of Shards: Symbols vs Names

Our system creates two separate types of shards because they require different processing:

**Symbol Resolution Shards**:

- Input: Rows that have company names but missing stock symbols
- Task: Use company name to find stock symbol (harder, needs AI matching)
- API: Finnhub search endpoint with our scoring algorithm

**Name Resolution Shards**:

- Input: Rows that have stock symbols but missing company names
- Task: Use stock symbol to find company name (easier, direct lookup)
- API: Finnhub profile endpoint (simple key-value lookup)

#### Smart Shard Size Selection

The shard size dramatically affects performance:

**Too Small (shard_size = 1)**:

- Problem: Too much overhead from creating/managing many workers
- Example: 1,000 items = 1,000 separate workers = system overwhelmed

**Too Large (shard_size = 1000)**:

- Problem: No parallelism benefit, back to sequential processing
- Example: 1,000 items = 1 worker = no speed improvement

**Just Right (shard_size = 50-100)**:

- Sweet spot: Good parallelism without excessive overhead
- Example: 1,000 items = 10-20 workers = optimal resource usage

#### Our Adaptive Configuration

```python
# Development/Demo: Small shards for quick testing
shard_size = 2  # Easy to see individual shard progress

# Production: Optimized shards for real workloads
shard_size = 50-100  # Balance between parallelism and efficiency
```

#### Load Balancing Across Shards

Our sharding automatically provides perfect load balancing:

**Even Distribution**: If you have 1,003 items with shard_size=100:

- Shards 1-10: Each gets exactly 100 items
- Shard 11: Gets 3 items (automatically handles remainder)

**No Manual Balancing**: The algorithm automatically ensures work is distributed evenly without any complex load balancing logic.

#### Real-World Example: Portfolio with Mixed Missing Data

```python
# Original portfolio: 500 rows total
portfolio_data = {
    0: {"Name": "Apple Inc.", "Symbol": ""},        # Missing symbol
    1: {"Name": "", "Symbol": "MSFT"},              # Missing name
    2: {"Name": "Google", "Symbol": ""},            # Missing symbol
    3: {"Name": "", "Symbol": "AMZN"},              # Missing name
    4: {"Name": "Tesla Inc.", "Symbol": "TSLA"},    # Complete (skip)
    # ... 495 more rows
}

# After analysis:
missing_symbols = [0, 2, 8, 12, 15, ...]  # 200 rows missing symbols
missing_names = [1, 3, 9, 14, 18, ...]    # 150 rows missing names

# Sharding with shard_size = 50:
symbol_shards = [
    [0, 2, 8, 12, 15, ...],    # Symbol shard 0: 50 rows
    [67, 72, 89, 91, 95, ...], # Symbol shard 1: 50 rows
    [134, 145, 167, 178, ...], # Symbol shard 2: 50 rows
    [234, 245, 267, 278]       # Symbol shard 3: 50 rows
]

name_shards = [
    [1, 3, 9, 14, 18, ...],    # Name shard 0: 50 rows
    [45, 56, 67, 78, 89, ...], # Name shard 1: 50 rows
    [123, 134, 145, 156]       # Name shard 2: 50 rows
]

# Parallel execution:
# 7 workers total (4 symbol + 3 name) run simultaneously
# Each worker handles ~50 rows independently
# Total time: ~50 seconds instead of 350 seconds sequential
```

#### Shard Independence: Why It Works

Each shard is completely independent:

**No Shared State**: Each shard loads the portfolio data independently and only updates its assigned rows.

**Atomic Updates**: Each shard writes its results using the FileLock system, so updates never conflict.

**Failure Isolation**: If shard 3 crashes, shards 1, 2, 4, and 5 continue working normally.

**Resume Capability**: You can restart just the failed shard without affecting completed work.

#### Monitoring Shard Progress

Each shard reports progress independently:

```python
# Shard 2 progress log:
"📊 Progress: shard symbol_2: processed 25/50 (50.0%); successes: 20; failures: 5"
"📊 Progress: shard symbol_2: processed 50/50 (100.0%); successes: 42; failures: 8"

# System can track overall progress across all shards:
# Total: 7 shards, 4 completed, 3 in progress = 57% done
```

#### Benefits of Our Sharding Approach

**Scalability**: Add more CPU cores = process more shards simultaneously = faster completion

**Reliability**: Individual shard failures don't break the entire job

**Efficiency**: Optimal resource utilization without overwhelming the system

**Flexibility**: Adjust shard size based on dataset size and available resources

**Simplicity**: Clean, understandable algorithm that's easy to debug and maintain

The sharding system transforms a slow, sequential process into a fast, parallel operation while maintaining data integrity and providing excellent error recovery capabilities.
