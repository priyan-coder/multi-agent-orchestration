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

## 📊 Intelligent Sharding System

### Optimal Shard Creation (`create_shards()`)

```python
Algorithm:
- Input: List of row indices requiring processing
- Shard Size: Configurable (2 for demo, 50-100 for production)
- Output: List of balanced shards for parallel execution
- Load Balancing: Even distribution across shards
```

### Parallel Task Architecture

```python
Symbol Resolution Shards: Process missing symbols
Name Resolution Shards: Process missing names
Independent Execution: No inter-shard dependencies
State Synchronization: File-lock coordinated updates
```

## 🎯 Task System Implementation

### Task A: CSV Analysis & Sharding (`task_a_csv_reader.py`)

```python
Workflow:
1. Load CSV → pandas DataFrame
2. Data normalization (fillna, strip, uppercase symbols)
3. Missing data identification:
   - Missing symbols: has name, no symbol
   - Missing names: has symbol, no name
4. Intelligent shard creation for parallel processing
5. State persistence with full validation
6. Shard plan generation for orchestrator

Output:
- portfolio_state.json with all row data
- Sharding plan with optimal task distribution
```

### Task B: Symbol Resolution (`task_b_symbol_resolver.py`)

```python
Shard Processing:
1. Load state (no global variables)
2. Validate shard row existence
3. Process each row:
   - Extract company name
   - Apply three-tier resolution strategy
   - Batch updates for atomic persistence
4. Progress monitoring with heartbeat logging
5. Error recovery and reporting
6. Final statistics generation

Features:
- Rate-limited API calls (0.5s between requests)
- Real-time progress reporting
- Graceful error handling per row
- Batch state updates for efficiency
```

### Task C: Name Resolution (`task_c_name_resolver.py`)

```python
Shard Processing:
1. Load state independently
2. Validate symbols exist
3. Process each row:
   - Extract stock symbol
   - Call Finnhub profile API
   - Validate response format
   - Batch updates for atomic persistence
4. Progress monitoring and heartbeat
5. Error recovery with detailed logging

Features:
- Symbol normalization (uppercase, strip)
- Response validation and error classification
- Atomic batch updates
- Performance monitoring
```

### Task D: CSV Writer & Validation (`task_d_csv_writer.py`)

```python
Final Assembly:
1. Load complete enriched state
2. Convert to structured DataFrame
3. Comprehensive validation analysis:
   - Symbol success rate calculation
   - Name success rate calculation
   - Overall completeness metrics
   - Gap identification for debugging
4. CSV generation with clean column mapping
5. Detailed validation reporting

Validation Metrics:
- Total rows processed
- Symbol/Name success rates (%)
- Overall completeness percentage
- Specific row gaps identification
```

## 🚀 Orchestration Flow (`orchestrator.py`)

### Phase 1: Analysis & Planning

```python
orchestrator.py → task_a_csv_reader.py
- CSV ingestion and analysis
- Missing data pattern identification
- Optimal shard size calculation
- Parallel execution plan generation
- Amp instruction document creation
```

### Phase 2: Distributed Parallel Execution

```python
Concurrent Shard Processing:
- Multiple Task B instances (symbol resolution)
- Multiple Task C instances (name resolution)
- Independent execution with shared state
- File-lock coordination for updates
- Real-time progress monitoring
```

### Phase 3: Final Assembly & Validation

```python
task_d_csv_writer.py
- State consolidation and validation
- Final CSV generation
- Comprehensive reporting
- Success metrics calculation
```

## 📈 Performance & Scalability

### Parallel Processing Characteristics

- **Concurrent Shards**: N shards execute simultaneously
- **Optimal Shard Size**: 2 (demo) → 50-100 (production)
- **Memory Efficiency**: Row-level processing, no dataset duplication
- **State Coordination**: File-lock based synchronization

### API Efficiency Optimizations

- **Cache Hit Rate**: ~70% reduction via local company universe
- **Smart Fallbacks**: Multi-tier resolution prevents API exhaustion
- **Rate Compliance**: 90% buffer with automatic throttling
- **Semantic Enhancement**: Tiktoken-based scoring improves accuracy

### Error Resilience & Recovery

- **Atomic State**: No partial corruption possible
- **Graceful Degradation**: Multiple fallback strategies
- **Progress Persistence**: Resume from any interruption point
- **Shard Independence**: Individual shard failures don't affect others

## 🔧 Advanced Features

### Tiktoken Semantic Matching

```python
SemanticMatcher Class:
- Tokenizer: cl100k_base encoding (GPT-4 compatible)
- Similarity: Jaccard coefficient on token sets
- Normalization: Corporate suffix removal, special character handling
- Fallback: Graceful degradation when tiktoken unavailable
- Bonus Scoring: Word overlap, exact matches, acronym detection
```

### Company Universe Caching

```python
Cache Management:
- Source: Finnhub /stock/symbol (US exchanges)
- Refresh: 7-day TTL with automatic updates
- Filtering: Common Stock only, symbol length ≤ 6
- Storage: JSON with timestamp validation
- Size: ~8000-10000 US companies
```

### Monitoring & Observability

```python
Dual Logging System:
- Text Logs: Structured logging to monitoring.log
- Markdown Reports: Real-time monitoring.md updates
- Progress Tracking: Per-shard heartbeat monitoring
- Error Classification: Detailed error categorization
- Performance Metrics: Timing and success rate tracking
```

## 🎯 Production Architecture Principles

1. **Zero Global State**: All data passed explicitly between functions
2. **Atomic Operations**: File-lock coordinated state management
3. **Horizontal Scalability**: Independent shard processing
4. **API Resilience**: Multi-tier fallback with intelligent retry
5. **Semantic Intelligence**: Tiktoken-enhanced matching accuracy
6. **Error Recovery**: Comprehensive error handling and reporting
7. **Monitoring**: Real-time observability and progress tracking
8. **Configuration**: Environment-based API key management
