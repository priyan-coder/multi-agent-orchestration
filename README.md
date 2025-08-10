# Portfolio Enrichment Multi-Agent System

## 📊 Multi-Agent Workflow Logic

### Phase 1: Data Analysis & Intelligent Sharding

**Task A Logic:**

- Analyzes portfolio completeness and identifies enrichment opportunities
- Creates optimally-sized shards based on API rate limits and worker capacity
- Initializes atomic shared state with file locking for concurrent access
- Generates parallel execution plan for maximum efficiency

### Phase 2: Concurrent Resolution Pipeline

**Parallel Resolution Logic:**

- **Symbol Resolution**: Multi-term search with corporate suffix handling and special case mappings
- **Scoring Engine**: String similarity + word overlap + keyword matching with configurable weights
- **API Strategy**: Exponential backoff, rate limiting, and intelligent fallback search terms
- **State Management**: Atomic batch updates with file locking to prevent race conditions

### Phase 3: Data Consolidation & Validation

## 🚀 Quick Start

### Prerequisites & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API access (required)
echo "FINNHUB_API_KEY=your_key_here" > .env

# Optional: Configure advanced settings
echo "MAX_RETRIES=5" >> .env
echo "DEFAULT_SHARD_SIZE=10" >> .env
echo "MIN_SCORE_THRESHOLD=0.3" >> .env
```

### Execution Workflow

```bash
# 1. Generate parallel task instructions
python orchestrator.py

# 2. Execute generated tasks in Amp for parallel processing
# Copy-paste Task commands from orchestrator output

# 3. Monitor progress and view results
tail -f monitoring.log
cat portfolio_output.csv
```
