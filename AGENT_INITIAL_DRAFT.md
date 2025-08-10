# Portfolio Enrichment System

## System Overview

**Goal:** Read a CSV portfolio file, identify missing company names or stock symbols, use Finnhub API to fill in the missing data, and write the complete data back to a new CSV file.

### Data Flow (Amp-aware)

```text
CSV Input → Global Dictionary (portfolio_data)
           ↓
Global Dictionary → API Enrichment (symbols and names)
           ↓
Enriched Global Dictionary → CSV Output
           ↓
All tasks → Monitoring log (monitoring.md + monitoring.log) with periodic progress updates
```

### Global State

Maintain a single dictionary:

```python
portfolio_data: Dict[int, Dict[str, str]]
```

where each key is a row index and each value contains all the row data.

## Shared Artifacts (for task independence & orchestration)

- **portfolio_state.json**: serialized snapshot of portfolio_data so tasks can run independently (read/refresh/write as needed)
- **monitoring.md and monitoring.log**: append-only logs from each task
- **Optional**: simple file lock (e.g., portfolio_state.lock) if you choose to run symbol/name resolvers concurrently in Amp

## Orchestration with Amp (Multi-Agent)

**Concept:** Treat each function as an independent task an Amp controller can trigger. Tasks can run sequentially or (optionally) in parallel batches. All tasks read/write the same state file and report to a shared monitoring log.

### Amp Roles / Tasks

#### Controller (Orchestrator)

- Decides the run sequence
- Spawns tasks with prompts below
- Passes file paths and API key as parameters
- Optionally shards large lists into batches for parallel agents

#### Task A – CSV Reader & Missing Identifier

- Reads the CSV, populates portfolio_data, identifies missing
- Writes portfolio_state.json
- Logs counts

#### Task B – Symbol Resolver

- Loads portfolio_state.json
- Resolves symbols for indices listed as "missing_symbols"
- Periodically logs progress (every N rows and/or every T seconds)
- Writes back updated portfolio_state.json

#### Task C – Name Resolver

- Same as Task B, but for "missing_names"

#### Task D – CSV Writer

- Loads portfolio_state.json and writes portfolio_output.csv
- Logs file location

#### Task E – Monitor (optional)

- Periodically summarizes progress by reading monitoring.log and appending a top-level status section in monitoring.md

### Amp Controller – Example Invocation (pseudocode)

```python
Task(description="Read CSV & identify missing",
     prompt="Run Task A with input_file=portfolio_input.csv, state_file=portfolio_state.json, logs=monitoring.*")

Task(description="Resolve missing symbols",
     prompt="Run Task B with state_file=portfolio_state.json, api_key=$FINNHUB_API_KEY, batch_size=50, progress_every=10, logs=monitoring.*")

Task(description="Resolve missing names",
     prompt="Run Task C with state_file=portfolio_state.json, api_key=$FINNHUB_API_KEY, batch_size=50, progress_every=10, logs=monitoring.*")

Task(description="Write updated CSV",
     prompt="Run Task D with state_file=portfolio_state.json, output_file=portfolio_output.csv, logs=monitoring.*")
```

**Parallelization tip (optional):** The controller may shard missing_symbols and missing_names into batches and spawn multiple instances of Task B/C with distinct shard IDs, each updating non-overlapping row indices. If you do this, enable file locking or write updates shard-local then merge.

## Logging & Monitoring (Periodic)

### Requirements for every task:

- Initialize a simple logger (logging module) that writes to both console and monitoring.log
- Also append a human-readable section to monitoring.md:
  - **Start**: task name, timestamp, key inputs
  - **Progress**: every progress_every rows (default 10) and/or every heartbeat_seconds (default 30), log a line with totals
  - **End**: summary of results, counts, and elapsed time

### Markdown Format (suggested):

```markdown
## [YYYY-MM-DD HH:MM:SS] Task: {TASK_NAME} (id={TASK_ID})

- Input: ...
- Start row count: ...
- Progress: processed X/Y (p%); successes: S; failures: F
- End: updated rows: U; elapsed: 12.3s
```

### Log Event Fields:

- timestamp_iso, task_id, task_name, event ∈ {start|progress|end|warning|error}, message, extra (optional dict)

### Config (per task):

- progress_every: int (default 10)
- heartbeat_seconds: int (default 30)
- rate_limit_sleep: float seconds (default 0.1 for lookups)
- log_files: monitoring.log and monitoring.md

---

# Function Specifications

## Function 1: CSV Reader and Missing Data Identifier

**Function Name:** `read_csv_and_identify_missing`

### AI Agent Prompt:

Create a function that reads a CSV file and identifies rows with missing data. Here are the exact requirements:

### INPUT:

- input_file: string path to CSV file
- OPTIONAL: state_file: path to JSON file to persist 'portfolio_data' for other tasks (default: "portfolio_state.json")
- OPTIONAL: log_files: dict with {"text": "monitoring.log", "md": "monitoring.md"}
- OPTIONAL: task_id: str for logging (autogenerate if not provided)

CSV has columns: Name, Symbol, Price, # of Shares, Market Value

### OUTPUT:

- Return a dictionary with two keys:
  - 'missing_symbols': list of row indices where Symbol is empty but Name exists
  - 'missing_names': list of row indices where Name is empty but Symbol exists
- Side-effects:
  - Globally update 'portfolio_data'
  - Serialize 'portfolio_data' to state_file (JSON)
  - Periodically log to monitoring.\* (start/end entries at minimum)

### LOGIC:

1. Log task START (task_id, input_file).
2. Read CSV using pandas.read_csv()
3. Fill NaN values with empty strings using fillna("")
4. Store ALL row data in global dictionary called 'portfolio_data'
   - Key: row index (int)
   - Value: dictionary with keys ['Name', 'Symbol', 'Price', '# of Shares', 'Market Value']
   - Strip whitespace from Name, convert Symbol to uppercase
5. Loop through portfolio_data to identify missing values:
   - Missing symbol: Symbol is empty AND Name is not empty
   - Missing name: Name is empty AND Symbol is not empty
6. Print and log count of missing symbols and missing names
7. Serialize 'portfolio_data' to state_file (JSON)
8. Log task END (summary) and return the dictionary with the two lists

**ERROR HANDLING:** Basic try/catch, print and log errors, continue.

## Function 2: Symbol Resolver

**Function Name:** `resolve_symbol`

### AI Agent Prompt:

Create a function that gets a stock symbol for a company name using Finnhub search API. Keep it simple and direct.

### INPUT:

- company_name: string (company name to search for)
- api_key: string (Finnhub API key)
- OPTIONAL: session kwargs such as timeout=10

### OUTPUT:

- Return string (stock symbol in uppercase) if found
- Return None if not found or error occurs

### API DETAILS:

- Endpoint: <https://finnhub.io/api/v1/search>
- Method: GET
- Parameters: {'q': company_name, 'token': api_key}
- Response format: {'result': [{'symbol': 'AAPL', 'description': '...', ...}, ...]}

### LOGIC:

1. Return None if company_name is empty
2. Make GET request to Finnhub search endpoint with company name and API key
3. If response status is 429 (rate limited):
   - Print/log "Rate limited, waiting 1 second..."
   - Sleep for 1 second
   - Retry the same request once
4. Raise for status to handle other HTTP errors
5. Parse JSON response
6. If result array exists and has at least 1 item:
   - Get 'symbol' from first result
   - Convert to uppercase
   - Print/log "Found symbol 'SYMBOL' for 'COMPANY_NAME'"
   - Return the symbol
7. If no results:
   - Print/log "No symbol found for 'COMPANY_NAME'"
   - Return None

**ERROR HANDLING:**

- Catch all exceptions, print/log error message with company name, return None
- Use 10 second timeout for requests

## Function 3: Name Resolver

**Function Name:** `resolve_name`

### AI Agent Prompt:

Create a function that gets a company name for a stock symbol using Finnhub profile API. Keep it simple and direct.

### INPUT:

- symbol: string (stock symbol to look up)
- api_key: string (Finnhub API key)
- OPTIONAL: session kwargs such as timeout=10

### OUTPUT:

- Return string (company name) if found
- Return None if not found or error occurs

### API DETAILS:

- Endpoint: <https://finnhub.io/api/v1/stock/profile2>
- Method: GET
- Parameters: {'symbol': symbol.upper(), 'token': api_key}
- Response format: {'name': 'Apple Inc', 'ticker': 'AAPL', ...}

### LOGIC:

1. Return None if symbol is empty
2. Convert symbol to uppercase
3. Make GET request to Finnhub profile2 endpoint with symbol and API key
4. If response status is 429 (rate limited):
   - Print/log "Rate limited, waiting 1 second..."
   - Sleep for 1 second
   - Retry the same request once
5. Raise for status to handle other HTTP errors
6. Parse JSON response
7. Get 'name' field from response and strip whitespace
8. If name exists and is not empty:
   - Print/log "Found name 'COMPANY_NAME' for 'SYMBOL'"
   - Return the company name
9. If no name found:
   - Print/log "No name found for 'SYMBOL'"
   - Return None

**ERROR HANDLING:**

- Catch all exceptions, print/log error message with symbol, return None
- Use 10 second timeout for requests

## Function 4: CSV Writer

**Function Name:** `write_updated_csv`

### AI Agent Prompt:

Create a function that writes the global portfolio data dictionary back to a CSV file.

### INPUT:

- output_file: string path for output CSV file
- OPTIONAL: state_file: path to JSON file containing 'portfolio_data' snapshot (default: "portfolio_state.json")
- OPTIONAL: log_files: dict with {"text": "monitoring.log", "md": "monitoring.md"}
- OPTIONAL: task_id: str for logging (autogenerate if not provided)

### OUTPUT:

- No return value (void function)
- Side effect: Creates/overwrites CSV file with updated data

### LOGIC:

1. Log task START.
2. Load 'portfolio_data' from state_file if global is empty; otherwise use global.
3. Convert dictionary to list of dictionaries:
   - Sort the row indices (keys) in ascending order
   - For each sorted index, append portfolio_data[index] to a list
4. Create pandas DataFrame from the list of dictionaries
5. Write DataFrame to CSV using to_csv() with index=False
6. Print/log "Updated portfolio saved to: OUTPUT_FILE"
7. Log task END.

**ERROR HANDLING:** Basic, print/log any errors that occur

## Function 5: Main Orchestrator

**Function Name:** `process_portfolio`

Create a main orchestrator function that runs the entire portfolio enrichment workflow in sequence. Also add optional Amp orchestration parameters so each step can be run as an independent task.

### INPUT:

- input_file: string path to input CSV
- output_file: string path to output CSV
- api_key: string Finnhub API key
- OPTIONAL: state_file: path to JSON file for shared state (default: "portfolio_state.json")
- OPTIONAL: progress_every: int (default 10)
- OPTIONAL: heartbeat_seconds: int (default 30)
- OPTIONAL: rate_limit_sleep: float (default 0.1)
- OPTIONAL: log_files: dict with {"text": "monitoring.log", "md": "monitoring.md"}
- OPTIONAL: amp_mode: bool (default False). If True, print instructions for Amp tasks and exit after Task A so the controller can schedule B, C, D separately. If False, run all steps inline.

### OUTPUT:

- No return value (void function)
- Side effects: Reads input CSV, makes API calls, writes output CSV, periodic logs

### WORKFLOW:

1. Print/log "=== Portfolio Enrichment Process ==="

2. **STEP 1 - Read and identify:**

   - Print/log "\nStep 1: Reading CSV and identifying missing values..."
   - Call read_csv_and_identify_missing(input_file, state_file=state_file, log_files=log_files)
   - Store returned dictionary in variable called 'missing_data'
   - If amp_mode=True: print/log a short guide detailing which Amp tasks to run next with parameters and then return immediately.

3. **STEP 2 - Resolve missing symbols:**

   - Print/log "\nStep 2: Resolving missing symbols..."
   - Loop through each row_idx in missing_data['missing_symbols']:
     - Get company_name from portfolio_data[row_idx]['Name']
     - Print/log "Row ROW_IDX: Looking up symbol for 'COMPANY_NAME'"
     - Call resolve_symbol(company_name, api_key)
     - If symbol is returned (not None):
       - Update portfolio_data[row_idx]['Symbol'] = symbol
     - Sleep for rate_limit_sleep seconds (simple rate limiting)
     - Periodically (every 'progress_every' rows and/or every 'heartbeat_seconds'), append progress to monitoring.\*
   - Serialize 'portfolio_data' back to state_file after the loop.

4. **STEP 3 - Resolve missing names:**

   - Print/log "\nStep 3: Resolving missing names..."
   - Loop through each row_idx in missing_data['missing_names']:
     - Get symbol from portfolio_data[row_idx]['Symbol']
     - Print/log "Row ROW_IDX: Looking up name for 'SYMBOL'"
     - Call resolve_name(symbol, api_key)
     - If company_name is returned (not None):
       - Update portfolio_data[row_idx]['Name'] = company_name
     - Sleep for rate_limit_sleep seconds
     - Periodically log progress as in Step 2
   - Serialize 'portfolio_data' back to state_file after the loop.

5. **STEP 4 - Write output:**

   - Print/log "\nStep 4: Writing updated CSV..."
   - Call write_updated_csv(output_file, state_file=state_file, log_files=log_files)

6. Print/log "\n=== Process Complete ==="

**ERROR HANDLING:** Let individual functions handle their own errors

---

# Global Variables and Imports

## AI Agent Prompt for Setup:

Create the module setup with these exact specifications:

### IMPORTS NEEDED:

```python
import pandas as pd
import requests
import time
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
```

### GLOBAL VARIABLES:

```python
portfolio_data: Dict[int, Dict[str, str]] = {}
```

(This stores all portfolio data, key=row_index, value=row_data_dict)

### LOGGING SETUP:

- Configure logging to INFO level
- StreamHandler (console) and FileHandler ("monitoring.log")
- Simple formatter with timestamp, level, message
- Provide helper: append_markdown(md_path, text: str)
- Provide helper: log_event(task_name, task_id, event, message, md_path, \*\*extras)

## State Persistence Helpers (for Task Independence)

### AI Agent Prompt:

Add two helpers for state management so each task can run independently:

1. **save_state(state_file: str) -> None**

   - Serialize 'portfolio_data' to JSON at 'state_file'

2. **load_state(state_file: str) -> None**
   - If 'portfolio_data' is empty and 'state_file' exists, load JSON into 'portfolio_data'
   - If both exist, prefer in-memory but log a warning if they differ in length

(Optional: note a simple file lock if parallel writes are enabled.)

---

# Example Usage Section

## AI Agent Prompt:

Create a **main** block and example usage that demonstrates how to use the system:

1. Add a **main** block that shows example file paths and API key setup
2. Show how to get API key from environment variable as fallback
3. Add validation to ensure API key is set before running (unless amp_mode=True and you only run Task A)
4. Call the main process_portfolio function with example parameters
5. Add helpful error messages if API key is not configured
6. Show how to run in 'amp_mode' for multi-agent orchestration:
   - Example: First run with amp_mode=True to produce missing lists and state file
   - Then the Amp controller schedules Task B and Task C (optionally sharded) with the same state_file
   - Finally schedule Task D to write CSV

### Example file names to use:

- Input: "portfolio_input.csv"
- Output: "portfolio_output.csv"
- Environment variable: "FINNHUB_API_KEY"
- State: "portfolio_state.json"
- Logs: "monitoring.log", "monitoring.md"

---

# Amp Task Prompts

## Task A – Read & Identify

Run Task A (read_csv_and_identify_missing):

- input_file=portfolio_input.csv
- state_file=portfolio_state.json
- log_files={"text":"monitoring.log", "md":"monitoring.md"}

**Expectations:**

- writes portfolio_state.json
- logs counts of missing symbols and names
- appends START/END sections in monitoring.md

## Task B – Resolve Symbols

Run Task B (symbol resolution pass):

- state_file=portfolio_state.json
- api_key=$FINNHUB_API_KEY
- progress_every=10
- heartbeat_seconds=30
- rate_limit_sleep=0.1
- log_files={"text":"monitoring.log", "md":"monitoring.md"}

**Expectations:**

- loads state
- iterates missing_symbols
- updates Symbol field
- logs progress periodically and on completion
- writes updated state back to portfolio_state.json

## Task C – Resolve Names

Run Task C (name resolution pass):

- state_file=portfolio_state.json
- api_key=$FINNHUB_API_KEY
- progress_every=10
- heartbeat_seconds=30
- rate_limit_sleep=0.1
- log_files={"text":"monitoring.log", "md":"monitoring.md"}

**Expectations:**

- loads state
- iterates missing_names
- updates Name field
- logs progress periodically and on completion
- writes updated state back to portfolio_state.json

## Task D – Write CSV

Run Task D (write_updated_csv):

- state_file=portfolio_state.json
- output_file=portfolio_output.csv
- log_files={"text":"monitoring.log", "md":"monitoring.md"}

**Expectations:**

- loads state
- writes final CSV
- logs file path and completion

## Task E – Monitor (optional)

Run Task E (monitoring summary):

- read monitoring.log
- append a summarized status block to monitoring.md with counts of INFO/ERROR and last progress lines

---

# Key Design Principles (Updated)

1. **Simplicity First:** Straightforward HTTP calls, minimal retries.

2. **Clear Separation:** Each function does exactly one job, and is runnable as its own task.

3. **Global State + Serialized State:** portfolio_data in-memory and portfolio_state.json on disk to allow independent tasks.

4. **Basic Rate Limiting:** time.sleep() between calls.

5. **Informative Logging:** Console + monitoring.log + human-readable monitoring.md. Periodic progress per N rows / T seconds.

6. **Fail Gracefully:** Return None on lookup failure; continue other rows.

7. **Direct API Calls:** Use requests directly; 10s timeout.

8. **Idempotence Where Possible:** Re-running a task should not corrupt state; always load, update, and write cleanly.

9. **Optional Parallelism:** When sharding tasks in Amp, ensure non-overlapping row sets or use simple file locking.
