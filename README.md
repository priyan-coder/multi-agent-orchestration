# Multi-Agent Portfolio Enrichment System

A multi-agent orchestration system that enriches portfolio data by resolving missing company names and stock symbols.

## 🔧 Setup

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

2. **Environment Configuration**:
   Create a `.env` file with your API keys:

```bash
FINNHUB_API_KEY=your_finnhub_api_key_here
OPEN_AI_API_KEY=your_openai_api_key_here
```

## 🏃‍♂️ Quick Start

### Prerequisites

1. **Install the Amp Extension**: You must have the [Amp extension](https://marketplace.visualstudio.com/items?itemName=amp.amp) installed in VS Code
2. **Activate Virtual Environment**:

```bash
# Create virtual environment if not exists
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

### Execution Steps

1. **Prepare Your Data**: Place your portfolio CSV file (see format below)

2. **Run the Orchestrator**:

```bash
python orchestrator.py
```

3. **Execute with Amp**: After step 2 generates `amp_instructions.md`, create a new Amp thread and paste this prompt:

```
Activate the venv, run all tasks in amp_instructions.md, and review the generated output CSV.
```

4. **Get Results**: Find your enriched portfolio in `portfolio_output.csv`

## 🤖 Amp Multi-Agent Execution

After running the orchestrator, you'll have generated `amp_instructions.md` with parallel task definitions. Here's how to execute them:

### Step 1: Install Amp Extension

- Install the [Amp VS Code Extension](https://ampcode.com/how-i-use-amp)
- Restart VS Code if needed

### Step 2: Create Amp Thread

1. Open VS Code in your project directory
2. Open a new Amp thread (Ctrl/Cmd + Shift + P → "Amp: New Thread")
3. Paste this exact prompt:

```
Activate the venv, run all tasks in amp_instructions.md, and review the generated output CSV.
```

### Step 3: Let Amp Execute

Amp will automatically:

- ✅ Activate the virtual environment
- ✅ Parse the `amp_instructions.md` file
- ✅ Execute all symbol resolution tasks in parallel
- ✅ Execute all name resolution tasks in parallel
- ✅ Generate the final `portfolio_output.csv`
- ✅ Provide a summary of results

### Expected Output

After successful execution, you'll have:

- `portfolio_output.csv` - Your enriched portfolio data
- `portfolio_state.json` - Complete processing state
- `monitoring.md` - Detailed execution logs

## 📁 Input Format

Your CSV should have columns for company information:

```csv
Name,Symbol,Price,# of Shares,Market Value
Apple Inc.,AAPL,210.5,50,10525.0
Microsoft Corporation,,425.2,10,4252.0
,GOOGL,165.1,8,1320.8
Berkshire Hathaway Inc. Class B,,430.0,3,1290.0
,KO,62.1,20,1242.0
```

## 📄 Output Format

The system produces a complete enriched portfolio with all original data preserved:

```csv
Row,Company_Name,Symbol,Holdings,Market_Value
0,Apple Inc.,AAPL,50,10525.0
1,Microsoft Corporation,MSFT,10,4252.0
2,Alphabet Inc. Class A,GOOGL,8,1320.8
3,Berkshire Hathaway Inc. Class B,BRK.B,3,1290.0
4,Coca-Cola Co,KO,20,1242.0
```

## 🔄 Core Workflow

### Phase 1: CSV Analysis & Sharding

```
Input CSV → Analyze Missing Data → Create Shards → Generate Instructions
```

**Task A (`task_a_csv_reader.py`)**:

- Reads portfolio CSV and identifies missing symbols/names
- Creates shards for parallel processing
- Saves initial state to `portfolio_state.json`
- Generates parallel task instructions in `amp_instructions.md`

### Phase 2: Parallel Resolution

```
Symbol Shards (Parallel) + Name Shards (Parallel) → Enhanced Data
```

**Task B (`task_b_symbol_resolver.py`)** - **Two-Step Approach**:

1. **Try FinnHub API** with full company name
2. **If fails → Ask OpenAI directly** for ticker symbol

**Task C (`task_c_name_resolver.py`)**:

- Resolves missing company names using FinnHub API
- Updates state atomically with file locking

### Phase 3: Final Output

```
Enhanced State → Validated CSV Output
```

**Task D (`task_d_csv_writer.py`)**:

- Writes complete enriched portfolio to CSV
- Provides comprehensive validation reporting
- Maintains all original data with enrichments

## 🧠 Symbol Resolution Logic

✅ **Two-step process:**

```python
def resolve_symbol(company_name):
    # Step 1: Try FinnHub API
    result = try_finnhub_api(company_name)
    if result:
        return result

    # Step 2: Ask OpenAI directly
    return ask_openai_for_ticker(company_name)
```

#### OpenAI Integration

```python
prompt = f"""What is the exact stock ticker symbol for "{company_name}"?
Requirements:
- Provide ONLY the ticker symbol (like "PG" or "BRK.B")
- Use the symbol traded on major US exchanges (NYSE, NASDAQ)
- For companies with multiple share classes, match the class mentioned in the name
- If the company doesn't exist or isn't publicly traded, respond with "NONE"
- No explanation, just the symbol"""
```

## 🏗️ System Architecture

### State Management

- **Atomic Updates**: File locking prevents race conditions
- **Incremental Processing**: Each shard updates state independently
- **Persistence**: JSON state file maintains progress across restarts

### Parallel Execution

```
Orchestrator → [Symbol Shard 0, Symbol Shard 1, Name Shard 0] → CSV Writer
                     ↓              ↓              ↓
                 BRK.B, V      JPM, PG      Coca-Cola, Netflix
```

### Error Handling

- API Fallback
- Retry Logic

### Key Parameters

- **Shard Size**: 2 (optimal for demo, 50+ for production)
- **Timeout**: 30s for API calls
- **Retry Attempts**: 2 with exponential backoff
- **OpenAI Model**: GPT-4

## 📁 File Structure

```
├── orchestrator.py           # Main coordination script
├── task_a_csv_reader.py      # CSV analysis & sharding
├── task_b_symbol_resolver.py # Simplified symbol resolution
├── task_c_name_resolver.py   # Company name resolution
├── task_d_csv_writer.py      # Final CSV output
├── core_functions.py         # Shared utilities & OpenAI integration
├── portfolio_state.json     # State persistence
├── amp_instructions.md       # Generated task instructions
└── portfolio_output.csv     # Final enriched portfolio
```
