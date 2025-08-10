#!/usr/bin/env python3
"""
Improved Task A - CSV Reader and Missing Data Identifier

Uses improved core functions without global variables.
"""

import pandas as pd
import uuid
from core_functions import (
    setup_logging, save_state, log_event, create_shards
)

def read_csv_and_identify_missing(
    input_file: str,
    state_file: str = "portfolio_state.json",
    log_files: dict = None,
    task_id: str = None,
    shard_size: int = 10
) -> dict:
    """
    Task A: Read CSV file and identify rows with missing data.
    
    Args:
        input_file: Path to input CSV file
        state_file: Path to JSON state file
        log_files: Logging configuration
        task_id: Unique task identifier
        shard_size: Size of shards for parallel processing
        
    Returns:
        Dictionary with missing data information and sharding details
    """
    if log_files is None:
        log_files = {"text": "monitoring.log", "md": "monitoring.md"}
    
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]
    
    md_path = log_files["md"]
    
    try:
        # Log task start
        log_event("CSV_READER", task_id, "start", f"input_file={input_file}", md_path)
        
        # Read CSV
        df = pd.read_csv(input_file)
        df = df.fillna("")
        
        # Store all row data in portfolio dictionary (local, not global)
        portfolio_data = {}
        for idx, row in df.iterrows():
            portfolio_data[idx] = {
                'Name': str(row.get('Name', '')).strip(),
                'Symbol': str(row.get('Symbol', '')).upper().strip(),
                'Price': str(row.get('Price', '')),
                '# of Shares': str(row.get('# of Shares', '')),
                'Market Value': str(row.get('Market Value', ''))
            }
        
        # Identify missing values
        missing_symbols = []
        missing_names = []
        
        for row_idx, row_data in portfolio_data.items():
            name = row_data['Name']
            symbol = row_data['Symbol']
            
            if not symbol and name:  # Missing symbol but has name
                missing_symbols.append(row_idx)
            elif not name and symbol:  # Missing name but has symbol
                missing_names.append(row_idx)
        
        # Create shards for parallel processing
        symbol_shards = create_shards(missing_symbols, shard_size) if missing_symbols else []
        name_shards = create_shards(missing_names, shard_size) if missing_names else []
        
        # Log counts
        message = f"Total rows: {len(portfolio_data)}, Missing symbols: {len(missing_symbols)}, Missing names: {len(missing_names)}"
        print(message)
        
        if symbol_shards:
            print(f"Symbol resolution: {len(symbol_shards)} shards of size ≤{shard_size}")
        if name_shards:
            print(f"Name resolution: {len(name_shards)} shards of size ≤{shard_size}")
        
        # Save state
        success = save_state(portfolio_data, state_file)
        if not success:
            raise ValueError(f"Failed to save state to {state_file}")
        
        # Prepare result
        result = {
            'missing_symbols': missing_symbols,
            'missing_names': missing_names,
            'symbol_shards': symbol_shards,
            'name_shards': name_shards,
            'total_rows': len(portfolio_data)
        }
        
        # Log task end
        log_event("CSV_READER", task_id, "end", 
                 f"processed {len(portfolio_data)} rows, created {len(symbol_shards)} symbol shards, {len(name_shards)} name shards", 
                 md_path)
        
        return result
        
    except Exception as e:
        error_msg = f"Error reading CSV: {e}"
        print(error_msg)
        log_event("CSV_READER", task_id, "error", error_msg, md_path)
        return {'missing_symbols': [], 'missing_names': [], 'symbol_shards': [], 'name_shards': [], 'total_rows': 0}

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Run Task A
    result = read_csv_and_identify_missing(
        input_file="Sample_Portfolio_Holdings.csv",
        shard_size=2  # Small shards for demo
    )
    
    print(f"\n=== Task A Results ===")
    print(f"Total rows: {result['total_rows']}")
    print(f"Missing symbols: {len(result['missing_symbols'])}")
    print(f"Missing names: {len(result['missing_names'])}")
    print(f"Symbol shards: {result['symbol_shards']}")
    print(f"Name shards: {result['name_shards']}")
