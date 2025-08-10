#!/usr/bin/env python3
"""
Improved Task C - Name Resolver (Sharded)

Uses improved core functions without global variables.
"""

import time
import uuid
import sys
import json
from core_functions import (
    setup_logging, load_state, update_state_batch, resolve_name, 
    log_event, get_api_key
)

def run_name_resolver_shard(
    shard_rows: list,
    shard_id: str,
    state_file: str = "portfolio_state.json",
    api_key: str = None,
    progress_every: int = 1,
    heartbeat_seconds: int = 30,
    rate_limit_sleep: float = 0.5,
    log_files: dict = None,
    task_id: str = None
):
    """
    Task C: Resolve missing company names for a specific shard of rows.
    
    Args:
        shard_rows: List of row indices to process
        shard_id: Unique identifier for this shard
        state_file: Path to JSON state file
        api_key: Finnhub API key
        progress_every: Log progress every N rows
        heartbeat_seconds: Log progress every N seconds
        rate_limit_sleep: Sleep between API calls
        log_files: Logging configuration
        task_id: Unique task identifier
    """
    if log_files is None:
        log_files = {"text": "monitoring.log", "md": "monitoring.md"}
    
    if task_id is None:
        task_id = f"{shard_id}_{str(uuid.uuid4())[:8]}"
    
    if api_key is None:
        api_key = get_api_key()
    
    md_path = log_files["md"]
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Log task start
        log_event("NAME_RESOLVER_SHARD", task_id, "start", 
                 f"shard_id={shard_id}, rows={shard_rows}, count={len(shard_rows)}", md_path)
        
        # Load state directly (no global variables)
        portfolio_data = load_state(state_file)
        if portfolio_data is None:
            raise ValueError(f"Failed to load portfolio data from {state_file}")
        
        print(f"✅ Loaded {len(portfolio_data)} rows from state file")
        
        if not shard_rows:
            log_event("NAME_RESOLVER_SHARD", task_id, "end", "No rows to process", md_path)
            return {"shard_id": shard_id, "processed": 0, "successes": 0, "failures": 0, "elapsed": 0}
        
        print(f"🎯 Shard {shard_id}: Processing {len(shard_rows)} rows for name resolution")
        
        # Validate that all rows exist
        missing_rows = [row for row in shard_rows if row not in portfolio_data]
        if missing_rows:
            raise ValueError(f"Rows {missing_rows} not found in portfolio data")
        
        # Show what we're about to process
        print(f"📋 Rows to process:")
        for row_idx in shard_rows:
            symbol = portfolio_data[row_idx]['Symbol']
            current_name = portfolio_data[row_idx]['Name']
            print(f"   Row {row_idx}: Symbol '{symbol}' (current name: '{current_name}')")
        
        # Resolve names for this shard
        start_time = time.time()
        last_heartbeat = start_time
        successes = 0
        failures = 0
        updates = {}
        
        for i, row_idx in enumerate(shard_rows):
            symbol = portfolio_data[row_idx]['Symbol']
            current_name = portfolio_data[row_idx]['Name']
            
            # Skip if already has name
            if current_name.strip():
                print(f"⏭️ Row {row_idx}: Already has name '{current_name}', skipping")
                continue
            
            print(f"\n🏢 Row {row_idx}: Looking up name for symbol '{symbol}'")
            
            company_name = resolve_name(symbol, api_key)
            if company_name:
                # Store update for batch processing
                updates[row_idx] = portfolio_data[row_idx].copy()
                updates[row_idx]['Name'] = company_name
                successes += 1
                print(f"✅ Success: '{symbol}' → '{company_name}'")
            else:
                failures += 1
                print(f"❌ Failed: '{symbol}' → No name found")
            
            time.sleep(rate_limit_sleep)
            
            # Progress logging
            current_time = time.time()
            if (i + 1) % progress_every == 0 or (current_time - last_heartbeat) >= heartbeat_seconds:
                percent = ((i + 1) / len(shard_rows)) * 100
                progress_msg = f"shard {shard_id}: processed {i + 1}/{len(shard_rows)} ({percent:.1f}%); successes: {successes}; failures: {failures}"
                print(f"📊 Progress: {progress_msg}")
                log_event("NAME_RESOLVER_SHARD", task_id, "progress", progress_msg, md_path)
                last_heartbeat = current_time
        
        # Batch update state file
        if updates:
            print(f"\n💾 Saving {len(updates)} updates to state file...")
            success = update_state_batch(portfolio_data, state_file, updates)
            if not success:
                print("⚠️ Warning: Failed to save some updates")
        else:
            print(f"\n📝 No updates to save")
        
        # Log task end
        elapsed = time.time() - start_time
        end_msg = f"shard {shard_id}: updated {successes} names; failures: {failures}; elapsed: {elapsed:.1f}s"
        log_event("NAME_RESOLVER_SHARD", task_id, "end", end_msg, md_path)
        
        print(f"\n🎉 Name resolution shard {shard_id} complete!")
        print(f"   ✅ Successes: {successes}")
        print(f"   ❌ Failures: {failures}")
        print(f"   ⏱️ Time: {elapsed:.1f}s")
        
        return {
            'shard_id': shard_id,
            'processed': len(shard_rows),
            'successes': successes,
            'failures': failures,
            'elapsed': elapsed
        }
        
    except Exception as e:
        error_msg = f"Error in name resolver shard {shard_id}: {e}"
        print(f"💥 {error_msg}")
        logger.error(error_msg)
        log_event("NAME_RESOLVER_SHARD", task_id, "error", error_msg, md_path)
        raise

if __name__ == "__main__":
    # Example usage: python task_c_name_resolver.py shard_0 "[10, 11]"
    if len(sys.argv) >= 3:
        shard_id = sys.argv[1]
        try:
            shard_rows = json.loads(sys.argv[2])  # Safe JSON parsing
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format for shard_rows: {e}")
            sys.exit(1)
        
        print(f"🚀 Starting name resolver for shard {shard_id} with rows {shard_rows}")
        
        result = run_name_resolver_shard(
            shard_rows=shard_rows,
            shard_id=shard_id,
            progress_every=1  # Log every row for demo
        )
        
        print(f"\n📊 Final Results for Shard {shard_id}:")
        print(f"   Processed: {result['processed']}")
        print(f"   Successes: {result['successes']}")
        print(f"   Failures: {result['failures']}")
        print(f"   Elapsed: {result['elapsed']:.1f}s")
        
    else:
        print("Usage: python task_c_name_resolver.py <shard_id> <shard_rows>")
        print("Example: python task_c_name_resolver.py shard_0 '[10, 11]'")
