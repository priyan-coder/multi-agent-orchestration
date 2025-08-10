#!/usr/bin/env python3
"""
Task D: CSV Writer with Validated Output

Writes enriched portfolio data to CSV with comprehensive validation reporting.
"""

import pandas as pd
import uuid
from typing import Dict
from core_functions import (
    setup_logging, load_state, log_event
)

def write_updated_csv(
    output_file: str = "portfolio_output.csv",
    state_file: str = "portfolio_state.json",
    md_path: str = "monitoring.md"
) -> Dict[str, any]:
    """
    Write enriched portfolio data to CSV with comprehensive validation.
    
    Args:
        output_file: Path for output CSV file
        state_file: Path to portfolio state JSON file
        md_path: Path to markdown monitoring file
        
    Returns:
        Dictionary with results summary
    """
    # Setup
    setup_logging()
    task_id = str(uuid.uuid4())[:8]
    
    log_event("CSV_WRITER", task_id, "start", 
              f"Writing to {output_file} from {state_file}", md_path)
    
    try:
        # Load final state
        portfolio_data = load_state(state_file)
        if not portfolio_data:
            error_msg = f"Failed to load state from {state_file}"
            log_event("CSV_WRITER", task_id, "error", error_msg, md_path)
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        rows = []
        for row_idx, row_data in sorted(portfolio_data.items()):
            rows.append({
            'Row': row_idx,
            'Company_Name': row_data.get('Name', ''),
            'Symbol': row_data.get('Symbol', ''),
            'Holdings': row_data.get('# of Shares', ''),
            'Market_Value': row_data.get('Market Value', '')
            })
        
        df = pd.DataFrame(rows)
        
        # Write CSV
        df.to_csv(output_file, index=False)
        
        # Perform comprehensive validation
        total_rows = len(df)
        missing_symbols = df[df['Symbol'].isna() | (df['Symbol'] == '')].shape[0]
        missing_names = df[df['Company_Name'].isna() | (df['Company_Name'] == '')].shape[0]
        
        # Calculate success rates
        symbol_success_rate = ((total_rows - missing_symbols) / total_rows * 100) if total_rows > 0 else 0
        name_success_rate = ((total_rows - missing_names) / total_rows * 100) if total_rows > 0 else 0
        overall_completeness = ((total_rows * 2 - missing_symbols - missing_names) / (total_rows * 2) * 100) if total_rows > 0 else 0
        
        # Generate detailed validation report
        validation_report = {
            'total_rows': total_rows,
            'missing_symbols': missing_symbols,
            'missing_names': missing_names,
            'symbol_success_rate': round(symbol_success_rate, 1),
            'name_success_rate': round(name_success_rate, 1),
            'overall_completeness': round(overall_completeness, 1),
            'output_file': output_file
        }
        
        # Log detailed results
        log_event("CSV_WRITER", task_id, "progress", 
                  f"Validation complete: {total_rows} rows processed", md_path)
        log_event("CSV_WRITER", task_id, "progress", 
                  f"Symbol success: {symbol_success_rate:.1f}% ({total_rows - missing_symbols}/{total_rows})", md_path)
        log_event("CSV_WRITER", task_id, "progress", 
                  f"Name success: {name_success_rate:.1f}% ({total_rows - missing_names}/{total_rows})", md_path)
        log_event("CSV_WRITER", task_id, "progress", 
                  f"Overall completeness: {overall_completeness:.1f}%", md_path)
        
        # Identify specific gaps for debugging
        if missing_symbols > 0:
            symbol_gaps = df[df['Symbol'].isna() | (df['Symbol'] == '')]['Row'].tolist()
            log_event("CSV_WRITER", task_id, "warning", 
                      f"Missing symbols in rows: {symbol_gaps}", md_path)
        
        if missing_names > 0:
            name_gaps = df[df['Company_Name'].isna() | (df['Company_Name'] == '')]['Row'].tolist()
            log_event("CSV_WRITER", task_id, "warning", 
                      f"Missing names in rows: {name_gaps}", md_path)
        
        # Print summary to console
        print(f"\n✅ CSV Writing Complete!")
        print(f"   📄 Output file: {output_file}")
        print(f"   📊 Total rows: {total_rows}")
        print(f"   🎯 Symbol success: {symbol_success_rate:.1f}% ({total_rows - missing_symbols}/{total_rows})")
        print(f"   🏢 Name success: {name_success_rate:.1f}% ({total_rows - missing_names}/{total_rows})")
        print(f"   📈 Overall completeness: {overall_completeness:.1f}%")
        
        if missing_symbols > 0 or missing_names > 0:
            print(f"   ⚠️ Remaining gaps:")
            if missing_symbols > 0:
                print(f"     - Missing symbols: {missing_symbols} rows")
            if missing_names > 0:
                print(f"     - Missing names: {missing_names} rows")
        else:
            print(f"   🎉 Perfect enrichment - no missing data!")
        
        log_event("CSV_WRITER", task_id, "end", 
                  f"Successfully wrote {total_rows} rows with {overall_completeness:.1f}% completeness", md_path)
        
        return validation_report
        
    except Exception as e:
        error_msg = f"Failed to write CSV: {e}"
        print(f"💥 {error_msg}")
        log_event("CSV_WRITER", task_id, "error", error_msg, md_path)
        raise

if __name__ == "__main__":
    # Run CSV writer
    try:
        result = write_updated_csv()
        print(f"\n📋 Final Report:")
        print(f"   File: {result['output_file']}")
        print(f"   Rows: {result['total_rows']}")
        print(f"   Completeness: {result['overall_completeness']}%")
        
    except Exception as e:
        print(f"💥 Task D failed: {e}")
        exit(1)
