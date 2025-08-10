#!/usr/bin/env python3
"""
Orchestrator for Portfolio Enrichment System

Demonstrates the complete parallel multi-agent workflow.
"""

import os
from task_a_csv_reader import read_csv_and_identify_missing
from core_functions import setup_logging

def run_orchestration():
    """
    Run the complete portfolio enrichment workflow.
    """
    print("="*80)
    print("🚀 PORTFOLIO ENRICHMENT SYSTEM")
    print("="*80)
    
    # Setup logging
    logger = setup_logging()
    
    print("\n📊 PHASE 1: CSV Reading & Intelligent Sharding")
    print("-" * 50)
    
    # Run Task A with optimal shard sizing
    result = read_csv_and_identify_missing(
        input_file="Sample_Portfolio_Holdings.csv",
        shard_size=2  # Small for demo, use 50+ for production
    )
    
    print(f"✅ Portfolio Analysis Complete:")
    print(f"   📋 Total rows: {result['total_rows']}")
    print(f"   🔍 Missing symbols: {len(result['missing_symbols'])}")
    print(f"   🏢 Missing names: {len(result['missing_names'])}")
    print(f"   🧩 Symbol shards: {len(result['symbol_shards'])}")
    print(f"   🧩 Name shards: {len(result['name_shards'])}")
    
    if result['symbol_shards']:
        print(f"\n🔍 Symbol Resolution Shards:")
        for i, shard in enumerate(result['symbol_shards']):
            print(f"   Shard {i}: rows {shard}")
    
    if result['name_shards']:
        print(f"\n🏢 Name Resolution Shards:")
        for i, shard in enumerate(result['name_shards']):
            print(f"   Shard {i}: rows {shard}")
    
    print(f"\n📊 PHASE 2: Amp Parallel Task Instructions")
    print("-" * 50)
    
    # Generate Amp instructions
    instructions = generate_amp_instructions(result)
    
    # Save to file
    with open("amp_instructions.md", "w") as f:
        f.write("# Amp Multi-Agent Instructions\n\n")
        f.write(instructions)
    
    print("✅ Instructions generated: amp_instructions.md")
    
    print(f"\n🎯 PHASE 3: System Status & Next Steps")
    print("-" * 50)
    
    print(f"🔄 Ready for Parallel Execution:")
    total_parallel_tasks = len(result['symbol_shards']) + len(result['name_shards'])
    print(f"   • {len(result['symbol_shards'])} symbol resolution shards")
    print(f"   • {len(result['name_shards'])} name resolution shards")
    print(f"   • 1 final CSV writer")
    print(f"   • Total parallel tasks: {total_parallel_tasks}")
        
    print(f"\n🔧 Ready for Production with:")
    print(f"   • Parallel execution: {total_parallel_tasks} concurrent agents")
    print(f"   • State persistence: Atomic updates with file locking")
    print(f"   • Real-time monitoring: Per-shard progress tracking")
    print(f"   • Clean architecture: No global variable dependencies")
    
    print("\n" + "="*80)
    print("🎉 SYSTEM READY - Use Amp Task tool with generated instructions!")
    print("="*80)

def generate_amp_instructions(result):
    """Generate clean Amp task instructions for the system."""
    
    instructions = []
    
    instructions.append("## 🚀 Parallel Multi-Agent Execution\n")
    instructions.append("Execute all of these tasks and verify that the output CSV file is generated.\n")
    
    # Symbol Resolution Phase
    if result['symbol_shards']:
        instructions.append("### Phase 1: Symbol Resolution (Parallel)")
        instructions.append("**All symbol shards run SIMULTANEOUSLY:**\n")
        
        for i, shard in enumerate(result['symbol_shards']):
            instructions.append(f"#### Symbol Shard {i}")
            instructions.append("```python")
            instructions.append("Task(")
            instructions.append(f'    description="Symbol Resolution - Shard {i}",')
            instructions.append("    prompt='''")
            instructions.append(f"🔍 Execute symbol resolution for shard {i}:")
            instructions.append(f"")
            instructions.append(f"Features:")
            instructions.append(f"- ✅ No global variable dependencies")
            instructions.append(f"- ✅ Advanced API fallback strategies") 
            instructions.append(f"- ✅ Comprehensive error handling")
            instructions.append(f"- ✅ Real-time progress tracking")
            instructions.append(f"")
            instructions.append(f"Parameters:")
            instructions.append(f"- shard_rows={shard}")
            instructions.append(f"- shard_id=symbol_shard_{i}")
            instructions.append(f"- state_file=portfolio_state.json")
            instructions.append(f"- api_key=${{FINNHUB_API_KEY}} (from .env)")
            instructions.append(f"")
            instructions.append(f"Command:")
            instructions.append(f"python task_b_symbol_resolver.py symbol_shard_{i} '{shard}'")
            instructions.append("'''")
            instructions.append(")")
            instructions.append("```\n")
    
    # Name Resolution Phase
    if result['name_shards']:
        instructions.append("### Phase 2: Name Resolution (Parallel)")
        instructions.append("**All name shards run SIMULTANEOUSLY:**\n")
        
        for i, shard in enumerate(result['name_shards']):
            instructions.append(f"#### Name Shard {i}")
            instructions.append("```python")
            instructions.append("Task(")
            instructions.append(f'    description="Name Resolution - Shard {i}",')
            instructions.append("    prompt='''")
            instructions.append(f"🏢 Execute name resolution for shard {i}:")
            instructions.append(f"")
            instructions.append(f"Features:")
            instructions.append(f"- ✅ Direct state loading")
            instructions.append(f"- ✅ Atomic batch updates")
            instructions.append(f"- ✅ Enhanced error recovery")
            instructions.append(f"")
            instructions.append(f"Parameters:")
            instructions.append(f"- shard_rows={shard}")
            instructions.append(f"- shard_id=name_shard_{i}")
            instructions.append(f"- state_file=portfolio_state.json")
            instructions.append(f"- api_key=${{FINNHUB_API_KEY}} (from .env)")
            instructions.append(f"")
            instructions.append(f"Command:")
            instructions.append(f"python task_c_name_resolver.py name_shard_{i} '{shard}'")
            instructions.append("'''")
            instructions.append(")")
            instructions.append("```\n")
    
    # Final CSV Writing
    instructions.append("### Phase 3: Final Output & Validation")
    instructions.append("**Execute AFTER all parallel tasks complete:**\n")
    instructions.append("#### CSV Writer")
    instructions.append("```python")
    instructions.append("Task(")
    instructions.append('    description="Generate validated CSV output",')
    instructions.append("    prompt='''")
    instructions.append("📄 Write final enriched portfolio with validation:")
    instructions.append("")
    instructions.append("Features:")
    instructions.append("- ✅ Comprehensive validation reporting")
    instructions.append("- ✅ Gap analysis and accuracy metrics")
    instructions.append("- ✅ Row-level error identification")
    instructions.append("")
    instructions.append("Parameters:")
    instructions.append("- output_file=portfolio_output.csv")
    instructions.append("- state_file=portfolio_state.json")
    instructions.append("")
    instructions.append("Command:")
    instructions.append("python task_d_csv_writer.py")
    instructions.append("'''")
    instructions.append(")")
    instructions.append("```\n")
    
    return "\n".join(instructions)

if __name__ == "__main__":
    run_orchestration()
