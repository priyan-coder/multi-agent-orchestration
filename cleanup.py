#!/usr/bin/env python3
"""
Cleanup Script for Portfolio Enrichment System

Removes unnecessary files and resets the system for fresh runs.
"""

import os


def cleanup_project():
    """
    Remove all generated and temporary files to reset system state.

    Cleans up logs, cache files, state files, and output files while
    preserving essential source code and configuration files.
    """

    print("🧹 Starting Portfolio Enrichment System Cleanup...")

    # Essential files to keep
    essential_files = {
        "core_functions.py",
        "task_a_csv_reader.py",
        "task_b_symbol_resolver.py",
        "task_c_name_resolver.py",
        "task_d_csv_writer.py",
        "orchestrator.py",
        "Sample_Portfolio_Holdings.csv",
        "requirements.txt",
        ".env",
        ".gitignore",
        "README.md",
        "cleanup.py",
        "AGENT_INITIAL_DRAFT.md",
    }

    # Get all files in current directory
    all_files = []
    for item in os.listdir("."):
        if os.path.isfile(item):
            all_files.append(item)

    removed_count = 0

    print(f"\n🔍 Scanning directory for non-essential files...")

    for file_path in all_files:
        if file_path not in essential_files:
            try:
                os.remove(file_path)
                print(f"   ✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {file_path}: {e}")
        else:
            print(f"   🔒 Keeping essential file: {file_path}")

    # Also remove Python cache directories
    cache_dirs = ["__pycache__"]
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            try:
                import shutil

                shutil.rmtree(cache_dir)
                print(f"   ✅ Removed directory: {cache_dir}")
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {cache_dir}: {e}")

    print(f"\n📊 Cleanup Summary:")
    print(f"   🗑️ Files removed: {removed_count}")

    # Show remaining essential files
    print(f"\n📁 Essential Files Remaining:")
    for file in sorted(essential_files):
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")

    print(f"\n🎯 System ready for fresh run!")
    print(f"   Next: python orchestrator.py")


if __name__ == "__main__":
    cleanup_project()
