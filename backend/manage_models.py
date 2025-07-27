#!/usr/bin/env python3
"""
Model Version Management Utility

This script provides easy commands to manage different versions of trained models.
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ml.model_trainer import ModelTrainer

def format_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def get_directory_size(path):
    """Calculate total size of directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

async def list_versions():
    """List all available model versions"""
    trainer = ModelTrainer(feature_count=50, create_model_dir=False)
    versions = trainer.list_model_versions()
    
    if not versions:
        print("No model versions found.")
        return
    
    print(f"Found {len(versions)} model versions:\n")
    print(f"{'Version':<17} {'Created':<20} {'Models':<8} {'Size':<8} {'Sharpe Ratio':<12}")
    print("-" * 70)
    
    for version in versions:
        created = datetime.fromisoformat(version['created']).strftime('%Y-%m-%d %H:%M:%S')
        model_count = len(version.get('available_models', []))
        
        # Calculate directory size
        version_path = Path('models') / version['version']
        size = format_size(get_directory_size(version_path)) if version_path.exists() else "N/A"
        
        # Get best Sharpe ratio from performance metrics
        sharpe_ratio = "N/A"
        if 'performance_metrics' in version:
            sharpe_ratios = [float(perf.get('sharpe_ratio', 0)) 
                           for perf in version['performance_metrics'].values() 
                           if 'sharpe_ratio' in perf]
            if sharpe_ratios:
                sharpe_ratio = f"{max(sharpe_ratios):.3f}"
        
        print(f"{version['version']:<17} {created:<20} {model_count:<8} {size:<8} {sharpe_ratio:<12}")
    
    # Show current version
    latest_link = Path('models/latest')
    if latest_link.exists():
        current_version = latest_link.readlink().name
        print(f"\nCurrent version (latest): {current_version}")

async def show_version_details(version):
    """Show detailed information about a specific version"""
    trainer = ModelTrainer(feature_count=50, create_model_dir=False)
    versions = trainer.list_model_versions()
    
    version_info = next((v for v in versions if v['version'] == version), None)
    if not version_info:
        print(f"Version {version} not found.")
        return
    
    print(f"Model Version: {version}")
    print(f"Created: {datetime.fromisoformat(version_info['created']).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Path: {version_info['path']}")
    
    if 'available_models' in version_info:
        print(f"\nAvailable Models: {', '.join(version_info['available_models'])}")
    
    if 'performance_metrics' in version_info:
        print("\nPerformance Metrics:")
        for model_name, metrics in version_info['performance_metrics'].items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"    Win Rate: {metrics.get('win_rate', 'N/A'):.4f}")
            print(f"    Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")
    
    if 'ensemble_weights' in version_info:
        print("\nEnsemble Weights:")
        for model_name, weight in version_info['ensemble_weights'].items():
            print(f"  {model_name}: {weight:.3f}")

async def load_version(version):
    """Load a specific model version"""
    trainer = ModelTrainer(feature_count=50, create_model_dir=False)
    
    print(f"Loading model version: {version}")
    await trainer.load_models(version)
    
    loaded_models = list(trainer.models.keys())
    if loaded_models:
        print(f"Successfully loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
        print(f"Current version: {trainer.get_current_version()}")
    else:
        print("No models were loaded.")

async def delete_version(version):
    """Delete a specific model version"""
    if version == "latest":
        print("Cannot delete 'latest' - it's a symlink to the current version.")
        return
    
    # Confirm deletion
    response = input(f"Are you sure you want to delete version {version}? (y/N): ")
    if response.lower() != 'y':
        print("Deletion cancelled.")
        return
    
    trainer = ModelTrainer(feature_count=50, create_model_dir=False)
    success = await trainer.delete_model_version(version)
    
    if success:
        print(f"Successfully deleted version {version}")
    else:
        print(f"Failed to delete version {version}")

async def cleanup_old_versions(keep_count=5):
    """Keep only the most recent N versions"""
    trainer = ModelTrainer(feature_count=50, create_model_dir=False)
    versions = trainer.list_model_versions()
    
    if len(versions) <= keep_count:
        print(f"Only {len(versions)} versions found, nothing to clean up.")
        return
    
    versions_to_delete = versions[keep_count:]
    print(f"Will delete {len(versions_to_delete)} old versions (keeping {keep_count} most recent):")
    
    for version in versions_to_delete:
        print(f"  - {version['version']} (created: {version['created']})")
    
    response = input("\nProceed with cleanup? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    deleted_count = 0
    for version in versions_to_delete:
        success = await trainer.delete_model_version(version['version'])
        if success:
            deleted_count += 1
    
    print(f"Successfully deleted {deleted_count} old versions.")

def main():
    parser = argparse.ArgumentParser(description='Model Version Management Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List versions command
    subparsers.add_parser('list', help='List all model versions')
    
    # Show version details command
    show_parser = subparsers.add_parser('show', help='Show details of a specific version')
    show_parser.add_argument('version', help='Version to show details for')
    
    # Load version command
    load_parser = subparsers.add_parser('load', help='Load a specific model version')
    load_parser.add_argument('version', help='Version to load (use "latest" for most recent)')
    
    # Delete version command
    delete_parser = subparsers.add_parser('delete', help='Delete a specific model version')
    delete_parser.add_argument('version', help='Version to delete')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Delete old versions, keeping only the most recent N')
    cleanup_parser.add_argument('--keep', type=int, default=5, help='Number of recent versions to keep (default: 5)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == 'list':
        asyncio.run(list_versions())
    elif args.command == 'show':
        asyncio.run(show_version_details(args.version))
    elif args.command == 'load':
        asyncio.run(load_version(args.version))
    elif args.command == 'delete':
        asyncio.run(delete_version(args.version))
    elif args.command == 'cleanup':
        asyncio.run(cleanup_old_versions(args.keep))

if __name__ == '__main__':
    main()