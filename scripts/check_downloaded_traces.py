#!/usr/bin/env python3
"""
Script to check downloaded traces and export to YAML.

Usage:
    python scripts/check_downloaded_traces.py --dir=$TBENCH_DIR/traces [--model=model-name]
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add the src directory to Python path to import trace utilities
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from trace_utils import scan_traces_directory, export_traces_to_yaml




def main():
    parser = argparse.ArgumentParser(description='Check downloaded traces and export to YAML')
    parser.add_argument('--dir', default=f"{BASE_DIR}/traces", help='Directory containing trace folders')
    parser.add_argument('--model', help='Model name to filter by (optional)')
    parser.add_argument('--output', help='Output YAML file path (optional)')
    
    args = parser.parse_args()
    
    # Find traces, optionally filtered by model
    traces = scan_traces_directory(args.dir, args.model)
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        base_dir = Path(args.dir).parent  # Assuming traces dir is under project root
        if args.model:
            # Default to config/downloaded_traces/model_{model_name}.yaml
            output_file = base_dir / "config" / "downloaded_traces" / f"model_{args.model}.yaml"
        else:
            # Default to config/downloaded_traces/all_traces.yaml
            output_file = base_dir / "config" / "downloaded_traces" / "all_traces.yaml"
    
    # Export to YAML
    export_traces_to_yaml(traces, str(output_file), include_model_name=True)


if __name__ == "__main__":
    main()