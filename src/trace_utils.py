#!/usr/bin/env python3
"""
Shared utilities for working with downloaded trace directories.

This module provides common functions for validating trace structure,
extracting trace information, and exporting data to YAML format.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def validate_trace_structure(trace_dir: Path) -> bool:
    """Validate that a trace directory has the correct structure for a complete download."""
    # Check for config.json
    config_file = trace_dir / "config.json"
    if not config_file.exists():
        return False
    
    # Check for agent directory
    agent_dir = trace_dir / "agent"
    if not agent_dir.exists() or not agent_dir.is_dir():
        return False
    
    # Check for at least one episode directory with prompt.txt
    episode_dirs = list(agent_dir.glob("episode-*"))
    if not episode_dirs:
        return False
    
    # Check that at least episode-0 exists and has prompt.txt
    episode_0_dir = agent_dir / "episode-0"
    if not episode_0_dir.exists() or not episode_0_dir.is_dir():
        return False
    
    prompt_file = episode_0_dir / "prompt.txt"
    if not prompt_file.exists():
        return False
    
    return True


def get_trace_info(trace_dir: Path) -> Optional[Dict[str, Any]]:
    """Extract trial information from a trace directory."""
    trial_id = trace_dir.name
    
    # First validate the trace structure
    if not validate_trace_structure(trace_dir):
        return None
    
    # Check if result.json exists
    result_file = trace_dir / "result.json"
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r') as f:
            result_data = json.load(f)
        
        # Extract model name (prefer from agent_info, fallback to config)
        model_name = None
        if 'agent_info' in result_data and 'model_info' in result_data['agent_info']:
            model_name = result_data['agent_info']['model_info']['name']
        elif 'config' in result_data and 'agent' in result_data['config']:
            full_model_name = result_data['config']['agent'].get('model_name', '')
            # Remove provider prefix if present (e.g., "anthropic/claude-opus-4-1-20250805")
            if '/' in full_model_name:
                model_name = full_model_name.split('/', 1)[1]
            else:
                model_name = full_model_name
        
        # Extract task name
        task_name = result_data.get('task_name', '')
        
        # Extract reward score
        reward_score = 0.0
        if result_data and 'verifier_result' in result_data and result_data['verifier_result'] and 'reward' in result_data['verifier_result']:
            reward_score = result_data['verifier_result']['reward']
        
        return {
            'trial_id': trial_id,
            'model_name': model_name,
            'task_name': task_name,
            'reward_score': reward_score
        }
    
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Warning: Could not parse {result_file}: {e}")
        return None


def export_traces_to_yaml(traces: List[str], output_file: str, include_model_name: bool = True):
    """Export trace list to YAML file with proper comments."""
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("task_ids:\n")
        for trace in traces:
            # Parse the trace entry to separate trial_id from comments
            if include_model_name:
                parts = trace.split(' #', 3)  # Split on first three '#' only
                trial_id = parts[0]
                task_name = parts[1] if len(parts) > 1 else ""
                reward_score = parts[2] if len(parts) > 2 else ""
                model_name = parts[3] if len(parts) > 3 else ""
                
                # Write as: - "trial_id" #task_name #reward_score #model_name
                f.write(f'  - "{trial_id}" #{task_name} #{reward_score} #{model_name}\n')
            else:
                parts = trace.split(' #', 2)  # Split on first two '#' only
                trial_id = parts[0]
                task_name = parts[1] if len(parts) > 1 else ""
                reward_score = parts[2] if len(parts) > 2 else ""
                
                # Write as: - "trial_id" #task_name #reward_score
                f.write(f'  - "{trial_id}" #{task_name} #{reward_score}\n')
    
    print(f"Exported {len(traces)} traces to {output_file}")


def scan_traces_directory(traces_dir: str, model_name: Optional[str] = None) -> List[str]:
    """
    Scan a traces directory for valid traces, optionally filtered by model.
    
    Args:
        traces_dir: Directory containing trace folders
        model_name: Optional model name to filter by
        
    Returns:
        List of trace entries formatted as "trial_id #task_name #reward_score #model_name"
    """
    traces_path = Path(traces_dir).resolve()
    matching_traces = []
    model_counts = {}  # Track counts per model
    
    if not traces_path.exists():
        print(f"Error: Traces directory {traces_dir} does not exist")
        return []
    
    if model_name:
        print(f"Scanning {traces_dir} for model '{model_name}'...")
    else:
        print(f"Scanning {traces_dir} for all traces...")
    
    total_dirs = 0
    invalid_structure = 0
    wrong_model = 0
    
    for trace_dir in traces_path.iterdir():
        if not trace_dir.is_dir():
            continue
            
        total_dirs += 1
        trace_info = get_trace_info(trace_dir)
        
        if trace_info is None:
            invalid_structure += 1
            continue
            
        # Count traces per model (for summary when no specific model requested)
        if trace_info['model_name']:
            model_counts[trace_info['model_name']] = model_counts.get(trace_info['model_name'], 0) + 1
            
        if model_name is None or trace_info['model_name'] == model_name:
            # Format: "trial_id" #task_name #reward_score #model_name
            entry = f"{trace_info['trial_id']} #{trace_info['task_name']} #{trace_info['reward_score']} #{trace_info['model_name']}"
            matching_traces.append(entry)
        else:
            wrong_model += 1
    
    print(f"Scan complete:")
    print(f"  - Total directories scanned: {total_dirs}")
    if model_name:
        print(f"  - Valid traces for '{model_name}': {len(matching_traces)}")
        print(f"  - Different model: {wrong_model}")
    else:
        print(f"  - Valid traces (all models): {len(matching_traces)}")
        if model_counts:
            print(f"  - Traces per model:")
            for model, count in sorted(model_counts.items()):
                print(f"    - {model}: {count} traces")
    print(f"  - Invalid/incomplete trace structure: {invalid_structure}")
    
    return matching_traces