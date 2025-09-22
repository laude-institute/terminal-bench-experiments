#!/usr/bin/env python3
"""
Get traces and info of unsolved tasks.
Can download the traces.

This script:
1. Uses get_model_task_scores from plot_task_scores.py to get agent/model/task/trial data
2. Identifies tasks where the reward_score <= threshold 
3. Prints traces for failed trials using get_trace_by_trial_id.py
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import argparse
import subprocess
from datetime import datetime
import json


# Add the scripts directory to Python path to import plot_task_scores
BASE_DIR = Path(__file__).parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))
from util_scores import connect_to_database, get_model_task_scores

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Load environment variables
load_dotenv()


def get_failed_trials(
    client, 
    agent_name: str ="terminus-2", 
    model_name: str ="gpt-5", 
    threshold: float =0, 
    dataset_name: str ="terminal-bench", 
    dataset_version: str ="2.0",
    trial_filter="with_timeouts",
    limit_trials: int = None,
    limit_tasks: int = None,
):
    """
    Get trial IDs for tasks where the agent/model performance is <= threshold.
    
    Args:
        client: Supabase client
        agent_name: Name of the agent (default: "terminus-2")
        model_name: Name of the model (default: "gpt-5") 
        threshold: Reward score threshold (default: 0)
        dataset_name: Dataset name (default: "terminal-bench")
        trial_filter: Trial filter (default: "with_timeouts"). Options: "with_timeouts", "only_timeouts"
        dataset_version: Dataset version (default: "2.0")
        limit_trials: Number of trials to limit per task
        limit_tasks: Number of tasks to limit
    Returns:
        Dictionary mapping task names to lists of trial IDs for failed attempts
    """
    print(f"\nGetting failed trials for {agent_name}/{model_name}/{trial_filter} with reward <= {threshold}")
    
    # Get model-task scores data with trial details
    df, trial_details = get_model_task_scores(client,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        trial_filter=trial_filter,
        agent_name=agent_name,
        model_name=model_name,
        return_trial_details=True)
    
    if df is None or df.empty:
        print("No data available")
        return {}
    
    # We want to print the number of total trials for the agent and model
    print(f"Total trials for {agent_name}/{model_name}/{trial_filter}: {len(df)}")

    # Filter for the specific agent and model
    # Note: trial_filter was already applied in get_model_task_scores, so we don't need to filter by it again
    filtered_df = df[
        (df['agent_name'] == agent_name) & 
        (df['model_name'] == model_name) &
        (df['avg_reward'] <= threshold)
    ]
    
    print(f"Found {len(filtered_df)} failed task combinations")
    
    if filtered_df.empty:
        print(f"No failed tasks found for {agent_name}/{model_name}/{trial_filter}")
        return {}
    
    # Now get the actual trial IDs for these failed tasks from the trial details
    failed_trials = {}
    
    # Apply limit_tasks if specified
    tasks_to_process = filtered_df
    if limit_tasks is not None:
        tasks_to_process = filtered_df.head(limit_tasks)
        print(f"Limiting tasks to {limit_tasks}")
    
    for _, row in tasks_to_process.iterrows():
        task_checksum = row['task_checksum']
        task_name = row['task_name']
        
        # Get trials from the already-filtered trial_details
        if task_checksum in trial_details:
            task_trial_ids = []
            for trial_info in trial_details[task_checksum]:
                # Filter by agent/model and reward threshold
                if (trial_info["agent_name"] == agent_name and 
                    trial_info["model_name"] == model_name and
                    trial_info["reward"] <= threshold):
                    task_trial_ids.append(trial_info["trial_id"])
            
            # Apply limit_trials if specified
            if limit_trials is not None:
                task_trial_ids = task_trial_ids[:limit_trials]
                print(f"Limiting trials to {limit_trials}")
            
            if task_trial_ids:
                failed_trials[task_name] = task_trial_ids
    
    # sort by number of failed trials
    failed_trials = dict(sorted(failed_trials.items(), key=lambda item: len(item[1]), reverse=True))
    return failed_trials


def show_trial_info(client, trial_id):
    """
    Show basic information about a trial from the database.
    
    Args:
        client: Supabase client
        trial_id: Trial ID to query
    """
    try:
        # Get trial details from database
        trial_response = (
            client.table("trial")
            .select("""
                id,
                agent_name,
                agent_version,
                task_checksum,
                reward,
                exception_info,
                config,
                created_at,
                trial_model(
                    model_name,
                    model_provider
                ),
                task(
                    name
                )
            """)
            .eq("id", trial_id)
            .execute()
        )
        
        if not trial_response.data:
            print(f"No trial found with ID {trial_id}")
            return
        
        trial = trial_response.data[0]
        
        print(f"Trial ID: {trial['id']}")
        print(f"Agent: {trial['agent_name']} v{trial.get('agent_version', 'unknown')}")
        
        if trial.get('trial_model'):
            model = trial['trial_model'][0] if isinstance(trial['trial_model'], list) else trial['trial_model']
            print(f"Model: {model['model_name']} ({model.get('model_provider', 'unknown')})")
        
        if trial.get('task'):
            task = trial['task']
            print(f"Task: {task['name']}")
        
        print(f"Reward: {trial.get('reward', 'None')}")
        print(f"Created: {trial.get('created_at', 'Unknown')}")
        
        if trial.get('exception_info'):
            exception = trial['exception_info']
            print(f"Exception: {exception.get('exception_type', 'Unknown')} - {exception.get('message', 'No message')}")
        
        # Try to get some basic configuration info
        config = trial.get('config', {})
        if config:
            print(f"Config keys: {list(config.keys())}")
        
        print(f"\nFull trial data available. Note: File traces may be stored separately.")
        
    except Exception as e:
        print(f"Error querying trial {trial_id}: {e}")


def download_trial_trace_simple(client, trial_id, output_dir="traces"):
    """
    Download terminus trajectory files from episode subdirectories.
    
    Args:
        client: Supabase client
        trial_id: Trial ID to download
        output_dir: Output directory for traces
    
    Returns:
        Path to downloaded directory or None if failed
    """
    try:
        # First show trial info from database
        show_trial_info(client, trial_id)
        
        output_path = Path(output_dir) / trial_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        def download_files_recursive(base_path, output_base):
            """Recursively download files from storage."""
            try:
                files = client.storage.from_("trials").list(path=str(base_path))
                
                for file in files:
                    file_name = file["name"]
                    file_id = file["id"]
                    
                    is_file = file_id is not None
                    
                    file_path = Path(base_path) / file_name
                    out_path = output_base / file_name
                    
                    if is_file:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        data = client.storage.from_("trials").download(str(file_path))
                        out_path.write_bytes(data)
                        
                        # Print content of trajectory files
                        if file_name in ["prompt.txt", "response.txt"]:
                            print(f"\n--- {file_path} ---")
                            try:
                                content = data.decode('utf-8')
                                print(content[:1500])  # Print first 1500 chars
                                if len(content) > 1500:
                                    print("... [truncated]")
                            except UnicodeDecodeError:
                                print(f"[Binary file, {len(data)} bytes]")
                        else:
                            print(f"Downloaded: {file_path}")
                    else:
                        out_path.mkdir(parents=True, exist_ok=True)
                        download_files_recursive(file_path, out_path)
                        
            except Exception as e:
                print(f"Error processing path {base_path}: {e}")
        
        print(f"\nDownloading trajectory for trial {trial_id}...")
        download_files_recursive(trial_id, output_path)
        
        # Check if we downloaded any files
        if any(output_path.rglob("*")):
            return output_path
        else:
            print(f"No files found for trial {trial_id}")
            return None
                
    except Exception as e:
        print(f"Error downloading trial {trial_id}: {e}")
        return None


def print_traces_for_failed_trials(client, failed_trials, download=False):
    """
    Print traces for a subset of failed trials.
    
    Args:
        client: Supabase client
        failed_trials: Dictionary mapping task names to trial IDs
        num_trials: Maximum number of trials to print per task (default: 4)
        num_tasks: Maximum number of tasks to process (default: 2)
        download: Whether to download trial traces (default: False)
    """
    if not failed_trials:
        print("No failed trials to process")
        return
    
    task_count = 0
    for task_name, trial_ids in failed_trials.items():            
        print(f"\n{'='*80}")
        print(f"TASK: {task_name}")
        print(f"{'='*80}")
        
        for i, trial_id in enumerate(trial_ids, 1):
            print(f"\n{'-'*60}")
            print(f"Trial {i}/{len(trial_ids)}: {trial_id}")
            print(f"{'-'*60}")
            
            if download:
                # Download and print the trace directly
                download_trial_trace_simple(client, trial_id)
            else:
                # Just show trial info without downloading
                show_trial_info(client, trial_id)
        
        task_count += 1


def main(args):

    print("Terminal Bench Failure Trace Generator")
    print("=" * 80)
    print(f"Agent: {args.agent_name}")
    print(f"Model: {args.model_name}")
    print(f"Threshold: {args.threshold}")
    print(f"Dataset: {args.dataset_name} v{args.dataset_version}")
    print("=" * 80)
    
    # Connect to database
    client = connect_to_database()
    
    if client is None:
        print("Failed to connect to database. Exiting.")
        return
    
    filename = None  # Initialize filename variable
    try:
        # Get failed trials
        failed_trials = get_failed_trials(
            client=client,
            agent_name=args.agent_name,
            model_name=args.model_name, 
            threshold=args.threshold,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            trial_filter=args.trial_filter,
            limit_trials=args.num_trials,
            limit_tasks=args.num_tasks,
        )
        
        if failed_trials:
            print(f"\nFound failed trials for {len(failed_trials)} tasks")
            # sort by number of failed trials

            # Print summary
            for task_name, trial_ids in failed_trials.items():
                print(f"  {task_name}: {len(trial_ids)} failed trials")
            
            # Export to json: including all the args:
            # the num_trials and num_tasks should be i
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. "20250917_231523"
            filename  = Path(OUTPUT_DIR / f"failed_trials_{timestamp}.json")
            with open(filename, "w") as f:
                json.dump({**args.__dict__, "failed_trials": failed_trials}, f)

            # Print traces for subset of failed trials
            print_traces_for_failed_trials(
                client,
                failed_trials, 
                download=args.download
            )
        else:
            print("No failed trials found matching criteria")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if client:
            print("\nSupabase client session completed.")

    if filename:
        print(f"\nSaved failed trials to {filename}")
    return filename


if __name__ == "__main__":
    """Main function to get and print failure traces."""
    parser = argparse.ArgumentParser(description="Get traces of unsolved tasks")
    parser.add_argument("--agent-name", default="terminus-2",
                       help="Agent name (default: terminus-2)")
    parser.add_argument("--model-name", default="gpt-5", 
                       help="Model name (default: gpt-5)")
    parser.add_argument("--threshold", type=float, default=0,
                       help="Reward score threshold (default: 0)")
    parser.add_argument("--dataset-name", default="terminal-bench",
                       help="Dataset name (default: terminal-bench)")
    parser.add_argument("--dataset-version", default="2.0",
                       help="Dataset version (default: 2.0)")
    parser.add_argument("--num-trials", type=int, default=None,
                       help="Number of trials to print per task (default: 4)")
    parser.add_argument("--num-tasks", type=int, default=None,
                       help="Number of tasks to process (default: 2)")
    parser.add_argument("--download", action="store_true", default=False,
                       help="Download trial traces (default: False, just show info)")
    parser.add_argument("--trial-filter", default="with_timeouts",
                       choices=["with_timeouts", "only_timeouts", "no_exceptions"],
                       help="Trial filter: with_timeouts (all failed), only_timeouts (timeout failures only), no_exceptions (non-timeout failures only)")

    args = parser.parse_args()
    main(args)