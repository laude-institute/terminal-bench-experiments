#!/usr/bin/env python3
"""
Get traces and info of unsolved tasks.
Can download the traces.

This script supports two modes:
1. Config file mode (recommended): Download specific trial IDs from a YAML config file
2. Database query mode: Query database for failed trials based on model/threshold criteria

Usage:
  # Download specific trial IDs (recommended)
  python get_failure_traces.py --config config/failures.yaml --download
  
  # Query database for failed trials
  python get_failure_traces.py --model-name "gpt-5" --threshold 0.3 --download
"""

import os
import sys
from pathlib import Path
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed


# Add the scripts directory to Python path to import plot_task_scores
BASE_DIR = Path(__file__).parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))
from util_scores import connect_to_database, get_model_task_scores
from trace_utils import validate_trace_structure


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
    print(f"\nGetting failed trials for {agent_name}/{model_name}/{trial_filter} with reward <= {threshold}", flush=True)
    
    # Get model-task scores data with trial details
    df, trial_details = get_model_task_scores(client,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        trial_filter=trial_filter,
        agent_name=agent_name,
        model_name=model_name,
        return_trial_details=True)
    
    if df is None or df.empty:
        print("No data available", flush=True)
        return {}
    
    print(f"Total trials for {agent_name}/{model_name}/{trial_filter}: {len(df)}", flush=True)

    # Filter for tasks below threshold (agent/model already filtered by get_model_task_scores)
    failed_tasks_df = df[df['avg_reward'] <= threshold]
    
    print(f"Found {len(failed_tasks_df)} failed task combinations", flush=True)
    
    if failed_tasks_df.empty:
        print(f"No failed tasks found for {agent_name}/{model_name}/{trial_filter}", flush=True)
        return {}
    
    # Apply limit_tasks if specified
    if limit_tasks is not None:
        failed_tasks_df = failed_tasks_df.head(limit_tasks)
        print(f"Limiting tasks to {limit_tasks}", flush=True)
    
    # Get trial IDs for failed tasks
    failed_trials = {}
    for _, row in failed_tasks_df.iterrows():
        task_checksum = row['task_checksum']
        task_name = row['task_name']
        
        if task_checksum in trial_details:
            # Get trial IDs for this task (already filtered by agent/model in get_model_task_scores)
            task_trial_ids = [trial_info["trial_id"] for trial_info in trial_details[task_checksum]
                            if trial_info["reward"] <= threshold]
            
            # Apply limit_trials if specified
            if limit_trials is not None:
                task_trial_ids = task_trial_ids[:limit_trials]
            
            if task_trial_ids:
                failed_trials[task_name] = task_trial_ids
    
    # Sort by number of failed trials (descending)
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
            print(f"No trial found with ID {trial_id}", flush=True)
            return
        
        trial = trial_response.data[0]
        
        print(f"Trial ID: {trial['id']}", flush=True)
        print(f"Agent: {trial['agent_name']} v{trial.get('agent_version', 'unknown')}", flush=True)
        
        if trial.get('trial_model'):
            model = trial['trial_model'][0] if isinstance(trial['trial_model'], list) else trial['trial_model']
            print(f"Model: {model['model_name']} ({model.get('model_provider', 'unknown')})", flush=True)
        
        if trial.get('task'):
            task = trial['task']
            print(f"Task: {task['name']}", flush=True)
        
        print(f"Reward: {trial.get('reward', 'None')}", flush=True)
        print(f"Created: {trial.get('created_at', 'Unknown')}", flush=True)
        
        if trial.get('exception_info'):
            exception = trial['exception_info']
            print(f"Exception: {exception.get('exception_type', 'Unknown')} - {exception.get('message', 'No message')}", flush=True)
        
        # Try to get some basic configuration info
        config = trial.get('config', {})
        if config:
            print(f"Config keys: {list(config.keys())}", flush=True)
        
        print(f"\nFull trial data available. Note: File traces may be stored separately.", flush=True)
        
    except Exception as e:
        print(f"Error querying trial {trial_id}: {e}", flush=True)


def download_trial_trace_simple(client, trial_id, output_dir="traces", verbose=False):
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
                        
                        if verbose:
                            # Show content of key trajectory files
                            if file_name in ["prompt.txt", "response.txt"]:
                                print(f"\n--- {file_path} ---", flush=True)
                                try:
                                    content = data.decode('utf-8')
                                    print(content[:1500], flush=True)
                                    if len(content) > 1500:
                                        print("... [truncated]", flush=True)
                                except UnicodeDecodeError:
                                    print(f"[Binary file, {len(data)} bytes]", flush=True)
                            else:
                                print(f"Downloaded: {file_path}", flush=True)
                    else:
                        out_path.mkdir(parents=True, exist_ok=True)
                        download_files_recursive(file_path, out_path)
                        
            except Exception as e:
                print(f"Error processing path {base_path}: {e}", flush=True)
        
        print(f"\nDownloading trajectory for trial {trial_id}...", flush=True)

        # Check if trial has already been downloaded with valid structure
        if output_path.exists() and validate_trace_structure(output_path):
            print(f"Trial {trial_id} already downloaded", flush=True)
            return output_path
        
        download_files_recursive(trial_id, output_path)
        
        # Validate the downloaded trace structure
        if validate_trace_structure(output_path):
            return output_path
        else:
            print(f"Download incomplete or invalid structure for trial {trial_id}", flush=True)
            return None
                
    except Exception as e:
        print(f"Error downloading trial {trial_id}: {e}", flush=True)
        return None


def load_task_ids_from_config(config_path):
    """
    Load task IDs from a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        List of task IDs to process
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        task_ids = config.get('task_ids', [])
        if not task_ids:
            print(f"No task_ids found in {config_path}", flush=True)
            return []
        
        print(f"Loaded {len(task_ids)} task IDs from {config_path}", flush=True)
        return task_ids
        
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}", flush=True)
        return []


def download_trials_parallel(client, trial_ids, verbose=False, max_workers=5):
    """
    Download multiple trials in parallel.
    
    Args:
        client: Supabase client
        trial_ids: List of trial IDs to download
        verbose: Show verbose output
        max_workers: Maximum number of parallel downloads
    """
    if not trial_ids:
        return
        
    print(f"\nStarting parallel download of {len(trial_ids)} trials with {max_workers} workers", flush=True)
    
    def download_single_trial(trial_id):
        """Download a single trial and return result."""
        try:
            result = download_trial_trace_simple(client, trial_id, verbose=verbose)
            return trial_id, "success", result
        except Exception as e:
            return trial_id, "error", str(e)
    
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_trial = {executor.submit(download_single_trial, trial_id): trial_id 
                          for trial_id in trial_ids}
        
        # Process completed downloads
        for future in as_completed(future_to_trial):
            trial_id, status, result = future.result()
            completed += 1
            
            if status == "success":
                print(f"[{completed}/{len(trial_ids)}] ✓ Downloaded {trial_id}", flush=True)
            else:
                print(f"[{completed}/{len(trial_ids)}] ✗ Failed {trial_id}: {result}", flush=True)
    
    print(f"\nParallel download completed: {len(trial_ids)} trials processed", flush=True)


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
        print("No failed trials to process", flush=True)
        return
    
    task_count = 0
    for task_name, trial_ids in failed_trials.items():            
        print(f"\n{'='*80}", flush=True)
        print(f"TASK: {task_name}", flush=True)
        print(f"{'='*80}", flush=True)
        
        for i, trial_id in enumerate(trial_ids, 1):
            print(f"\n{'-'*60}", flush=True)
            print(f"Trial {i}/{len(trial_ids)}: {trial_id}", flush=True)
            print(f"{'-'*60}", flush=True)
            
            if download:
                # Download and print the trace directly
                download_trial_trace_simple(client, trial_id)
            else:
                # Just show trial info without downloading
                show_trial_info(client, trial_id)
        
        task_count += 1


def main(args):

    print("Terminal Bench Failure Trace Generator", flush=True)
    print("=" * 80, flush=True)
    
    # Connect to database
    client = connect_to_database()
    
    if client is None:
        print("Failed to connect to database. Exiting.", flush=True)
        return
    
    try:
        # Check if config file is provided
        if args.config:
            # Config file mode - process specific trial IDs
            print(f"Using config file mode: {args.config}", flush=True)
            print("Note: All other parameters (model-name, threshold, etc.) are ignored when using config file", flush=True)
            task_ids = load_task_ids_from_config(args.config)
            
            if not task_ids:
                print("No valid task IDs found in config file. Exiting.", flush=True)
                return
            
            print(f"Downloaded traces will be saved to: traces/", flush=True)
            print("=" * 80, flush=True)
            
            if args.download:
                # Use parallel downloading for better performance
                download_trials_parallel(client, task_ids, verbose=args.verbose, max_workers=args.max_workers)
            else:
                # Show info sequentially (info mode doesn't need parallelization)
                for i, task_id in enumerate(task_ids, 1):
                    print(f"\n{'-'*60}", flush=True)
                    print(f"Processing trial {i}/{len(task_ids)}: {task_id}", flush=True)
                    print(f"{'-'*60}", flush=True)
                    show_trial_info(client, task_id)
                    
            print(f"\nProcessed {len(task_ids)} trials from config file", flush=True)
        
        else:
            # Database query mode - discover failed trials based on criteria
            print(f"Using database query mode", flush=True)
            print(f"Agent: {args.agent_name}", flush=True)
            print(f"Model: {args.model_name}", flush=True)
            print(f"Threshold: {args.threshold}", flush=True)
            print(f"Dataset: {args.dataset_name} v{args.dataset_version}", flush=True)
            print("=" * 80, flush=True)
            
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
                print(f"\nFound failed trials for {len(failed_trials)} tasks", flush=True)
                
                # Print summary
                for task_name, trial_ids in failed_trials.items():
                    print(f"  {task_name}: {len(trial_ids)} failed trials", flush=True)
                
                # Collect all trial IDs for parallel processing
                all_trial_ids = []
                for trial_ids in failed_trials.values():
                    all_trial_ids.extend(trial_ids)
                
                if args.download:
                    # Use parallel downloading for better performance  
                    download_trials_parallel(client, all_trial_ids, verbose=args.verbose, max_workers=args.max_workers)
                else:
                    # Show info sequentially (preserving task grouping for readability)
                    print_traces_for_failed_trials(client, failed_trials, download=False)
            else:
                print("No failed trials found matching criteria", flush=True)
            
    except Exception as e:
        print(f"Error in main execution: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    finally:
        if client:
            print("\nSupabase client session completed.", flush=True)

    print("\nCompleted processing traces.", flush=True)


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
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file containing task_ids to process (overrides all other parameters - recommended approach)")
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Show verbose output when downloading traces (default: False)")
    parser.add_argument("--max-workers", type=int, default=5,
                       help="Maximum number of parallel download workers (default: 5)")

    args = parser.parse_args()
    main(args)