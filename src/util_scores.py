import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from supabase import create_client

# Load environment variables
load_dotenv()

def connect_to_database():
    """Connect to the Terminal Bench Supabase database."""
    # Get Supabase credentials from environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_PUBLISHABLE_KEY")
    
    if not all([supabase_url, supabase_key]):
        print("Error: Missing Supabase environment variables.")
        print("Required: SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY")
        return None
    
    try:
        client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_PUBLISHABLE_KEY"],
        )
        print(f"Successfully connected to Supabase at {supabase_url}")
        return client
    except Exception as e:
        raise Exception(f"Failed to connect to Supabase: {e}")


filter_descriptions = {
    "no_exceptions": "Completed Trials Only",
    "with_timeouts": "Completed + Timeout Trials", 
    "with_all_exceptions": "All Trials",
    "only_timeouts": "Timeout Trials Only"
}


def get_model_task_scores(client,
        dataset_name="terminal-bench",
        dataset_version="2.0",
        trial_filter="with_timeouts",
        agent_name=None,
        model_name=None,
        return_trial_details=False,
        filter_by_creation_date=True,
        creation_date_cutoff="2025-09-17T01:13:33.950824+00:00"):
    """
    Get model performance scores for each task.
    
    Args:
        client: Supabase client
        dataset_name: Name of the dataset
        dataset_version: Version of the dataset
        trial_filter: Which trials to include:
            - "no_exceptions": Include all completed trials (no exceptions)
            - "with_timeouts": Include completed trials + timeout errors
            - "with_all_exceptions": Include completed trials + all exceptions
            - "only_timeouts": Include only timeout errors
        agent_name: Optional agent name to filter by
        model_name: Optional model name to filter by
        return_trial_details: If True, return (df, trial_details) where trial_details 
                             maps task_checksum to list of trial IDs used in aggregation
        filter_by_creation_date: If True, only include jobs created after creation_date_cutoff
        creation_date_cutoff: ISO format datetime string for job creation date filter
    
    Returns a DataFrame with columns: model_name, task_name, avg_reward
    Or if return_trial_details=True: (DataFrame, dict mapping task_checksum to trial IDs)
    """
    try:
        print("Fetching trial data from database...")
        
        # Get ALL trials with their associated data using pagination
        # Querying every limit 
        trials = []
        offset = 0
        limit = 1000
        
        # Build the select query including job info if needed
        select_query = """
            id,
            agent_name,
            agent_version,
            task_checksum,
            reward,
            exception_info,
            config,
            trial_model(
                model_name,
                model_provider
            ),
            job!inner(
                id,
                created_at
            )
        """ if filter_by_creation_date else """
            id,
            agent_name,
            agent_version,
            task_checksum,
            reward,
            exception_info,
            config,
            trial_model(
                model_name,
                model_provider
            )
        """
        
        while True:
            query = client.table("trial").select(select_query)
            
            # Apply creation date filter if enabled
            if filter_by_creation_date:
                query = query.gte("job.created_at", creation_date_cutoff)
                
            trials_response = (
                query
                .range(offset, offset + limit - 1)
                .execute()
            )
            
            if not trials_response.data:
                break
                
            trials.extend(trials_response.data)
            offset += limit
            
            if len(trials_response.data) < limit:
                break
        
        if filter_by_creation_date:
            print(f"Found {len(trials)} total trials (filtered by job creation date >= {creation_date_cutoff})")
        else:
            print(f"Found {len(trials)} total trials without filtering")

        dataset_tasks_response = (
            client.table("dataset_task")
            .select("task_checksum")
            .eq("dataset_name", dataset_name)
            .eq("dataset_version", dataset_version)
            .execute()
        )
        dataset_task_checksums = {task["task_checksum"] for task in dataset_tasks_response.data}
        print(f"Found {len(dataset_task_checksums)} tasks in dataset {dataset_name} v{dataset_version}")
        
        # Debug: Check if checksums actually match
        trial_checksums = {trial["task_checksum"] for trial in trials if trial["task_checksum"]}
        overlap = dataset_task_checksums & trial_checksums
        print(f"Debug: {len(trial_checksums)} unique checksums in trials, {len(overlap)} overlap with dataset")
        
        # If no overlap, likely a data mismatch - use all trials instead
        if len(overlap) == 0:
            print("Warning: No checksum overlap found. Using all trials for analysis.")
            dataset_task_checksums = trial_checksums
            
        # Filter trials to only include those in our dataset based on filter setting
        # A trial can run and complete successfully, or it can run and timeout
        # A trial can also run and fail with an exception
        def meets_basic_criteria(trial):
            """Check if trial meets basic filtering criteria"""
            if trial["task_checksum"] not in dataset_task_checksums:
                return False
            if not trial.get("trial_model"):  # Must have model data
                return False
            if trial.get("config", {}).get("agent", {}).get("kwargs", {}).get("parser_name"):
                return False
            
            # Apply agent_name filter if specified
            if agent_name and trial.get("agent_name") != agent_name:
                return False
                
            # Apply model_name filter if specified
            if model_name:
                trial_model_list = trial.get("trial_model", [])
                if not trial_model_list:
                    return False
                trial_model = trial_model_list[0] if isinstance(trial_model_list, list) else trial_model_list
                if trial_model.get("model_name") != model_name:
                    return False
            
            return True
        
        if trial_filter == "no_exceptions":
            print(f"Including all completed trials (no exceptions)")
            filtered_trials = [
                trial for trial in trials 
                if meets_basic_criteria(trial)
                and not trial.get("exception_info")   # only trials with no exceptions
                and trial.get("reward") is not None   # only trials that completed with a reward
            ]
        elif trial_filter == "with_timeouts":
            print(f"Including all completed trials + unsuccessful trials with timeout errors")
            filtered_trials = [
                trial for trial in trials 
                if meets_basic_criteria(trial)
                and (
                    not trial.get("exception_info") or  # successful trials
                    (trial.get("exception_info") and trial["exception_info"].get("exception_type") in ["AgentTimeoutError", "VerifierTimeoutError"])  # timeout errors
                )
            ]
        elif trial_filter == "with_all_exceptions":
            print(f"Including all completed trials + unsuccessful trials with all exceptions)")
            filtered_trials = [
                trial for trial in trials 
                if meets_basic_criteria(trial)
            ]
        elif trial_filter == "only_timeouts":
            print(f"Including unsuccessful trials (only timeout errors)")
            filtered_trials = [
                trial for trial in trials 
                if meets_basic_criteria(trial)
                and trial.get("exception_info")
                and trial["exception_info"].get("exception_type") == "AgentTimeoutError"  # timeout errors only
            ]
        else:
            raise ValueError(f"Invalid trial_filter value: {trial_filter}. Must be 'no_exceptions', 'with_timeouts', 'with_all_exceptions', or 'only_timeouts'")

        
        print(f"Filtered from {len(trials)} total trials to {len(filtered_trials)} valid trials by filtering criteria {trial_filter}")
        
        # Group by agent, model and task to calculate average rewards
        if len(filtered_trials) == 0:
            raise ValueError("No trials found in the dataset after filtering")
        model_task_rewards = {}
        trial_details = {}  # Maps task_checksum to list of trial IDs
        
        for trial in filtered_trials:
            # trial_model is a list, take the first element
            trial_model_list = trial["trial_model"]
            if not trial_model_list:
                continue
            trial_model = trial_model_list[0] if isinstance(trial_model_list, list) else trial_model_list
            
            model_name = trial_model["model_name"]
            agent_name = trial["agent_name"]
            task_checksum = trial["task_checksum"]
            trial_id = trial["id"]
            # Handle None rewards by converting to 0
            reward = trial.get("reward")
            if reward is None:
                reward = 0
            
            key = (agent_name, model_name, task_checksum)
            if key not in model_task_rewards:
                model_task_rewards[key] = []
            model_task_rewards[key].append(reward)
            
            # Store trial details for later use
            if task_checksum not in trial_details:
                trial_details[task_checksum] = []
            trial_details[task_checksum].append({
                "trial_id": trial_id, 
                "reward": reward,
                "agent_name": agent_name,
                "model_name": model_name
            })
        
        # Calculate average rewards per task per model
        avg_scores = []
        for (agent_name, model_name, task_checksum), rewards in model_task_rewards.items():
            avg_reward = np.mean(rewards)
            avg_scores.append({
                "agent_name": agent_name,
                "model_name": model_name,
                "task_checksum": task_checksum,
                "avg_reward": avg_reward,
                "num_trials": len(rewards)
            })
        
        # Get task names
        task_checksums = list(set(score["task_checksum"] for score in avg_scores))
        tasks_response = (
            client.table("task")
            .select("checksum, name")
            .in_("checksum", task_checksums)
            .execute()
        )
        
        task_names = {task["checksum"]: task["name"] for task in tasks_response.data}
        
        # Add task names to scores
        for score in avg_scores:
            score["task_name"] = task_names.get(score["task_checksum"], "Unknown")
        
        # Create DataFrame
        df = pd.DataFrame(avg_scores)
        
        # Print summary statistics
        print(f"\nModel-Task Score Summary:")
        print(f"- Number of agents: {df['agent_name'].nunique()}")
        print(f"- Number of models: {df['model_name'].nunique()}")
        print(f"- Number of tasks: {df['task_name'].nunique()}")
        print(f"- Total agent-model-task combinations: {len(df)}")
        
        # Calculate theoretical total
        n_agents = df['agent_name'].nunique()
        n_models = df['model_name'].nunique()
        n_tasks = df['task_name'].nunique()
        theoretical_total = n_agents * n_models * n_tasks
        missing_total = theoretical_total - len(df)
        missing_pct = (missing_total / theoretical_total) * 100
        
        print(f"\nCombination Analysis:")
        print(f"- Theoretical combinations: {n_agents} × {n_models} × {n_tasks} = {theoretical_total:,}")
        print(f"- Actual combinations: {len(df):,}")
        print(f"- Missing combinations: {missing_total:,} ({missing_pct:.1f}%)")
        
        # Analyze which agents were tested with which models
        agent_model_coverage = df.groupby('agent_name')['model_name'].nunique()
        print(f"\nAgent Coverage (models tested per agent out of {n_models} total):")
        for agent in sorted(agent_model_coverage.index):
            models_tested = agent_model_coverage[agent]
            models_missing = n_models - models_tested
            print(f"  {agent:20s}: {models_tested:2d} models tested, {models_missing:2d} missing")
        
        # Identify which agent-model pairs are completely missing
        all_agents = set(df['agent_name'].unique())
        all_models = set(df['model_name'].unique())
        existing_pairs = set(df.groupby(['agent_name', 'model_name']).size().index)
        theoretical_pairs = {(a, m) for a in all_agents for m in all_models}
        missing_pairs = theoretical_pairs - existing_pairs
        
        if missing_pairs:
            print(f"\nCompletely Missing Agent-Model Pairs ({len(missing_pairs)} pairs with 0 tasks):")
            missing_by_agent = {}
            for agent, model in sorted(missing_pairs):
                if agent not in missing_by_agent:
                    missing_by_agent[agent] = []
                missing_by_agent[agent].append(model)
            
            for agent in sorted(missing_by_agent.keys()):
                models_list = missing_by_agent[agent]
                if len(models_list) <= 3:
                    print(f"  {agent}: missing {', '.join(models_list)}")
                else:
                    print(f"  {agent}: missing {len(models_list)} models ({', '.join(models_list[:3])}, ...)")
        
        # group by agent and model name:
        models_summary = df.groupby(['agent_name', 'model_name']).agg({
            'task_name': 'count',
            'avg_reward': 'mean'
        }).rename(columns={'task_name': 'tasks_attempted', 'avg_reward': 'overall_avg_reward'})
        
        print(f"\nAgent-Model Pairs with Incomplete Task Coverage (<{n_tasks} tasks):")
        incomplete_pairs = []
        for (agent, model), stats in models_summary.iterrows():
            if stats['tasks_attempted'] < n_tasks:
                incomplete_pairs.append((agent, model, int(stats['tasks_attempted'])))
        
        if incomplete_pairs:
            for agent, model, task_count in sorted(incomplete_pairs):
                missing_tasks = n_tasks - task_count
                print(f"  {agent} + {model}: {task_count}/{n_tasks} tasks (missing {missing_tasks})")
        else:
            print("  All existing agent-model pairs have complete task coverage.")
        
        print(f"\nDetailed Agent-Model Coverage:")
        for model, stats in models_summary.iterrows():
            print(f"  {model}: {stats['tasks_attempted']} tasks, avg reward: {stats['overall_avg_reward']:.3f}")
        
        if return_trial_details:
            return df, trial_details
        else:
            return df
        
    except Exception as e:
        print(f"Error fetching model-task scores: {e}")
        import traceback
        traceback.print_exc()
        return None


