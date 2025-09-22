#!/usr/bin/env python3
"""
Read and visualize task categories from Terminal Bench database.

This script:
1. Connects to the Terminal Bench PostgreSQL database.
2. Extracts task descriptions and categories.
3. Creates a pie chart visualization of task category distribution.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from supabase import create_client
from dotenv import load_dotenv
import requests
import time
import argparse

# Load environment variables
load_dotenv()

def fetch_solution_from_github(task_name):
    """Fetch solution.sh content for a given task from the terminal-bench GitHub repository."""
    github_url = f"https://raw.githubusercontent.com/laude-institute/terminal-bench/main/tasks/{task_name}/solution.sh"
    
    try:
        response = requests.get(github_url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
        elif response.status_code == 404:
            print(f"  Warning: No solution.sh found for task '{task_name}' (404)")
            return None
        else:
            print(f"  Warning: HTTP {response.status_code} when fetching solution for task '{task_name}'")
            return None
    except requests.RequestException as e:
        print(f"  Error fetching solution for task '{task_name}': {e}")
        return None

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


def get_sample_task_data():
    """Provide sample task data when database is unavailable."""
    return [
        {'name': 'adaptive-rejection-sampler', 'checksum': 'abc123', 'instruction': 'Implement an adaptive rejection sampling algorithm', 'source': 'terminal-bench', 'path': 'sampling/adaptive.py'},
        {'name': 'binary-search-tree', 'checksum': 'def456', 'instruction': 'Implement a binary search tree with insert and search operations', 'source': 'terminal-bench', 'path': 'trees/bst.py'},
    ]

def get_agent_model_results(client, dataset_name, dataset_version):
    """
    Returns a list of agent-model results using Supabase.
    Each result is a dictionary with the following keys:
    - agent_name
    - agent_version
    - model_name
    - model_provider
    - avg_accuracy
    - stderr
    - num_tasks
    - total_trials
    - avg_cost_dollars
    - task_details
    """
    try:
        # This is a complex query that would be difficult to replicate exactly with Supabase API calls
        # For now, we'll implement a simplified version that gets basic trial data
        # and performs the aggregation in Python
        
        print("Getting agent-model results from Supabase...")
        
        # Get trials with their associated data
        trials_response = (
            client.table("trial")
            .select("""
                agent_name,
                agent_version,
                task_checksum,
                reward,
                exception_info,
                config,
                trial_model!inner(
                    model_name,
                    model_provider,
                    n_input_tokens,
                    n_output_tokens,
                    model!inner(
                        cents_per_million_input_tokens,
                        cents_per_million_output_tokens
                    )
                )
            """)
            .execute()
        )
        
        trials = trials_response.data
        
        # Filter trials for the specified dataset
        dataset_tasks_response = (
            client.table("dataset_task")
            .select("task_checksum")
            .eq("dataset_name", dataset_name)
            .eq("dataset_version", dataset_version)
            .execute()
        )
        
        dataset_task_checksums = {task["task_checksum"] for task in dataset_tasks_response.data}
        
        # Filter trials to only include those in our dataset
        filtered_trials = [
            trial for trial in trials 
            if trial["task_checksum"] in dataset_task_checksums
            and (
                not trial.get("exception_info") or 
                trial["exception_info"].get("exception_type") in [None, "AgentTimeoutError"]
            )
            and (
                not trial.get("config", {}).get("agent", {}).get("kwargs", {}).get("parser_name")
            )
        ]
        
        print(f"Found {len(filtered_trials)} trials for dataset {dataset_name} v{dataset_version}")
        
        # Group by agent, model, and task
        agent_model_task_scores = {}
        
        for trial in filtered_trials:
            trial_model = trial["trial_model"]
            model = trial_model["model"]
            
            # Calculate cost per trial
            trial_cost = None
            if (trial_model.get("n_input_tokens") and 
                trial_model.get("n_output_tokens") and
                model.get("cents_per_million_input_tokens") and
                model.get("cents_per_million_output_tokens")):
                
                input_cost = (trial_model["n_input_tokens"] / 1_000_000.0) * model["cents_per_million_input_tokens"] / 100.0
                output_cost = (trial_model["n_output_tokens"] / 1_000_000.0) * model["cents_per_million_output_tokens"] / 100.0
                trial_cost = input_cost + output_cost
            
            key = (
                trial["agent_name"],
                trial["agent_version"],
                trial_model["model_name"],
                trial_model["model_provider"],
                trial["task_checksum"]
            )
            
            if key not in agent_model_task_scores:
                agent_model_task_scores[key] = {
                    "rewards": [],
                    "costs": [],
                    "agent_name": trial["agent_name"],
                    "agent_version": trial["agent_version"],
                    "model_name": trial_model["model_name"],
                    "model_provider": trial_model["model_provider"],
                    "task_checksum": trial["task_checksum"]
                }
            
            agent_model_task_scores[key]["rewards"].append(trial.get("reward", 0))
            if trial_cost is not None:
                agent_model_task_scores[key]["costs"].append(trial_cost)
        
        # Calculate task-level scores
        task_scores = []
        for key, data in agent_model_task_scores.items():
            rewards = data["rewards"]
            costs = data["costs"]
            
            avg_score = sum(rewards) / len(rewards)
            variance = None
            if len(rewards) > 1:
                variance = avg_score * (1 - avg_score) / (len(rewards) - 1)
            
            avg_cost = sum(costs) / len(costs) if costs else 0
            total_cost = sum(costs) if costs else 0
            
            task_scores.append({
                "agent_name": data["agent_name"],
                "agent_version": data["agent_version"],
                "model_name": data["model_name"],
                "model_provider": data["model_provider"],
                "task_checksum": data["task_checksum"],
                "avg_task_score": avg_score,
                "variance": variance,
                "trial_count": len(rewards),
                "avg_task_cost_dollars": avg_cost,
                "total_task_cost_dollars": total_cost
            })
        
        # Get task names
        task_checksums = list(set(score["task_checksum"] for score in task_scores))
        tasks_response = (
            client.table("task")
            .select("checksum, name")
            .in_("checksum", task_checksums)
            .execute()
        )
        
        task_names = {task["checksum"]: task["name"] for task in tasks_response.data}
        
        # Add task names to scores
        for score in task_scores:
            score["task_name"] = task_names.get(score["task_checksum"], "Unknown")
        
        # Group by agent-model combinations
        agent_model_groups = {}
        for score in task_scores:
            key = (score["agent_name"], score["agent_version"], score["model_name"], score["model_provider"])
            if key not in agent_model_groups:
                agent_model_groups[key] = []
            agent_model_groups[key].append(score)
        
        # Calculate final results
        results = []
        total_tasks = len(dataset_task_checksums)
        
        for key, scores in agent_model_groups.items():
            # Only include agent-model combos that tried all tasks
            if len(scores) == total_tasks:
                agent_name, agent_version, model_name, model_provider = key
                
                avg_accuracy = sum(score["avg_task_score"] for score in scores) / len(scores) * 100
                
                # Calculate standard error
                variances = [score["variance"] for score in scores if score["variance"] is not None]
                stderr = None
                if len(variances) == len(scores):
                    stderr = (sum(variances) ** 0.5) / len(scores) * 100
                
                total_trials = sum(score["trial_count"] for score in scores)
                total_cost = sum(score["total_task_cost_dollars"] for score in scores)
                
                # Create task details
                task_details = []
                for score in sorted(scores, key=lambda x: x["task_name"]):
                    task_details.append({
                        "task_name": score["task_name"],
                        "task_checksum": score["task_checksum"],
                        "avg_score": round(score["avg_task_score"], 4),
                        "trial_count": score["trial_count"],
                        "avg_cost_dollars": round(score["avg_task_cost_dollars"], 2),
                        "total_cost_dollars": round(score["total_task_cost_dollars"], 2)
                    })
                
                results.append({
                    "agent_name": agent_name,
                    "agent_version": agent_version,
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "avg_accuracy": round(avg_accuracy, 2),
                    "stderr": round(stderr, 2) if stderr else None,
                    "num_tasks": len(scores),
                    "total_trials": total_trials,
                    "avg_cost_dollars": round(total_cost, 2),
                    "task_details": task_details
                })
        
        # Sort by accuracy
        results.sort(key=lambda x: x["avg_accuracy"], reverse=True)
        
        print(f"\nTotal agent-model results: {len(results)}")
        for result in results:
            print(f"{result['agent_name']} ({result['model_name']}) → {result['avg_accuracy']}% acc, "
                  f"stderr={result['stderr']}, cost=${result['avg_cost_dollars']}")
            print(f"  Tasks: {len(result['task_details'])}")
            # Print first few task details
            for task in result['task_details'][:3]:
                print(f"    - {task['task_name']} ({task['avg_score']*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"Error extracting agent-model results: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_tasks(conn, dataset_name, dataset_version, fetch_solutions=False):
    """Extract task descriptions and categories from the database or sample data."""

    if conn is None:
        raise Exception("No database connection")
        #print("\nUsing sample data for demonstration...")
        #tasks = get_sample_task_data()
        #print(f"\nTotal tasks found: {len(tasks)}")
    # Use Supabase to get tasks for the specified dataset
    try:
        # Query tasks with dataset_task join using Supabase
        response = (
            conn.table("task")
            .select("*, dataset_task!inner(*)")
            .eq("dataset_task.dataset_name", dataset_name)
            .eq("dataset_task.dataset_version", dataset_version)
            .execute()
        )
        
        tasks = response.data
        print(f"\nTotal tasks found (restricted to dataset {dataset_name} v{dataset_version}): {len(tasks)}")
        
    except Exception as e:
        print(f"Error querying tasks from Supabase: {e}")
        raise Exception(f"Failed to query tasks: {e}")
                    

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(tasks)
    df = df.drop(columns=['dataset_task'])
    
    ## Metadata is a column where each row is a dict, so we need to normalize it, and then make sure the elements are columns:
    # First, collect all unique keys from all metadata dictionaries
    all_metadata_keys = set()
    for metadata_dict in df['metadata']:
        if isinstance(metadata_dict, dict):
            all_metadata_keys.update(metadata_dict.keys())
    
    # Normalize the metadata, ensuring all keys are included
    metadata_df = pd.json_normalize(df['metadata'])
    
    # Add any missing columns with NaN values
    for key in all_metadata_keys:
        if key not in metadata_df.columns:
            metadata_df[key] = None
    
    df = df.join(metadata_df)
    df = df.drop(columns=['metadata'])

    # Drop columns that are not needed:
    df.pop('checksum')
    df.pop('created_at')
    df.pop('author_name')
    df.pop('author_email')
    df.pop('git_url')
    df.pop('git_commit_id')

    # Fetch solutions from GitHub for each task (only if flag is set)
    if fetch_solutions:
        print(f"\nFetching solutions from GitHub for {len(df)} tasks...")
        solutions = []
        
        for i, row in df.iterrows():
            task_name = row['name']
            print(f"  Fetching solution for task {i+1}/{len(df)}: {task_name}")
            
            solution = fetch_solution_from_github(task_name)
            solutions.append(solution)
            
            # Add a small delay to be respectful to GitHub's API
            time.sleep(0.1)
        
        df['solution'] = solutions
        
        # Count successful fetches
        successful_fetches = sum(1 for solution in solutions if solution is not None)
        print(f"\nSuccessfully fetched {successful_fetches}/{len(df)} solutions")
    else:
        print(f"\nSkipping solution fetching (--fetch-solutions not set)")
        df['solution'] = [None] * len(df)

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # export df as json
    df.to_json("data/tasks.json", orient='records', lines=True)

    print(f'\nTask data with solutions exported to data/tasks.json')
    return df


def create_category_piechart(df, output_path):
    """Create and save a pie chart of task category distribution."""
    # Calculate category distribution
    category_counts = df['category'].value_counts()
    
    # Group small categories (less than 2% of total) into "Other"
    total_tasks = len(df)
    threshold = 0.02 * total_tasks
    
    main_categories = category_counts[category_counts >= threshold]
    other_categories = category_counts[category_counts < threshold]
    
    if len(other_categories) > 0:
        main_categories['other (misc)'] = other_categories.sum()
    
    # Sort by count for consistent ordering
    main_categories = main_categories.sort_values(ascending=False)
    
    # Create the pie chart
    plt.figure(figsize=(12, 8))
    
    # Create custom colors for better visibility
    colors = plt.cm.Set3(range(len(main_categories)))
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(
        main_categories.values,
        labels=main_categories.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85
    )
    
    # Enhance text readability
    for text in texts:
        text.set_fontsize(10)
        text.set_weight('bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_weight('bold')
    
    # Add title and subtitle
    plt.title('Terminal Bench Task Categories Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.text(0, -1.3, f'Total Tasks: {total_tasks}', ha='center', fontsize=12)
    
    # Add legend with task counts
    legend_labels = [f'{cat}: {count} tasks' for cat, count in main_categories.items()]
    plt.legend(
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )
    
    # Ensure the plot is circular
    plt.axis('equal')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPie chart saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close()

def main():
    """Main function to orchestrate the task category analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Read and visualize task categories from Terminal Bench database')
    parser.add_argument('--fetch-solutions', action='store_true', 
                       help='Fetch solution.sh files from GitHub for each task')
    args = parser.parse_args()
    
    print("Terminal Bench Task Category Analysis")
    print("=" * 80)
    
    # Connect to database
    conn = connect_to_database()
    
    try:
        dataset_name = "terminal-bench"
        dataset_version = "2.0"
        # Extract task categories (works with or without connection)
        df = extract_tasks(conn, dataset_name, dataset_version, fetch_solutions=args.fetch_solutions)
        
        if df is not None and not df.empty:
            # Create output directory if it doesn't exist
            output_dir = Path(os.environ.get("TBENCH_DIR", ".")) / "plots"
            output_dir.mkdir(exist_ok=True)
            
            # Create pie chart
            output_path = output_dir / "task_categories.png"
            create_category_piechart(df, output_path)
            
            # Print summary statistics
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            
            category_counts = df['category'].value_counts()
            print(f"\nTotal number of tasks: {len(df)}")
            print(f"Number of categories: {len(category_counts)}")
            print(f"\nTop 10 categories by task count:")
            for category, count in category_counts.head(10).items():
                percentage = (count / len(df)) * 100
                print(f"  {category:20s}: {count:3d} tasks ({percentage:5.1f}%)")
            
    finally:
        # Supabase client doesn't need explicit connection closing
        if conn:
            print("\nSupabase client session completed.")

if __name__ == "__main__":
    main()