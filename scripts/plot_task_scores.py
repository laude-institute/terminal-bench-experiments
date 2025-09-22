#!/usr/bin/env python3
"""
Plot heatmap of reward scores for models across tasks.

This script:
1. Queries the database to extract model names, task names, and reward scores.
2. Computes per-task average rewards for each model.
3. Creates a heatmap visualization showing model performance across tasks.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import argparse
import sys

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
from util_scores import connect_to_database, get_model_task_scores, filter_descriptions


def create_agent_model_heatmap(df, output_path, trial_filter="with_timeouts"):
    """
    Create a heatmap showing agent performance across different models.
    
    Args:
        df: DataFrame with columns [agent_name, model_name, task_name, avg_reward]
        output_path: Path to save the heatmap image
        trial_filter: Trial filtering mode for title display
    """
    print("\n" + "="*80)
    print("Creating agent vs model performance heatmap...")
    print("="*80)
    
    # Calculate average performance for each agent-model combination across all tasks
    agent_model_performance = df.groupby(['agent_name', 'model_name'])['avg_reward'].mean().reset_index()
    
    # Create pivot table with agents as rows and models as columns
    matrix = agent_model_performance.pivot_table(
        index='agent_name',
        columns='model_name',
        values='avg_reward',
        aggfunc='mean'
    )
    
    if matrix.empty:
        print("No data available for agent-model heatmap")
        return False
    
    print(f"Matrix shape: {matrix.shape} (agents x models)")
    
    # Sort models by maximum performance (best to worst), ignoring NaN
    model_max_performance = matrix.max(axis=0, skipna=True).sort_values(ascending=False)
    model_order = model_max_performance.index
    
    # Sort agents by maximum performance (best to worst), ignoring NaN
    agent_max_performance = matrix.max(axis=1, skipna=True).sort_values(ascending=False)
    agent_order = agent_max_performance.index
    
    # Also keep average for reporting
    agent_avg_performance = matrix.mean(axis=1, skipna=True)
    model_avg_performance = matrix.mean(axis=0, skipna=True)
    
    # Reorder matrix with sorted agents and models (keep NaN values)
    reordered_matrix = matrix.loc[agent_order, model_order]
    
    # Calculate figure size based on matrix dimensions
    fig_width = max(10, len(model_order) * 0.8)
    fig_height = max(6, len(agent_order) * 0.8)
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap with NaN values shown as blank
    sns.heatmap(
        reordered_matrix,
        cmap='coolwarm_r',
        vmin=0,
        vmax=1,
        cbar_kws={
            'label': 'Average Reward Score',
            'shrink': 0.8
        },
        linewidths=1,
        linecolor='gray',
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 9},
        mask=reordered_matrix.isna()  # Mask NaN values to show as blank
    )
    
    # Create title with trial filter information
    filter_desc = filter_descriptions.get(trial_filter, trial_filter)
    
    # Customize the plot
    plt.suptitle(f'Agent Performance Across Models ({filter_desc})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.xlabel('Models (sorted by max performance)', fontsize=12)
    plt.ylabel('Agents (sorted by max performance)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Agent-Model heatmap saved to: {output_path}")
    
    # Print insights
    print("\nAgent-Model Performance Insights:")
    print(f"- Best agent by max score: {agent_order[0]} "
          f"(max: {agent_max_performance.loc[agent_order[0]]:.3f}, "
          f"avg: {agent_avg_performance.loc[agent_order[0]]:.3f})")
    print(f"- Best model by max score: {model_order[0]} "
          f"(max: {model_max_performance.loc[model_order[0]]:.3f}, "
          f"avg: {model_avg_performance.loc[model_order[0]]:.3f})")
    
    # Find best agent-model combination
    best_value = reordered_matrix.max().max()
    best_location = reordered_matrix.stack().idxmax()
    print(f"- Best agent-model combination: {best_location[0]} + {best_location[1]} "
          f"(score: {best_value:.3f})")
    
    # Close the figure
    plt.close()
    
    return True


def create_model_task_heatmap(df, output_path, trial_filter="with_timeouts", specific_agent=None):
    """
    Create a heatmap showing model performance across tasks (models x tasks matrix).
    
    Args:
        df: DataFrame with columns [agent_name, model_name, task_name, avg_reward]
        output_path: Path to save the heatmap image
        trial_filter: Trial filtering mode for title display
        specific_agent: If provided, only include data for this agent
    """
    print("\n" + "="*80)
    if specific_agent:
        print(f"Creating model vs task performance heatmap for agent: {specific_agent}...")
    else:
        print("Creating model vs task performance heatmap (all agents)...")
    print("="*80)
    
    # Filter for specific agent if provided
    if specific_agent:
        df = df[df['agent_name'] == specific_agent].copy()
        if df.empty:
            print(f"No data available for agent: {specific_agent}")
            return False
    
    # Calculate average performance for each model-task combination
    # If specific_agent is provided, this will be for that agent only
    # Otherwise, it's across all agents
    model_task_performance = df.groupby(['model_name', 'task_name'])['avg_reward'].mean().reset_index()
    
    # Create pivot table with models as rows and tasks as columns
    matrix = model_task_performance.pivot_table(
        index='model_name',
        columns='task_name', 
        values='avg_reward',
        aggfunc='mean'
    )
    
    if matrix.empty:
        print("No data available for model-task heatmap")
        return False
    
    print(f"Matrix shape: {matrix.shape} (models x tasks)")
    
    # Fill NaN values with 0 for processing
    matrix_filled = matrix.fillna(0)
    
    # Sort tasks by average difficulty (easiest to hardest)
    task_avg_performance = matrix_filled.mean(axis=0).sort_values(ascending=False)
    col_order = [matrix_filled.columns.get_loc(task) for task in task_avg_performance.index]
    
    # Sort models by average performance (best at top)
    model_avg_performance = matrix_filled.mean(axis=1).sort_values(ascending=False)
    row_order = [matrix_filled.index.get_loc(model) for model in model_avg_performance.index]
    
    # Create final matrix with sorted models and tasks
    reordered_matrix = matrix_filled.iloc[row_order, col_order]
    
    # Calculate figure size based on matrix dimensions
    fig_width = min(max(12, matrix_filled.shape[1] * 0.3), 50)
    fig_height = min(max(8, matrix_filled.shape[0] * 0.4), 30)
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(
        reordered_matrix,
        cmap='coolwarm_r',
        vmin=0,
        vmax=1,
        cbar_kws={
            'label': 'Average Reward Score',
            'shrink': 0.8
        },
        linewidths=0.5,
        linecolor='gray',
        xticklabels=True,
        yticklabels=True,
        annot=False
    )
    
    # Create title with trial filter and agent information
    filter_desc = filter_descriptions.get(trial_filter, trial_filter)
    
    # Customize the plot
    if specific_agent:
        title = f'Model Performance Across Tasks - {specific_agent} ({filter_desc})'
    else:
        title = f'Model Performance Across Tasks - All Agents ({filter_desc})'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.xlabel('Tasks (sorted by difficulty: easy → hard →)', fontsize=12)
    plt.ylabel('Models (sorted by performance: best → worst ↓)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Model-Task heatmap saved to: {output_path}")
    
    # Print insights
    print("\nModel-Task Performance Insights:")
    print(f"- Best performing model: {model_avg_performance.index[0]} "
          f"(avg score: {model_avg_performance.iloc[0]:.3f})")
    print(f"- Worst performing model: {model_avg_performance.index[-1]} "
          f"(avg score: {model_avg_performance.iloc[-1]:.3f})")
    print(f"- Easiest task: {task_avg_performance.index[0]} "
          f"(avg score: {task_avg_performance.iloc[0]:.3f})")
    print(f"- Hardest task: {task_avg_performance.index[-1]} "
          f"(avg score: {task_avg_performance.iloc[-1]:.3f})")
    
    # Close the figure
    plt.close()
    
    return True


def main():
    """Main function to generate the task scores heatmap."""
    parser = argparse.ArgumentParser(description="Generate task scores heatmap")
    parser.add_argument("--dataset-name", default="terminal-bench", 
                       help="Dataset name (default: terminal-bench)")
    parser.add_argument("--dataset-version", default="2.0",
                       help="Dataset version (default: 2.0)")
    parser.add_argument("--agent-name", default=None,
                       help="Agent name to filter by (optional)")
    parser.add_argument("--model-name", default=None,
                       help="Model name to filter by (optional)")
    parser.add_argument("--trial-filter", default="with_timeouts",
                       choices=["no_exceptions", "with_timeouts", "with_all_exceptions", "only_timeouts"],
                       help="Which trials to include: 'no_exceptions' (only completed), \
                        'with_timeouts' (completed + timeouts), \
                        'with_all_exceptions' (all), \
                        'only_timeouts' (only timeouts) (default: with_timeouts)")
    # Always produces two heatmaps:
    #   1. Agent x Model performance matrix
    #   2. Model x Task performance matrix
   
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_version = args.dataset_version
    agent_name = args.agent_name
    model_name = args.model_name
    trial_filter = args.trial_filter
    
    print("Terminal Bench Task Scores Heatmap Generator")
    print("=" * 80)
    print(f"Dataset: {dataset_name} v{dataset_version}")
    if agent_name:
        print(f"Agent filter: {agent_name}")
    if model_name:
        print(f"Model filter: {model_name}")
    print("=" * 80)
    
    # Connect to database
    client = connect_to_database()
    
    if client is None:
        print("Failed to connect to database. Exiting.")
        return
    
    try:
        # Get model-task scores - always get all data unless specific filters provided
        if not agent_name and not model_name:
            # Get all data for comprehensive analysis
            df = get_model_task_scores(client, dataset_name=dataset_name, dataset_version=dataset_version, 
                                       trial_filter=trial_filter, agent_name=None, model_name=None)
        else:
            # Get filtered data based on provided filters
            df = get_model_task_scores(client, dataset_name=dataset_name, dataset_version=dataset_version, 
                                       trial_filter=trial_filter, agent_name=agent_name, model_name=model_name)
        
        if df is not None and not df.empty:
            # Create output directory if it doesn't exist
            output_dir = Path(os.environ.get("TBENCH_DIR", ".")) / "plots"
            output_dir.mkdir(exist_ok=True)
            
            # Generate agent x model and model x task heatmaps
            filter_suffix_map = {
                "no_exceptions": "_successful_only",
                "with_timeouts": "_with_timeouts",
                "with_all_exceptions": "_all_trials",
                "only_timeouts": "_only_timeouts"
            }
            filename_suffix = filter_suffix_map.get(trial_filter, f"_{trial_filter}")
            
            # 1. Create agent x model heatmap
            agent_model_path = output_dir / f"agent_model_performance{filename_suffix}.png"
            create_agent_model_heatmap(df, agent_model_path, trial_filter)
            
            # 2. Create model x task heatmap (aggregated across all agents)
            model_task_path = output_dir / f"model_task_performance{filename_suffix}.png"
            create_model_task_heatmap(df, model_task_path, trial_filter)
            
            # 3. Create individual model x task heatmaps for each agent
            if not agent_name:  # Only create per-agent heatmaps if no specific agent was requested
                unique_agents = df['agent_name'].unique()
                print(f"\nCreating individual model-task heatmaps for {len(unique_agents)} agents...")
                for agent in sorted(unique_agents):
                    agent_model_task_path = output_dir / f"model_task_performance_{agent}{filename_suffix}.png"
                    print(f"  - Creating heatmap for {agent}...")
                    create_model_task_heatmap(df, agent_model_task_path, trial_filter, specific_agent=agent)
            elif agent_name:
                # If specific agent requested, also create that agent's individual heatmap
                agent_model_task_path = output_dir / f"model_task_performance_{agent_name}{filename_suffix}.png"
                print(f"\nCreating individual model-task heatmap for {agent_name}...")
                create_model_task_heatmap(df, agent_model_task_path, trial_filter, specific_agent=agent_name)
            
            print("\n" + "="*80)
            print("Heatmap generation complete!")
            print("="*80)
        else:
            print("No data available to create heatmap.")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Supabase client doesn't need explicit connection closing
        if client:
            print("\nSupabase client session completed.")


if __name__ == "__main__":
    main()