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
from name_remapping import model_names, agent_name_map


def create_agent_model_heatmap(df, output_path, trial_filter="with_timeouts"):
    """
    Create a heatmap showing model performance across different agents (transposed).
    
    Args:
        df: DataFrame with columns [agent_name, model_name, task_name, avg_reward]
        output_path: Path to save the heatmap image
        trial_filter: Trial filtering mode for title display
    """
    print("\n" + "="*80)
    print("Creating model vs agent performance heatmap (transposed)...")
    print("="*80)
    
    # Calculate average performance for each agent-model combination across all tasks
    agent_model_performance = df.groupby(['agent_name', 'model_name'])['avg_reward'].mean().reset_index()
    
    # Convert to failure rate (1 - avg_reward)
    agent_model_performance['failure_rate'] = 1 - agent_model_performance['avg_reward']
    
    # Apply name remapping
    agent_model_performance['model_display_name'] = agent_model_performance['model_name'].map(model_names).fillna(agent_model_performance['model_name'])
    agent_model_performance['agent_display_name'] = agent_model_performance['agent_name'].map(agent_name_map).fillna(agent_model_performance['agent_name'])
    
    # Create pivot table with models as rows and agents as columns (transposed)
    matrix = agent_model_performance.pivot_table(
        index='model_display_name',
        columns='agent_display_name',
        values='failure_rate',
        aggfunc='mean'
    )
    
    if matrix.empty:
        print("No data available for agent-model heatmap")
        return False
    
    print(f"Matrix shape: {matrix.shape} (models x agents)")
    
    # Sort agents by minimum failure rate (best to worst), ignoring NaN
    agent_min_failure = matrix.min(axis=0, skipna=True).sort_values(ascending=True)
    agent_order = agent_min_failure.index
    
    # Sort models by minimum failure rate (best to worst), ignoring NaN
    model_min_failure = matrix.min(axis=1, skipna=True).sort_values(ascending=True)
    model_order = model_min_failure.index
    
    # Also keep average for reporting
    model_avg_failure = matrix.mean(axis=1, skipna=True)
    agent_avg_failure = matrix.mean(axis=0, skipna=True)
    
    # Reorder matrix with sorted models and agents (keep NaN values)
    reordered_matrix = matrix.loc[model_order, agent_order]
    
    # Calculate figure size based on matrix dimensions
    fig_width = max(12, len(agent_order) * 1.2)
    fig_height = max(8, len(model_order) * 0.8)
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap with NaN values shown as blank
    ax = sns.heatmap(
        reordered_matrix,
        cmap='RdBu_r',  # Seaborn's RdBu_r: red for high failure rate, blue for low failure rate
        vmin=0,
        vmax=1,
        cbar_kws={
            'label': 'Failure Rate',
            'shrink': 0.8,
            'orientation': 'horizontal',
            'pad': 0.1
        },
        linewidths=1.5,
        linecolor='white',
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 12, 'weight': 'bold'},
        mask=reordered_matrix.isna()  # Mask NaN values to show as blank
    )
    
    # Move colorbar to top
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Create title with trial filter information
    filter_desc = filter_descriptions.get(trial_filter, trial_filter)
    
    # Customize the plot
    plt.suptitle(f'Model Performance Across Agents ({filter_desc})', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.xlabel('Agents (sorted by best performance)', fontsize=14, fontweight='bold')
    plt.ylabel('Models (sorted by best performance)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Agent-Model heatmap saved to: {output_path}")
    
    # Print insights
    print("\nModel-Agent Performance Insights:")
    print(f"- Best agent by min failure rate: {agent_order[0]} "
          f"(min: {agent_min_failure.loc[agent_order[0]]:.3f}, "
          f"avg: {agent_avg_failure.loc[agent_order[0]]:.3f})")
    print(f"- Best model by min failure rate: {model_order[0]} "
          f"(min: {model_min_failure.loc[model_order[0]]:.3f}, "
          f"avg: {model_avg_failure.loc[model_order[0]]:.3f})")
    
    # Find best model-agent combination (lowest failure rate)
    best_value = reordered_matrix.min().min()
    best_location = reordered_matrix.stack().idxmin()
    print(f"- Best model-agent combination: {best_location[0]} + {best_location[1]} "
          f"(failure rate: {best_value:.3f})")
    
    # Close the figure
    plt.close()
    
    return True


def create_model_task_heatmap(df, output_path, trial_filter="with_timeouts", specific_agent=None):
    """
    Create a heatmap showing task performance across models (tasks x models matrix - transposed).
    
    Args:
        df: DataFrame with columns [agent_name, model_name, task_name, avg_reward]
        output_path: Path to save the heatmap image
        trial_filter: Trial filtering mode for title display
        specific_agent: If provided, only include data for this agent
    """
    print("\n" + "="*80)
    if specific_agent:
        print(f"Creating task vs model performance heatmap for agent: {specific_agent} (transposed)...")
    else:
        print("Creating task vs model performance heatmap (all agents, transposed)...")
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
    
    # Convert to failure rate (1 - avg_reward)
    model_task_performance['failure_rate'] = 1 - model_task_performance['avg_reward']
    
    # Apply model name remapping
    model_task_performance['model_display_name'] = model_task_performance['model_name'].map(model_names).fillna(model_task_performance['model_name'])
    
    # Create pivot table with tasks as rows and models as columns (transposed)
    matrix = model_task_performance.pivot_table(
        index='task_name',
        columns='model_display_name', 
        values='failure_rate',
        aggfunc='mean'
    )
    
    if matrix.empty:
        print("No data available for model-task heatmap")
        return False
    
    print(f"Matrix shape: {matrix.shape} (tasks x models)")
    
    # Fill NaN values with 1 for processing (highest failure rate for missing data)
    matrix_filled = matrix.fillna(1)
    
    # Sort models by average failure rate (best to worst)
    model_avg_failure = matrix_filled.mean(axis=0).sort_values(ascending=True)
    col_order = model_avg_failure.index
    
    # Sort tasks by average difficulty (hardest to easiest - highest failure rate first)
    task_avg_failure = matrix_filled.mean(axis=1).sort_values(ascending=False)
    row_order = task_avg_failure.index
    
    # Create final matrix with sorted tasks and models
    reordered_matrix = matrix_filled.loc[row_order, col_order]
    
    # Calculate figure size based on matrix dimensions
    fig_width = min(max(14, matrix_filled.shape[1] * 0.8), 50)
    fig_height = min(max(10, matrix_filled.shape[0] * 0.3), 30)
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create heatmap
    ax = sns.heatmap(
        reordered_matrix,
        cmap='RdBu_r',  # Seaborn's RdBu_r: red for high failure rate, blue for low failure rate
        vmin=0,
        vmax=1,
        cbar_kws={
            'label': 'Failure Rate',
            'shrink': 0.8,
            'orientation': 'horizontal',
            'pad': 0.1
        },
        linewidths=1.0,
        linecolor='white',
        xticklabels=True,
        yticklabels=True,
        annot=False
    )
    
    # Move colorbar to top
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Create title with trial filter and agent information
    filter_desc = filter_descriptions.get(trial_filter, trial_filter)
    
    # Customize the plot
    if specific_agent:
        # Apply agent name mapping for title
        agent_display_name = agent_name_map.get(specific_agent, specific_agent)
        title = f'Task Performance Across Models - {agent_display_name} ({filter_desc})'
    else:
        title = f'Task Performance Across Models - All Agents ({filter_desc})'
    plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    plt.xlabel('Models (sorted by performance: best → worst →)', fontsize=14, fontweight='bold')
    plt.ylabel('Tasks (sorted by difficulty: hard → easy ↓)', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Model-Task heatmap saved to: {output_path}")
    
    # Print insights
    print("\nTask-Model Performance Insights:")
    print(f"- Best performing model: {model_avg_failure.index[0]} "
          f"(avg failure rate: {model_avg_failure.iloc[0]:.3f})")
    print(f"- Worst performing model: {model_avg_failure.index[-1]} "
          f"(avg failure rate: {model_avg_failure.iloc[-1]:.3f})")
    print(f"- Hardest task: {task_avg_failure.index[0]} "
          f"(avg failure rate: {task_avg_failure.iloc[0]:.3f})")
    print(f"- Easiest task: {task_avg_failure.index[-1]} "
          f"(avg failure rate: {task_avg_failure.iloc[-1]:.3f})")
    
    # Close the figure
    plt.close()
    
    return True


def create_trial_count_heatmap(df, output_path, trial_filter="with_timeouts"):
    """
    Create a heatmap showing number of trials per task per agent/model combination.
    
    Args:
        df: DataFrame with columns [agent_name, model_name, task_name, num_trials]
        output_path: Path to save the heatmap image
        trial_filter: Trial filtering mode for title display
    """
    print("\n" + "="*80)
    print("Creating trial count heatmap...")
    print("="*80)
    
    if 'num_trials' not in df.columns:
        print("No num_trials column found in data - cannot create trial count heatmap")
        return False
    
    # Create agent_model combination column
    df['agent_model'] = df['agent_name'] + ' + ' + df['model_name']
    
    # Create pivot table with tasks as columns and agent_model as rows
    matrix = df.pivot_table(
        index='agent_model',
        columns='task_name',
        values='num_trials',
        aggfunc='first',  # Should be only one value per combination
        fill_value=0
    )
    
    if matrix.empty:
        print("No data available for trial count heatmap")
        return False
    
    print(f"Matrix shape: {matrix.shape} (agent_model combinations x tasks)")
    print(f"Total non-zero entries: {(matrix > 0).sum().sum()}")
    print(f"Max trials per combination: {matrix.max().max()}")
    print(f"Min trials per combination (excluding 0): {matrix[matrix > 0].min().min()}")
    
    # Sort tasks by total number of trials (most trials first)
    task_totals = matrix.sum(axis=0).sort_values(ascending=False)
    col_order = task_totals.index
    
    # Sort agent_model combinations by total number of trials (most trials first)
    agent_model_totals = matrix.sum(axis=1).sort_values(ascending=False)
    row_order = agent_model_totals.index
    
    # Reorder matrix
    reordered_matrix = matrix.loc[row_order, col_order]
    
    # Calculate figure size based on matrix dimensions
    fig_width = min(max(15, matrix.shape[1] * 0.2), 50)
    fig_height = min(max(10, matrix.shape[0] * 0.3), 30)
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Using seaborn's Blues colormap which handles 0 values well
    
    # Create heatmap
    ax = sns.heatmap(
        reordered_matrix,
        cmap='Blues',  # Seaborn's Blues colormap
        vmin=0,
        vmax=reordered_matrix.max().max(),
        cbar_kws={
            'label': 'Number of Trials',
            'shrink': 0.8,
            'orientation': 'horizontal',
            'pad': 0.1
        },
        linewidths=1.0,
        linecolor='white',
        xticklabels=True,
        yticklabels=True,
        annot=False,  # Don't annotate due to size
        fmt='d'
    )
    
    # Move colorbar to top
    cbar = ax.collections[0].colorbar
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    # Create title with trial filter information
    filter_desc = filter_descriptions.get(trial_filter, trial_filter)
    
    # Customize the plot
    plt.suptitle(f'Number of Trials per Task per Agent/Model Combination ({filter_desc})', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.xlabel('Tasks (sorted by total trial count: high → low →)', fontsize=12)
    plt.ylabel('Agent + Model Combinations (sorted by total trial count: high → low ↓)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Trial count heatmap saved to: {output_path}")
    
    # Print insights
    print("\nTrial Count Insights:")
    print(f"- Task with most trials: {task_totals.index[0]} ({int(task_totals.iloc[0])} total trials)")
    print(f"- Task with fewest trials: {task_totals.index[-1]} ({int(task_totals.iloc[-1])} total trials)")
    print(f"- Agent/model with most trials: {agent_model_totals.index[0]} ({int(agent_model_totals.iloc[0])} total trials)")
    print(f"- Agent/model with fewest trials: {agent_model_totals.index[-1]} ({int(agent_model_totals.iloc[-1])} total trials)")
    
    # Find the combination with max trials
    max_trials = reordered_matrix.max().max()
    max_loc = reordered_matrix.stack().idxmax()
    print(f"- Max trials for single combination: {max_loc[0]} × {max_loc[1]} ({int(max_trials)} trials)")
    
    # Show trial count distribution
    all_trials = reordered_matrix.values.flatten()
    unique_counts = np.unique(all_trials[all_trials > 0])
    print(f"- Trial count values seen: {unique_counts.tolist()}")
    
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
            
            # 1. Create model x agent heatmap (transposed)
            model_agent_path = output_dir / f"model_agent_performance{filename_suffix}.png"
            create_agent_model_heatmap(df, model_agent_path, trial_filter)
            
            # 2. Create task x model heatmap (transposed, aggregated across all agents)
            task_model_path = output_dir / f"task_model_performance{filename_suffix}.png"
            create_model_task_heatmap(df, task_model_path, trial_filter)
            
            # 3. Create individual task x model heatmaps for each agent (transposed)
            if not agent_name:  # Only create per-agent heatmaps if no specific agent was requested
                unique_agents = df['agent_name'].unique()
                print(f"\nCreating individual task-model heatmaps for {len(unique_agents)} agents...")
                for agent in sorted(unique_agents):
                    agent_task_model_path = output_dir / f"task_model_performance_{agent}{filename_suffix}.png"
                    print(f"  - Creating heatmap for {agent}...")
                    create_model_task_heatmap(df, agent_task_model_path, trial_filter, specific_agent=agent)
            elif agent_name:
                # If specific agent requested, also create that agent's individual heatmap
                agent_task_model_path = output_dir / f"task_model_performance_{agent_name}{filename_suffix}.png"
                print(f"\nCreating individual task-model heatmap for {agent_name}...")
                create_model_task_heatmap(df, agent_task_model_path, trial_filter, specific_agent=agent_name)
            
            # 4. Create trial count heatmap
            trial_count_path = output_dir / f"trial_counts{filename_suffix}.png"
            create_trial_count_heatmap(df, trial_count_path, trial_filter)
            
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