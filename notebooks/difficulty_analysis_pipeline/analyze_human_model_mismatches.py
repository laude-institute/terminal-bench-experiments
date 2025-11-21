#!/usr/bin/env python3
"""
Analyze mismatches between human difficulty assessments and model performance.
"""

import pandas as pd
import yaml
from pathlib import Path

def load_human_difficulty_labels():
    human_labels = {}
    # TODO: Set path to your terminal-bench repository
    terminal_bench_path = Path.home() / "path/to/terminal-bench" / "tasks"
    
    if terminal_bench_path.exists():
        for task_dir in terminal_bench_path.iterdir():
            if not task_dir.is_dir():
                continue
            task_yaml_path = task_dir / "task.yaml"
            if not task_yaml_path.exists():
                continue
            try:
                with open(task_yaml_path, 'r') as f:
                    task_config = yaml.safe_load(f)
                task_name = task_dir.name
                difficulty = task_config.get('difficulty', 'unknown')
                if difficulty != 'unknown':
                    human_labels[task_name] = difficulty.lower()
            except:
                continue
    
    return human_labels

def load_data():
    difficulty_file = "difficulty_analysis_results/task_difficulty.csv"
    if not Path(difficulty_file).exists():
        return None, None, None
    
    model_df = pd.read_csv(difficulty_file)
    human_labels = load_human_difficulty_labels()
    
    time_file = "../time_analysis_pipeline/time_analysis_results/human_model_time_comparison.csv"
    human_df = None
    if Path(time_file).exists():
        human_df = pd.read_csv(time_file)
    
    return model_df, human_df, human_labels

def analyze_all_tasks(model_df, human_labels):
    model_df_enhanced = model_df.copy()
    model_df_enhanced['human_difficulty'] = model_df_enhanced['task_name'].map(human_labels)
    
    with_human_labels = model_df_enhanced[model_df_enhanced['human_difficulty'].notna()].copy()
    without_human_labels = model_df_enhanced[model_df_enhanced['human_difficulty'].isna()].copy()
    universal_failures = model_df_enhanced[model_df_enhanced['model_resolve_rate'] == 0.0].copy()
    
    if len(with_human_labels) > 0:
        difficulty_order = {'easy': 1, 'medium': 2, 'hard': 3}
        with_human_labels['model_difficulty_num'] = with_human_labels['model_difficulty'].map(difficulty_order)
        with_human_labels['human_difficulty_num'] = with_human_labels['human_difficulty'].map(difficulty_order)
        with_human_labels['mismatch_score'] = with_human_labels['human_difficulty_num'] - with_human_labels['model_difficulty_num']
        
        with_human_labels['mismatch_type'] = 'aligned'
        with_human_labels.loc[with_human_labels['mismatch_score'] > 0, 'mismatch_type'] = 'model_easier'
        with_human_labels.loc[with_human_labels['mismatch_score'] < 0, 'mismatch_type'] = 'human_easier'
    
    return model_df_enhanced, with_human_labels, without_human_labels, universal_failures

def create_edge_case_record(row, case_type, human_df=None):
    task_name = row['task_name']
    
    expert_time = 'N/A'
    junior_time = 'N/A'
    if human_df is not None:
        human_time_data = human_df[human_df['task_name'] == task_name]
        if len(human_time_data) > 0:
            expert_time = human_time_data.iloc[0].get('expert_time_estimate_min', 'N/A')
            junior_time = human_time_data.iloc[0].get('junior_time_estimate_min', 'N/A')
    
    resolving_models = eval(row['resolving_models']) if row['resolving_models'] != '[]' else []
    
    return {
        'type': case_type,
        'task_name': task_name,
        'human_difficulty': row.get('human_difficulty', 'unknown'),
        'model_difficulty': row['model_difficulty'],
        'model_resolve_rate': row['model_resolve_rate'],
        'models_that_solve': len(resolving_models),
        'total_models_tested': row['total_models_tested'],
        'expert_time_min': expert_time,
        'junior_time_min': junior_time,
        'mismatch_score': row.get('mismatch_score', 'N/A')
    }

def find_edge_cases(all_tasks_df, with_human_labels_df, without_human_labels_df, universal_failures_df, human_df=None):
    edge_cases = []
    
    # Universal failures with human labels
    universal_with_human = universal_failures_df[universal_failures_df['human_difficulty'].notna()]
    for _, row in universal_with_human.iterrows():
        edge_case = create_edge_case_record(row, 'universal_failure_with_human_label', human_df)
        edge_cases.append(edge_case)
    
    # Universal failures without human labels
    universal_without_human = universal_failures_df[universal_failures_df['human_difficulty'].isna()]
    for _, row in universal_without_human.iterrows():
        edge_case = create_edge_case_record(row, 'universal_failure_no_human_label', human_df)
        edge_cases.append(edge_case)
    
    if len(with_human_labels_df) > 0:
        # Tasks where models find it easier than humans
        model_easier = with_human_labels_df[with_human_labels_df['mismatch_type'] == 'model_easier'].copy()
        model_easier = model_easier.sort_values('mismatch_score', ascending=False)
        for _, row in model_easier.iterrows():
            edge_case = create_edge_case_record(row, 'model_easier', human_df)
            edge_cases.append(edge_case)
        
        # Tasks where humans find it easier than models
        human_easier = with_human_labels_df[with_human_labels_df['mismatch_type'] == 'human_easier'].copy()
        human_easier = human_easier.sort_values('mismatch_score', ascending=True)
        for _, row in human_easier.iterrows():
            edge_case = create_edge_case_record(row, 'human_easier', human_df)
            edge_cases.append(edge_case)
    
    return edge_cases

def save_results(edge_cases, all_tasks_df):
    output_dir = Path("difficulty_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save edge cases
    edge_cases_df = pd.DataFrame(edge_cases)
    edge_cases_file = output_dir / "comprehensive_human_model_mismatches.csv"
    edge_cases_df.to_csv(edge_cases_file, index=False)
    
    # Save full analysis
    all_tasks_file = output_dir / "all_tasks_human_model_comparison.csv"
    all_tasks_df.to_csv(all_tasks_file, index=False)
    
    return len(edge_cases)

def main():
    model_df, human_df, human_labels = load_data()
    if model_df is None:
        return
    
    all_tasks_df, with_human_labels_df, without_human_labels_df, universal_failures_df = analyze_all_tasks(model_df, human_labels)
    
    edge_cases = find_edge_cases(all_tasks_df, with_human_labels_df, without_human_labels_df, universal_failures_df, human_df)
    
    edge_case_count = save_results(edge_cases, all_tasks_df)
    
    print(f"Analyzed {len(all_tasks_df)} tasks, found {edge_case_count} edge cases")

if __name__ == "__main__":
    main()
