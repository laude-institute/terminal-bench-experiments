"""
Read and visualize task category classifications with multi-judge comparison.
We want to use two claude judges.

The visualization should include a piechart and also print some examples of tasks assigned a each category for paper appendix
"""

import os
import sys
import json
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.task_classes import TASK_CLASSES_RUBRIC, get_category_names, get_skills_list

class TaskCategoryVisualizer:
    def __init__(self):
        self.data_dir = Path(os.environ['TBENCH_DIR']) / 'data'
        self.output_dir = Path(os.environ['TBENCH_DIR']) / 'plots'
        self.output_dir.mkdir(exist_ok=True)
        self.categories = get_category_names()
        self.predefined_skills = get_skills_list()
    
    def get_available_judges(self):
        """Find all judge databases."""
        judge_dbs = list(self.data_dir.glob('task_classifications_*.db'))
        judges = []
        for db_path in judge_dbs:
            judge_id = db_path.stem.replace('task_classifications_', '')
            judges.append({
                'judge_id': judge_id,
                'db_path': db_path
            })
        return judges
    
    def load_classifications(self, judge_id: str) -> pd.DataFrame:
        """Load classifications for a specific judge."""
        db_path = self.data_dir / f'task_classifications_{judge_id}.db'
        
        if not db_path.exists():
            return pd.DataFrame()
        
        conn = sqlite3.connect(db_path)
        
        df = pd.read_sql_query('''
            SELECT 
                task_id,
                task_name,
                task_summary,
                primary_category,
                primary_confidence,
                primary_reasoning,
                secondary_category,
                secondary_confidence,
                secondary_reasoning,
                required_skills,
                llm_model,
                llm_provider,
                response_time_seconds
            FROM task_classifications
            ORDER BY task_name
        ''', conn)
        
        conn.close()
        
        if len(df) > 0:
            # Parse JSON fields
            df['required_skills'] = df['required_skills'].apply(json.loads)
            df['judge_id'] = judge_id
        
        return df
    
    def create_primary_category_comparison(self, all_judge_data: dict):
        """Create comparison of primary categories across judges."""
        
        n_judges = len(all_judge_data)
        if n_judges == 0:
            print("No data to visualize")
            return
        
        # Create subplots for each judge
        fig_cols = min(3, n_judges)
        fig_rows = (n_judges - 1) // fig_cols + 1
        
        fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(6*fig_cols, 6*fig_rows))
        if n_judges == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_judges > 1 else [axes]
        
        for idx, (judge_id, df) in enumerate(all_judge_data.items()):
            if len(df) == 0:
                continue
                
            ax = axes[idx]
            
            # Count primary categories
            primary_counts = df['primary_category'].value_counts()
            
            # Map to display names
            labels = [self.categories.get(cat, cat) for cat in primary_counts.index]
            sizes = primary_counts.values
            
            # Create pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Title with model info
            model_info = df.iloc[0]['llm_model'] if 'llm_model' in df.columns else judge_id
            ax.set_title(f'{judge_id}\n({model_info})\nTotal: {len(df)} tasks')
            
            # Adjust text size
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(8)
        
        # Hide unused subplots
        for idx in range(len(all_judge_data), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Primary Category Distribution - Judge Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure with judge names in filename
        judge_names = "_".join(all_judge_data.keys())
        output_path = self.output_dir / f'primary_category_comparison_{judge_names}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_path}")
        plt.show()
    
    def create_judge_agreement_matrix(self, all_judge_data: dict):
        """Create agreement matrix between judges."""
        
        if len(all_judge_data) < 2:
            print("Need at least 2 judges for agreement analysis")
            return
        
        # Get common tasks across all judges
        judge_ids = list(all_judge_data.keys())
        common_tasks = set(all_judge_data[judge_ids[0]]['task_id'])
        
        for judge_id in judge_ids[1:]:
            common_tasks &= set(all_judge_data[judge_id]['task_id'])
        
        print(f"Found {len(common_tasks)} common tasks across all judges")
        
        if len(common_tasks) == 0:
            print("No common tasks found")
            return
        
        # Create agreement matrix
        n_judges = len(judge_ids)
        agreement_matrix = np.zeros((n_judges, n_judges))
        
        for i, judge1 in enumerate(judge_ids):
            df1 = all_judge_data[judge1]
            df1_common = df1[df1['task_id'].isin(common_tasks)]
            
            for j, judge2 in enumerate(judge_ids):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    df2 = all_judge_data[judge2]
                    df2_common = df2[df2['task_id'].isin(common_tasks)]
                    
                    # Merge on task_id to compare classifications
                    merged = pd.merge(df1_common[['task_id', 'primary_category']], 
                                    df2_common[['task_id', 'primary_category']], 
                                    on='task_id', suffixes=('_1', '_2'))
                    
                    # Calculate agreement
                    agreement = (merged['primary_category_1'] == merged['primary_category_2']).mean()
                    agreement_matrix[i, j] = agreement
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(agreement_matrix, 
                   xticklabels=judge_ids, 
                   yticklabels=judge_ids,
                   annot=True, 
                   fmt='.3f',
                   cmap='Blues',
                   ax=ax)
        
        ax.set_title('Judge Agreement Matrix\n(Primary Category Classifications)')
        plt.tight_layout()
        
        # Save figure with judge names in filename
        judge_names = "_".join(judge_ids)
        output_path = self.output_dir / f'judge_agreement_matrix_{judge_names}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved agreement matrix to: {output_path}")
        plt.show()
    
    def print_category_examples(self, all_judge_data: dict, examples_per_category: int = 5):
        """Print examples of tasks for each category for paper appendix."""
        
        if not all_judge_data:
            print("No judge data available for examples")
            return
        
        # Use the first judge's data for examples (or combine if multiple)
        primary_judge_id = list(all_judge_data.keys())[0]
        df = all_judge_data[primary_judge_id]
        
        print("\n" + "="*80)
        print("TASK EXAMPLES BY CATEGORY")
        print("="*80)
        
        for category_key, category_info in TASK_CLASSES_RUBRIC.items():
            category_name = category_info['name']
            category_tasks = df[df['primary_category'] == category_key]
            
            print(f"\n{category_name.upper()} ({len(category_tasks)} tasks)")
            print("-" * 60)
            print(f"Description: {category_info['description']}")
            
            if len(category_tasks) > 0:
                print(f"\nExample tasks:")
                # Sort by confidence and take top examples
                top_examples = category_tasks.nlargest(examples_per_category, 'primary_confidence')
                
                for i, (_, task) in enumerate(top_examples.iterrows(), 1):
                    print(f"{i}. {task['task_name']}")
                    print(f"   Summary: {task['task_summary']}")
                    print(f"   Confidence: {task['primary_confidence']:.2f}")
                    print(f"   Skills: {', '.join(task['required_skills'][:3])}...")
                    print()
            else:
                print("   No tasks classified in this category")
    
    def create_skills_distribution(self, all_judge_data: dict):
        """Create visualization of skills distribution across categories."""
        
        if not all_judge_data:
            print("No judge data available for skills analysis")
            return
        
        # Combine data from all judges
        all_skills = Counter()
        category_skills = defaultdict(Counter)
        
        for judge_id, df in all_judge_data.items():
            for _, row in df.iterrows():
                skills = row['required_skills']
                category = row['primary_category']
                
                for skill in skills:
                    all_skills[skill] += 1
                    category_skills[category][skill] += 1
        
        # Create skills frequency chart
        top_skills = dict(all_skills.most_common(15))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall skills distribution
        skills_list = list(top_skills.keys())
        counts = list(top_skills.values())
        
        ax1.barh(skills_list, counts)
        ax1.set_xlabel('Number of Tasks')
        ax1.set_title('Most Common Skills Across All Tasks')
        ax1.grid(axis='x', alpha=0.3)
        
        # Skills by category heatmap
        categories = list(category_skills.keys())
        skills_matrix = np.zeros((len(categories), len(skills_list)))
        
        for i, category in enumerate(categories):
            for j, skill in enumerate(skills_list):
                skills_matrix[i, j] = category_skills[category][skill]
        
        # Normalize by row to show relative importance
        skills_matrix_norm = skills_matrix / (skills_matrix.sum(axis=1, keepdims=True) + 1e-8)
        
        sns.heatmap(skills_matrix_norm,
                   xticklabels=[skill.replace('_', ' ') for skill in skills_list],
                   yticklabels=[self.categories.get(cat, cat) for cat in categories],
                   annot=False,
                   cmap='Blues',
                   ax=ax2)
        
        ax2.set_title('Skills Distribution by Category\n(Normalized)')
        ax2.set_xlabel('Skills')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure with judge names in filename
        judge_names = "_".join(all_judge_data.keys())
        output_path = self.output_dir / f'skills_distribution_{judge_names}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved skills distribution to: {output_path}")
        plt.show()


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize task category classifications')
    parser.add_argument('--judges', nargs='+', help='Specific judge IDs to analyze')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples per category')
    args = parser.parse_args()
    
    visualizer = TaskCategoryVisualizer()
    
    # Get available judges
    available_judges = visualizer.get_available_judges()
    
    if not available_judges:
        print("No classification databases found!")
        print("Run build_task_taxonomy.py first to generate classifications.")
        return
    
    print(f"Found {len(available_judges)} judge databases:")
    for judge in available_judges:
        print(f"  - {judge['judge_id']}")
    
    # Filter judges if specified
    if args.judges:
        available_judges = [j for j in available_judges if j['judge_id'] in args.judges]
        print(f"\nFiltered to {len(available_judges)} specified judges")
    
    # Load all judge data
    all_judge_data = {}
    for judge in available_judges:
        judge_id = judge['judge_id']
        df = visualizer.load_classifications(judge_id)
        
        if len(df) > 0:
            all_judge_data[judge_id] = df
            print(f"Loaded {len(df)} classifications from {judge_id}")
        else:
            print(f"No data found for judge {judge_id}")
    
    if not all_judge_data:
        print("No classification data found!")
        return
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Primary category comparison (pie charts)
    visualizer.create_primary_category_comparison(all_judge_data)
    
    # Judge agreement matrix (if multiple judges)
    if len(all_judge_data) > 1:
        visualizer.create_judge_agreement_matrix(all_judge_data)
    
    # Skills distribution
    visualizer.create_skills_distribution(all_judge_data)
    
    # Print category examples for paper
    visualizer.print_category_examples(all_judge_data, args.examples)
    
    print(f"\nPlots saved to: {visualizer.output_dir}")


if __name__ == "__main__":
    main()