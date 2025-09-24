#!/usr/bin/env python3
"""
Simplified plotting utilities for Terminal Bench failure analysis.
"""

from pathlib import Path
from typing import Dict, Optional, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style once
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Constants
MAST_CATEGORIES = {
    "Specification Issues": ["1.1", "1.2", "1.3", "1.4", "1.5"],
    "Communication Misalignment": ["2.1", "2.2", "2.3", "2.4", "2.5", "2.6"],
    "Task Verification": ["3.1", "3.2", "3.3"]
}

MAST_LABELS = {
    "1.1": "Disobey task specification",
    "1.2": "Disobey role specification",
    "1.3": "Step repetition",
    "1.4": "Loss of conversation history",
    "1.5": "Unaware of termination conditions",
    "2.1": "Conversation reset",
    "2.2": "Fail to ask for clarification",
    "2.3": "Task derailment",
    "2.4": "Information withholding",
    "2.5": "Ignored previous outputs",
    "2.6": "Reasoning-action mismatch",
    "3.1": "Premature termination",
    "3.2": "No or incomplete verification",
    "3.3": "Incorrect verification"
}

CATEGORY_COLORS = {
    "Specification Issues": "#FF1744",      # Red
    "Communication Misalignment": "#2196F3", # Blue
    "Task Verification": "#00E676"          # Green
}


class FailureDataProcessor:
    """Handles data transformation for failure analysis."""
    
    def __init__(self, categories: Dict = None, labels: Dict = None):
        self.categories = categories
        self.labels = labels
        self.colors = self._generate_color_map()
    
    def _generate_color_map(self) -> Dict[str, str]:
        """Generate colors for each failure code based on category."""
        color_map = {}
        
        if self.categories:
            # Use provided categories
            for i, (category, codes) in enumerate(self.categories.items()):
                # Get base color for category
                if category in CATEGORY_COLORS:
                    base_color = CATEGORY_COLORS[category]
                else:
                    # Generate distinct colors for custom categories
                    palette = sns.color_palette("husl", len(self.categories))
                    base_color = palette[i]
                
                n_codes = len(codes)
                
                # Create color shades for subcategories
                if n_codes > 1:
                    shades = sns.light_palette(base_color, n_colors=n_codes+2)[1:-1][::-1]
                else:
                    shades = [base_color]
                
                for j, code in enumerate(codes):
                    color_map[code] = shades[j]
        else:
            # No categories provided - generate unique colors for each code
            # First collect all unique codes from test data if available
            color_map = {}  # Will be populated dynamically in to_dataframe
        
        return color_map
    
    def to_dataframe(self, failure_counts: Dict[str, Dict[str, int]]) -> pd.DataFrame:
        """Convert failure counts to DataFrame with percentages."""
        rows = []
        
        # Collect all unique failure codes if no predefined colors
        if not self.colors and not self.categories:
            all_codes = set()
            for counts in failure_counts.values():
                all_codes.update(counts.keys())
            # Generate colors for all codes
            palette = sns.color_palette("husl", len(all_codes))
            self.colors = {code: palette[i] for i, code in enumerate(sorted(all_codes))}
        
        for model, counts in failure_counts.items():
            total = sum(counts.values())
            if total == 0:
                continue
                
            for code, count in counts.items():
                if count > 0:
                    # Find category for this code
                    category = None
                    if self.categories:
                        for cat_name, cat_codes in self.categories.items():
                            if code in cat_codes:
                                category = cat_name
                                break
                    
                    # Use labels if provided, otherwise use code as label
                    if self.labels:
                        label = self.labels.get(code, code)
                    else:
                        label = code
                    
                    rows.append({
                        'model': model,
                        'failure_code': code,
                        'failure_mode': label,
                        'category': category or "General",  # Default category name
                        'count': count,
                        'percentage': (count / total) * 100,
                        'color': self.colors.get(code, "#808080")
                    })
        
        df = pd.DataFrame(rows)
        
        if not df.empty and self.categories:
            # Sort DataFrame by category and failure code for consistent ordering
            # Create category order
            category_order = list(self.categories.keys())
            df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)
            
            # Create failure code order within categories
            code_order = []
            for cat_codes in self.categories.values():
                code_order.extend(cat_codes)
            df['failure_code'] = pd.Categorical(df['failure_code'], categories=code_order, ordered=True)
            
            # Sort by category first, then by failure code
            df = df.sort_values(['category', 'failure_code'])
        
        return df


class FailurePlotter:
    """Creates various failure analysis plots."""
    
    def __init__(self, processor: FailureDataProcessor = None, 
                 categories: Dict = None, labels: Dict = None):
        """Initialize plotter with optional processor or create one.
        
        Args:
            processor: Existing FailureDataProcessor instance
            categories: Category definitions if creating new processor
            labels: Label definitions if creating new processor
        """
        if processor:
            self.processor = processor
        elif categories or labels:
            self.processor = FailureDataProcessor(categories, labels)
        else:
            # Default to MAST if nothing provided
            self.processor = FailureDataProcessor(MAST_CATEGORIES, MAST_LABELS)
    
    def plot_by_category(self, 
                         failure_counts: Dict[str, Dict[str, int]], 
                         title: str = "Failure Analysis",
                         save_path: Optional[Path] = None):
        """Create subplots for each failure category with grouped legends."""
        df = self.processor.to_dataframe(failure_counts)
        if df.empty:
            print(f"No data to plot for {title}")
            return
        
        categories = df['category'].unique()
        n_cats = len(categories)
        
        fig, axes = plt.subplots(n_cats, 1, figsize=(12, 4*n_cats), sharex=True)
        if n_cats == 1:
            axes = [axes]
        
        for ax, category in zip(axes, categories):
            cat_df = df[df['category'] == category]
            
            # Get unique colors for this category's failure modes
            palette = {mode: cat_df[cat_df['failure_mode'] == mode].iloc[0]['color'] 
                      for mode in cat_df['failure_mode'].unique()}
            
            sns.barplot(data=cat_df, x='model', y='percentage', 
                       hue='failure_mode', ax=ax, palette=palette)
            
            ax.set_title(category, fontweight='bold', fontsize=14)
            ax.set_ylabel("Percentage (%)", fontsize=12)
            
            # Create grouped legend if we have categories and labels
            if self.processor.categories and self.processor.labels:
                handles, labels = ax.get_legend_handles_labels()
                
                # For each subplot, show only the relevant subcategory items
                # but maintain the grouped structure
                grouped_handles = []
                grouped_labels = []
                
                # Find which category codes are in this subplot
                if category in self.processor.categories:
                    category_codes = self.processor.categories[category]
                    
                    for code in category_codes:
                        failure_mode_label = self.processor.labels.get(code, code)
                        if failure_mode_label in labels:
                            idx = labels.index(failure_mode_label)
                            grouped_handles.append(handles[idx])
                            grouped_labels.append(failure_mode_label)
                    
                    ax.legend(grouped_handles, grouped_labels,
                             bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
                else:
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            else:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
            
            ax.tick_params(axis='x', rotation=45, labelsize=12)
        
        plt.suptitle(title, fontweight='bold', fontsize=18)
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def plot_combined(self,
                     failure_counts: Dict[str, Dict[str, int]],
                     title: str = "Failure Analysis",
                     save_path: Optional[Path] = None):
        """Create single plot with all failure modes."""
        df = self.processor.to_dataframe(failure_counts)
        if df.empty:
            print(f"No data to plot for {title}")
            return
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Create ordered list of failure modes based on category order
        if self.processor.categories:
            # Order failure modes by category
            ordered_modes = []
            for cat_codes in self.processor.categories.values():
                for code in cat_codes:
                    label = self.processor.labels.get(code, code) if self.processor.labels else code
                    if label not in ordered_modes and label in df['failure_mode'].unique():
                        ordered_modes.append(label)
        else:
            # Use natural order from sorted dataframe
            ordered_modes = df['failure_mode'].unique()
        
        # Pivot data with ordered columns
        pivot_df = df.pivot_table(index='model', columns='failure_mode', 
                                  values='percentage', fill_value=0)
        
        # Reorder columns based on category grouping
        pivot_df = pivot_df[ordered_modes]
        
        # Get colors for all failure modes in the correct order
        colors = []
        for mode in pivot_df.columns:
            mode_df = df[df['failure_mode'] == mode]
            if not mode_df.empty:
                colors.append(mode_df.iloc[0]['color'])
            else:
                colors.append("#808080")
        
        # Create grouped bar chart
        pivot_df.plot(kind='bar', stacked=False, ax=ax, width=0.6, color=colors)
        
        # Customize with larger fonts (matching original v2)
        ax.set_ylabel("Percentage of Failures (%)", fontsize=16)
        ax.set_xlabel("Model", fontsize=16)
        ax.set_title(title, fontweight='bold', fontsize=20, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Create grouped legend if we have categories
        if self.processor.categories:
            handles, labels = ax.get_legend_handles_labels()
            
            # Group legend items by category
            grouped_handles = []
            grouped_labels = []
            
            for category_name, category_codes in self.processor.categories.items():
                # Add category title
                category_added = False
                
                # Add subcategories for this category
                for code in category_codes:
                    failure_mode_label = self.processor.labels.get(code, code) if self.processor.labels else code
                    if failure_mode_label in labels:
                        if not category_added:
                            grouped_handles.append(plt.Rectangle((0,0),1,1,fc="none", edgecolor="none"))
                            grouped_labels.append(f"\n{category_name}")
                            category_added = True
                        idx = labels.index(failure_mode_label)
                        grouped_handles.append(handles[idx])
                        grouped_labels.append(f"  {failure_mode_label}")
            
            # Create grouped legend with larger fonts
            legend = ax.legend(grouped_handles, grouped_labels, 
                              bbox_to_anchor=(1.05, 1), loc='upper left',
                              frameon=True, fontsize=12, title_fontsize=14)
            
            # Style category titles in legend
            for i, text in enumerate(legend.get_texts()):
                if text.get_text().startswith('\n'):  # Category titles
                    text.set_weight('bold')
                    text.set_fontsize(14)
        else:
            # Simple legend if no categories
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def plot_aggregated(self,
                       failure_counts: Dict[str, Dict[str, int]],
                       title: str = "Failure Analysis by Category",
                       save_path: Optional[Path] = None):
        """Create plot with failures aggregated by category."""
        df = self.processor.to_dataframe(failure_counts)
        if df.empty:
            print(f"No data to plot for {title}")
            return
        
        # Aggregate by category
        agg_df = df.groupby(['model', 'category'])['percentage'].sum().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        # Use category colors if available, otherwise generate palette
        categories = agg_df['category'].unique()
        if all(cat in CATEGORY_COLORS for cat in categories):
            palette = {cat: CATEGORY_COLORS[cat] for cat in categories}
        else:
            # Generate distinct colors for all categories
            colors = sns.color_palette("husl", len(categories))
            palette = {cat: colors[i] for i, cat in enumerate(categories)}
        
        sns.barplot(data=agg_df, x='model', y='percentage',
                   hue='category', palette=palette)
        
        plt.title(title, fontweight='bold', fontsize=16)
        plt.ylabel("Percentage (%)")
        plt.xlabel("Model")
        plt.legend(title="Category")
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def plot_comparison(self,
                       data1: Dict[str, Dict[str, int]],
                       data2: Dict[str, Dict[str, int]],
                       labels: Tuple[str, str] = ("Set 1", "Set 2"),
                       title: str = "Failure Comparison",
                       save_path: Optional[Path] = None):
        """Create side-by-side comparison plots with a single shared legend."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
        
        all_handles = []
        all_labels = []
        
        for ax, data, label in zip([ax1, ax2], [data1, data2], labels):
            df = self.processor.to_dataframe(data)
            
            if df.empty:
                ax.set_title(f"{label} (No Data)", fontweight='bold', fontsize=16)
                continue
            
            palette = {mode: df[df['failure_mode'] == mode].iloc[0]['color']
                      for mode in df['failure_mode'].unique()}
            
            sns.barplot(data=df, x='model', y='percentage',
                       hue='failure_mode', ax=ax, palette=palette)
            
            ax.set_title(label, fontweight='bold', fontsize=16)
            ax.set_ylabel("Percentage of Failures (%)", fontsize=14)
            ax.set_xlabel("Model", fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            # Collect handles and labels for shared legend
            if not all_handles:  # Only collect from first plot
                handles, labels_list = ax.get_legend_handles_labels()
                all_handles = handles
                all_labels = labels_list
            
            # Remove individual subplot legends
            ax.get_legend().remove()
        
        # Create single grouped legend if we have categories
        if self.processor.categories and all_handles:
            # Group legend items by category
            grouped_handles = []
            grouped_labels = []
            
            for category_name, category_codes in self.processor.categories.items():
                # Check if any codes from this category are in the plot
                category_has_data = False
                for code in category_codes:
                    failure_mode_label = self.processor.labels.get(code, code) if self.processor.labels else code
                    if failure_mode_label in all_labels:
                        category_has_data = True
                        break
                
                if category_has_data:
                    # Add category title
                    grouped_handles.append(plt.Rectangle((0,0),1,1,fc="none", edgecolor="none"))
                    grouped_labels.append(f"\n{category_name}")
                    
                    # Add subcategories for this category
                    for code in category_codes:
                        failure_mode_label = self.processor.labels.get(code, code) if self.processor.labels else code
                        if failure_mode_label in all_labels:
                            idx = all_labels.index(failure_mode_label)
                            grouped_handles.append(all_handles[idx])
                            grouped_labels.append(f"  {failure_mode_label}")
            
            # Create single shared legend positioned to the right of both plots
            legend = fig.legend(grouped_handles, grouped_labels,
                               loc='center left', bbox_to_anchor=(0.98, 0.5),
                               fontsize=11, frameon=True)
            
            # Style category titles in legend
            for i, text in enumerate(legend.get_texts()):
                if text.get_text().startswith('\n'):  # Category titles
                    text.set_weight('bold')
                    text.set_fontsize(12)
        elif all_handles:
            # Simple shared legend if no categories
            fig.legend(all_handles, all_labels,
                      loc='center left', bbox_to_anchor=(0.98, 0.5),
                      fontsize=11, frameon=True)
        
        plt.suptitle(title, fontweight='bold', fontsize=18, y=0.98)
        plt.tight_layout()
        self._save_or_show(save_path)
    
    def _save_or_show(self, save_path: Optional[Path]):
        """Save plot to file or display it."""
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
            plt.close()
        else:
            plt.show()


# Convenience functions for backward compatibility
def create_failure_plot(failure_counts: Dict, title: str, filename: str, 
                        output_dir: Optional[Path] = None, plot_type: str = "combined",
                        categories: Dict = None, labels: Dict = None):
    """Simplified interface for creating failure plots.
    
    Args:
        failure_counts: Dict mapping model names to failure mode counts
        title: Plot title
        filename: Output filename
        output_dir: Directory to save plot
        plot_type: Type of plot ("combined", "by_category", "aggregated")
        categories: Optional custom category definitions
        labels: Optional custom label definitions
    """
    plotter = FailurePlotter(categories=categories, labels=labels)
    save_path = output_dir / filename if output_dir else None
    
    if plot_type == "combined":
        plotter.plot_combined(failure_counts, title, save_path)
    elif plot_type == "by_category":
        plotter.plot_by_category(failure_counts, title, save_path)
    elif plot_type == "aggregated":
        plotter.plot_aggregated(failure_counts, title, save_path)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


if __name__ == "__main__":
    # Test data matching original code
    failure_counts_mast = {
        "claude-opus-4.1": {
            "1.1": 15, "1.2": 8, "1.3": 3, "1.4": 6, "1.5": 2,
            "2.1": 12, "2.2": 7, "2.3": 9, "2.4": 4, "2.5": 5, "2.6": 3,
            "3.1": 10, "3.2": 6, "3.3": 4
        },
        "gpt-5": {
            "1.1": 12, "1.2": 10, "1.3": 5, "1.4": 4, "1.5": 3,
            "2.1": 14, "2.2": 8, "2.3": 6, "2.4": 7, "2.5": 4, "2.6": 5,
            "3.1": 8, "3.2": 9, "3.3": 3
        },
        "claude-sonnet-3.5": {
            "1.1": 18, "1.2": 6, "1.3": 7, "1.4": 5, "1.5": 4,
            "2.1": 10, "2.2": 11, "2.3": 8, "2.4": 6, "2.5": 7, "2.6": 4,
            "3.1": 12, "3.2": 5, "3.3": 6
        }
    }
    
    no_timeout_counts = {
        "claude-opus-4.1": {
            "1.1": 12, "1.2": 7, "1.3": 3, "1.4": 5, "1.5": 2,
            "2.1": 10, "2.2": 6, "2.3": 8, "2.4": 3, "2.5": 4, "2.6": 2,
            "3.1": 8, "3.2": 5, "3.3": 3
        },
        "gpt-5": {
            "1.1": 10, "1.2": 8, "1.3": 4, "1.4": 3, "1.5": 2,
            "2.1": 11, "2.2": 6, "2.3": 5, "2.4": 6, "2.5": 3, "2.6": 4,
            "3.1": 6, "3.2": 7, "3.3": 2
        },
        "claude-sonnet-3.5": {
            "1.1": 14, "1.2": 5, "1.3": 6, "1.4": 4, "1.5": 3,
            "2.1": 8, "2.2": 9, "2.3": 7, "2.4": 5, "2.5": 6, "2.6": 3,
            "3.1": 10, "3.2": 4, "3.3": 5
        }
    }
    
    # Create test plots with matching names from original
    output_dir = Path("plots/test_plots")
    plotter = FailurePlotter()
    
    print("Test 1: MAST failure mode plot with subplots...")
    plotter.plot_by_category(failure_counts_mast, 
                            "MAST Failure Analysis by Category",
                            output_dir / "mast_failures_test.png")
    
    print("Test 2: MAST failure mode plot v2 (all in one plot)...")
    plotter.plot_combined(failure_counts_mast,
                         "MAST Failure Analysis - All Categories",
                         output_dir / "mast_failures_v2_test.png")
    
    print("Test 3: MAST failure mode plot v3 (3 category subplots)...")
    plotter.plot_aggregated(failure_counts_mast,
                           "MAST Failure Analysis - Category Aggregation",
                           output_dir / "mast_failures_v3_test.png")
    
    print("Test 4: Comparison plot with MAST categories...")
    plotter.plot_comparison(no_timeout_counts, no_timeout_counts,
                           ("Non-Timeout Failures", "Timeout Failures"),
                           "MAST Non-Timeout vs Timeout Failure Comparison",
                           output_dir / "mast_comparison_test.png")
    
    print("All tests completed successfully!")
    
    # Test 5: Custom categories (non-MAST)
    print("\nTest 5: Custom categories...")
    custom_categories = {
        "Performance": ["slow", "timeout", "memory"],
        "Correctness": ["wrong_output", "missing_data", "format_error"],
        "Stability": ["crash", "hang", "exception"]
    }
    
    custom_data = {
        "Model-X": {
            "slow": 10, "timeout": 5, "memory": 3,
            "wrong_output": 8, "missing_data": 4, "format_error": 2,
            "crash": 6, "hang": 3, "exception": 2
        },
        "Model-Y": {
            "slow": 8, "timeout": 7, "memory": 4,
            "wrong_output": 6, "missing_data": 5, "format_error": 3,
            "crash": 4, "hang": 2, "exception": 3
        }
    }
    
    custom_plotter = FailurePlotter(categories=custom_categories)
    custom_plotter.plot_combined(custom_data, "Custom Categories Test",
                                 output_dir / "custom_categories_test.png")
    
    # Test 6: No categories provided (auto-generate)
    print("\nTest 6: No categories (auto-generated colors)...")
    simple_data = {
        "System-A": {"error_1": 15, "error_2": 10, "error_3": 8, "error_4": 5},
        "System-B": {"error_1": 12, "error_2": 14, "error_3": 6, "error_4": 9},
        "System-C": {"error_1": 10, "error_2": 11, "error_3": 9, "error_4": 7}
    }
    
    simple_plotter = FailurePlotter(categories=None, labels=None)
    simple_plotter.plot_combined(simple_data, "Simple Errors (No Categories)",
                                 output_dir / "no_categories_test.png")
    
    print("\nAll extended tests completed!")