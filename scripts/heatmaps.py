import os

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text

load_dotenv()

db_user = os.environ["SANDBOXES_POSTGRES_USER"]
db_password = os.environ["SANDBOXES_POSTGRES_PASSWORD"]
db_host = os.environ["SANDBOXES_POSTGRES_HOST"]
db_port = os.environ["SANDBOXES_POSTGRES_PORT"]
db_name = os.environ["SANDBOXES_POSTGRES_NAME"]

engine = create_engine(
    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

query = """
SELECT COALESCE(r.agent_display_name, r.agent_name) AS agent,
  r.model_display_names[1] AS model_display_name,
  t.name AS task_name,
  r.resolution_rate AS p_hat,
  r.n_errors::numeric / NULLIF(r.n_trials, 0) AS error_rate,
  COALESCE(r.avg_input_tokens, 0) + COALESCE(r.avg_output_tokens, 0) AS avg_n_tokens
FROM resolution_rates r
  JOIN dataset_task dt ON dt.task_checksum = r.task_checksum
  JOIN task t ON t.checksum = r.task_checksum
WHERE r.verified = true
  AND dt.dataset_name = 'terminal-bench'
  AND dt.dataset_version = '2.0'
  AND r.agent_name != 'ante'
  AND NOT EXISTS (
    SELECT 1
    FROM unnest(r.model_keys) AS mk
    WHERE mk LIKE '%gpt-5.1%'
      OR mk LIKE '%gpt-5-codex%'
  )
  AND NOT (
    r.agent_name = 'openhands'
    AND r.model_keys @> ARRAY ['xai/grok-code-fast-1']
  )
  AND NOT r.model_keys @> ARRAY ['minimax/minimax-m2.1']
  AND NOT r.model_keys @> ARRAY ['openai/minimax-m2.1']
  AND COALESCE(r.avg_input_tokens, 0) >= 0
ORDER BY agent, model_display_name, task_name;
"""

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn)

print(f"Loaded {len(df)} rows")

# Create pivot table for resolution rate
pivot_df = df.pivot_table(
    index=["agent", "model_display_name"],
    columns="task_name",
    values="p_hat",
    aggfunc="mean",
)

# Sort rows by highest average p-hat across tasks
agent_model_avg_p_hat = pivot_df.mean(axis=1)
sorted_agent_models = agent_model_avg_p_hat.sort_values(ascending=False)
pivot_sorted = pivot_df.reindex(sorted_agent_models.index)

# Clean and sort by task performance
pivot_clean = pivot_sorted.fillna(0).astype(float)
task_avg_p_hat = pivot_clean.mean(axis=0)
sorted_tasks = task_avg_p_hat.sort_values(ascending=False)
pivot_clean_sorted = pivot_clean.reindex(columns=sorted_tasks.index)

# Transpose: tasks as rows, agent-model as columns
pivot_transposed = pivot_clean_sorted.T
formatted_columns = [f"{model} ({agent})" for agent, model in pivot_transposed.columns]


def create_heatmap(data, cbar_label, filename, cmap="RdYlBu"):
    plt.figure(figsize=(24, 28))
    heatmap = sns.heatmap(
        data,
        annot=False,
        cmap=cmap,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="white",
        xticklabels=formatted_columns,
    )
    plt.xlabel("Model (Agent), Sorted by Resolution Rate", fontsize=16)
    plt.ylabel("Task, Sorted by Resolution Rate", fontsize=16)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    cbar = heatmap.collections[0].colorbar
    cbar.set_label(cbar_label, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


# 1. Resolution Rate Heatmap
create_heatmap(
    pivot_transposed,
    "Resolution Rate",
    "outputs/resolution_rate_heatmap.pdf",
)

# 2. Error/Timeout Rate Heatmap
pivot_error_df = df.pivot_table(
    index=["agent", "model_display_name"],
    columns="task_name",
    values="error_rate",
    aggfunc="mean",
)
pivot_error_sorted = pivot_error_df.reindex(
    index=sorted_agent_models.index, columns=sorted_tasks.index
)
pivot_error_final = pivot_error_sorted.fillna(0).astype(float)
pivot_error_transposed = pivot_error_final.T

create_heatmap(
    pivot_error_transposed,
    "Timeout Rate",
    "outputs/timeout_heatmap.pdf",
)

# 3. Token Usage Heatmap
pivot_tokens_df = df.pivot_table(
    index=["agent", "model_display_name"],
    columns="task_name",
    values="avg_n_tokens",
    aggfunc="mean",
)
pivot_tokens_sorted = pivot_tokens_df.reindex(
    index=sorted_agent_models.index, columns=sorted_tasks.index
)
pivot_tokens_final = pivot_tokens_sorted.fillna(0).astype(float)
pivot_tokens_transposed = pivot_tokens_final.T

create_heatmap(
    pivot_tokens_transposed,
    "Average Token Usage",
    "outputs/token_heatmap.pdf",
)

print("Done!")
