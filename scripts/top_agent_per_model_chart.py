import os

import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
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
WITH ranked AS (
  SELECT COALESCE(r.agent_display_name, r.agent_name) AS agent,
    r.agent_name,
    r.model_display_names[1] AS model_display_name,
    r.model_keys[1] AS model_key,
    avg(r.resolution_rate) AS score,
    1.96 / count(*) * sqrt(sum(r.resolution_rate * (1 - r.resolution_rate) / NULLIF(r.n_trials - 1, 0))) AS ci95
  FROM resolution_rates r
    JOIN dataset_task dt ON dt.task_checksum = r.task_checksum
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
  GROUP BY COALESCE(r.agent_display_name, r.agent_name),
    r.agent_name,
    r.model_display_names[1],
    r.model_keys[1]
  HAVING avg(r.resolution_rate) > 0
)
SELECT DISTINCT ON (model_key) *
FROM ranked
ORDER BY model_key, score DESC;
"""

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn)

# Convert score to percentage
df["accuracy"] = df["score"] * 100
df["ci95_pct"] = df["ci95"].fillna(0) * 100

# Create label from model display name and agent
df["label"] = df["model_display_name"] + " (" + df["agent"] + ")"

# Sort by accuracy descending (highest at top)
df_sorted = df.sort_values("accuracy", ascending=False)

sns.set_theme(font="Verdana")

fig, ax = plt.subplots(figsize=(10, max(7, len(df_sorted) * 0.30)))

bars = sns.barplot(
    x="accuracy",
    y="label",
    data=df_sorted,
    ax=ax,
    color=sns.color_palette("tab20c")[1],
    edgecolor="black",
    alpha=0.6,
)

# Add error bars
ax.errorbar(
    x=df_sorted["accuracy"],
    y=range(len(df_sorted)),
    xerr=df_sorted["ci95_pct"],
    fmt="none",
    color="black",
    capsize=0,
)

ax.set_xlabel("Resolution Rate")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color("black")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)
ax.set_facecolor("white")
ax.grid(True, axis="both", linestyle=":", color="lightgray")

plt.tight_layout()
plt.savefig("outputs/top_agent_per_model.pdf")
