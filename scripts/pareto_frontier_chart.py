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
SELECT COALESCE(r.agent_display_name, r.agent_name) AS agent,
  r.agent_name,
  r.model_display_names[1] AS model_display_name,
  r.model_org_display_names[1] AS model_org,
  r.model_keys[1] AS model_key,
  avg(r.resolution_rate) AS score,
  sum(r.avg_cost_cents) AS total_cost_cents
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
  AND NOT EXISTS (
    SELECT 1
    FROM unnest(r.model_keys) AS mk
    WHERE mk LIKE 'xai/%'
  )
  AND COALESCE(r.avg_input_tokens, 0) >= 0
GROUP BY COALESCE(r.agent_display_name, r.agent_name),
  r.agent_name,
  r.model_display_names[1],
  r.model_org_display_names[1],
  r.model_keys[1]
HAVING avg(r.resolution_rate) >= 0.01
  AND sum(r.avg_cost_cents) IS NOT NULL
ORDER BY model_org_display_names[1], model_display_names[1];
"""

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn)

# Convert to useful units
df["resolution_rate"] = df["score"] * 100
df["cost_usd"] = df["total_cost_cents"] / 100

# Get unique agents and models
agents = df["agent_name"].unique()
agent_display_names = {
    row["agent_name"]: row["agent"]
    for _, row in df.drop_duplicates("agent_name").iterrows()
}
models = df["model_display_name"].unique()

# Define markers for agents
markers = ["o", "s", "^", "D", "v", "p", "h", "*", "X", "P"]
agent_markers = {agent: markers[i % len(markers)] for i, agent in enumerate(agents)}

# Use tab20c for model colors with specific mapping
palette = sns.color_palette("tab20c", 20)
model_color_map = {
    "gpt-5.2": 0,
    "gpt-5": 1,
    "gpt-5-mini": 2,
    "gpt-5-nano": 3,
    "claude opus 4.5": 4,
    "claude opus 4.1": 5,
    "claude sonnet 4.5": 6,
    "claude haiku 4.5": 7,
    "gemini 3 pro": 8,
    "gemini 2.5 pro": 9,
    "gemini 3 flash": 10,
    "gemini 2.5 flash": 11,
    "minimax m2": 12,
    "kimi k2 instruct": 13,
    "kimi k2 thinking": 14,
    "glm 4.6": 15,
    "qwen 3 coder 480b": 16,
    "gpt-oss-120b": 17,
    "gpt-oss-20b": 18,
}


def get_model_color(model_name):
    model_lower = model_name.lower()
    for key, idx in model_color_map.items():
        if key.lower() in model_lower:
            return palette[idx]
    return palette[18]  # Default color for unmapped models


model_colors = {model: get_model_color(model) for model in models}


# Sort models by color map order for legend
def get_model_sort_key(model_name):
    model_lower = model_name.lower()
    for key, idx in model_color_map.items():
        if key.lower() in model_lower:
            return idx
    return 99  # Unmapped models at end


models_sorted = sorted(models, key=get_model_sort_key)

sns.set_theme(font="Verdana")

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each point
for _, row in df.iterrows():
    ax.scatter(
        row["cost_usd"],
        row["resolution_rate"],
        marker=agent_markers[row["agent_name"]],
        color=model_colors[row["model_display_name"]],
        s=100,
        alpha=0.8,
    )

# Compute and draw Pareto frontier (minimize cost, maximize resolution rate)
# Sort by cost ascending
df_sorted = df.sort_values("cost_usd")
pareto_points = []
pareto_labels = []
max_resolution = -1
for _, row in df_sorted.iterrows():
    if row["resolution_rate"] > max_resolution:
        pareto_points.append((row["cost_usd"], row["resolution_rate"]))
        pareto_labels.append(f"{row['model_display_name']} ({row['agent']})")
        max_resolution = row["resolution_rate"]

if pareto_points:
    pareto_x, pareto_y = zip(*pareto_points)
    ax.plot(pareto_x, pareto_y, color="black", linestyle="-", linewidth=2, alpha=0.7)

    # Add labels to Pareto frontier points
    for x, y, label in zip(pareto_x, pareto_y, pareto_labels):
        if (
            "gemini" in label.lower()
            and "3" in label.lower()
            and "pro" in label.lower()
        ):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(-10, -10),
                fontsize=11,
                ha="right",
            )
        elif "opus" in label.lower() and "4.5" in label:
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(-10, 10),
                fontsize=11,
                ha="right",
            )
        else:
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(-10, 0),
                fontsize=11,
                ha="right",
            )

ax.set_xlabel("Cost (USD)")
ax.set_ylabel("")
ax.set_xscale("log")
ax.set_xlim(left=0.03)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:g}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))

# Axes styling
ax.spines["left"].set_visible(True)
ax.spines["left"].set_color("black")
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_visible(True)
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(0.8)
ax.set_facecolor("white")
ax.grid(True, axis="both", linestyle=":", color="lightgray")

# Create combined legend outside the plot
from matplotlib.lines import Line2D

# Agent section
agent_title = Line2D([0], [0], linestyle="None", label=r"$\bf{Agent}$")
agent_handles = [
    Line2D(
        [0],
        [0],
        marker=agent_markers[agent],
        color="gray",
        linestyle="None",
        markersize=8,
        label=agent_display_names[agent],
    )
    for agent in agents
]

# Model section
model_title = Line2D([0], [0], linestyle="None", label=r"$\bf{Model}$")
model_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color=model_colors[model],
        linestyle="None",
        markersize=8,
        label=model,
    )
    for model in models_sorted
]

# Combine handles with blank spacer
blank = Line2D([0], [0], linestyle="None", label="")
all_handles = [agent_title] + agent_handles + [blank, model_title] + model_handles

ax.legend(
    handles=all_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True
)

plt.tight_layout()
plt.savefig("outputs/pareto_frontier.pdf", bbox_inches="tight")
