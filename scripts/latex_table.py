import os

import pandas as pd
from dotenv import load_dotenv
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
  avg(r.resolution_rate) AS score,
  1.96 / count(*) * sqrt(sum(r.resolution_rate * (1 - r.resolution_rate) / NULLIF(r.n_trials - 1, 0))) AS ci95,
  sum(COALESCE(r.avg_input_tokens, 0)) AS total_input_tokens,
  sum(COALESCE(r.avg_output_tokens, 0)) AS total_output_tokens
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
  r.model_display_names[1]
HAVING avg(r.resolution_rate) >= 0.01
ORDER BY score DESC;
"""

with engine.connect() as conn:
    df = pd.read_sql(text(query), conn)

# Format columns
df["resolution_rate"] = df.apply(
    lambda row: f"{row['score']*100:.1f}\\% $\\pm$ {row['ci95']*100:.1f}\\%" if pd.notna(row['ci95']) else f"{row['score']*100:.1f}\\%",
    axis=1
)
df["input_tokens"] = (df["total_input_tokens"] / 1e6).apply(lambda x: f"{x:.1f}M")
df["output_tokens"] = (df["total_output_tokens"] / 1e6).apply(lambda x: f"{x:.1f}M")

# Select and rename columns for table
table_df = df[["model_display_name", "agent", "resolution_rate", "input_tokens", "output_tokens"]]
table_df.columns = ["Model", "Agent", "Resolution Rate", "Input Tokens", "Output Tokens"]

# Generate LaTeX
latex = table_df.to_latex(index=False, escape=False)
print(latex)

# Save to file
with open("outputs/results_table.tex", "w") as f:
    f.write(latex)

print("\nSaved to outputs/results_table.tex")
