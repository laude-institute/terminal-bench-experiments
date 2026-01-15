-- Leaderboard scores for terminal-bench 2.0
-- Aggregates resolution rates per agent/model combination with full metadata
SELECT COALESCE(r.agent_display_name, r.agent_name) AS agent,
  r.agent_name,
  r.agent_version,
  r.agent_org_display_name,
  r.agent_url,
  r.model_names[1] AS model_name,
  r.model_providers[1] AS model_provider,
  r.model_display_names[1] AS model_display_name,
  r.model_org_display_names[1] AS model_org_display_name,
  r.model_keys[1] AS model_key,
  r.verified,
  avg(r.resolution_rate) AS score,
  1.96 / count(*) * sqrt(sum(r.resolution_rate * (1 - r.resolution_rate) / NULLIF(r.n_trials - 1, 0))) AS ci95,
  count(*) AS n_tasks,
  sum(r.n_trials) AS total_trials,
  sum(r.n_errors) AS total_errors,
  sum(COALESCE(r.avg_input_tokens, 0)) AS total_input_tokens,
  sum(COALESCE(r.avg_output_tokens, 0)) AS total_output_tokens,
  sum(COALESCE(r.avg_cache_tokens, 0)) AS total_cache_tokens,
  sum(r.avg_cost_cents) AS total_cost_cents,
  avg(r.avg_execution_time_seconds) AS avg_execution_time_seconds
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
  r.agent_version,
  r.agent_org_display_name,
  r.agent_url,
  r.model_names[1],
  r.model_providers[1],
  r.model_display_names[1],
  r.model_org_display_names[1],
  r.model_keys[1],
  r.verified
HAVING avg(r.resolution_rate) > 0
ORDER BY score DESC;