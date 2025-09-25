-- Query to find tasks solved by codex agent with gpt-5 but not by claude-opus-4-1-20250805 with openhands
-- Only for terminal-bench dataset version 2.0
WITH solved_by_codex AS (
    -- Find tasks solved by codex agent with gpt-5
    SELECT DISTINCT t.task_checksum,
        tsk.name as task_name
    FROM trial t
        JOIN trial_model tm ON t.id = tm.trial_id
        JOIN task tsk ON t.task_checksum = tsk.checksum
        JOIN dataset_task dt ON t.task_checksum = dt.task_checksum
        JOIN dataset d ON dt.dataset_name = d.name
        AND dt.dataset_version = d.version
        AND dt.dataset_registry_uri = d.registry_uri
    WHERE t.agent_name = 'codex'
        AND tm.model_name = 'gpt-5'
        AND t.reward > 0 -- Assuming reward > 0 means solved
        AND d.name = 'terminal-bench'
        AND d.version = '2.0'
),
solved_by_openhands AS (
    -- Find tasks solved by claude-opus-4-1-20250805 with openhands
    SELECT DISTINCT t.task_checksum
    FROM trial t
        JOIN trial_model tm ON t.id = tm.trial_id
        JOIN dataset_task dt ON t.task_checksum = dt.task_checksum
        JOIN dataset d ON dt.dataset_name = d.name
        AND dt.dataset_version = d.version
        AND dt.dataset_registry_uri = d.registry_uri
    WHERE t.agent_name = 'openhands'
        AND tm.model_name = 'claude-opus-4-1-20250805'
        AND t.reward > 0 -- Assuming reward > 0 means solved
        AND d.name = 'terminal-bench'
        AND d.version = '2.0'
) -- Find tasks solved by codex but not by openhands
SELECT task_name
FROM solved_by_codex
WHERE task_checksum NOT IN (
        SELECT task_checksum
        FROM solved_by_openhands
    )
ORDER BY task_name;