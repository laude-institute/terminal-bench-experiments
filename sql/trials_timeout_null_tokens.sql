-- Query to show for each agent x model combination, how many tasks have 0 trials with non-null token counts
WITH task_trial_summary AS (
    SELECT t.agent_name,
        t.agent_version,
        tm.model_name,
        tm.model_provider,
        t.task_checksum,
        COUNT(*) as total_trials_for_task,
        COUNT(
            CASE
                WHEN tm.n_input_tokens IS NOT NULL
                AND tm.n_output_tokens IS NOT NULL THEN 1
            END
        ) as trials_with_non_null_tokens
    FROM trial t
        JOIN trial_model tm ON t.id = tm.trial_id
    WHERE t.agent_name IN ('openhands', 'gemini-cli', 'swe-agent-mini')
        AND t.created_at >= '2025-09-17 01:13:33.950824+00'::timestamptz
        AND (
            t.exception_info IS NULL
            OR t.exception_info->>'exception_type' in ('AgentTimeoutError', 'VerifierTimeoutError')
        )
    GROUP BY t.agent_name,
        t.agent_version,
        tm.model_name,
        tm.model_provider,
        t.task_checksum
)
SELECT agent_name,
    model_name,
    COUNT(
        CASE
            WHEN trials_with_non_null_tokens = 0 THEN 1
        END
    ) as tasks_with_zero_non_null_token_trials
FROM task_trial_summary
GROUP BY agent_name,
    agent_version,
    model_name,
    model_provider
ORDER BY tasks_with_zero_non_null_token_trials DESC,
    agent_name,
    model_name,
    model_provider;