WITH dataset_task_count AS (
    -- Count total number of tasks in the dataset
    SELECT COUNT(DISTINCT task_checksum) AS total_tasks
    FROM dataset_task
    WHERE dataset_task.dataset_name = 'terminal-bench'
        AND dataset_task.dataset_version = '2.0'
),
trial_aggregates AS (
    -- First aggregate tokens and costs per trial (across all models)
    SELECT t.id AS trial_id,
        t.agent_name,
        task.name AS task_name,
        COALESCE(t.reward, 0) AS reward,
        t.created_at,
        jsonb_array_length(t.agent_metadata->'api_request_times_msec') AS n_api_calls,
        CASE
            WHEN t.exception_info IS NULL THEN 0
            ELSE 1
        END AS is_error,
        t.agent_execution_ended_at,
        t.agent_execution_started_at,
        SUM(tm.n_input_tokens) AS total_input_tokens,
        SUM(tm.n_output_tokens) AS total_output_tokens,
        SUM(COALESCE(tm.n_cache_tokens, 0)) AS total_cache_tokens,
        SUM(
            -- Regular input tokens (excluding cache tokens)
            (tm.n_input_tokens - tm.n_cache_tokens) / 1000000.0 * m.cents_per_million_input_tokens -- Output tokens
            + tm.n_output_tokens / 1000000.0 * m.cents_per_million_output_tokens -- Cache tokens
            + COALESCE(tm.n_cache_tokens, 0) / 1000000.0 * COALESCE(
                m.cents_per_million_cache_tokens,
                m.cents_per_million_input_tokens
            )
        ) AS total_cost_cents,
        COALESCE(
            array_agg(
                DISTINCT tm.model_name
                ORDER BY tm.model_name
            ) FILTER (
                WHERE tm.model_name IS NOT NULL
            ),
            ARRAY []::TEXT []
        ) AS model_names,
        COALESCE(
            array_agg(
                DISTINCT tm.model_provider
                ORDER BY tm.model_provider
            ) FILTER (
                WHERE tm.model_provider IS NOT NULL
            ),
            ARRAY []::TEXT []
        ) AS model_providers,
        COALESCE(
            array_agg(
                DISTINCT m.display_name
                ORDER BY m.display_name
            ) FILTER (
                WHERE m.display_name IS NOT NULL
            ),
            ARRAY []::TEXT []
        ) AS model_display_names,
        COALESCE(
            array_agg(
                DISTINCT m.org_display_name
                ORDER BY m.org_display_name
            ) FILTER (
                WHERE m.org_display_name IS NOT NULL
            ),
            ARRAY []::TEXT []
        ) AS model_org_display_names,
        ag.display_name AS agent_display_name,
        ag.org_display_name AS agent_org_display_name,
        ag.url AS agent_url
    FROM trial AS t
        INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
        INNER JOIN task ON task.checksum = dt.task_checksum
        LEFT JOIN trial_model AS tm ON tm.trial_id = t.id
        LEFT JOIN model AS m ON m.name = tm.model_name
        AND m.provider = tm.model_provider
        INNER JOIN job AS j ON j.id = t.job_id
        LEFT JOIN agent AS ag ON ag.name = t.agent_name
        AND ag.version = t.agent_version
    WHERE dt.dataset_name = 'terminal-bench'
        and m.provider != 'xai'
        AND dt.dataset_version = '2.0'
        AND j.include_on_leaderboard IS TRUE
        AND j.created_at <= '2025-11-04 18:58:26.409036+00'::timestamptz
        AND (
            t.exception_info IS NULL
            OR t.exception_info->>'exception_type' IN (
                'AgentTimeoutError',
                'VerifierTimeoutError'
            )
        )
    GROUP BY t.id,
        t.agent_name,
        task.name,
        t.reward,
        t.created_at,
        t.agent_metadata,
        t.exception_info,
        t.agent_execution_ended_at,
        t.agent_execution_started_at,
        ag.display_name,
        ag.org_display_name,
        ag.url
),
p_hats AS (
    SELECT ta.agent_name,
        ta.model_names,
        ta.model_providers,
        ta.model_display_names,
        ta.model_org_display_names,
        ta.agent_display_name,
        ta.agent_org_display_name,
        ta.agent_url,
        ta.task_name,
        AVG(ta.reward) AS p_hat,
        COUNT(*) AS n_trials,
        MIN(ta.created_at) AS earliest_trial_date,
        AVG(ta.n_api_calls) AS avg_api_calls,
        SUM(ta.is_error) AS n_errors,
        CASE
            WHEN COUNT(*) > 1 THEN AVG(ta.reward) * (1 - AVG(ta.reward)) / (COUNT(*) - 1)
            ELSE NULL
        END AS partial_var,
        AVG(ta.total_input_tokens) AS avg_n_input_tokens,
        AVG(ta.total_output_tokens) AS avg_n_output_tokens,
        AVG(ta.total_cost_cents) AS avg_cost_cents,
        AVG(
            EXTRACT(
                EPOCH
                FROM (
                        ta.agent_execution_ended_at - ta.agent_execution_started_at
                    )
            )
        ) AS avg_execution_time_seconds
    FROM trial_aggregates ta
    WHERE ta.model_display_names IS NOT NULL
        AND array_length(ta.model_display_names, 1) > 0
    GROUP BY ta.agent_name,
        ta.model_names,
        ta.model_providers,
        ta.model_display_names,
        ta.model_org_display_names,
        ta.agent_display_name,
        ta.agent_org_display_name,
        ta.agent_url,
        ta.task_name
),
aggregated_scores AS (
    SELECT ph.agent_name,
        ph.model_names,
        ph.model_providers,
        ph.model_display_names,
        ph.model_org_display_names,
        ph.agent_display_name,
        ph.agent_org_display_name,
        ph.agent_url,
        AVG(ph.p_hat) AS accuracy,
        CASE
            WHEN COUNT(*) > COUNT(ph.partial_var) THEN NULL
            ELSE SQRT(SUM(ph.partial_var)) / COUNT(*)
        END AS stderr,
        MIN(ph.earliest_trial_date) AS created_at,
        COUNT(DISTINCT ph.task_name) AS tasks_evaluated,
        SUM(ph.avg_cost_cents) AS avg_total_cost_cents
    FROM p_hats ph
    GROUP BY ph.agent_name,
        ph.model_names,
        ph.model_providers,
        ph.model_display_names,
        ph.model_org_display_names,
        ph.agent_display_name,
        ph.agent_org_display_name,
        ph.agent_url
    HAVING AVG(ph.p_hat) > 0.01
        AND COUNT(DISTINCT ph.task_name) = (
            SELECT total_tasks
            FROM dataset_task_count
        )
)
SELECT RANK() OVER (
        ORDER BY a.accuracy DESC
    ) AS rank,
    a.agent_name,
    a.model_names,
    a.model_providers,
    a.accuracy,
    a.stderr,
    a.agent_display_name,
    a.model_display_names,
    a.agent_org_display_name,
    a.model_org_display_names,
    a.agent_url,
    a.created_at,
    a.avg_total_cost_cents
FROM aggregated_scores a
WHERE NOT (
        a.agent_name = 'openhands'
        AND 'grok-code-fast-1' = ANY(a.model_names)
    )
ORDER BY a.accuracy DESC;