-- Query to compare token usage between timeout and non-timeout trials for the same task
-- Specifically for Terminus agent, grouped by model, showing median percentage difference across all models
WITH task_trial_tokens AS (
    SELECT t.task_checksum,
        t.id as trial_id,
        t.agent_name,
        t.agent_version,
        tm.model_name,
        CASE
            WHEN t.exception_info->>'exception_type' = 'AgentTimeoutError' THEN 'timeout'
            ELSE 'no_timeout'
        END as trial_type,
        tm.n_input_tokens,
        tm.n_output_tokens,
        COALESCE(tm.n_input_tokens, 0) + COALESCE(tm.n_output_tokens, 0) as total_tokens
    FROM trial t
        JOIN trial_model tm ON t.id = tm.trial_id
    WHERE t.agent_name = 'terminus-2'
        AND t.created_at >= '2025-09-01'::timestamptz -- Adjust date filter as needed
        AND tm.n_input_tokens IS NOT NULL
        AND tm.n_output_tokens IS NOT NULL
),
task_summary AS (
    SELECT task_checksum,
        model_name,
        -- Timeout trials
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY CASE
                    WHEN trial_type = 'timeout' THEN total_tokens
                END
        ) as median_timeout_tokens,
        COUNT(
            CASE
                WHEN trial_type = 'timeout' THEN 1
            END
        ) as timeout_trial_count,
        -- Non-timeout trials  
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY CASE
                    WHEN trial_type = 'no_timeout' THEN total_tokens
                END
        ) as median_no_timeout_tokens,
        COUNT(
            CASE
                WHEN trial_type = 'no_timeout' THEN 1
            END
        ) as no_timeout_trial_count
    FROM task_trial_tokens
    GROUP BY task_checksum,
        model_name
    HAVING COUNT(
            CASE
                WHEN trial_type = 'timeout' THEN 1
            END
        ) > 0
        AND COUNT(
            CASE
                WHEN trial_type = 'no_timeout' THEN 1
            END
        ) > 0
),
percentage_differences AS (
    SELECT task_checksum,
        model_name,
        median_timeout_tokens,
        median_no_timeout_tokens,
        timeout_trial_count,
        no_timeout_trial_count,
        CASE
            WHEN median_no_timeout_tokens > 0 THEN ROUND(
                (
                    (median_timeout_tokens - median_no_timeout_tokens) / median_no_timeout_tokens
                )::numeric * 100::numeric,
                2
            )
            ELSE NULL
        END as percentage_difference
    FROM task_summary
    WHERE median_timeout_tokens IS NOT NULL
        AND median_no_timeout_tokens IS NOT NULL
        AND median_no_timeout_tokens > 0
)
SELECT COUNT(*) as tasks_with_both_timeout_and_no_timeout,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY percentage_difference
        )::numeric,
        2
    ) as median_percentage_difference,
    ROUND(MIN(percentage_difference)::numeric, 2) as min_percentage_difference,
    ROUND(MAX(percentage_difference)::numeric, 2) as max_percentage_difference,
    ROUND(STDDEV(percentage_difference)::numeric, 2) as stddev_percentage_difference,
    SUM(timeout_trial_count) as total_timeout_trials,
    SUM(no_timeout_trial_count) as total_no_timeout_trials,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY median_timeout_tokens
        )::numeric,
        0
    ) as overall_median_timeout_tokens,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY median_no_timeout_tokens
        )::numeric,
        0
    ) as overall_median_no_timeout_tokens
FROM percentage_differences;