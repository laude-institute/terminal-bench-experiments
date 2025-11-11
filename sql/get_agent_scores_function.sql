DROP FUNCTION IF EXISTS get_agent_scores(TEXT, TEXT);
CREATE OR REPLACE FUNCTION get_agent_scores(
        p_dataset_name TEXT,
        p_dataset_version TEXT
    ) RETURNS TABLE (
        rank BIGINT,
        agent_name TEXT,
        agent_version TEXT,
        model_name TEXT,
        model_provider TEXT,
        accuracy NUMERIC,
        stderr NUMERIC,
        agent_display_name TEXT,
        model_display_name TEXT,
        agent_org_display_name TEXT,
        model_org_display_name TEXT,
        agent_url TEXT,
        created_at TIMESTAMP WITH TIME ZONE
    ) AS $$ BEGIN RETURN QUERY WITH p_hats AS (
        SELECT t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name AS task_name,
            AVG(COALESCE(t.reward, 0)) AS p_hat,
            COUNT(*) AS n_trials,
            MIN(t.created_at) AS earliest_trial_date,
            AVG(
                jsonb_array_length(t.agent_metadata->'api_request_times_msec')
            ) AS avg_api_calls,
            SUM(
                CASE
                    WHEN t.exception_info IS NULL THEN 0
                    ELSE 1
                END
            ) AS n_errors,
            CASE
                WHEN COUNT(*) > 1 THEN AVG(COALESCE(t.reward, 0)) * (1 - AVG(COALESCE(t.reward, 0))) / (COUNT(*) - 1)
                ELSE NULL
            END AS partial_var,
            AVG(tm.n_input_tokens) AS avg_n_input_tokens,
            AVG(tm.n_output_tokens) AS avg_n_output_tokens,
            AVG(
                tm.n_input_tokens / 1000000.0 * m.cents_per_million_input_tokens + tm.n_output_tokens / 1000000.0 * m.cents_per_million_output_tokens
            ) AS avg_cost_cents,
            AVG(
                EXTRACT(
                    EPOCH
                    FROM (
                            t.agent_execution_ended_at - t.agent_execution_started_at
                        )
                )
            ) AS avg_execution_time_seconds
        FROM trial AS t
            INNER JOIN dataset_task AS dt ON dt.task_checksum = t.task_checksum
            INNER JOIN task ON task.checksum = dt.task_checksum
            INNER JOIN trial_model AS tm ON tm.trial_id = t.id
            INNER JOIN model AS m ON m.name = tm.model_name
            AND m.provider = tm.model_provider
            INNER JOIN job AS j ON j.id = t.job_id
        WHERE dt.dataset_name = p_dataset_name
            AND dt.dataset_version = p_dataset_version
            AND (
                t.exception_info IS NULL
                OR t.exception_info->>'exception_type' IN (
                    'AgentTimeoutError',
                    'VerifierTimeoutError'
                )
            )
        GROUP BY t.agent_name,
            t.agent_version,
            tm.model_name,
            tm.model_provider,
            task.name
    ),
    aggregated_scores AS (
        SELECT ph.agent_name,
            ph.agent_version,
            ph.model_name,
            ph.model_provider,
            AVG(ph.p_hat) AS accuracy,
            CASE
                WHEN COUNT(*) > COUNT(ph.partial_var) THEN NULL
                ELSE SQRT(SUM(ph.partial_var)) / COUNT(*)
            END AS stderr,
            MIN(ph.earliest_trial_date) AS created_at
        FROM p_hats ph
        GROUP BY ph.agent_name,
            ph.agent_version,
            ph.model_name,
            ph.model_provider
        HAVING AVG(ph.p_hat) > 0.01
    )
SELECT RANK() OVER (
        ORDER BY a.accuracy DESC
    ) AS rank,
    a.agent_name,
    a.agent_version,
    a.model_name,
    a.model_provider,
    a.accuracy,
    a.stderr,
    ag.display_name AS agent_display_name,
    mo.display_name AS model_display_name,
    ag.org_display_name AS agent_org_display_name,
    mo.org_display_name AS model_org_display_name,
    ag.url AS agent_url,
    a.created_at
FROM aggregated_scores a
    LEFT JOIN agent ag ON ag.name = a.agent_name
    AND ag.version = a.agent_version
    LEFT JOIN model mo ON mo.name = a.model_name
    AND mo.provider = a.model_provider
ORDER BY a.accuracy DESC;
END;
$$ LANGUAGE plpgsql;
select *
from get_agent_scores('terminal-bench', '2.0');