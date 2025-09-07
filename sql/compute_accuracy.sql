WITH dataset_tasks AS (
    SELECT DISTINCT dt.task_checksum
    FROM dataset_task AS dt
    WHERE dt.dataset_name = 'tbc-pre-release'
        AND dt.dataset_version = '2.0'
),
task_count AS (
    SELECT COUNT(DISTINCT task_checksum) AS total_tasks
    FROM dataset_tasks
),
agent_model_trials AS (
    SELECT t.agent_name,
        t.agent_version,
        tm.model_name,
        tm.model_provider,
        t.task_checksum,
        t.reward
    FROM trial AS t
        JOIN trial_model AS tm ON t.id = tm.trial_id -- TODO add instruction eventually.
),
agent_model_task_scores AS (
    SELECT amt.agent_name,
        amt.agent_version,
        amt.model_name,
        amt.model_provider,
        amt.task_checksum,
        AVG(COALESCE(amt.reward, 0)) AS avg_task_score,
        case
            when count(*) = 1 then null
            else AVG(COALESCE(amt.reward, 0)) * (1 - AVG(COALESCE(amt.reward, 0))) / (count(*) - 1)
        end as variance,
        COUNT(*) AS trial_count
    FROM agent_model_trials AS amt
        JOIN dataset_tasks AS dt ON amt.task_checksum = dt.task_checksum
    GROUP BY amt.agent_name,
        amt.agent_version,
        amt.model_name,
        amt.model_provider,
        amt.task_checksum
),
complete_agent_models AS (
    SELECT agent_name,
        agent_version,
        model_name,
        model_provider,
        COUNT(DISTINCT task_checksum) AS tasks_evaluated
    FROM agent_model_task_scores
    GROUP BY agent_name,
        agent_version,
        model_name,
        model_provider
    HAVING COUNT(DISTINCT task_checksum) = (
            SELECT total_tasks
            FROM task_count
        )
),
task_level_scores AS (
    -- Get task-level average scores for complete agent-model combos
    SELECT amts.agent_name,
        amts.agent_version,
        amts.model_name,
        amts.model_provider,
        t.name AS task_name,
        amts.task_checksum,
        amts.avg_task_score,
        amts.variance,
        amts.trial_count
    FROM agent_model_task_scores AS amts
        JOIN complete_agent_models AS cam ON amts.agent_name = cam.agent_name
        AND amts.agent_version = cam.agent_version
        AND amts.model_name = cam.model_name
        AND amts.model_provider = cam.model_provider
        JOIN task AS t ON amts.task_checksum = t.checksum
) -- Final results: overall average and per-task breakdown
SELECT agent_name,
    agent_version,
    model_name,
    model_provider,
    -- Overall average score across all tasks
    ROUND(AVG(avg_task_score) * 100, 2) AS avg_accuracy,
    case
        when COUNT(variance) = COUNT(*) then ROUND(
            sqrt(SUM(variance)) / COUNT(*) * 100,
            2
        )
        else null
    end as stderr,
    -- Total number of tasks
    COUNT(DISTINCT task_checksum) AS num_tasks,
    -- Total trials across all tasks
    SUM(trial_count) AS total_trials,
    -- Task-level scores as JSON
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'task_name',
            task_name,
            'task_checksum',
            task_checksum,
            'avg_score',
            ROUND(avg_task_score::numeric, 4),
            'trial_count',
            trial_count
        )
        ORDER BY task_name
    ) AS task_scores
FROM task_level_scores
GROUP BY agent_name,
    agent_version,
    model_name,
    model_provider
ORDER BY avg_accuracy DESC,
    agent_name,
    agent_version,
    model_name,
    model_provider;