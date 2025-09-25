select agent_name,
    model_name,
    task.name as task_name,
    avg(coalesce(reward, 0)) as p_hat,
    count(*) as n_trials,
    sum(
        case
            when exception_info is null then 0
            else 1
        end
    ) as n_errors,
    sum(
        case
            when exception_info is null then 0
            else 1
        end
    )::numeric / count(*) as error_rate,
    avg(tm.n_input_tokens) + avg(tm.n_output_tokens) as avg_n_tokens
from trial as t
    inner join dataset_task as dt on dt.task_checksum = t.task_checksum
    inner join task on task.checksum = dt.task_checksum
    inner join trial_model as tm on tm.trial_id = t.id
    inner join job as j on j.id = t.job_id
where dataset_name = 'terminal-bench'
    and dataset_version = '2.0'
    and agent_name = 'terminus-2'
    and (
        exception_info is null
        or exception_info->>'exception_type' in (
            'AgentTimeoutError',
            'VerifierTimeoutError'
        )
    )
    and j.created_at >= '2025-09-17 01:13:33.950824+00'::timestamptz
group by agent_name,
    model_name,
    task_name
order by agent_name,
    model_name,
    task_name;