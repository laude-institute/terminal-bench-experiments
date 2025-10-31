-- Maximum gap between models within a single agent
with agent_model_scores as (
    select agent_name,
        model_name,
        round(avg(p_hat) * 100, 2) as accuracy
    from (
            select agent_name,
                model_name,
                task.name as task_name,
                avg(coalesce(reward, 0)) as p_hat
            from trial as t
                inner join dataset_task as dt on dt.task_checksum = t.task_checksum
                inner join task on task.checksum = dt.task_checksum
                inner join trial_model as tm on tm.trial_id = t.id
                inner join model as m on m.name = tm.model_name
                and m.provider = tm.model_provider
                inner join job as j on j.id = t.job_id
            where dataset_name = 'terminal-bench'
                and dataset_version = '2.0'
                and j.created_at >= '2025-09-17 01:13:33.950824+00'::timestamptz
                and (
                    exception_info is null
                    or exception_info->>'exception_type' in (
                        'AgentTimeoutError',
                        'VerifierTimeoutError'
                    )
                )
            group by agent_name,
                model_name,
                task_name
        ) p_hats
    group by agent_name,
        model_name
    having avg(p_hat) > 0.02
),
model_gaps as (
    select agent_name,
        max(accuracy) - min(accuracy) as max_gap,
        max(accuracy) as best_model_accuracy,
        min(accuracy) as worst_model_accuracy,
        count(*) as n_models,
        (
            array_agg(
                model_name
                order by accuracy desc
            )
        ) [1] as best_model_name,
        (
            array_agg(
                model_name
                order by accuracy asc
            )
        ) [1] as worst_model_name
    from agent_model_scores
    group by agent_name
    having count(*) > 1
)
select 'Max gap between models within a single agent:' as analysis_type,
    agent_name,
    max_gap,
    best_model_accuracy,
    worst_model_accuracy,
    best_model_name,
    worst_model_name,
    n_models
from model_gaps
order by max_gap desc
limit 1;
-- Maximum gap between agents with a single model
with agent_model_scores as (
    select agent_name,
        model_name,
        round(avg(p_hat) * 100, 2) as accuracy
    from (
            select agent_name,
                model_name,
                task.name as task_name,
                avg(coalesce(reward, 0)) as p_hat
            from trial as t
                inner join dataset_task as dt on dt.task_checksum = t.task_checksum
                inner join task on task.checksum = dt.task_checksum
                inner join trial_model as tm on tm.trial_id = t.id
                inner join model as m on m.name = tm.model_name
                and m.provider = tm.model_provider
                inner join job as j on j.id = t.job_id
            where dataset_name = 'terminal-bench'
                and dataset_version = '2.0'
                and j.created_at >= '2025-09-17 01:13:33.950824+00'::timestamptz
                and (
                    exception_info is null
                    or exception_info->>'exception_type' in (
                        'AgentTimeoutError',
                        'VerifierTimeoutError'
                    )
                )
            group by agent_name,
                model_name,
                task_name
        ) p_hats
    group by agent_name,
        model_name
    having avg(p_hat) > 0.02
),
agent_gaps as (
    select model_name,
        max(accuracy) - min(accuracy) as max_gap,
        max(accuracy) as best_agent_accuracy,
        min(accuracy) as worst_agent_accuracy,
        count(*) as n_agents,
        (
            array_agg(
                agent_name
                order by accuracy desc
            )
        ) [1] as best_agent_name,
        (
            array_agg(
                agent_name
                order by accuracy asc
            )
        ) [1] as worst_agent_name
    from agent_model_scores
    group by model_name
    having count(*) > 1
)
select 'Max gap between agents with a single model:' as analysis_type,
    model_name,
    max_gap,
    best_agent_accuracy,
    worst_agent_accuracy,
    best_agent_name,
    worst_agent_name,
    n_agents
from agent_gaps
order by max_gap desc
limit 1;