with p_hats as (
    select agent_name,
        model_name,
        task.name as task_name,
        avg(coalesce(reward, 0)) as p_hat,
        count(*) as n_trials,
        avg(
            jsonb_array_length(agent_metadata->'api_request_times_msec')
        ) as avg_api_calls,
        sum(
            case
                when exception_info is null then 0
                else 1
            end
        ) as n_errors,
        case
            when count(*) > 1 then avg(coalesce(reward, 0)) * (1 - avg(coalesce(reward, 0))) / (count(*) - 1)
            else null
        end as partial_var,
        avg(n_input_tokens) as avg_n_input_tokens,
        avg(n_output_tokens) as avg_n_output_tokens,
        avg(
            n_input_tokens / 1000000.0 * m.cents_per_million_input_tokens + n_output_tokens / 1000000.0 * m.cents_per_million_output_tokens
        ) as avg_cost_cents,
        avg(
            extract(
                epoch
                from (
                        agent_execution_ended_at - agent_execution_started_at
                    )
            )
        ) as avg_execution_time_seconds
    from trial as t
        inner join dataset_task as dt on dt.task_checksum = t.task_checksum
        inner join task on task.checksum = dt.task_checksum
        inner join trial_model as tm on tm.trial_id = t.id
        inner join model as m on m.name = tm.model_name
        and m.provider = tm.model_provider
        inner join job as j on j.id = t.job_id
    where dataset_name = 'terminal-bench'
        and dataset_version = '2.0'
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
)
select agent_name,
    model_name,
    round(avg(p_hat) * 100, 2) as accuracy,
    round(sum(n_errors) / sum(n_trials), 2) as error_probability,
    round(sum(avg_api_calls)) as total_avg_api_calls,
    round(sum(avg_n_input_tokens) / 1000000.0, 2) || 'M' as total_avg_n_input_tokens,
    round(sum(avg_n_output_tokens) / 1000000.0, 2) || 'M' as total_avg_n_output_tokens,
    '$' || to_char(sum(avg_cost_cents) / 100.0, 'FM999999990.00') as total_avg_cost_usd,
    round(avg(avg_execution_time_seconds), 2) as avg_execution_time_sec,
    case
        when count(*) > count(partial_var) then null
        else round(sqrt(sum(partial_var)) / count(*) * 100, 2)
    end as stderr,
    case
        when count(*) > count(partial_var) then null
        else round(
            sqrt(sum(partial_var * n_trials)) / count(*) * 100,
            2
        )
    end as stddev,
    count(distinct task_name) as n_tasks
from p_hats
group by agent_name,
    model_name
having avg(p_hat) > 0.01
order by accuracy desc;