with
  p_hats as (
    select
      agent_name,
      model_name,
      task.name as task_name,
      avg(coalesce(reward, 0)) as p_hat,
      count(*) as n_trials,
      avg(jsonb_array_length(agent_metadata -> 'api_request_times_msec')) as avg_api_calls,
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
      t.config -> 'environment' ->> 'type' as env_type,
      j.config -> 'orchestrator' -> 'n_concurrent_trials' as n_concurrent_trials
    from
      trial as t
      inner join dataset_task as dt on dt.task_checksum = t.task_checksum
      inner join task on task.checksum = dt.task_checksum
      inner join trial_model as tm on tm.trial_id = t.id
      inner join job as j on j.id = t.job_id
    where
      dataset_name = 'terminal-bench'
      and dataset_version = '2.0'
      and j.created_at >= '2025-09-17 01:13:33.950824+00'::timestamptz
      and j.job_name like 'reliability%'
      and agent_name = 'terminus-2'
      and t.config -> 'environment' ->> 'type' in ('docker', 'daytona')
      and (
        exception_info is null
        or exception_info ->> 'exception_type' in (
          'AgentTimeoutError',
          'VerifierTimeoutError',
          'RuntimeError'
        )
      )
    group by
      agent_name,
      model_name,
      task_name,
      env_type,
      n_concurrent_trials
  )
select
  agent_name,
  model_name,
  env_type,
  round(avg(p_hat) * 100, 2) as accuracy,
  round(sum(n_errors) / sum(n_trials), 2) as error_probability,
  round(sum(avg_api_calls)) as total_avg_api_calls,
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
  count(distinct task_name) as n_tasks,
  n_concurrent_trials
from
  p_hats
group by
  agent_name,
  model_name,
  env_type,
  n_concurrent_trials
order by
  model_name,
  n_concurrent_trials asc;