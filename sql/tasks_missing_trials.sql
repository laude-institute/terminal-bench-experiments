with task_trial_counts as (
    select "name" as task_name,
        agent_name,
        model_name,
        count(
            distinct case
                when trial.exception_info is null
                or trial.exception_info->>'exception_type' in ('AgentTimeoutError', 'VerifierTimeoutError') then trial.id
            end
        ) as n_trials
    from task
        inner join dataset_task on task.checksum = dataset_task.task_checksum
        inner join trial on task.checksum = trial.task_checksum
        inner join trial_model on trial.id = trial_model.trial_id
    where dataset_name = 'terminal-bench'
        and dataset_version = '2.0'
    group by "name",
        agent_name,
        model_name
)
select task_name,
    agent_name,
    model_name,
    n_trials
from task_trial_counts
where n_trials < 4;