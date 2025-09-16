import os
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sandboxes.models.environment_type import EnvironmentType
from sandboxes.models.job.config import JobConfig
from sandboxes.models.trial.config import TrialConfig
from supabase import create_client

load_dotenv()

client = create_client(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SECRET_KEY"],
)

N_REQUIRED_VALID_TRIALS = 5


def group_trials() -> dict[tuple[str, str, str], list[dict]]:
    response = (
        client.table("dataset_task")
        .select("*,task(*,trial(*,trial_model(*)))")
        .eq("dataset_name", "terminal-bench")
        .eq("dataset_version", "2.0")
        .eq("task.trial.agent_name", "terminus-2")
        .execute()
    )

    trial_configs_per_agent_model = defaultdict(list)

    for dataset in response.data:
        task = dataset.get("task")
        if not task:
            continue

        task_name = task.get("name", "N/A")
        task_trials = task.get("trial", [])

        for trial in task_trials:
            agent_name = trial["agent_name"]
            trial_models = trial.get("trial_model", [])

            if len(trial_models) != 1:
                print(
                    f"Trial {trial.get('id')} has {len(trial_models)} trial models. "
                    "This should never happen."
                )
                continue

            trial_model = trial_models[0]
            model_name = trial_model.get("model_name")

            trial_configs_per_agent_model[(agent_name, model_name, task_name)].append(
                trial
            )

    return trial_configs_per_agent_model


def get_remaining_trial_configs(
    grouped_trials: dict[tuple[str, str, str], list[dict]],
) -> list[TrialConfig]:
    remaining_trial_configs = []

    for (agent_name, model_name, task_name), trials in grouped_trials.items():
        n_valid_trials = sum(
            1
            for trial in trials
            if (
                trial.get("exception_info") is None
                or trial.get("exception_info", {}).get("exception_type")
                == "AgentTimeoutError"
            )
        )

        if n_valid_trials < 5:
            remaining_trial_configs.extend(
                [TrialConfig.model_validate(trials[0]["config"])]
                * (N_REQUIRED_VALID_TRIALS - n_valid_trials)
            )

    return remaining_trial_configs


def create_job_config(trial_configs: list[TrialConfig]) -> JobConfig:
    return JobConfig(
        trial_configs=trial_configs,
    )


def main():
    grouped_trials = group_trials()
    remaining_trial_configs = get_remaining_trial_configs(grouped_trials)
    job_config = create_job_config(remaining_trial_configs)

    job_config.orchestrator.n_concurrent_trials = 400
    job_config.orchestrator.quiet = True
    job_config.environment.type = EnvironmentType.DAYTONA
    job_config.environment.force_build = False
    job_config.environment.delete = True
    job_config.agents = []

    output_path = Path(f"configs/{job_config.job_name}.yaml")

    output_path.write_text(
        yaml.safe_dump(job_config.model_dump(mode="json"), sort_keys=False)
    )

    print(f"Job config written to {output_path}")
    print(f"Number of trials: {len(job_config.trial_configs)}")


if __name__ == "__main__":
    main()
