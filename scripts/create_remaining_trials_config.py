import os
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv
from sandboxes.models.environment_type import EnvironmentType
from sandboxes.models.job.config import (
    JobConfig,
    OrchestratorConfig,
    OrchestratorType,
    RetryConfig,
)
from sandboxes.models.trial.config import EnvironmentConfig, TrialConfig
from supabase import create_client

load_dotenv()

client = create_client(
    supabase_url=os.environ["SUPABASE_URL"],
    supabase_key=os.environ["SUPABASE_SECRET_KEY"],
)

N_REQUIRED_VALID_TRIALS = 1
AGENT_NAME = "terminus-2"
DATASET_NAME = "terminal-bench"
DATASET_VERSION = "2.0"
JOB_NAME = "terminus-2-rerun"


def group_trials() -> dict[tuple[str, str, str], list[dict]]:
    response = (
        client.table("dataset_task")
        .select(
            """
            task!inner(
                name,
                trial(
                    id,
                    exception_info,
                    config,
                    agent_name,
                    trial_model(
                        model_name
                    ),
                    job!inner(
                        job_name
                    )
                )
            )
            """
        )
        .eq("dataset_name", DATASET_NAME)
        .eq("dataset_version", DATASET_VERSION)
        .eq("task.trial.agent_name", AGENT_NAME)
        .eq("task.trial.job.job_name", JOB_NAME)
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

        if n_valid_trials < N_REQUIRED_VALID_TRIALS:
            remaining_trial_configs.extend(
                [TrialConfig.model_validate(trials[0]["config"])]
                * (N_REQUIRED_VALID_TRIALS - n_valid_trials)
            )

    return remaining_trial_configs


def create_job_config(trial_configs: list[TrialConfig]) -> JobConfig:
    return JobConfig(
        trial_configs=trial_configs,
        orchestrator=OrchestratorConfig(
            type=OrchestratorType.LOCAL,
            n_concurrent_trials=200,
            retry=RetryConfig(
                max_retries=5,
                exclude_exceptions={"AgentTimeoutError"},
                max_wait_sec=60.0,
                min_wait_sec=10.0,
                wait_multiplier=2.0,
            ),
            quiet=True,
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType.DAYTONA,
            force_build=False,
            delete=True,
        ),
        agents=[],
        datasets=[],
        tasks=[],
    )


def main():
    grouped_trials = group_trials()
    remaining_trial_configs = get_remaining_trial_configs(grouped_trials)
    job_config = create_job_config(remaining_trial_configs)

    output_path = Path("configs") / AGENT_NAME / f"{job_config.job_name}.yaml"

    output_path.write_text(
        yaml.safe_dump(job_config.model_dump(mode="json"), sort_keys=False)
    )

    print(f"Job config written to {output_path}")
    print(f"Number of trials: {len(job_config.trial_configs)}")


if __name__ == "__main__":
    main()
