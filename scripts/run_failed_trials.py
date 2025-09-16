import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import sandboxes
from dotenv import load_dotenv
from litellm import model_cost
from sandboxes.job import Job
from sandboxes.models.job.config import JobConfig, OrchestratorConfig
from sandboxes.models.orchestrator_type import OrchestratorType
from sandboxes.models.trial.config import TrialConfig
from sandboxes.models.trial.result import TrialResult
from sandboxes.orchestrators.base import OrchestratorEvent
from supabase import create_client

sys.path.append(str(Path(__file__).resolve().parent.parent))

from db.schema_public_latest import (
    AgentInsert,
    JobInsert,
    ModelInsert,
    TrialInsert,
    TrialModelInsert,
)

load_dotenv()


def insert_trial_into_db(result: TrialResult):
    """Insert trial result into the database."""
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    agent_insert = AgentInsert(
        name=result.agent_info.name,
        version=result.agent_info.version,
    )

    client.table("agent").upsert(
        agent_insert.model_dump(mode="json", by_alias=True, exclude_none=True)
    ).execute()

    trial_insert = TrialInsert(
        id=result.id,
        agent_name=result.agent_info.name,
        agent_version=result.agent_info.version,
        config=result.config.model_dump(mode="json"),
        task_checksum=result.task_checksum,
        trial_name=result.trial_name,
        trial_uri=result.trial_uri,
        agent_execution_started_at=(
            result.agent_execution.started_at if result.agent_execution else None
        ),
        agent_execution_ended_at=(
            result.agent_execution.finished_at if result.agent_execution else None
        ),
        agent_setup_started_at=(
            result.agent_setup.started_at if result.agent_setup else None
        ),
        agent_setup_ended_at=(
            result.agent_setup.finished_at if result.agent_setup else None
        ),
        environment_setup_started_at=(
            result.environment_setup.started_at if result.environment_setup else None
        ),
        environment_setup_ended_at=(
            result.environment_setup.finished_at if result.environment_setup else None
        ),
        verifier_started_at=result.verifier.started_at if result.verifier else None,
        verifier_ended_at=result.verifier.finished_at if result.verifier else None,
        exception_info=(
            result.exception_info.model_dump(mode="json")
            if result.exception_info
            else None
        ),
        job_id=result.config.job_id,
        reward=(
            Decimal(result.verifier_result.reward)
            if result.verifier_result and result.verifier_result.reward is not None
            else None
        ),
        started_at=result.started_at,
        ended_at=result.finished_at,
        agent_metadata=result.agent_result.metadata if result.agent_result else None,
    )

    client.table("trial").insert(
        trial_insert.model_dump(mode="json", by_alias=True, exclude_none=True)
    ).execute()

    if result.agent_info.model_info:
        name = result.agent_info.model_info.name
        provider = result.agent_info.model_info.provider

        key = f"{provider}/{name}"
        token_costs = model_cost.get(key) or model_cost.get(name)

        input_cost_per_token = (
            token_costs.get("input_cost_per_token") if token_costs else None
        )
        output_cost_per_token = (
            token_costs.get("output_cost_per_token") if token_costs else None
        )

        if input_cost_per_token is None or output_cost_per_token is None:
            print(f"Could not find token costs for model: {key} or {name}")

        client.table("model").upsert(
            ModelInsert(
                name=name,
                provider=provider,
                cents_per_million_input_tokens=(
                    round(input_cost_per_token * 1e8) if input_cost_per_token else None
                ),
                cents_per_million_output_tokens=(
                    round(output_cost_per_token * 1e8)
                    if output_cost_per_token
                    else None
                ),
            ).model_dump(mode="json", by_alias=True, exclude_none=True)
        ).execute()

        client.table("trial_model").insert(
            TrialModelInsert(
                trial_id=result.id,
                model_name=name,  # type: ignore
                model_provider=provider,  # type: ignore
                n_input_tokens=(
                    result.agent_result.n_input_tokens if result.agent_result else None
                ),
                n_output_tokens=(
                    result.agent_result.n_output_tokens if result.agent_result else None
                ),
            ).model_dump(mode="json", by_alias=True, exclude_none=True)
        ).execute()


def insert_job_into_db(job_insert: JobInsert):
    """Insert job into the database."""
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    client.table("job").upsert(
        job_insert.model_dump(mode="json", by_alias=True, exclude_none=True)
    ).execute()


def get_all_agent_model_task_combinations():
    """Get all agent-model-task combinations for running 5 trials each."""
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    trials = (
        client.table("dataset_task")
        .select("*, task(*, trial(*, trial_model(*)))")
        .eq("dataset_name", "tbc-pre-release")
        .eq("dataset_version", "2.0")
        .neq("task.trial.agent_name", "oracle")
        .execute()
    )

    # Collect all unique agent-model-task combinations
    all_combinations = {}

    for dataset_task in trials.data:
        task = dataset_task.get("task")
        if not task:
            continue

        task_name = task.get("name", "N/A")
        task_trials = task.get("trial", [])

        for trial in task_trials:
            agent_name = trial.get("agent_name", "N/A")
            trial_models = trial.get("trial_model", [])

            # Skip if there are no trial_models
            if not trial_models:
                continue

            for trial_model in trial_models:
                model_name = trial_model.get("model_name", "N/A")
                key = (agent_name, model_name, task_name)

                # Store one representative trial config for each combination
                # Only store if we haven't seen this combination before, or if this trial
                # is not a timeout/null exception (prefer non-timeout trials as templates)
                if key not in all_combinations:
                    # Always store the first trial we see for this combination
                    all_combinations[key] = trial
                else:
                    # If we already have a trial for this combination, prefer non-timeout trials
                    existing_trial = all_combinations[key]
                    existing_exception = existing_trial.get("exception_info")
                    existing_is_timeout_or_null = (
                        existing_exception is None
                        or existing_exception.get("exception_type")
                        == "AgentTimeoutError"
                    )

                    current_exception = trial.get("exception_info")
                    current_is_timeout_or_null = (
                        current_exception is None
                        or current_exception.get("exception_type")
                        == "AgentTimeoutError"
                    )

                    # Replace existing trial if current one is not timeout/null and existing one is
                    if existing_is_timeout_or_null and not current_is_timeout_or_null:
                        all_combinations[key] = trial

    return all_combinations


def parse_trial_config(trial):
    return TrialConfig.model_validate(trial["config"])


def main():
    """Main function to run 5 trials for all agent-model-task combinations."""
    print("Fetching all agent-model-task combinations from database...")
    all_combinations = get_all_agent_model_task_combinations()

    if not all_combinations:
        print("No agent-model-task combinations found.")
        return

    print(f"Found {len(all_combinations)} unique agent-model-task combinations")

    # Create 5 trial configs for each combination
    trial_configs: list[TrialConfig] = []
    for (agent_name, model_name, task_name), trial in all_combinations.items():
        try:
            base_trial_config = parse_trial_config(trial)

            # Create 5 trials for this combination
            for _ in range(5):
                trial_configs.append(base_trial_config)

            print(
                f"Added 5 trial configs for: {agent_name} + {model_name} + {task_name}"
            )
        except Exception as e:
            print(
                f"Error parsing trial config for {agent_name} + {model_name} + {task_name}: {e}"
            )
            continue

    if not trial_configs:
        print("No valid trial configs could be parsed.")
        return

    print(
        f"Created {len(trial_configs)} total trial configs to run ({len(all_combinations)} combinations × 5 trials each)"
    )

    for trial_config in trial_configs:
        trial_config.agent.kwargs["max_episodes"] = 200

    job_config = JobConfig(
        orchestrator=OrchestratorConfig(
            type=OrchestratorType.LOCAL,
            n_concurrent_trials=50,
            quiet=False,
        ),
        trial_configs=trial_configs,
    )

    job = Job(config=job_config)

    job_insert = JobInsert(
        id=job._id,
        config=job_config.model_dump(mode="json"),
        job_name=job_config.job_name,
        n_trials=len(job._trial_configs),
        username=os.environ.get("USER", "unknown"),
        git_commit_id=(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(sandboxes.__file__).parent,
            )
            .decode("utf-8")
            .strip()
        ),
        package_version=sandboxes.__version__,
    )

    insert_job_into_db(job_insert)

    # Add hook to insert trial results
    job._orchestrator.add_hook(
        event=OrchestratorEvent.TRIAL_COMPLETED,
        hook=lambda result: insert_trial_into_db(result),
    )

    print("Starting job execution...")
    result = asyncio.run(job.run())

    # Update job with final results
    job_insert.started_at = result.started_at
    job_insert.ended_at = result.finished_at
    job_insert.metrics = (
        [
            metric.model_dump(mode="json", by_alias=True, exclude_none=True)
            for metric in result.metrics
        ]
        if result.metrics
        else None
    )
    job_insert.stats = result.stats.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )

    insert_job_into_db(job_insert)

    print("Job completed successfully!")


if __name__ == "__main__":
    main()
