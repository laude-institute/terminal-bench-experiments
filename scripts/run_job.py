import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import sandboxes
import yaml
from litellm import model_cost
from sandboxes.job import Job
from sandboxes.models.job.config import JobConfig
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


def insert_trial_into_db(result: TrialResult):
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
    )

    client.table("trial").insert(
        trial_insert.model_dump(mode="json", by_alias=True, exclude_none=True)
    ).execute()

    if result.agent_info.model_info:
        name = result.agent_info.model_info.name
        provider = result.agent_info.model_info.provider

        key = f"{name}/{provider}"
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
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    client.table("job").upsert(
        job_insert.model_dump(mode="json", by_alias=True, exclude_none=True)
    ).execute()


if __name__ == "__main__":
    with open("configs/job.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = JobConfig.model_validate(config_dict)

    job = Job(config=config)

    job_insert = JobInsert(
        id=job._id,
        config=config.model_dump(mode="json"),
        job_name=config.job_name,
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

    job._orchestrator.add_hook(
        event=OrchestratorEvent.TRIAL_COMPLETED,
        hook=lambda result: insert_trial_into_db(result),
    )

    result = asyncio.run(job.run())

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
