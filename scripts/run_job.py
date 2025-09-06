import asyncio
import os
from decimal import Decimal

import yaml
from litellm import model_cost
from sandboxes.job import Job
from sandboxes.models.job.config import JobConfig
from sandboxes.models.trial.result import TrialResult
from sandboxes.orchestrators.base import OrchestratorEvent
from supabase import create_client

from db.schema_public_latest import (
    AgentInsert,
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

    client.table("agent").insert(
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
                    input_cost_per_token * 1e8 if input_cost_per_token else None
                ),
                cents_per_million_output_tokens=(
                    output_cost_per_token * 1e8 if output_cost_per_token else None
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


if __name__ == "__main__":
    with open("configs/job.yaml") as f:
        config_dict = yaml.safe_load(f)

    config = JobConfig.model_validate(config_dict)

    job = Job(config=config)

    job._orchestrator.add_hook(
        event=OrchestratorEvent.TRIAL_COMPLETED,
        hook=lambda result: insert_trial_into_db(result),
    )

    asyncio.run(job.run())
