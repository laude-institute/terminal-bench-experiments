import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import harbor
import yaml
from harbor.job import Job
from harbor.models.job.config import JobConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from harbor.orchestrators.base import OrchestratorEvent
from litellm import model_cost
from supabase import acreate_client
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

sys.path.append(str(Path(__file__).resolve().parent.parent))

from db.schema_public_latest import (
    AgentInsert,
    JobInsert,
    ModelInsert,
    TrialInsert,
    TrialModelInsert,
)


async def upload_trial_to_storage(result: TrialResult) -> str | None:
    """
    Upload trial directory as a tar.gz archive to Supabase storage and return public URL.
    Also uploads trajectory.json if it exists.

    Returns:
        Public URL of the trial archive in storage, or None if upload failed.
    """
    trial_path = Path(urlparse(result.trial_uri).path)

    if not trial_path.exists():
        print(f"Trial directory not found: {trial_path}")
        return None

    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    bucket_name = "trials"
    trial_id = str(result.id)

    # Create a temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Create tar.gz archive of the trial directory
        print(f"Creating tar.gz archive for trial {trial_id}...")
        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(trial_path, arcname=trial_path.name)

        archive_size = tmp_path.stat().st_size
        print(f"Archive created: {archive_size / (1024 * 1024):.2f} MB")

        # Upload the archive with retry logic
        async def upload_archive():
            """Upload the tar.gz file with retry logic."""
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                reraise=True,
            ):
                with attempt:
                    with open(tmp_path, "rb") as f:
                        storage_path = f"{trial_id}.tar.gz"
                        response = await client.storage.from_(bucket_name).upload(
                            file=f, path=storage_path, file_options={"upsert": "true"}
                        )
                    return response

        await upload_archive()
        print(f"Successfully uploaded {trial_id}.tar.gz")

        # Upload trajectory.json if it exists
        trajectory_path = trial_path / "agent" / "trajectory.json"
        if trajectory_path.exists():
            print(f"Found trajectory.json for trial {trial_id}, uploading...")

            async def upload_trajectory():
                """Upload the trajectory.json file with retry logic."""
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                    reraise=True,
                ):
                    with attempt:
                        with open(trajectory_path, "rb") as f:
                            storage_path = f"{trial_id}-traj.json"
                            response = await client.storage.from_(bucket_name).upload(
                                file=f,
                                path=storage_path,
                                file_options={"upsert": "true"},
                            )
                        return response

            try:
                await upload_trajectory()
                print(f"Successfully uploaded {trial_id}-traj.json")
            except Exception as e:
                print(f"Failed to upload trajectory.json for trial {trial_id}: {e}")

        # Get the public URL for the archive
        public_url = await client.storage.from_(bucket_name).get_public_url(
            f"{trial_id}.tar.gz"
        )

        return public_url

    except Exception as e:
        print(f"Failed to upload trial archive {trial_id}.tar.gz: {e}")
        return None

    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()


async def insert_trial_into_db(result: TrialResult):
    storage_url = await upload_trial_to_storage(result)

    trial_uri = storage_url
    if storage_url:
        pass
    else:
        print("Upload failed - trial_uri will be null")

    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    async def insert_agent():
        agent_insert = AgentInsert(
            name=result.agent_info.name,
            version=result.agent_info.version,
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("agent")
                    .upsert(
                        agent_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    async def get_dataset_task():
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("dataset_task")
                    .select("*, task!inner(name)")
                    .eq("dataset_name", "terminal-bench")
                    .eq("dataset_version", "2.0")
                    .eq("task.name", result.task_name)
                    .single()
                    .execute()
                )

    async def insert_trial():
        trial_insert = TrialInsert(
            id=result.id,
            agent_name=result.agent_info.name,
            agent_version=result.agent_info.version,
            config=result.config.model_dump(mode="json"),
            task_checksum=result.task_checksum,
            trial_name=result.trial_name,
            trial_uri=trial_uri,
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
                result.environment_setup.started_at
                if result.environment_setup
                else None
            ),
            environment_setup_ended_at=(
                result.environment_setup.finished_at
                if result.environment_setup
                else None
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
                Decimal(result.verifier_result.rewards.get("reward", 0))
                if result.verifier_result and result.verifier_result.rewards is not None
                else None
            ),
            started_at=result.started_at,
            ended_at=result.finished_at,
            agent_metadata=(
                result.agent_result.metadata if result.agent_result else None
            ),
        )
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("trial")
                    .insert(
                        trial_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    async def insert_model(name, provider, input_cost_per_token, output_cost_per_token):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("model")
                    .upsert(
                        ModelInsert(
                            name=name,
                            provider=provider,
                            cents_per_million_input_tokens=(
                                round(input_cost_per_token * 1e8)
                                if input_cost_per_token
                                else None
                            ),
                            cents_per_million_output_tokens=(
                                round(output_cost_per_token * 1e8)
                                if output_cost_per_token
                                else None
                            ),
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                    .execute()
                )

    async def insert_trial_model(name, provider):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("trial_model")
                    .insert(
                        TrialModelInsert(
                            trial_id=result.id,
                            model_name=name,  # type: ignore
                            model_provider=provider,  # type: ignore
                            n_input_tokens=(
                                result.agent_result.n_input_tokens
                                if result.agent_result
                                else None
                            ),
                            n_output_tokens=(
                                result.agent_result.n_output_tokens
                                if result.agent_result
                                else None
                            ),
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                    .execute()
                )

    try:
        await insert_agent()
        await insert_trial()

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

            await insert_model(
                name, provider, input_cost_per_token, output_cost_per_token
            )
            await insert_trial_model(name, provider)

    except Exception as e:
        print(
            f"Failed to insert trial {result.trial_name} into database after 3 retries: {e}"
        )
        return


async def insert_job_into_db(job_insert: JobInsert):
    client = await acreate_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    async def insert_job():
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
        ):
            with attempt:
                return await (
                    client.table("job")
                    .upsert(
                        job_insert.model_dump(
                            mode="json", by_alias=True, exclude_none=True
                        )
                    )
                    .execute()
                )

    try:
        await insert_job()
    except Exception as e:
        print(
            f"Failed to insert job {getattr(job_insert, 'job_name', 'unknown')} into database after 3 retries: {e}"
        )
        return


async def main():
    parser = argparse.ArgumentParser(
        description="Run a job with configurable job config"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="configs/job.yaml",
        help="Path to the job configuration file (default: configs/job.yaml)",
    )
    parser.add_argument(
        "-f",
        "--filter-error-types",
        action="append",
        help="Filter error types",
    )

    args = parser.parse_args()

    config_path = args.config
    config_text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        config_dict = json.loads(config_text)
    else:
        config_dict = yaml.safe_load(config_text)

    config = JobConfig.model_validate(config_dict)

    job_path = config.jobs_dir / config.job_name

    if (job_path / "config.json").exists() and args.filter_error_types:
        existing_config = JobConfig.model_validate_json(
            (job_path / "config.json").read_text()
        )

        if existing_config != config:
            raise ValueError(
                f"Job directory {job_path} already exists and cannot be "
                "resumed with a different config."
            )

        filter_error_types_set = set(args.filter_error_types)
        for trial_dir in job_path.iterdir():
            if not trial_dir.is_dir():
                continue

            trial_paths = TrialPaths(trial_dir)

            if not trial_paths.result_path.exists():
                continue

            try:
                trial_result = TrialResult.model_validate_json(
                    trial_paths.result_path.read_text()
                )
            except Exception:
                print(
                    f"Failed to parse trial result {trial_dir.name}. Removing trial "
                    "directory."
                )
                shutil.rmtree(trial_dir)
                continue

            if (
                trial_result.exception_info is not None
                and trial_result.exception_info.exception_type in filter_error_types_set
            ):
                print(
                    f"Removing trial directory with "
                    f"{trial_result.exception_info.exception_type}: {trial_dir.name}"
                )
                shutil.rmtree(trial_dir)

    job = Job(config=config)

    job_insert = JobInsert(
        id=job._id,
        config=config.model_dump(mode="json"),
        job_name=config.job_name,
        n_trials=len(job),
        username=os.environ.get("USER", "unknown"),
        git_commit_id=(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(harbor.__file__).parent,
            )
            .decode("utf-8")
            .strip()
        ),
        package_version=harbor.__version__,
    )

    if not job.is_resuming:
        await insert_job_into_db(job_insert)

    job._orchestrator.add_hook(
        event=OrchestratorEvent.TRIAL_COMPLETED,
        hook=insert_trial_into_db,
    )

    result = await job.run()

    job_insert.started_at = result.started_at
    job_insert.ended_at = result.finished_at
    job_insert.stats = result.stats.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )

    await insert_job_into_db(job_insert)


if __name__ == "__main__":
    asyncio.run(main())
