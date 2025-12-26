import os
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import harbor
import yaml
from dotenv import load_dotenv
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.models.job.config import JobConfig, LocalDatasetConfig
from harbor.models.registry import Dataset, RegistryTaskId
from harbor.models.task.task import Task
from harbor.models.trial.result import TrialResult
from harbor.registry.client import RegistryClient
from litellm import model_cost
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential
from typer import Option, Typer

load_dotenv()

app = Typer(no_args_is_help=True)
console = Console()


def validate_job_config(config: JobConfig) -> list[str]:
    """
    Validate a job config meets terminal-bench submission requirements.

    Returns a list of validation errors (empty if valid).
    """
    errors: list[str] = []

    # Check timeout modifications
    if config.timeout_multiplier != 1.0:
        errors.append(
            f"timeout_multiplier is {config.timeout_multiplier}, expected 1.0"
        )

    # Check agent timeout overrides
    for i, agent in enumerate(config.agents):
        if agent.override_timeout_sec is not None:
            errors.append(
                f"agents[{i}].override_timeout_sec is set to {agent.override_timeout_sec}"
            )
        if agent.max_timeout_sec is not None:
            errors.append(
                f"agents[{i}].max_timeout_sec is set to {agent.max_timeout_sec}"
            )

    # Check verifier timeout overrides
    if config.verifier.override_timeout_sec is not None:
        errors.append(
            f"verifier.override_timeout_sec is set to {config.verifier.override_timeout_sec}"
        )
    if config.verifier.max_timeout_sec is not None:
        errors.append(
            f"verifier.max_timeout_sec is set to {config.verifier.max_timeout_sec}"
        )

    # Check resource modifications
    env = config.environment
    if env.override_cpus is not None:
        errors.append(f"environment.override_cpus is set to {env.override_cpus}")
    if env.override_memory_mb is not None:
        errors.append(
            f"environment.override_memory_mb is set to {env.override_memory_mb}"
        )
    if env.override_storage_mb is not None:
        errors.append(
            f"environment.override_storage_mb is set to {env.override_storage_mb}"
        )

    return errors


@app.command()
def generate_schemas():
    postgres_host = os.environ["SANDBOXES_POSTGRES_HOST"]
    postgres_port = os.environ["SANDBOXES_POSTGRES_PORT"]
    postgres_name = os.environ["SANDBOXES_POSTGRES_NAME"]
    postgres_user = os.environ["SANDBOXES_POSTGRES_USER"]
    postgres_password = os.environ["SANDBOXES_POSTGRES_PASSWORD"]

    postgres_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{
        postgres_port
    }/{postgres_name}"

    subprocess.run(
        [
            "sb-pydantic",
            "gen",
            "--type",
            "sqlalchemy",
            "--type",
            "pydantic",
            "--dir",
            "db",
            "--db-url",
            postgres_url,
        ]
    )

    src_dir = Path("db") / "fastapi"
    dst_dir = Path("db")

    if src_dir.exists():
        for src_file in src_dir.iterdir():
            if src_file.is_file():
                dst_file = dst_dir / src_file.name
                shutil.move(str(src_file), str(dst_file))
                console.print(f"[green]Moved {src_file.name} to {dst_dir}")

        try:
            shutil.rmtree(src_dir)
            console.print(f"[green]Removed directory {src_dir}")
        except Exception as e:
            console.print(f"[yellow]Could not remove {src_dir}: {e}")


@app.command()
def validate(
    job_path: Annotated[
        Path,
        Option("--job-path", help="Path to the job directory containing config.json"),
    ],
):
    """
    Validate that a job meets terminal-bench submission requirements.

    Checks:
      - No timeout modifications (timeout_multiplier=1.0, no override_timeout_sec)
      - No resource modifications (no override_cpus, override_memory_mb, override_storage_mb)
    """
    config_path = job_path / "config.json"
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = JobConfig.model_validate_json(config_path.read_text())
    errors = validate_job_config(config)

    # Print results
    if errors:
        console.print(f"\n[bold red]Validation failed with {len(errors)} error(s):")
        for error in errors:
            console.print(f"  [red]✗[/red] {error}")
        raise SystemExit(1)
    else:
        console.print("[bold green]Validation passed!")


@app.command()
def upload_dataset(
    name: Annotated[str, Option("-n", "--name", help="Name of the dataset to upload")],
    version: Annotated[
        str, Option("-v", "--version", help="Version of the dataset to upload")
    ],
    dataset_path: Annotated[
        Path | None,
        Option(
            "-p",
            "--path",
            help="Path to local dataset directory (alternative to registry)",
        ),
    ] = None,
    registry_path: Annotated[
        Path | None,
        Option("-r", "--registry-path", help="Path to local registry directory"),
    ] = None,
    registry_url: Annotated[
        str | None, Option("-u", "--registry-url", help="URL of remote registry")
    ] = None,
    config_file: Annotated[
        Path | None,
        Option(
            "-c",
            "--config",
            help="Path to job config YAML file containing dataset configuration",
        ),
    ] = None,
):
    try:
        from db.schema_public_latest import (
            DatasetInsert,
            DatasetTaskInsert,
            TaskInsert,
        )
    except ImportError:
        console.print(
            "[yellow]Could not import schema_public_latest. Please run `tbx "
            "generate-schemas` to generate the schema."
        )
        raise

    dataset_config = None
    # Handle config file if provided
    if config_file is not None:
        if not config_file.exists():
            console.print(f"[red]Config file not found: {config_file}")
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        # Extract dataset configuration from job config
        datasets_config = config_dict.get("datasets", [])
        if not datasets_config:
            console.print("[red]No datasets found in config file")
            raise ValueError("No datasets found in config file")

        # Use the first dataset if multiple are defined
        first_dataset = datasets_config[0]

        # Create dataset config object to use get_task_configs()
        if "path" in first_dataset:
            config_dataset_path = Path(first_dataset["path"])
            # Resolve relative paths from config file location
            if not config_dataset_path.is_absolute():
                config_dataset_path = (
                    config_file.parent / config_dataset_path
                ).resolve()

            dataset_config = LocalDatasetConfig(
                path=config_dataset_path,
                task_names=first_dataset.get("task_names"),
                exclude_task_names=first_dataset.get("exclude_task_names"),
            )

            # Override dataset_path if not already provided
            if dataset_path is None:
                dataset_path = config_dataset_path
                console.print(f"[cyan]Using dataset path from config: {dataset_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task(
            f"[cyan]Uploading dataset {name}@{version}...", total=5
        )

        errors = []

        try:
            progress.update(
                main_task, description="[yellow]Configuring registry URI..."
            )
            if registry_path is not None:
                registry_uri = registry_path.expanduser().resolve().as_uri()
            elif registry_url is not None:
                registry_uri = registry_url
            elif dataset_path is None:
                raise ValueError(
                    "Either registry_path or registry_url or dataset_path must be provided"
                )
            else:
                registry_uri = "none"

            progress.update(main_task, advance=1)

            dataset: Dataset | None = None

            if dataset_path is None:
                progress.update(
                    main_task, description="[yellow]Connecting to registry..."
                )
                registry_client = RegistryClient(
                    url=registry_url,
                    path=registry_path,
                )

                try:
                    dataset = registry_client.datasets[name][version]
                except KeyError:
                    progress.update(main_task, description="[red]✗ Dataset not found")
                    raise ValueError(
                        f"Dataset {name}@{version} not found in registry: {registry_uri}"
                    )

                progress.update(
                    main_task, description="[yellow]Downloading dataset items..."
                )
                try:
                    downloaded_dataset_items = registry_client.download_dataset(
                        name=name,
                        version=version,
                        overwrite=True,
                        output_dir=dataset_path,
                    )
                except Exception as e:
                    progress.update(main_task, description="[red]✗ Download failed")
                    raise RuntimeError(f"Failed to download dataset: {e}")

            else:
                progress.update(
                    main_task, description="[yellow]Loading local dataset items..."
                )
                # Use get_task_configs() if we have a dataset config from the config file
                if dataset_config is not None:
                    task_configs = dataset_config.get_task_configs()
                    downloaded_dataset_items = [
                        DownloadedDatasetItem(
                            id=RegistryTaskId(
                                name=task_config.path.name, path=task_config.path
                            ),
                            downloaded_path=task_config.path,
                        )
                        for task_config in task_configs
                    ]
                else:
                    downloaded_dataset_items = [
                        DownloadedDatasetItem(
                            id=RegistryTaskId(name=task_path.name, path=task_path),
                            downloaded_path=task_path,
                        )
                        for task_path in dataset_path.iterdir()
                        if task_path.is_dir()
                    ]

            progress.update(main_task, advance=1)
            console.print(
                f"[green]✓ Found {len(downloaded_dataset_items)} dataset items"
            )

            progress.update(main_task, description="[yellow]Connecting to database...")
            try:
                client = create_client(
                    supabase_url=os.environ["SUPABASE_URL"],
                    supabase_key=os.environ["SUPABASE_SECRET_KEY"],
                )
                progress.update(main_task, advance=1)
            except Exception as e:
                progress.update(
                    main_task, description="[red]✗ Database connection failed"
                )
                raise RuntimeError(f"Failed to connect to database: {e}")

            progress.update(
                main_task, description="[yellow]Uploading dataset metadata..."
            )
            try:
                client.table("dataset").upsert(
                    DatasetInsert(
                        name=name,
                        version=version,
                        registry_uri=registry_uri,
                        description=(
                            dataset.description
                            if dataset and dataset.description
                            else None
                        ),
                    ).model_dump(mode="json", by_alias=True, exclude_none=True)
                ).execute()
                progress.update(main_task, advance=1)
                console.print("[green]✓ Dataset metadata uploaded")
            except Exception as e:
                progress.update(main_task, description="[red]✗ Metadata upload failed")
                raise RuntimeError(f"Failed to upload dataset metadata: {e}")

            progress.update(main_task, description="[yellow]Processing tasks...")

            task_processing = progress.add_task(
                "[cyan]Processing tasks...", total=len(downloaded_dataset_items)
            )

            task_inserts = []
            dataset_task_inserts = []
            failed_tasks = []

            for downloaded_dataset_item in downloaded_dataset_items:
                try:
                    task_path = (
                        downloaded_dataset_item.id.path.expanduser().resolve()
                        if downloaded_dataset_item.id.git_url is None
                        else downloaded_dataset_item.id.path
                    )
                    task = Task(downloaded_dataset_item.downloaded_path)
                    task_inserts.append(
                        TaskInsert(
                            checksum=task.checksum,
                            instruction=task.instruction,
                            name=task.name,
                            source=None,
                            agent_timeout_sec=Decimal(task.config.agent.timeout_sec),
                            verifier_timeout_sec=Decimal(
                                task.config.verifier.timeout_sec
                            ),
                            path=str(task_path),
                            git_url=downloaded_dataset_item.id.git_url,
                            git_commit_id=downloaded_dataset_item.id.git_commit_id,
                            metadata=task.config.metadata,
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                    dataset_task_inserts.append(
                        DatasetTaskInsert(
                            dataset_name=name,
                            dataset_version=version,
                            dataset_registry_uri=registry_uri,
                            task_checksum=task.checksum,
                        ).model_dump(mode="json", by_alias=True, exclude_none=True)
                    )
                except Exception as e:
                    failed_tasks.append((downloaded_dataset_item.id.name, str(e)))
                    errors.append(
                        f"Failed to process task {downloaded_dataset_item.id.name}: {e}"
                    )

                progress.update(task_processing, advance=1)

            if failed_tasks:
                console.print(f"[yellow]⚠ {len(failed_tasks)} tasks failed to process")
                for task_name, error in failed_tasks[:5]:
                    console.print(f"  • {task_name}: {error}")
                if len(failed_tasks) > 5:
                    console.print(f"  ... and {len(failed_tasks) - 5} more")

            progress.update(
                main_task, description="[yellow]Uploading tasks to database..."
            )
            try:
                if task_inserts:
                    response = client.table("task").upsert(task_inserts).execute()
                    client.table("dataset_task").upsert(dataset_task_inserts).execute()
                progress.update(main_task, advance=1)
            except Exception as e:
                progress.update(main_task, description="[red]✗ Task upload failed")
                raise RuntimeError(f"Failed to upload tasks: {e}")

            progress.update(main_task, description="[green]✓ Dataset upload complete!")

            console.print(
                f"\n[bold green]Successfully uploaded dataset {name}@{version}"
            )
            console.print(f"  • Tasks uploaded: {len(task_inserts)}")
            if failed_tasks:
                console.print(f"  • Tasks failed: {len(failed_tasks)}")
            console.print(f"  • Registry URI: {registry_uri}")

            if errors:
                console.print("\n[yellow]Warnings encountered:")
                for error in errors[:10]:
                    console.print(f"  • {error}")
                if len(errors) > 10:
                    console.print(f"  ... and {len(errors) - 10} more")

        except Exception as e:
            progress.update(main_task, description="[red]✗ Upload failed")
            console.print(f"\n[bold red]Dataset upload failed: {e}")
            if errors:
                console.print("\n[yellow]Errors encountered before failure:")
                for error in errors[:5]:
                    console.print(f"  • {error}")
            raise


@app.command(name="import")
def import_jobs(
    jobs_dir: Annotated[
        Path | None,
        Option("--jobs-dir", help="Directory containing job subdirectories"),
    ] = None,
    job_path: Annotated[
        list[Path] | None,
        Option(
            "--job-path",
            help="Specific job directory path (can be specified multiple times)",
        ),
    ] = None,
    job_id: Annotated[
        list[str] | None,
        Option(
            "--job-id",
            help="Specific job ID to import (can be specified multiple times, requires --jobs-dir)",
        ),
    ] = None,
    override_model_name: Annotated[
        list[str] | None,
        Option(
            "-m",
            "--override-model-name",
            help="Override the model name for imported trials (can be specified multiple times, must match length of --override-model-provider)",
        ),
    ] = None,
    override_model_provider: Annotated[
        list[str] | None,
        Option(
            "-p",
            "--override-model-provider",
            help="Override the model provider for imported trials (can be specified multiple times, must match length of --override-model-name)",
        ),
    ] = None,
    agent_display_name: Annotated[
        str | None,
        Option(
            "--adn",
            "--agent-display-name",
            help="Display name for the agent",
        ),
    ] = None,
    agent_url: Annotated[
        str | None,
        Option(
            "--url",
            "--agent-url",
            help="URL for the agent",
        ),
    ] = None,
    agent_org_display_name: Annotated[
        str | None,
        Option(
            "--ao",
            "--agent-org-display-name",
            help="Organization display name for the agent",
        ),
    ] = None,
    model_display_name: Annotated[
        str | None,
        Option(
            "--mdn",
            "--model-display-name",
            help="Display name for the model",
        ),
    ] = None,
    model_org_display_name: Annotated[
        str | None,
        Option(
            "--mo",
            "--model-org-display-name",
            help="Organization display name for the model",
        ),
    ] = None,
    include_on_leaderboard: Annotated[
        bool,
        Option(
            "--lb",
            "--leaderboard",
            help="Include the job on the leaderboard",
        ),
    ] = False,
):
    """
    Import jobs and their trials from disk into the database.

    Usage examples:
      # Import all jobs from a directory
      tbx import-jobs --jobs-dir ./jobs

      # Import specific job paths (no jobs-dir needed)
      tbx import-jobs --job-path ./jobs/2025-09-06__19-58-44

      # Import multiple specific job paths
      tbx import-jobs --job-path ./jobs/job1 --job-path ./jobs/job2

      # Import specific jobs by ID (requires --jobs-dir)
      tbx import-jobs --jobs-dir ./jobs --job-id 6af21ded-0f2e-4878-8856-8a73dbe472cf
    """
    try:
        from db.schema_public_latest import (
            AgentInsert,
            JobInsert,
            ModelInsert,
            TrialInsert,
            TrialModelInsert,
        )
    except ImportError:
        console.print(
            "[yellow]Could not import schema_public_latest. Please run `tbx "
            "generate-schemas` to generate the schema."
        )
        raise

    # Validate parameter combinations
    if not jobs_dir and not job_path:
        console.print("[red]Either --jobs-dir or --job-path must be provided")
        raise ValueError("Either --jobs-dir or --job-path must be provided")

    if job_id and not jobs_dir:
        console.print("[red]--job-id can only be used with --jobs-dir")
        raise ValueError("--job-id can only be used with --jobs-dir")

    # Validate model override lists
    if override_model_name and override_model_provider:
        if len(override_model_name) != len(override_model_provider):
            console.print(
                f"[red]--override-model-name and --override-model-provider must have the same length. "
                f"Got {len(override_model_name)} names and {len(override_model_provider)} providers."
            )
            raise ValueError(
                "override_model_name and override_model_provider must have the same length"
            )
    elif override_model_name or override_model_provider:
        console.print(
            "[red]Both --override-model-name and --override-model-provider must be provided together"
        )
        raise ValueError(
            "Both override_model_name and override_model_provider must be provided together"
        )

    if jobs_dir and not jobs_dir.exists():
        console.print(f"[red]Jobs directory not found: {jobs_dir}")
        raise FileNotFoundError(f"Jobs directory not found: {jobs_dir}")

    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    # Discover jobs to import
    jobs_to_import: list[Path] = []

    if job_path:
        # Use specific job paths
        jobs_to_import.extend(job_path)
    elif job_id and jobs_dir:
        # Find jobs by ID within jobs_dir
        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir():
                continue
            config_path = job_dir / "config.json"
            if config_path.exists():
                try:
                    config = JobConfig.model_validate_json(config_path.read_text())
                    # Check if this job's trials contain the requested job_id
                    # The job_id is stored in trial configs
                    for trial_dir in job_dir.iterdir():
                        if not trial_dir.is_dir():
                            continue
                        trial_config_path = trial_dir / "config.json"
                        if trial_config_path.exists():
                            trial_config = yaml.safe_load(trial_config_path.read_text())
                            if trial_config.get("job_id") in job_id:
                                jobs_to_import.append(job_dir)
                                break
                except Exception as e:
                    console.print(f"[yellow]Failed to read config from {job_dir}: {e}")
                    continue
    elif jobs_dir:
        # Import all jobs from jobs_dir
        for job_dir in jobs_dir.iterdir():
            if job_dir.is_dir() and (job_dir / "config.json").exists():
                jobs_to_import.append(job_dir)

    if not jobs_to_import:
        console.print("[yellow]No jobs found to import")
        return

    console.print(f"[cyan]Found {len(jobs_to_import)} job(s) to import")

    # Validate all jobs before importing
    console.print("[cyan]Validating job configurations...")
    validation_failed = False
    for job_dir in jobs_to_import:
        config_path = job_dir / "config.json"
        if not config_path.exists():
            continue
        config = JobConfig.model_validate_json(config_path.read_text())
        validation_errors = validate_job_config(config)
        if validation_errors:
            validation_failed = True
            console.print(f"\n[bold red]Validation failed for {job_dir.name}:")
            for error in validation_errors:
                console.print(f"  [red]✗[/red] {error}")

    if validation_failed:
        console.print("\n[bold red]Aborting import due to validation errors.")
        raise SystemExit(1)

    console.print("[green]✓ All jobs passed validation")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        job_task = progress.add_task(
            "[cyan]Importing jobs...", total=len(jobs_to_import)
        )

        total_trials_imported = 0
        total_jobs_imported = 0
        errors = []

        for job_dir in jobs_to_import:
            try:
                # Load job config
                config_path = job_dir / "config.json"
                if not config_path.exists():
                    errors.append(f"No config.json found in {job_dir}")
                    continue

                config = JobConfig.model_validate_json(config_path.read_text())

                # Find all trial directories
                trial_dirs = [d for d in job_dir.iterdir() if d.is_dir()]

                if not trial_dirs:
                    errors.append(f"No trial directories found in {job_dir}")
                    continue

                progress.update(
                    job_task,
                    description=f"[cyan]Importing job {job_dir.name} ({len(trial_dirs)} trials)...",
                )

                # First, get or create the job_id from a trial config
                job_id_val = None
                for trial_dir in trial_dirs:
                    trial_config_path = trial_dir / "config.json"
                    if trial_config_path.exists():
                        trial_config = yaml.safe_load(trial_config_path.read_text())
                        job_id_val = trial_config.get("job_id")
                        if job_id_val:
                            break

                if not job_id_val:
                    errors.append(
                        f"Could not find job_id in any trial config for {job_dir}"
                    )
                    continue

                # Insert/update job
                job_insert = JobInsert(
                    id=job_id_val,
                    config=config.model_dump(mode="json"),
                    job_name=config.job_name,
                    n_trials=len(trial_dirs),
                    username=os.environ.get("USER", "unknown"),
                    git_commit_id=(
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=Path(harbor.__file__).parent,
                        )
                        .decode("utf-8")
                        .strip()
                        if Path(harbor.__file__).parent.joinpath(".git").exists()
                        else None
                    ),
                    package_version=harbor.__version__,
                    verified=False,
                    include_on_leaderboard=include_on_leaderboard,
                )

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                )
                def insert_job():
                    return (
                        client.table("job")
                        .upsert(
                            job_insert.model_dump(
                                mode="json", by_alias=True, exclude_none=True
                            )
                        )
                        .execute()
                    )

                insert_job()
                total_jobs_imported += 1

                # Collect all data for batch upload
                agent_inserts: dict[tuple[str, str], dict] = {}
                trial_inserts: list[dict] = []
                model_inserts: dict[tuple[str, str], dict] = {}
                trial_model_inserts: list[dict] = []

                # Import trials
                trial_progress = progress.add_task(
                    f"[cyan]Processing trials from {job_dir.name}...",
                    total=len(trial_dirs),
                )

                for trial_dir in trial_dirs:
                    result_path = trial_dir / "result.json"
                    if not result_path.exists():
                        progress.update(trial_progress, advance=1)
                        continue

                    try:
                        result = TrialResult.model_validate_json(
                            result_path.read_text()
                        )

                        # Collect agent (dedupe by name/version)
                        agent_key = (result.agent_info.name, result.agent_info.version)
                        if agent_key not in agent_inserts:
                            agent_inserts[agent_key] = AgentInsert(
                                name=result.agent_info.name,
                                version=result.agent_info.version,
                                display_name=agent_display_name,
                                url=agent_url,
                                org_display_name=agent_org_display_name,
                            ).model_dump(mode="json", by_alias=True, exclude_none=True)

                        # Collect trial
                        trial_inserts.append(
                            TrialInsert(
                                id=result.id,
                                agent_name=result.agent_info.name,
                                agent_version=result.agent_info.version,
                                config=result.config.model_dump(mode="json"),
                                task_checksum=result.task_checksum,
                                trial_name=result.trial_name,
                                trial_uri=result.trial_uri,
                                agent_execution_started_at=(
                                    result.agent_execution.started_at
                                    if result.agent_execution
                                    else None
                                ),
                                agent_execution_ended_at=(
                                    result.agent_execution.finished_at
                                    if result.agent_execution
                                    else None
                                ),
                                agent_setup_started_at=(
                                    result.agent_setup.started_at
                                    if result.agent_setup
                                    else None
                                ),
                                agent_setup_ended_at=(
                                    result.agent_setup.finished_at
                                    if result.agent_setup
                                    else None
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
                                verifier_started_at=(
                                    result.verifier.started_at
                                    if result.verifier
                                    else None
                                ),
                                verifier_ended_at=(
                                    result.verifier.finished_at
                                    if result.verifier
                                    else None
                                ),
                                exception_info=(
                                    result.exception_info.model_dump(mode="json")
                                    if result.exception_info
                                    else None
                                ),
                                job_id=result.config.job_id,
                                reward=(
                                    Decimal(
                                        result.verifier_result.rewards.get("reward", 0)
                                    )
                                    if result.verifier_result
                                    and result.verifier_result.rewards is not None
                                    else None
                                ),
                                started_at=result.started_at,
                                ended_at=result.finished_at,
                                agent_metadata=(
                                    result.agent_result.metadata
                                    if result.agent_result
                                    else None
                                ),
                            ).model_dump(mode="json", by_alias=True, exclude_none=True)
                        )

                        # Collect model information if available or overrides provided
                        model_info = result.agent_info.model_info

                        # Build list of (name, provider) pairs to insert
                        model_pairs = []
                        if override_model_name and override_model_provider:
                            model_pairs = list(
                                zip(override_model_name, override_model_provider)
                            )
                        elif model_info and model_info.name and model_info.provider:
                            model_pairs = [(model_info.name, model_info.provider)]

                        for name, provider in model_pairs:
                            if not name or not provider:
                                continue

                            # Collect model (dedupe by name/provider)
                            model_key = (name, provider)
                            if model_key not in model_inserts:
                                key = f"{provider}/{name}"
                                token_costs = model_cost.get(key) or model_cost.get(
                                    name
                                )

                                input_cost_per_token = (
                                    token_costs.get("input_cost_per_token")
                                    if token_costs
                                    else None
                                )
                                output_cost_per_token = (
                                    token_costs.get("output_cost_per_token")
                                    if token_costs
                                    else None
                                )

                                model_inserts[model_key] = ModelInsert(
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
                                    display_name=model_display_name,
                                    org_display_name=model_org_display_name,
                                ).model_dump(
                                    mode="json", by_alias=True, exclude_none=True
                                )

                            # Collect trial_model
                            trial_model_inserts.append(
                                TrialModelInsert(
                                    trial_id=result.id,
                                    model_name=name,
                                    model_provider=provider,
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
                                ).model_dump(
                                    mode="json", by_alias=True, exclude_none=True
                                )
                            )

                        total_trials_imported += 1

                    except Exception as e:
                        errors.append(f"Failed to process trial {trial_dir.name}: {e}")

                    progress.update(trial_progress, advance=1)

                progress.remove_task(trial_progress)

                # Batch upload all collected data
                progress.update(
                    job_task,
                    description=f"[cyan]Uploading {len(trial_inserts)} trials from {job_dir.name}...",
                )

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                )
                def batch_insert_agents():
                    if agent_inserts:
                        return (
                            client.table("agent")
                            .upsert(list(agent_inserts.values()))
                            .execute()
                        )

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                )
                def batch_insert_trials():
                    if trial_inserts:
                        return client.table("trial").upsert(trial_inserts).execute()

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                )
                def batch_insert_models():
                    if model_inserts:
                        return (
                            client.table("model")
                            .upsert(list(model_inserts.values()))
                            .execute()
                        )

                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=2, max=10),
                )
                def batch_insert_trial_models():
                    if trial_model_inserts:
                        return (
                            client.table("trial_model")
                            .upsert(trial_model_inserts)
                            .execute()
                        )

                # Execute batch inserts (agents and models first due to FK constraints)
                batch_insert_agents()
                batch_insert_models()
                batch_insert_trials()
                batch_insert_trial_models()

            except Exception as e:
                errors.append(f"Failed to import job {job_dir}: {e}")

            progress.update(job_task, advance=1)

    console.print("\n[bold green]Import complete!")
    console.print(f"  • Jobs imported: {total_jobs_imported}")
    console.print(f"  • Trials imported: {total_trials_imported}")

    if errors:
        console.print(f"\n[yellow]Encountered {len(errors)} error(s):")
        for error in errors[:10]:
            console.print(f"  • {error}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")


def main():
    app()


if __name__ == "__main__":
    main()
