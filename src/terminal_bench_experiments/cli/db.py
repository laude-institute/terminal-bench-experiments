import os
import shutil
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import yaml
from dotenv import load_dotenv
from harbor.models.dataset_item import DownloadedDatasetItem
from harbor.models.job.config import LocalDatasetConfig
from harbor.models.registry import Dataset, RegistryTaskId
from harbor.models.task.task import Task
from harbor.registry.client import RegistryClient
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from supabase import create_client
from typer import Option, Typer

load_dotenv()

db_app = Typer(no_args_is_help=True)
console = Console()


@db_app.command()
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


@db_app.command()
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
            "[yellow]Could not import schema_public_latest. Please run `tbx db "
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
