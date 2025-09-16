import asyncio
from pathlib import Path

import yaml
from daytona import AsyncDaytona, CreateSnapshotParams, Image, Resources
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from sandboxes.models.job.config import JobConfig
from sandboxes.models.task.task import Task

console = Console()


async def create_snapshot(
    client: AsyncDaytona, task_config, progress, task_id, semaphore
):
    async with semaphore:
        task = Task(task_config.path)

        if not task.config.environment.docker_image:
            console.print(
                f"[yellow]Task {task.name} does not have a docker image[/yellow]"
            )
            progress.advance(task_id)
            return None

        progress.update(task_id, description=f"Building snapshot for: {task.name}")

        params = CreateSnapshotParams(
            name=task.name,
            image=Image.base(task.config.environment.docker_image),
            resources=Resources(
                cpu=2,
                memory=4,
                disk=10,
                gpu=0,
            ),
        )

        try:
            await client.snapshot.create(params=params, timeout=6000)
            progress.advance(task_id)
            return True
        except Exception as e:
            console.print(f"[red]Failed to create snapshot for {task.name}: {e}[/red]")
            progress.advance(task_id)
            return False


async def main():
    config = JobConfig.model_validate(
        yaml.safe_load(Path("configs/job.yaml").read_text())
    )

    client = AsyncDaytona()

    task_configs = list(config.datasets[0].get_task_configs())

    semaphore = asyncio.Semaphore(100)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Creating snapshots", total=len(task_configs))

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    create_snapshot(client, task_config, progress, task_id, semaphore)
                )
                for task_config in task_configs
            ]

        success_count = sum(1 for task in tasks if task.result() is True)
        console.print(
            f"\n[green]Successfully created {success_count} snapshots[/green]"
        )


if __name__ == "__main__":
    asyncio.run(main())
