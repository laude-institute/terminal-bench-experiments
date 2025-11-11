import asyncio
from pathlib import Path

from daytona import AsyncDaytona, CreateSnapshotParams, Image, Resources
from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task
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

console = Console()


async def create_snapshot(client: AsyncDaytona, path, progress, task_id, semaphore):
    async with semaphore:
        task = Task(path)

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
                cpu=task.config.environment.cpus,
                memory=int(task.config.environment.memory.strip("G")),
                disk=int(task.config.environment.storage.strip("G")),
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
    paths = [
        path
        for path in Path("../sandboxes/tasks/tb-2/sb").iterdir()
        if TaskPaths(path).is_valid()
    ]
    client = AsyncDaytona()

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
        task_id = progress.add_task("Creating snapshots", total=len(paths))

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    create_snapshot(client, path, progress, task_id, semaphore)
                )
                for path in paths
            ]

        success_count = sum(1 for task in tasks if task.result() is True)
        console.print(
            f"\n[green]Successfully created {success_count} snapshots[/green]"
        )


if __name__ == "__main__":
    asyncio.run(main())
