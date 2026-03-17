from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from harbor.models.registry import DatasetSpec, Registry, RegistryTaskId
from pydantic import BaseModel, Field, model_validator


class SourceDatasetConfig(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    sample_size: int = Field(gt=0)


class OutputConfig(BaseModel):
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    registry_path: Path


class SamplingManifest(BaseModel):
    source_registry: Path
    seed: str | int
    output: OutputConfig
    sources: list[SourceDatasetConfig]

    @model_validator(mode="after")
    def validate_sources(self) -> "SamplingManifest":
        if not self.sources:
            raise ValueError("sources must contain at least one dataset selector")

        seen: set[tuple[str, str]] = set()
        duplicates: list[str] = []
        for source in self.sources:
            key = (source.name, source.version)
            if key in seen:
                duplicates.append(f"{source.name}@{source.version}")
            seen.add(key)

        if duplicates:
            duplicate_text = ", ".join(sorted(set(duplicates)))
            raise ValueError(
                f"duplicate dataset selectors are not allowed: {duplicate_text}"
            )

        return self


@dataclass(frozen=True, slots=True)
class ResolvedSamplingManifest:
    manifest_path: Path
    source_registry_path: Path
    output_registry_path: Path
    seed: str
    output: OutputConfig
    sources: tuple[SourceDatasetConfig, ...]


@dataclass(frozen=True, slots=True)
class SelectedTask:
    task: RegistryTaskId
    fingerprint: str
    sample_rank: str


@dataclass(frozen=True, slots=True)
class SourceSample:
    name: str
    version: str
    requested_sample_size: int
    available_task_count: int
    selected_tasks: tuple[SelectedTask, ...]


@dataclass(frozen=True, slots=True)
class SamplingResult:
    manifest_path: Path
    source_registry_path: Path
    source_registry_sha256: str
    output_registry_path: Path
    output_lock_path: Path
    seed: str
    dataset: DatasetSpec
    source_samples: tuple[SourceSample, ...]

    @property
    def total_selected_tasks(self) -> int:
        return sum(len(source.selected_tasks) for source in self.source_samples)

    def to_registry_rows(self) -> list[dict[str, Any]]:
        return [self.dataset.model_dump(mode="json")]

    def to_lock_payload(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "source_registry": {
                "path": str(self.source_registry_path),
                "sha256": self.source_registry_sha256,
            },
            "seed": self.seed,
            "output": {
                "dataset_name": self.dataset.name,
                "dataset_version": self.dataset.version,
                "description": self.dataset.description,
                "registry_path": str(self.output_registry_path),
                "lock_path": str(self.output_lock_path),
            },
            "summary": {
                "source_dataset_count": len(self.source_samples),
                "selected_task_count": self.total_selected_tasks,
            },
            "sources": [
                {
                    "name": source.name,
                    "version": source.version,
                    "sample_size": source.requested_sample_size,
                    "available_task_count": source.available_task_count,
                    "selected_tasks": [
                        {
                            "name": selected.task.name,
                            "git_url": selected.task.git_url,
                            "git_commit_id": selected.task.git_commit_id,
                            "path": selected.task.path.as_posix(),
                            "fingerprint": selected.fingerprint,
                            "sample_rank": selected.sample_rank,
                        }
                        for selected in source.selected_tasks
                    ],
                }
                for source in self.source_samples
            ],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Output dataset: {self.dataset.name}@{self.dataset.version}",
            f"Source registry: {self.source_registry_path}",
            f"Seed: {self.seed}",
            f"Selected tasks: {self.total_selected_tasks}",
            f"Registry path: {self.output_registry_path}",
            f"Lock path: {self.output_lock_path}",
        ]

        for source in self.source_samples:
            lines.append(
                "  "
                f"- {source.name}@{source.version}: "
                f"{len(source.selected_tasks)}/{source.available_task_count}"
            )

        return lines


def load_manifest(manifest_path: Path) -> ResolvedSamplingManifest:
    manifest_path = manifest_path.expanduser().resolve()
    raw_manifest = yaml.safe_load(manifest_path.read_text())
    manifest_data = raw_manifest or {}
    manifest = SamplingManifest.model_validate(manifest_data)
    manifest_dir = manifest_path.parent

    return ResolvedSamplingManifest(
        manifest_path=manifest_path,
        source_registry_path=_resolve_path(manifest.source_registry, manifest_dir),
        output_registry_path=_resolve_path(manifest.output.registry_path, manifest_dir),
        seed=str(manifest.seed),
        output=manifest.output,
        sources=tuple(manifest.sources),
    )


def build_sampling_result(
    resolved_manifest: ResolvedSamplingManifest,
) -> SamplingResult:
    if not resolved_manifest.source_registry_path.exists():
        raise FileNotFoundError(
            f"source registry not found: {resolved_manifest.source_registry_path}"
        )

    source_registry = Registry.from_path(resolved_manifest.source_registry_path)
    dataset_lookup = _build_dataset_lookup(source_registry)
    source_samples: list[SourceSample] = []

    for source in resolved_manifest.sources:
        dataset_key = (source.name, source.version)
        if dataset_key not in dataset_lookup:
            raise ValueError(
                f"dataset not found in source registry: {source.name}@{source.version}"
            )

        dataset = dataset_lookup[dataset_key]
        available_task_count = len(dataset.tasks)
        if source.sample_size > available_task_count:
            raise ValueError(
                f"sample_size {source.sample_size} exceeds available tasks "
                f"({available_task_count}) for {source.name}@{source.version}"
            )

        source_samples.append(
            _sample_source_dataset(
                dataset=dataset,
                source=source,
                seed=resolved_manifest.seed,
            )
        )

    description = _build_output_description(
        user_description=resolved_manifest.output.description,
        source_samples=source_samples,
        seed=resolved_manifest.seed,
    )
    dataset = DatasetSpec(
        name=resolved_manifest.output.dataset_name,
        version=resolved_manifest.output.dataset_version,
        description=description,
        tasks=[
            selected.task
            for source_sample in source_samples
            for selected in source_sample.selected_tasks
        ],
    )

    output_lock_path = lock_path_for_registry(resolved_manifest.output_registry_path)
    return SamplingResult(
        manifest_path=resolved_manifest.manifest_path,
        source_registry_path=resolved_manifest.source_registry_path,
        source_registry_sha256=_sha256_file(resolved_manifest.source_registry_path),
        output_registry_path=resolved_manifest.output_registry_path,
        output_lock_path=output_lock_path,
        seed=resolved_manifest.seed,
        dataset=dataset,
        source_samples=tuple(source_samples),
    )


def run_from_manifest(manifest_path: Path, dry_run: bool = False) -> SamplingResult:
    resolved_manifest = load_manifest(manifest_path)
    result = build_sampling_result(resolved_manifest)

    if not dry_run:
        write_sampling_result(result)

    return result


def write_sampling_result(result: SamplingResult) -> None:
    result.output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_lock_path.parent.mkdir(parents=True, exist_ok=True)

    result.output_registry_path.write_text(
        json.dumps(result.to_registry_rows(), indent=2) + "\n"
    )
    result.output_lock_path.write_text(
        json.dumps(result.to_lock_payload(), indent=2) + "\n"
    )


def lock_path_for_registry(registry_path: Path) -> Path:
    if registry_path.suffix:
        return registry_path.with_suffix(".lock.json")
    return registry_path.parent / f"{registry_path.name}.lock.json"


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (base_dir / path).expanduser().resolve()


def _build_dataset_lookup(registry: Registry) -> dict[tuple[str, str], DatasetSpec]:
    lookup: dict[tuple[str, str], DatasetSpec] = {}
    for dataset in registry.datasets:
        key = (dataset.name, dataset.version)
        if key in lookup:
            raise ValueError(
                f"duplicate dataset entry in source registry: {dataset.name}@{dataset.version}"
            )
        lookup[key] = dataset
    return lookup


def _sample_source_dataset(
    dataset: DatasetSpec,
    source: SourceDatasetConfig,
    seed: str,
) -> SourceSample:
    ranked_tasks: list[SelectedTask] = []
    for task in dataset.tasks:
        fingerprint = _task_fingerprint(task)
        ranked_tasks.append(
            SelectedTask(
                task=task,
                fingerprint=fingerprint,
                sample_rank=_sample_rank(
                    seed=seed,
                    dataset_name=dataset.name,
                    dataset_version=dataset.version,
                    fingerprint=fingerprint,
                ),
            )
        )

    ranked_tasks.sort(key=lambda task: (task.sample_rank, task.fingerprint))
    selected_tasks = ranked_tasks[: source.sample_size]
    selected_tasks.sort(key=lambda task: (task.task.name, task.fingerprint))

    return SourceSample(
        name=source.name,
        version=source.version,
        requested_sample_size=source.sample_size,
        available_task_count=len(dataset.tasks),
        selected_tasks=tuple(selected_tasks),
    )


def _build_output_description(
    user_description: str,
    source_samples: Sequence[SourceSample],
    seed: str,
) -> str:
    source_summary = ", ".join(
        f"{source.name}@{source.version}:{source.requested_sample_size}"
        for source in source_samples
    )
    generated_text = (
        f"Generated adapter experiment sample with seed={seed}. "
        f"Source datasets: {source_summary}."
    )

    cleaned_description = user_description.strip()
    if cleaned_description:
        return f"{cleaned_description}\n\n{generated_text}"
    return generated_text


def _task_identity(task: RegistryTaskId) -> dict[str, Any]:
    return {
        "name": task.name,
        "git_url": task.git_url,
        "git_commit_id": task.git_commit_id,
        "path": task.path.as_posix(),
    }


def _task_fingerprint(task: RegistryTaskId) -> str:
    payload = json.dumps(_task_identity(task), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_rank(
    seed: str,
    dataset_name: str,
    dataset_version: str,
    fingerprint: str,
) -> str:
    raw = "\0".join([seed, dataset_name, dataset_version, fingerprint])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a synthetic Harbor registry for adapter experiments.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the sampling manifest YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved sample without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        result = run_from_manifest(args.manifest, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("Dry run: no files were written.")
    for line in result.summary_lines():
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
