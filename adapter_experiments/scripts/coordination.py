from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml
from pydantic import BaseModel, Field, model_validator

from subset_registry import build_subset_result, write_subset_result


FORBIDDEN_JOB_KEYS = {"job_name", "agents", "datasets"}
REPO_ROOT = Path(__file__).resolve().parents[2]


class ExperimentConfig(BaseModel):
    name: str = Field(min_length=1)
    output_root: Path
    registry_name: str = Field(default="local", min_length=1)


class TaskSetConfig(BaseModel):
    registry_path: Path
    lock_path: Path


class PhaseConfig(BaseModel):
    percent: float = Field(gt=0, le=100)
    dataset_name: str = Field(min_length=1)
    dataset_version: str | None = None
    description: str | None = None
    rounding: Literal["ceil", "floor"] = "ceil"
    minimum_tasks_per_source: int = Field(default=1, ge=0)


class AdapterConfig(BaseModel):
    agent: dict[str, Any] | None = None
    job_overrides: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_adapter(self) -> "AdapterConfig":
        if self.agent is not None and not self.agent:
            raise ValueError("adapter.agent must not be empty")
        _validate_forbidden_job_keys(self.job_overrides, location="adapter.job_overrides")
        return self


class ContributorConfig(BaseModel):
    adapters: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contributor(self) -> "ContributorConfig":
        if not self.adapters:
            raise ValueError("contributors must have at least one adapter")
        return self


class CoordinationManifest(BaseModel):
    experiment: ExperimentConfig
    task_set: TaskSetConfig
    phases: dict[str, PhaseConfig]
    job_template: dict[str, Any] | None = None
    adapters: dict[str, AdapterConfig]
    contributors: dict[str, ContributorConfig]

    @model_validator(mode="after")
    def validate_manifest(self) -> "CoordinationManifest":
        if not self.phases:
            raise ValueError("phases must contain at least one phase")
        if not self.adapters:
            raise ValueError("adapters must contain at least one adapter")
        if not self.contributors:
            raise ValueError("contributors must contain at least one contributor")

        if self.job_template is not None:
            _validate_forbidden_job_keys(self.job_template, location="job_template")

        requires_job_template = any(
            adapter.agent is not None for adapter in self.adapters.values()
        )
        if requires_job_template and self.job_template is None:
            raise ValueError(
                "job_template is required when any adapter defines an agent config"
            )

        duplicated_assignments: dict[str, list[str]] = {}
        assigned_counts: dict[str, int] = {}
        for contributor_name, contributor in self.contributors.items():
            seen_for_contributor: set[str] = set()
            for adapter_id in contributor.adapters:
                if adapter_id in seen_for_contributor:
                    raise ValueError(
                        f"contributor {contributor_name} lists adapter {adapter_id} more than once"
                    )
                seen_for_contributor.add(adapter_id)

                if adapter_id not in self.adapters:
                    raise ValueError(
                        f"contributor {contributor_name} references unknown adapter {adapter_id}"
                    )

                assigned_counts[adapter_id] = assigned_counts.get(adapter_id, 0) + 1
                duplicated_assignments.setdefault(adapter_id, []).append(contributor_name)

        duplicates = {
            adapter_id: contributor_names
            for adapter_id, contributor_names in duplicated_assignments.items()
            if assigned_counts.get(adapter_id, 0) > 1
        }
        if duplicates:
            duplicate_text = ", ".join(
                f"{adapter_id} -> {', '.join(contributor_names)}"
                for adapter_id, contributor_names in sorted(duplicates.items())
            )
            raise ValueError(f"adapters may only be assigned once: {duplicate_text}")

        return self


@dataclass(frozen=True, slots=True)
class ResolvedCoordinationManifest:
    manifest_path: Path
    experiment_name: str
    output_root: Path
    registry_name: str
    task_set_registry_path: Path
    task_set_lock_path: Path
    phases: tuple[tuple[str, PhaseConfig], ...]
    job_template: dict[str, Any] | None
    adapters: dict[str, AdapterConfig]
    contributors: dict[str, ContributorConfig]

    def phase_names(self) -> list[str]:
        return [phase_name for phase_name, _ in self.phases]

    def phase_config(self, phase_name: str) -> PhaseConfig:
        for current_phase_name, phase_config in self.phases:
            if current_phase_name == phase_name:
                return phase_config
        raise KeyError(phase_name)


@dataclass(frozen=True, slots=True)
class ContributorMaterialization:
    contributor_name: str
    phase_name: str
    registry_path: Path
    lock_path: Path
    assignment_path: Path
    config_paths: tuple[Path, ...]


def load_manifest(manifest_path: Path) -> ResolvedCoordinationManifest:
    manifest_path = manifest_path.expanduser().resolve()
    raw_manifest = yaml.safe_load(manifest_path.read_text()) or {}
    manifest = CoordinationManifest.model_validate(raw_manifest)
    manifest_dir = manifest_path.parent

    return ResolvedCoordinationManifest(
        manifest_path=manifest_path,
        experiment_name=manifest.experiment.name,
        output_root=_resolve_path(manifest.experiment.output_root, manifest_dir),
        registry_name=manifest.experiment.registry_name,
        task_set_registry_path=_resolve_path(
            manifest.task_set.registry_path,
            manifest_dir,
        ),
        task_set_lock_path=_resolve_path(
            manifest.task_set.lock_path,
            manifest_dir,
        ),
        phases=tuple(manifest.phases.items()),
        job_template=copy.deepcopy(manifest.job_template),
        adapters=manifest.adapters,
        contributors=manifest.contributors,
    )


def materialize_phase(
    manifest: ResolvedCoordinationManifest,
    phase_name: str,
) -> tuple[Path, Path, str, str]:
    phase_names = manifest.phase_names()
    if phase_name not in phase_names:
        raise ValueError(f"unknown phase: {phase_name}")

    if not manifest.task_set_registry_path.exists():
        raise FileNotFoundError(
            f"task set registry not found: {manifest.task_set_registry_path}"
        )
    if not manifest.task_set_lock_path.exists():
        raise FileNotFoundError(f"task set lock not found: {manifest.task_set_lock_path}")

    target_index = phase_names.index(phase_name)
    dataset_name = ""
    dataset_version = ""

    for current_index, current_phase_name in enumerate(phase_names[: target_index + 1]):
        current_phase = manifest.phase_config(current_phase_name)
        output_registry_path = phase_registry_path(manifest, current_phase_name)
        exclude_lock_paths = [
            phase_lock_path(manifest, previous_phase_name)
            for previous_phase_name in phase_names[:current_index]
        ]

        result = build_subset_result(
            source_registry_path=manifest.task_set_registry_path,
            source_lock_path=manifest.task_set_lock_path,
            exclude_lock_paths=exclude_lock_paths,
            output_registry_path=output_registry_path,
            percent=current_phase.percent,
            rounding=current_phase.rounding,
            minimum_tasks_per_source=current_phase.minimum_tasks_per_source,
            dataset_name=current_phase.dataset_name,
            dataset_version=current_phase.dataset_version,
            description=current_phase.description,
        )
        write_subset_result(result)

        dataset_name = result.dataset_name
        dataset_version = result.dataset_version

    return (
        phase_registry_path(manifest, phase_name),
        phase_lock_path(manifest, phase_name),
        dataset_name,
        dataset_version,
    )


def materialize_contributor(
    manifest: ResolvedCoordinationManifest,
    contributor_name: str,
    phase_name: str,
    runner_manifest: ResolvedCoordinationManifest | None = None,
    runner_contributor_name: str | None = None,
) -> ContributorMaterialization:
    contributor = manifest.contributors.get(contributor_name)
    if contributor is None:
        raise ValueError(f"unknown contributor: {contributor_name}")
    if runner_manifest is not None:
        inline_agent_adapters = [
            adapter_id
            for adapter_id in contributor.adapters
            if manifest.adapters[adapter_id].agent is not None
        ]
        if inline_agent_adapters:
            raise ValueError(
                "runner_manifest cannot be combined with inline adapter agents in the base "
                f"manifest: {', '.join(inline_agent_adapters)}"
            )

    registry_path, lock_path, dataset_name, dataset_version = materialize_phase(
        manifest=manifest,
        phase_name=phase_name,
    )

    output_dir = contributor_config_dir(manifest, contributor_name, phase_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_path = output_dir / 'assignment.yaml'
    assignment_payload = build_assignment_payload(
        manifest=manifest,
        contributor_name=contributor_name,
        phase_name=phase_name,
        registry_path=registry_path,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        runner_manifest=runner_manifest,
        runner_contributor_name=runner_contributor_name,
    )
    assignment_path.write_text(yaml.safe_dump(assignment_payload, sort_keys=False))

    config_paths: list[Path] = []
    if runner_manifest is not None:
        runner_adapter_ids = resolve_runner_adapter_ids(
            runner_manifest=runner_manifest,
            runner_contributor_name=runner_contributor_name,
        )
        phase_registry_row = load_single_dataset_registry_row(registry_path)
        for adapter_id in contributor.adapters:
            adapter = manifest.adapters[adapter_id]
            adapter_registry_path = output_dir / 'registries' / f'{adapter_id}.json'
            materialize_adapter_registry(
                adapter_id=adapter_id,
                adapter=adapter,
                phase_registry_row=phase_registry_row,
                phase_lock_path=lock_path,
                output_registry_path=adapter_registry_path,
            )
            for runner_adapter_id in runner_adapter_ids:
                runner_adapter = runner_manifest.adapters[runner_adapter_id]
                config_path = output_dir / f"{adapter_id}__{runner_adapter_id}.yaml"
                config_payload = build_composed_job_config(
                    benchmark_manifest=manifest,
                    runner_manifest=runner_manifest,
                    phase_name=phase_name,
                    benchmark_adapter_id=adapter_id,
                    benchmark_adapter=adapter,
                    runner_adapter_id=runner_adapter_id,
                    runner_adapter=runner_adapter,
                    config_path=config_path,
                    registry_path=adapter_registry_path,
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                )
                config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
                config_paths.append(config_path)
    elif manifest.job_template is not None:
        for adapter_id in contributor.adapters:
            adapter = manifest.adapters[adapter_id]
            if adapter.agent is None:
                continue
            config_path = output_dir / f"{adapter_id}.yaml"
            config_payload = build_job_config(
                manifest=manifest,
                phase_name=phase_name,
                adapter_id=adapter_id,
                adapter=adapter,
                config_path=config_path,
                registry_path=registry_path,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
            )
            config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False))
            config_paths.append(config_path)

    return ContributorMaterialization(
        contributor_name=contributor_name,
        phase_name=phase_name,
        registry_path=registry_path,
        lock_path=lock_path,
        assignment_path=assignment_path,
        config_paths=tuple(config_paths),
    )


def build_assignment_payload(
    manifest: ResolvedCoordinationManifest,
    contributor_name: str,
    phase_name: str,
    registry_path: Path,
    dataset_name: str,
    dataset_version: str,
    runner_manifest: ResolvedCoordinationManifest | None = None,
    runner_contributor_name: str | None = None,
) -> dict[str, Any]:
    contributor = manifest.contributors[contributor_name]
    job_config_mode = 'none'
    if runner_manifest is not None:
        job_config_mode = 'runner_manifest'
    elif any(manifest.adapters[adapter_id].agent is not None for adapter_id in contributor.adapters):
        job_config_mode = 'inline_adapters'
    adapters = []
    for adapter_id in contributor.adapters:
        adapter = manifest.adapters[adapter_id]
        adapters.append(
            {
                'adapter_id': adapter_id,
                'metadata': copy.deepcopy(adapter.metadata),
                'has_generated_job_config': (
                    runner_manifest is not None or adapter.agent is not None
                ),
            }
        )

    return {
        'experiment': manifest.experiment_name,
        'phase': phase_name,
        'dataset': {
            'name': dataset_name,
            'version': dataset_version,
            'registry_name': manifest.registry_name,
            'registry_path': str(registry_path),
        },
        'contributor': {
            'name': contributor_name,
            'metadata': copy.deepcopy(contributor.metadata),
        },
        'job_configs': {
            'mode': job_config_mode,
            'runner_manifest': (
                str(runner_manifest.manifest_path) if runner_manifest is not None else None
            ),
            'runner_contributor': runner_contributor_name,
        },
        'adapters': adapters,
    }


def build_job_config(
    manifest: ResolvedCoordinationManifest,
    phase_name: str,
    adapter_id: str,
    adapter: AdapterConfig,
    config_path: Path,
    registry_path: Path,
    dataset_name: str,
    dataset_version: str,
) -> dict[str, Any]:
    if manifest.job_template is None:
        raise ValueError('job_template is required to build job configs')
    if adapter.agent is None:
        raise ValueError(f'adapter {adapter_id} does not define an agent config')

    payload = copy.deepcopy(manifest.job_template)
    payload['job_name'] = _build_job_name(
        experiment_name=manifest.experiment_name,
        phase_name=phase_name,
        adapter_id=adapter_id,
    )
    payload['agents'] = [copy.deepcopy(adapter.agent)]
    payload['datasets'] = [
        {
            'registry': {
                'name': manifest.registry_name,
                'path': _repo_relative_path(registry_path),
            },
            'name': dataset_name,
            'version': dataset_version,
        }
    ]
    return _deep_merge(payload, copy.deepcopy(adapter.job_overrides))


def build_composed_job_config(
    benchmark_manifest: ResolvedCoordinationManifest,
    runner_manifest: ResolvedCoordinationManifest,
    phase_name: str,
    benchmark_adapter_id: str,
    benchmark_adapter: AdapterConfig,
    runner_adapter_id: str,
    runner_adapter: AdapterConfig,
    config_path: Path,
    registry_path: Path,
    dataset_name: str,
    dataset_version: str,
) -> dict[str, Any]:
    if runner_manifest.job_template is None:
        raise ValueError('runner manifest must define job_template to build job configs')
    if runner_adapter.agent is None:
        raise ValueError(
            f'runner adapter {runner_adapter_id} does not define an agent config'
        )

    payload = copy.deepcopy(runner_manifest.job_template)
    combined_adapter_id = f"{benchmark_adapter_id}__{runner_adapter_id}"
    payload['job_name'] = _build_job_name(
        experiment_name=benchmark_manifest.experiment_name,
        phase_name=phase_name,
        adapter_id=combined_adapter_id,
    )
    payload['agents'] = [copy.deepcopy(runner_adapter.agent)]
    payload['datasets'] = [
        {
            'registry': {
                'name': benchmark_manifest.registry_name,
                'path': _repo_relative_path(registry_path),
            },
            'name': dataset_name,
            'version': dataset_version,
        }
    ]

    payload = _deep_merge(payload, copy.deepcopy(runner_adapter.job_overrides))
    return _deep_merge(payload, benchmark_job_overrides(benchmark_adapter))


def resolve_runner_adapter_ids(
    runner_manifest: ResolvedCoordinationManifest,
    runner_contributor_name: str | None,
) -> tuple[str, ...]:
    if runner_manifest.job_template is None:
        raise ValueError('runner manifest must define job_template')

    if runner_contributor_name is None:
        adapter_ids = tuple(runner_manifest.adapters.keys())
    else:
        contributor = runner_manifest.contributors.get(runner_contributor_name)
        if contributor is None:
            raise ValueError(f"unknown runner contributor: {runner_contributor_name}")
        adapter_ids = tuple(contributor.adapters)

    if not adapter_ids:
        selection = runner_contributor_name or '<all>'
        raise ValueError(f"runner manifest does not define any adapters for {selection}")

    missing_agents = [
        adapter_id
        for adapter_id in adapter_ids
        if runner_manifest.adapters[adapter_id].agent is None
    ]
    if missing_agents:
        raise ValueError(
            'runner manifest adapters must define agent configs: '
            + ', '.join(missing_agents)
        )

    return adapter_ids


def materialize_adapter_registry(
    adapter_id: str,
    adapter: AdapterConfig,
    phase_registry_row: dict[str, Any],
    phase_lock_path: Path,
    output_registry_path: Path,
) -> None:
    source_keys = adapter_source_keys(adapter=adapter, adapter_id=adapter_id)
    raw_lock_payload = json.loads(phase_lock_path.read_text())
    source_payloads = raw_lock_payload.get('sources')
    if not isinstance(source_payloads, list):
        raise ValueError(f"phase lock must contain a sources list: {phase_lock_path}")

    matched_sources: list[tuple[str, str]] = []
    selected_tasks: list[dict[str, Any]] = []
    seen_tasks: set[tuple[str | None, str | None, str | None, str | None]] = set()

    for source_payload in source_payloads:
        if not isinstance(source_payload, dict):
            raise ValueError('phase lock sources entries must be objects')
        source_name = source_payload.get('name')
        source_version = source_payload.get('version')
        if not isinstance(source_name, str) or not source_name:
            raise ValueError('phase lock source entry is missing name')
        if not isinstance(source_version, str) or not source_version:
            raise ValueError('phase lock source entry is missing version')
        if (source_name, source_version) not in source_keys and (source_name, None) not in source_keys:
            continue

        matched_sources.append((source_name, source_version))
        source_tasks = source_payload.get('selected_tasks')
        if not isinstance(source_tasks, list):
            raise ValueError(
                'phase lock source entry must contain selected_tasks when composing '
                f'runner configs for {adapter_id}'
            )
        for raw_task in source_tasks:
            sanitized_task = sanitize_registry_task(raw_task)
            task_key = (
                sanitized_task.get('name'),
                sanitized_task.get('git_url'),
                sanitized_task.get('git_commit_id'),
                sanitized_task.get('path'),
            )
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)
            selected_tasks.append(sanitized_task)

    if not matched_sources:
        source_text = ', '.join(
            f"{name}@{version}" if version is not None else name
            for name, version in sorted(source_keys, key=lambda item: (item[0], item[1] or ''))
        )
        raise ValueError(
            f"adapter {adapter_id} source_datasets do not match any phase-lock sources: "
            f"{source_text}"
        )

    output_registry_row = copy.deepcopy(phase_registry_row)
    output_registry_row['tasks'] = selected_tasks
    output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    output_registry_path.write_text(json.dumps([output_registry_row], indent=2) + '\n')


def adapter_source_keys(
    adapter: AdapterConfig,
    adapter_id: str,
) -> set[tuple[str, str | None]]:
    raw_sources = adapter.metadata.get('source_datasets')
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(
            f"adapter {adapter_id} must define metadata.source_datasets to compose "
            'runner configs'
        )

    source_keys: set[tuple[str, str | None]] = set()
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(
                f"adapter {adapter_id} metadata.source_datasets[{index}] must be an object"
            )
        source_name = raw_source.get('name')
        if not isinstance(source_name, str) or not source_name:
            raise ValueError(
                f"adapter {adapter_id} metadata.source_datasets[{index}] is missing name"
            )
        source_version = raw_source.get('version')
        if source_version is not None:
            source_version = str(source_version)
        source_keys.add((source_name, source_version))

    return source_keys


def benchmark_job_overrides(adapter: AdapterConfig) -> dict[str, Any]:
    synthesized_overrides: dict[str, Any] = {
        'n_attempts': 5,
    }
    environment_name = adapter.metadata.get('environment')
    if isinstance(environment_name, str) and environment_name in {'docker', 'daytona'}:
        synthesized_overrides['environment'] = {'type': environment_name}
        synthesized_overrides['orchestrator'] = {
            'n_concurrent_trials': 4 if environment_name == 'docker' else 32,
        }
    return _deep_merge(synthesized_overrides, copy.deepcopy(adapter.job_overrides))


def load_single_dataset_registry_row(registry_path: Path) -> dict[str, Any]:
    registry_rows = json.loads(registry_path.read_text())
    if not isinstance(registry_rows, list) or len(registry_rows) != 1:
        raise ValueError(
            f"registry must contain exactly one dataset entry: {registry_path}"
        )
    dataset_row = registry_rows[0]
    if not isinstance(dataset_row, dict):
        raise ValueError(f"registry dataset entry must be an object: {registry_path}")
    return dataset_row


def sanitize_registry_task(raw_task: Any) -> dict[str, Any]:
    if not isinstance(raw_task, dict):
        raise ValueError('registry task entries must be objects')
    sanitized: dict[str, Any] = {}
    for key in ('name', 'git_url', 'git_commit_id', 'path'):
        value = raw_task.get(key)
        if value is not None:
            sanitized[key] = value
    if not sanitized.get('path'):
        raise ValueError('registry task entries must define path')
    return sanitized


def phase_registry_path(
    manifest: ResolvedCoordinationManifest,
    phase_name: str,
) -> Path:
    return manifest.output_root / 'phases' / phase_name / 'registry.json'


def phase_lock_path(
    manifest: ResolvedCoordinationManifest,
    phase_name: str,
) -> Path:
    return manifest.output_root / 'phases' / phase_name / 'registry.lock.json'


def contributor_config_dir(
    manifest: ResolvedCoordinationManifest,
    contributor_name: str,
    phase_name: str,
) -> Path:
    return manifest.output_root / 'contributors' / contributor_name / phase_name


def print_manifest_summary(
    manifest: ResolvedCoordinationManifest,
    contributor_name: str | None = None,
) -> None:
    print(f"Manifest: {manifest.manifest_path}")
    print(f"Experiment: {manifest.experiment_name}")
    print(f"Output root: {manifest.output_root}")
    print(f"Task set registry: {manifest.task_set_registry_path}")
    print(f"Task set lock: {manifest.task_set_lock_path}")
    print('Phases:')
    phase_names = manifest.phase_names()
    for index, (phase_name, phase) in enumerate(manifest.phases):
        previous_phases = phase_names[:index]
        exclusion_text = ''
        if previous_phases:
            exclusion_text = f" excluding {', '.join(previous_phases)}"
        dataset_version = phase.dataset_version or '<source version>'
        print(
            '  '
            f"- {phase_name}: {phase.percent:g}% -> "
            f"{phase.dataset_name}@{dataset_version}{exclusion_text}"
        )

    if contributor_name is not None:
        contributor = manifest.contributors.get(contributor_name)
        if contributor is None:
            raise ValueError(f"unknown contributor: {contributor_name}")
        print(f"Contributor: {contributor_name}")
        for adapter_id in contributor.adapters:
            adapter = manifest.adapters[adapter_id]
            adapter_type = adapter.metadata.get('adapter_type') or 'unspecified'
            environment = adapter.metadata.get('environment') or 'unspecified'
            paper_tasks = adapter.metadata.get('paper_task_count')
            paper_tasks_text = f", paper_tasks={paper_tasks}" if paper_tasks is not None else ''
            print(
                '  '
                f"- {adapter_id} [{adapter_type}, {environment}{paper_tasks_text}]"
            )
        return

    print('Contributors:')
    for current_contributor_name, contributor in manifest.contributors.items():
        print(
            '  '
            f"- {current_contributor_name}: {', '.join(contributor.adapters)}"
        )


def _validate_forbidden_job_keys(payload: dict[str, Any], location: str) -> None:
    forbidden = sorted(key for key in payload if key in FORBIDDEN_JOB_KEYS)
    if forbidden:
        raise ValueError(
            f"{location} may not set {', '.join(forbidden)}; those are generated automatically"
        )


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (base_dir / path).expanduser().resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        existing_value = merged.get(key)
        if isinstance(existing_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing_value, value)
        else:
            merged[key] = value
    return merged


def _relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start=start)).as_posix()


def _repo_relative_path(path: Path) -> str:
    # Job configs are executed from the repository root, not from each YAML's directory.
    return _relative_path(path.resolve(), REPO_ROOT)


def _build_job_name(experiment_name: str, phase_name: str, adapter_id: str) -> str:
    raw = f"{adapter_id}__{experiment_name}__{phase_name}"
    return raw.replace('/', '-').replace(' ', '-')


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Materialize contributor-local phase registries and assignment/config files from a '
            'single coordinated adapter experiment manifest.'
        )
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    show_parser = subparsers.add_parser(
        'show',
        help='Print the contributor assignments and phase plan.',
    )
    show_parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help='Path to the coordination manifest YAML file.',
    )
    show_parser.add_argument(
        '--contributor',
        type=str,
        default=None,
        help='Optional contributor name to inspect.',
    )

    phase_parser = subparsers.add_parser(
        'phase',
        help='Generate the local phase registry and lock for the requested phase.',
    )
    phase_parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help='Path to the coordination manifest YAML file.',
    )
    phase_parser.add_argument(
        '--phase',
        type=str,
        required=True,
        help='Phase name to materialize, such as phase2, phase3, or phase4.',
    )

    contributor_parser = subparsers.add_parser(
        'contributor',
        help='Generate the local phase registry plus assignment/config files for one contributor.',
    )
    contributor_parser.add_argument(
        '--manifest',
        type=Path,
        required=True,
        help='Path to the coordination manifest YAML file.',
    )
    contributor_parser.add_argument(
        '--phase',
        type=str,
        required=True,
        help='Phase name to materialize, such as phase2, phase3, or phase4.',
    )
    contributor_parser.add_argument(
        '--contributor',
        type=str,
        required=True,
        help='Contributor name defined in the coordination manifest.',
    )
    contributor_parser.add_argument(
        '--runner-manifest',
        type=Path,
        default=None,
        help=(
            'Optional coordination manifest whose adapters define runner/model agent '
            'configs. When provided, contributor configs are generated for the cartesian '
            'product of assigned benchmark adapters and selected runner adapters.'
        ),
    )
    contributor_parser.add_argument(
        '--runner-contributor',
        type=str,
        default=None,
        help=(
            'Optional contributor defined in --runner-manifest. When omitted, all runner '
            'adapters from that manifest are used.'
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)

        if args.command == 'show':
            print_manifest_summary(manifest, contributor_name=args.contributor)
            return 0

        if args.command == 'phase':
            registry_path, lock_path, dataset_name, dataset_version = materialize_phase(
                manifest=manifest,
                phase_name=args.phase,
            )
            print(f"Generated phase registry: {registry_path}")
            print(f"Generated phase lock: {lock_path}")
            print('Upload dataset metadata with:')
            print(
                '  '
                'uv run tbx upload-dataset '
                f"-n {shlex.quote(dataset_name)} "
                f"-v {shlex.quote(dataset_version)} "
                f"--registry-path {shlex.quote(str(registry_path))}"
            )
            return 0

        if args.command == 'contributor':
            runner_manifest = (
                load_manifest(args.runner_manifest)
                if args.runner_manifest is not None
                else None
            )
            materialization = materialize_contributor(
                manifest=manifest,
                contributor_name=args.contributor,
                phase_name=args.phase,
                runner_manifest=runner_manifest,
                runner_contributor_name=args.runner_contributor,
            )
            print(f"Contributor: {materialization.contributor_name}")
            print(f"Phase: {materialization.phase_name}")
            print(f"Registry: {materialization.registry_path}")
            print(f"Lock: {materialization.lock_path}")
            print(f"Assignment: {materialization.assignment_path}")
            if materialization.config_paths:
                print('Configs:')
                for config_path in materialization.config_paths:
                    print(f"  - {config_path}")
                print('Run jobs with:')
                for config_path in materialization.config_paths:
                    print(
                        '  '
                        'uv run python scripts/run_job.py '
                        f"-c {shlex.quote(str(config_path))}"
                    )
            else:
                print(
                    'No job configs were generated because the manifest does not define '
                    'inline adapter agent settings. Provide --runner-manifest to compose '
                    'benchmark assignments with a model/agent manifest, or use a manifest '
                    'that defines adapter.agent blocks.'
                )
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
