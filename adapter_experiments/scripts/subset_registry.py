from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SourceKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ExcludedLock:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceSelection:
    name: str
    version: str
    full_sample_size: int
    available_task_count: int
    selected_tasks: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class SubsetResult:
    source_registry_path: Path
    source_registry_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    exclude_locks: tuple[ExcludedLock, ...]
    output_registry_path: Path
    output_lock_path: Path
    dataset_name: str
    dataset_version: str
    description: str
    percent: float
    rounding: str
    minimum_tasks_per_source: int
    total_source_count: int
    total_selected_task_count: int
    output_registry_rows: tuple[dict[str, Any], ...]
    source_selections: tuple[SourceSelection, ...]

    def to_lock_payload(self) -> dict[str, Any]:
        return {
            "source_registry": {
                "path": str(self.source_registry_path),
                "sha256": self.source_registry_sha256,
            },
            "source_lock": {
                "path": str(self.source_lock_path),
                "sha256": self.source_lock_sha256,
            },
            "exclude_locks": [
                {
                    "path": str(excluded.path),
                    "sha256": excluded.sha256,
                }
                for excluded in self.exclude_locks
            ],
            "selection": {
                "percent": self.percent,
                "fraction": self.percent / 100.0,
                "rounding": self.rounding,
                "minimum_tasks_per_source": self.minimum_tasks_per_source,
            },
            "output": {
                "dataset_name": self.dataset_name,
                "dataset_version": self.dataset_version,
                "description": self.description,
                "registry_path": str(self.output_registry_path),
                "lock_path": str(self.output_lock_path),
            },
            "summary": {
                "source_dataset_count": self.total_source_count,
                "selected_task_count": self.total_selected_task_count,
            },
            "sources": [
                {
                    "name": source.name,
                    "version": source.version,
                    "full_sample_size": source.full_sample_size,
                    "available_task_count": source.available_task_count,
                    "selected_subset_size": len(source.selected_tasks),
                    "selected_tasks": list(source.selected_tasks),
                }
                for source in self.source_selections
            ],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Output dataset: {self.dataset_name}@{self.dataset_version}",
            f"Source registry: {self.source_registry_path}",
            f"Source lock: {self.source_lock_path}",
        ]
        if self.exclude_locks:
            lines.append(
                "Excluded locks: "
                + ", ".join(str(excluded.path) for excluded in self.exclude_locks)
            )
        lines.extend(
            [
                f"Selection: {self.percent:g}% per source "
                f"(rounding={self.rounding}, min={self.minimum_tasks_per_source})",
                f"Selected tasks: {self.total_selected_task_count}",
                f"Registry path: {self.output_registry_path}",
                f"Lock path: {self.output_lock_path}",
            ]
        )
        for source in self.source_selections:
            lines.append(
                "  "
                f"- {source.name}@{source.version}: "
                f"{len(source.selected_tasks)}/{source.full_sample_size}"
            )
        return lines


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text())


def build_subset_result(
    source_registry_path: Path,
    source_lock_path: Path,
    exclude_lock_paths: Sequence[Path],
    output_registry_path: Path,
    percent: float,
    rounding: str,
    minimum_tasks_per_source: int,
    dataset_name: str | None,
    dataset_version: str | None,
    description: str | None,
) -> SubsetResult:
    source_registry_path = source_registry_path.expanduser().resolve()
    source_lock_path = source_lock_path.expanduser().resolve()
    output_registry_path = output_registry_path.expanduser().resolve()
    output_lock_path = lock_path_for_registry(output_registry_path)

    registry_rows = load_json_file(source_registry_path)
    if not isinstance(registry_rows, list) or len(registry_rows) != 1:
        raise ValueError(
            "source registry must be a JSON array with exactly one dataset entry"
        )

    source_dataset = registry_rows[0]
    if not isinstance(source_dataset, dict):
        raise ValueError("source registry dataset entry must be a JSON object")

    source_tasks = source_dataset.get("tasks")
    if not isinstance(source_tasks, list):
        raise ValueError("source registry dataset entry must contain a tasks list")

    source_lock = load_json_file(source_lock_path)
    if not isinstance(source_lock, dict):
        raise ValueError("source lock must be a JSON object")

    source_specs = source_lock.get("sources")
    if not isinstance(source_specs, list) or not source_specs:
        raise ValueError("source lock must contain a non-empty sources list")

    _validate_source_registry_matches_lock(source_tasks, source_specs)
    exclude_locks, excluded_counts_by_source = _load_excluded_task_counts(
        source_specs=source_specs,
        exclude_lock_paths=exclude_lock_paths,
    )

    all_selected_counts: Counter[str] = Counter()
    source_selections: list[SourceSelection] = []

    for source in source_specs:
        source_name = _require_str(source, "name")
        source_version = _require_str(source, "version")
        source_key = (source_name, source_version)
        selected_tasks = _require_task_list(source, "selected_tasks")
        full_sample_size = len(selected_tasks)
        available_task_count = _require_int(source, "available_task_count")
        subset_size = _compute_subset_size(
            task_count=full_sample_size,
            percent=percent,
            rounding=rounding,
            minimum_tasks_per_source=minimum_tasks_per_source,
        )
        selected_counts = _ranked_selection_counts(
            selected_tasks=selected_tasks,
            subset_size=subset_size,
        )
        ordered_selected_tasks = _select_tasks_in_source_order(
            selected_tasks=selected_tasks,
            included_counts=selected_counts,
            excluded_counts=excluded_counts_by_source.get(source_key, Counter()),
        )

        for task in ordered_selected_tasks:
            all_selected_counts[_lock_task_fingerprint(task)] += 1

        source_selections.append(
            SourceSelection(
                name=source_name,
                version=source_version,
                full_sample_size=full_sample_size,
                available_task_count=available_task_count,
                selected_tasks=tuple(ordered_selected_tasks),
            )
        )

    output_tasks = _project_selected_tasks_to_registry(
        source_tasks=source_tasks,
        selected_counts=all_selected_counts,
    )

    output_dataset_name = dataset_name or output_registry_path.stem
    output_dataset_version = dataset_version or _require_str(source_dataset, "version")
    output_description = _build_output_description(
        source_dataset=source_dataset,
        source_selections=source_selections,
        percent=percent,
        rounding=rounding,
        minimum_tasks_per_source=minimum_tasks_per_source,
        description_override=description,
        source_lock_path=source_lock_path,
        exclude_locks=exclude_locks,
    )

    output_dataset = {
        "name": output_dataset_name,
        "version": output_dataset_version,
        "description": output_description,
        "tasks": output_tasks,
    }

    return SubsetResult(
        source_registry_path=source_registry_path,
        source_registry_sha256=_sha256_file(source_registry_path),
        source_lock_path=source_lock_path,
        source_lock_sha256=_sha256_file(source_lock_path),
        exclude_locks=exclude_locks,
        output_registry_path=output_registry_path,
        output_lock_path=output_lock_path,
        dataset_name=output_dataset_name,
        dataset_version=output_dataset_version,
        description=output_description,
        percent=percent,
        rounding=rounding,
        minimum_tasks_per_source=minimum_tasks_per_source,
        total_source_count=len(source_selections),
        total_selected_task_count=len(output_tasks),
        output_registry_rows=(output_dataset,),
        source_selections=tuple(source_selections),
    )


def write_subset_result(result: SubsetResult) -> None:
    result.output_registry_path.parent.mkdir(parents=True, exist_ok=True)
    result.output_lock_path.parent.mkdir(parents=True, exist_ok=True)

    result.output_registry_path.write_text(
        json.dumps(list(result.output_registry_rows), indent=2) + "\n"
    )
    result.output_lock_path.write_text(
        json.dumps(result.to_lock_payload(), indent=2) + "\n"
    )


def lock_path_for_registry(registry_path: Path) -> Path:
    if registry_path.suffix:
        return registry_path.with_suffix(".lock.json")
    return registry_path.parent / f"{registry_path.name}.lock.json"


def _validate_source_registry_matches_lock(
    source_tasks: Sequence[dict[str, Any]],
    source_specs: Sequence[dict[str, Any]],
) -> None:
    registry_counts = Counter(_task_fingerprint(task) for task in source_tasks)
    lock_counts = Counter()

    for source in source_specs:
        if not isinstance(source, dict):
            raise ValueError("each source entry in the lock must be a JSON object")
        for task in _require_task_list(source, "selected_tasks"):
            lock_counts[_lock_task_fingerprint(task)] += 1

    if registry_counts != lock_counts:
        raise ValueError(
            "source registry tasks do not exactly match the selected tasks "
            "recorded in the source lock"
        )


def _load_excluded_task_counts(
    source_specs: Sequence[dict[str, Any]],
    exclude_lock_paths: Sequence[Path],
) -> tuple[tuple[ExcludedLock, ...], dict[SourceKey, Counter[str]]]:
    base_counts_by_source: dict[SourceKey, Counter[str]] = {}
    excluded_counts_by_source: dict[SourceKey, Counter[str]] = {}

    for source in source_specs:
        source_key = (_require_str(source, "name"), _require_str(source, "version"))
        if source_key in base_counts_by_source:
            raise ValueError(
                f"duplicate source entry in the base lock: {source_key[0]}@{source_key[1]}"
            )

        selected_tasks = _require_task_list(source, "selected_tasks")
        base_counts = Counter(_lock_task_fingerprint(task) for task in selected_tasks)
        base_counts_by_source[source_key] = base_counts
        excluded_counts_by_source[source_key] = Counter()

    exclude_locks: list[ExcludedLock] = []

    for exclude_lock_path in exclude_lock_paths:
        resolved_path = exclude_lock_path.expanduser().resolve()
        exclude_lock = load_json_file(resolved_path)
        if not isinstance(exclude_lock, dict):
            raise ValueError("exclude lock must be a JSON object")

        exclude_sources = exclude_lock.get("sources")
        if not isinstance(exclude_sources, list):
            raise ValueError("exclude lock must contain a sources list")

        seen_source_keys: set[SourceKey] = set()
        for exclude_source in exclude_sources:
            if not isinstance(exclude_source, dict):
                raise ValueError("exclude lock sources must be JSON objects")

            source_key = (
                _require_str(exclude_source, "name"),
                _require_str(exclude_source, "version"),
            )
            if source_key in seen_source_keys:
                raise ValueError(
                    "duplicate source entry in exclude lock: "
                    f"{source_key[0]}@{source_key[1]}"
                )
            seen_source_keys.add(source_key)

            if source_key not in base_counts_by_source:
                raise ValueError(
                    "exclude lock contains a source not present in the base lock: "
                    f"{source_key[0]}@{source_key[1]}"
                )

            for task in _require_task_list(exclude_source, "selected_tasks"):
                excluded_counts_by_source[source_key][_lock_task_fingerprint(task)] += 1

        exclude_locks.append(
            ExcludedLock(
                path=resolved_path,
                sha256=_sha256_file(resolved_path),
            )
        )

    for source_key, excluded_counts in excluded_counts_by_source.items():
        base_counts = base_counts_by_source[source_key]
        overflow = [
            fingerprint
            for fingerprint, count in excluded_counts.items()
            if count > base_counts[fingerprint]
        ]
        if overflow:
            raise ValueError(
                "exclude locks overspecify tasks for source "
                f"{source_key[0]}@{source_key[1]}"
            )

    return tuple(exclude_locks), excluded_counts_by_source


def _ranked_selection_counts(
    selected_tasks: Sequence[dict[str, Any]],
    subset_size: int,
) -> Counter[str]:
    ranked_tasks: list[tuple[str, str, int]] = []
    for index, task in enumerate(selected_tasks):
        fingerprint = _lock_task_fingerprint(task)
        sample_rank = _require_str(task, "sample_rank")
        ranked_tasks.append((sample_rank, fingerprint, index))

    ranked_tasks.sort(key=lambda item: (item[0], item[1], item[2]))
    return Counter(fingerprint for _, fingerprint, _ in ranked_tasks[:subset_size])


def _select_tasks_in_source_order(
    selected_tasks: Sequence[dict[str, Any]],
    included_counts: Counter[str],
    excluded_counts: Counter[str],
) -> list[dict[str, Any]]:
    remaining_included = Counter(included_counts)
    remaining_excluded = Counter(excluded_counts)
    final_tasks: list[dict[str, Any]] = []

    for task in selected_tasks:
        fingerprint = _lock_task_fingerprint(task)
        if remaining_included[fingerprint] <= 0:
            continue

        remaining_included[fingerprint] -= 1
        if remaining_excluded[fingerprint] > 0:
            remaining_excluded[fingerprint] -= 1
            continue

        final_tasks.append(task)

    return final_tasks


def _project_selected_tasks_to_registry(
    source_tasks: Sequence[dict[str, Any]],
    selected_counts: Counter[str],
) -> list[dict[str, Any]]:
    remaining_counts = Counter(selected_counts)
    output_tasks: list[dict[str, Any]] = []

    for task in source_tasks:
        fingerprint = _task_fingerprint(task)
        if remaining_counts[fingerprint] <= 0:
            continue
        output_tasks.append(task)
        remaining_counts[fingerprint] -= 1

    missing = sorted(
        fingerprint for fingerprint, count in remaining_counts.items() if count > 0
    )
    if missing:
        raise ValueError(
            "unable to project all selected tasks back into the source registry: "
            f"{', '.join(missing[:3])}"
        )

    return output_tasks


def _compute_subset_size(
    task_count: int,
    percent: float,
    rounding: str,
    minimum_tasks_per_source: int,
) -> int:
    if task_count <= 0:
        return 0

    raw_size = task_count * percent / 100.0
    if rounding == "ceil":
        subset_size = math.ceil(raw_size)
    elif rounding == "floor":
        subset_size = math.floor(raw_size)
    else:
        raise ValueError(f"unsupported rounding mode: {rounding}")

    subset_size = max(subset_size, minimum_tasks_per_source)
    return min(task_count, subset_size)


def _build_output_description(
    source_dataset: dict[str, Any],
    source_selections: Sequence[SourceSelection],
    percent: float,
    rounding: str,
    minimum_tasks_per_source: int,
    description_override: str | None,
    source_lock_path: Path,
    exclude_locks: Sequence[ExcludedLock],
) -> str:
    base_description = description_override
    if base_description is None:
        base_description = source_dataset.get("description", "")

    source_summary = ", ".join(
        f"{source.name}@{source.version}:{len(source.selected_tasks)}/{source.full_sample_size}"
        for source in source_selections
    )
    exclusion_text = ""
    if exclude_locks:
        exclusion_text = (
            " Excluding tasks from: "
            + ", ".join(excluded.path.name for excluded in exclude_locks)
            + "."
        )

    generated_text = (
        f"Progressive subset derived from {source_lock_path.name} with "
        f"{percent:g}% per source dataset "
        f"(rounding={rounding}, minimum_tasks_per_source={minimum_tasks_per_source})."
        f"{exclusion_text} Selected tasks: {source_summary}."
    )

    cleaned_description = str(base_description).strip()
    if cleaned_description:
        return f"{cleaned_description}\n\n{generated_text}"
    return generated_text


def _require_task_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    tasks: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{key} entries must be JSON objects")
        tasks.append(item)
    return tasks


def _task_identity(task: dict[str, Any]) -> dict[str, str]:
    return {
        "name": _require_str(task, "name"),
        "git_url": _require_str(task, "git_url"),
        "git_commit_id": _require_str(task, "git_commit_id"),
        "path": _require_str(task, "path"),
    }


def _task_fingerprint(task: dict[str, Any]) -> str:
    payload = json.dumps(_task_identity(task), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _lock_task_fingerprint(task: dict[str, Any]) -> str:
    expected = task.get("fingerprint")
    computed = _task_fingerprint(task)
    if expected is None:
        return computed
    if not isinstance(expected, str):
        raise ValueError("lock task fingerprint must be a string")
    if expected != computed:
        raise ValueError(
            f"lock task fingerprint mismatch for task {_require_str(task, 'name')}"
        )
    return expected


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic progressive subset registry from a flattened "
            "full registry and its lock file."
        ),
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        required=True,
        help="Path to the flattened full registry JSON file.",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        required=True,
        help="Path to the lock file for the flattened full registry.",
    )
    parser.add_argument(
        "--exclude-lock",
        type=Path,
        action="append",
        default=[],
        help="Optional lock file to subtract from the selected tasks. Repeatable.",
    )
    parser.add_argument(
        "--output-registry",
        type=Path,
        required=True,
        help="Path to write the subset registry JSON file.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        required=True,
        help="Percentage of tasks to keep from each source dataset, such as 1, 10, or 100.",
    )
    parser.add_argument(
        "--rounding",
        choices=("ceil", "floor"),
        default="ceil",
        help="How to round per-source fractional task counts. Default: ceil.",
    )
    parser.add_argument(
        "--minimum-tasks-per-source",
        type=int,
        default=1,
        help="Minimum number of tasks to keep for each non-empty source dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional name for the output dataset entry. Defaults to the output file stem.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help="Optional version for the output dataset entry. Defaults to the source dataset version.",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description prefix for the output dataset entry.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved subset summary without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if not 0 < args.percent <= 100:
        parser.error("--percent must be greater than 0 and at most 100")
    if args.minimum_tasks_per_source < 0:
        parser.error("--minimum-tasks-per-source must be non-negative")

    try:
        result = build_subset_result(
            source_registry_path=args.source_registry,
            source_lock_path=args.source_lock,
            exclude_lock_paths=args.exclude_lock,
            output_registry_path=args.output_registry,
            percent=args.percent,
            rounding=args.rounding,
            minimum_tasks_per_source=args.minimum_tasks_per_source,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            description=args.description,
        )
        if not args.dry_run:
            write_subset_result(result)
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
