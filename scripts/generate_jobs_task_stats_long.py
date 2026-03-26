import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate trial-level results under a jobs directory into a long-form CSV "
            "grouped by dataset, task, agent, and model."
        )
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("jobs"),
        help="Path to the jobs directory. Defaults to ./jobs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jobs_task_stats_long.csv"),
        help="Output CSV path. Defaults to ./jobs_task_stats_long.csv.",
    )
    return parser.parse_args()


def safe_mean(values: list[float | int | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    return mean(present_values) if present_values else None


def infer_dataset(result_data: dict, result_path: Path) -> str:
    task_path = ((result_data.get("task_id") or {}).get("path")) or ""
    parts = task_path.split("/")
    if len(parts) >= 3 and parts[0] == "datasets":
        return parts[1]
    return result_path.parent.parent.name.split("__")[0]


def infer_error_type(exception_info: object) -> str:
    if isinstance(exception_info, dict):
        return exception_info.get("exception_type") or "UNKNOWN"
    if exception_info is None:
        return "OK"
    return type(exception_info).__name__


def format_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_trial_groups(
    jobs_dir: Path,
) -> dict[tuple[str, str, str, str], list[dict[str, object]]]:
    trial_groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )

    for result_path in sorted(jobs_dir.glob("*/*/result.json")):
        try:
            result_data = json.loads(result_path.read_text())
        except Exception as exc:
            print(f"Skipping unreadable JSON: {result_path} ({exc})")
            continue

        config = result_data.get("config") or {}
        agent_config = config.get("agent") or {}
        task_name = result_data.get("task_name")
        agent_name = agent_config.get("name")
        model_name = agent_config.get("model_name")
        if not (task_name and agent_name and model_name):
            continue

        dataset = infer_dataset(result_data, result_path)
        agent_result = result_data.get("agent_result") or {}
        verifier_result = result_data.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}

        key = (dataset, task_name, agent_name, model_name)
        trial_groups[key].append(
            {
                "reward": rewards.get("reward"),
                "input_tokens": agent_result.get("n_input_tokens"),
                "output_tokens": agent_result.get("n_output_tokens"),
                "cache_tokens": agent_result.get("n_cache_tokens"),
                "errortype": infer_error_type(result_data.get("exception_info")),
            }
        )

    return trial_groups


def write_long_csv(
    output_path: Path,
    trial_groups: dict[tuple[str, str, str, str], list[dict[str, object]]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "task",
                "agent",
                "model",
                "n_trials",
                "reward_mean",
                "input_mean",
                "output_mean",
                "cache_mean",
                "errortype",
            ]
        )

        for (dataset, task_name, agent_name, model_name), matches in sorted(
            trial_groups.items()
        ):
            err_counts = Counter(match["errortype"] for match in matches)
            err_summary = " | ".join(
                f"{err}:{count}"
                for err, count in sorted(
                    err_counts.items(), key=lambda item: (item[0] != "OK", item[0])
                )
            )

            writer.writerow(
                [
                    dataset,
                    task_name,
                    agent_name,
                    model_name,
                    len(matches),
                    format_float(
                        safe_mean([match["reward"] for match in matches]), digits=6
                    ),
                    format_float(
                        safe_mean([match["input_tokens"] for match in matches]),
                        digits=2,
                    ),
                    format_float(
                        safe_mean([match["output_tokens"] for match in matches]),
                        digits=2,
                    ),
                    format_float(
                        safe_mean([match["cache_tokens"] for match in matches]),
                        digits=2,
                    ),
                    err_summary,
                ]
            )


def main() -> None:
    args = parse_args()
    trial_groups = load_trial_groups(args.jobs_dir)
    write_long_csv(args.output, trial_groups)
    n_trials = sum(len(matches) for matches in trial_groups.values())
    print(
        f"Wrote {args.output} with {len(trial_groups)} aggregate rows "
        f"from {n_trials} trials."
    )


if __name__ == "__main__":
    main()
