import os
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client
from tabulate import tabulate

load_dotenv()


def main():
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    trials = (
        client.table("dataset_task")
        .select("*, task(*, trial(*, trial_model(*)))")
        .eq("dataset_name", "terminal-bench")
        .eq("dataset_version", "2.0")
        .neq("task.trial.agent_name", "oracle")
        .execute()
    )

    combinations = defaultdict(list)

    for dataset_task in trials.data:
        task = dataset_task.get("task")
        if not task:
            continue

        task_name = task.get("name", "N/A")
        task_trials = task.get("trial", [])

        for trial in task_trials:
            agent_name = trial.get("agent_name", "N/A")
            trial_models = trial.get("trial_model", [])

            if not trial_models:
                model_name = "N/A"
                key = (agent_name, model_name, task_name)
                combinations[key].append(trial)
            else:
                for trial_model in trial_models:
                    model_name = trial_model.get("model_name", "N/A")
                    key = (agent_name, model_name, task_name)
                    combinations[key].append(trial)

    agent_model_combinations = defaultdict(lambda: defaultdict(list))

    for (agent_name, model_name, task_name), trial_list in combinations.items():
        has_failed_trial = False
        exception_counts = defaultdict(int)

        for trial in trial_list:
            exception_info = trial.get("exception_info")

            if exception_info is None:
                has_failed_trial = True
                exception_counts["null"] += 1
            else:
                exception_type = exception_info.get("exception_type")
                if exception_type == "AgentTimeoutError":
                    has_failed_trial = True
                    exception_counts["AgentTimeoutError"] += 1
                elif exception_type:
                    exception_counts[exception_type] += 1
                else:
                    exception_counts["success"] += 1

        if not has_failed_trial and len(trial_list) > 0:
            agent_model_combinations[(agent_name, model_name)][task_name] = {
                "trial_count": len(trial_list),
                "exception_counts": exception_counts,
            }

    successful_combinations = []
    for (agent_name, model_name), tasks in agent_model_combinations.items():
        task_list = []
        total_trials = 0
        all_exception_counts = defaultdict(int)

        for task_name, task_data in tasks.items():
            trial_count = task_data["trial_count"]
            total_trials += trial_count
            task_list.append(f"{task_name} ({trial_count})")

            for exc_type, count in task_data["exception_counts"].items():
                all_exception_counts[exc_type] += count

        task_summary = ", ".join(sorted(task_list))
        exception_summary = ", ".join(
            [f"{k}: {v}" for k, v in all_exception_counts.items()]
        )

        successful_combinations.append(
            [agent_name, model_name, task_summary, exception_summary]
        )

    successful_combinations.sort(key=lambda x: (x[0], x[1]))

    headers = [
        "Agent Name",
        "Model Name",
        "Tasks (Count)",
        "Exception Types",
    ]
    print(
        f"Found {len(successful_combinations)} agent-model combinations with no failed trials:"
    )
    print(tabulate(successful_combinations, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
