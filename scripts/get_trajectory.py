import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

if __name__ == "__main__":
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_PUBLISHABLE_KEY"],
    )

    response = (
        client.table("trial")
        .select("*, task!inner(*, dataset_task!inner(*)), trial_model!inner(*)")
        .is_("exception_info", None)
        .eq("task.name", "cancel-async-tasks")
        .eq("task.dataset_task.dataset_name", "terminal-bench")
        .eq("task.dataset_task.dataset_version", "2.0")
        .eq("agent_name", "terminus-2")
        .eq("trial_model.model_name", "gpt-5")
        .limit(1)
        .single()
        .execute()
    )

    trial = response.data

    output_dir = Path("trajectories") / trial["id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_files_recursive(base_path, output_base):
        files = client.storage.from_("trials").list(path=str(base_path))

        for file in files:
            file_name = file["name"]
            file_id = file["id"]

            is_file = file_id is not None

            file_path = Path(base_path) / file_name
            out_path = output_base / file_name

            if is_file:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                data = client.storage.from_("trials").download(str(file_path))
                out_path.write_bytes(data)
            else:
                out_path.mkdir(parents=True, exist_ok=True)
                download_files_recursive(file_path, out_path)

    download_files_recursive(trial["id"], output_dir)