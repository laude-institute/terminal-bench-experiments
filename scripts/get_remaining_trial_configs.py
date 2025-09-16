import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

if __name__ == "__main__":
    client = create_client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SECRET_KEY"],
    )

    response = (
        client.table("dataset_task")
        .select("*, task(*, trial(*, trial_model(*)))")
        .eq("dataset_name", "terminal-bench")
        .eq("dataset_version", "2.0")
        .eq("task.trial.agent_name", "terminus-2")
        .eq("task.trial.trial_model.model_name", "openai/gpt-5-nano")
        .execute()
    )

    for dataset_task in response.data:
        pass
