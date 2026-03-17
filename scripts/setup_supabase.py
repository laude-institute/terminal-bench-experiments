import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

MANAGEMENT_API_BASE = "https://api.supabase.com/v1"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def run_schema_query(project_ref: str, schema_path: Path) -> None:
    access_token = require_env("SUPABASE_ACCESS_TOKEN")
    sql = schema_path.read_text(encoding="utf-8")

    request = urllib.request.Request(
        url=f"{MANAGEMENT_API_BASE}/projects/{project_ref}/database/query",
        data=json.dumps({"query": sql}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Failed to apply schema via Management API ({exc.code}): {error_body}"
        ) from exc

    print(f"Applied schema from {schema_path}")
    if body:
        print(body)


def ensure_bucket(bucket_name: str, public: bool) -> None:
    client = create_client(
        supabase_url=require_env("SUPABASE_URL"),
        supabase_key=require_env("SUPABASE_SECRET_KEY"),
    )

    try:
        client.storage.get_bucket(bucket_name)
        print(f"Bucket already exists: {bucket_name}")
        return
    except Exception:
        pass

    client.storage.create_bucket(
        bucket_name,
        options={"public": public},
    )
    print(f"Created bucket: {bucket_name} (public={public})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the Supabase schema and storage bucket for this repo."
    )
    parser.add_argument(
        "--project-ref",
        default=os.environ.get("SUPABASE_PROJECT_REF"),
        help="Supabase project ref. Defaults to SUPABASE_PROJECT_REF.",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=Path("db/schema.sql"),
        help="Path to the SQL schema file.",
    )
    parser.add_argument(
        "--bucket-name",
        default="trials",
        help="Storage bucket to create.",
    )
    parser.add_argument(
        "--bucket-private",
        action="store_true",
        help="Create the bucket as private instead of public.",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip the schema creation step.",
    )
    parser.add_argument(
        "--skip-bucket",
        action="store_true",
        help="Skip the bucket creation step.",
    )
    args = parser.parse_args()

    if not args.skip_schema:
        if not args.project_ref:
            raise RuntimeError(
                "Missing Supabase project ref. Pass --project-ref or set "
                "SUPABASE_PROJECT_REF."
            )
        run_schema_query(args.project_ref, args.schema_path)

    if not args.skip_bucket:
        ensure_bucket(args.bucket_name, public=not args.bucket_private)


if __name__ == "__main__":
    main()
