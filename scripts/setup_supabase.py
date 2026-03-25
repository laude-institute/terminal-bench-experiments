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
# Cloudflare can reject urllib's default client signature for this endpoint
# with a 1010 "browser signature" block, while the equivalent curl request
# succeeds. Send curl-like headers to match the documented manual check path.
MANAGEMENT_API_USER_AGENT = "curl/8.5.0"


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_management_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": MANAGEMENT_API_USER_AGENT,
    }


def sanitize_schema_sql(sql: str) -> tuple[str, list[str]]:
    removed_meta_commands: list[str] = []
    cleaned_lines: list[str] = []

    for line in sql.splitlines():
        if line.startswith("\\"):
            removed_meta_commands.append(line)
            continue
        if line == "CREATE SCHEMA public;":
            cleaned_lines.append("CREATE SCHEMA IF NOT EXISTS public;")
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines) + "\n", removed_meta_commands


def run_schema_query(project_ref: str, schema_path: Path) -> None:
    access_token = require_env("SUPABASE_ACCESS_TOKEN")
    sql, removed_meta_commands = sanitize_schema_sql(
        schema_path.read_text(encoding="utf-8")
    )

    request = urllib.request.Request(
        url=f"{MANAGEMENT_API_BASE}/projects/{project_ref}/database/query",
        data=json.dumps({"query": sql}).encode("utf-8"),
        headers=build_management_headers(access_token),
        method="POST",
    )

    if removed_meta_commands:
        print("Skipping psql meta-commands not supported by the Supabase Management API:")
        for command in removed_meta_commands:
            print(f"  {command}")

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403 and "1010" in error_body:
            error_body = (
                f"{error_body}\n"
                "Cloudflare rejected the request based on the client's browser "
                "signature before it reached Supabase. This is usually caused "
                "by bot protection rather than invalid SQL. The script now "
                "sends curl-style headers; if the error persists, retry the "
                "equivalent curl command from the same machine or apply the "
                "schema through the Supabase SQL Editor."
            )
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
