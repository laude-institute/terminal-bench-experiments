# terminal-bench-experiments

Use this repo to run Harbor experiments and upload results to Supabase.

## Setup

1. Keep Harbor as a sibling checkout:

```text
code/
  harbor/
  terminal-bench-experiments/
```

2. Fill `.env` in the repo root. For the included MiniMax example you need:

```dotenv
SUPABASE_ACCESS_TOKEN=
SUPABASE_PROJECT_REF=
SUPABASE_URL=
SUPABASE_SECRET_KEY=
DAYTONA_API_KEY=
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

Supabase variables are easy to mix up:

- `SUPABASE_ACCESS_TOKEN`: your Supabase Personal Access Token for the
  Management API. It should come from your Supabase account settings and
  usually starts with `sbp_`.
- `SUPABASE_PROJECT_REF`: your project ref. If
  `SUPABASE_URL=https://abcdefghijklmnop.supabase.co`, then
  `SUPABASE_PROJECT_REF=abcdefghijklmnop`.
- `SUPABASE_URL`: your project URL.
- `SUPABASE_SECRET_KEY`: your project service-role key from Supabase project
  settings. This is not the same thing as `SUPABASE_ACCESS_TOKEN`.

Do not swap these:

- `SUPABASE_ACCESS_TOKEN` is only for the Management API step that applies the
  SQL schema.
- `SUPABASE_SECRET_KEY` is for storage/database operations from this repo.

If you test the access token manually with `curl`, use double quotes so the
shell expands the variable:

```bash
curl -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN" \
  https://api.supabase.com/v1/projects
```

If that returns `JWT could not be decoded`, you are not sending a valid
Personal Access Token. If it returns `403`, the token/account does not have
permission for the project or your network is being blocked before the request
reaches Supabase.

3. Install Python dependencies:

```bash
uv sync
```

4. Bootstrap Supabase.

Use the Supabase Management API to apply `db/schema.sql`, then use the project
storage API to create the `trials` bucket:

```bash
uv run python scripts/setup_supabase.py
```

This command needs:

- `SUPABASE_ACCESS_TOKEN`: Personal Access Token
- `SUPABASE_PROJECT_REF`: project ref
- `SUPABASE_URL`: project URL
- `SUPABASE_SECRET_KEY`: service-role key

The script applies `db/schema.sql` through
`POST /v1/projects/{project_ref}/database/query` and creates a public `trials`
bucket with `supabase-py`.

If you only want one half:

```bash
uv run python scripts/setup_supabase.py --skip-bucket
uv run python scripts/setup_supabase.py --skip-schema
```

## Run The Example Experiment

1. Generate the sampled registry:

```bash
uv run python adapter_experiments/scripts/sample_registry.py \
  --manifest adapter_experiments/manifests/example.yaml
```

2. Upload dataset metadata:

```bash
uv run tbx upload-dataset \
  -n test-example \
  -v 1.0 \
  --registry-path adapter_experiments/registries/test-example.json
```

3. Run or resume the job:

```bash
uv run python scripts/run_job.py \
  -c configs/adapter_experiments/terminus-2__minimax-m2.5__test-example.yaml
```

4. Validate the finished job:

```bash
uv run tbx validate \
  --job-path jobs/terminus-2__minimax-m2.5__test-example
```

5. Import the finished job into Supabase:

```bash
uv run tbx import \
  --job-path jobs/terminus-2__minimax-m2.5__test-example
```

Rerun step 3 with the same config to resume an interrupted job.
