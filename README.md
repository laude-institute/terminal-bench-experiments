# terminal-bench-experiments

Use this repo to run Harbor experiments and upload results to Supabase.

## Setup

1. Keep Harbor as a sibling checkout:

```text
code/
  harbor/
  terminal-bench-experiments/
```

2. Fill `.env` in the repo root. For the included examples you need:

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
curl -H "Authorization: Bearer $SUPABASE_ACCESS_TOKEN"   https://api.supabase.com/v1/projects
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

For coordinated adapter experiments, the manager usually runs `scripts/setup_supabase.py` once to create or update the schema and storage bucket. Contributors still need local `.env` values for `tbx upload-dataset`, `tbx validate`, `tbx import`, and the model/provider credentials referenced by their generated job YAMLs. Contributors do not need to reapply the schema, but each contributor should upload the phase dataset metadata from the local registry they materialize before running jobs.

## Run A Single Experiment

1. Generate or choose a dataset registry:

```bash
uv run python adapter_experiments/scripts/sample_registry.py   --manifest adapter_experiments/manifests/<sampling-manifest>.yaml
```

2. Upload dataset metadata:

```bash
uv run tbx upload-dataset   -n <dataset-name>   -v <dataset-version>   --registry-path <registry-path>
```

3. Run or resume the job:

```bash
uv run python scripts/run_job.py   -c configs/<job-config>.yaml
```

4. Validate the finished job:

```bash
uv run tbx validate   --job-path jobs/<job-name>
```

5. Import the finished job into Supabase:

```bash
uv run tbx import   --job-path jobs/<job-name>
```

Rerun step 3 with the same config to resume an interrupted job.

## Coordinated Adapter Experiments

Use the coordinated flow when multiple contributors share one unified task set and
split benchmark adapters between themselves.

### Manager Pipeline

1. Fill `.env` with the Supabase management credentials needed by
   `scripts/setup_supabase.py`.
2. Bootstrap the schema and storage bucket once:

```bash
uv run python scripts/setup_supabase.py
```

3. Generate the unified task set once:

```bash
uv run python adapter_experiments/scripts/sample_registry.py   --manifest adapter_experiments/manifests/batch1_all.yaml
```

4. Create the benchmark coordination manifest that assigns adapters to
   contributors, for example `adapter_experiments/manifests/batch1_coordination.yaml`.
5. If you are composing benchmark assignments with a model or agent matrix,
   render a runner manifest from `.env`:

```bash
uv run python adapter_experiments/scripts/generate_model_coordination_manifest.py   --experiment-name batch1-runners   --task-set-registry-path adapter_experiments/registries/batch1_all.json   --task-set-lock-path adapter_experiments/registries/batch1_all.lock.json   --output outputs/adapter_experiments/batch1/runners.generated.yaml   --contributor local
```

6. Inspect the plan and materialize the phase registry when a phase opens:

```bash
uv run python adapter_experiments/scripts/coordination.py show   --manifest adapter_experiments/manifests/batch1_coordination.yaml

uv run python adapter_experiments/scripts/coordination.py phase   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --phase phase2
```

7. Share the phase name, benchmark manifest path, and runner manifest path with
   contributors.

### Contributor Pipeline

1. Fill `.env` with:
   - `SUPABASE_URL` and `SUPABASE_SECRET_KEY` for `tbx upload-dataset`,
     `tbx validate`, and `tbx import`
   - runtime/provider credentials needed by the generated jobs, such as
     `DAYTONA_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and any provider-specific
     keys referenced by the runner manifest

2. Inspect your assignment:

```bash
uv run python adapter_experiments/scripts/coordination.py show   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --contributor Haowei
```

3. Materialize your local phase outputs:

```bash
uv run python adapter_experiments/scripts/coordination.py contributor   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --phase phase2   --contributor Haowei   --runner-manifest outputs/adapter_experiments/batch1/runners.generated.yaml   --runner-contributor local
```

If your coordination manifest already defines inline `job_template` and
`adapters[*].agent`, omit `--runner-manifest` and `--runner-contributor`.

4. Upload the phase dataset metadata from the phase registry you just materialized:

```bash
uv run tbx upload-dataset   -n adapter-experiments-batch1-phase2   -v 1.0   --registry-path outputs/adapter_experiments/batch1/phases/phase2/registry.json
```

The dataset name, dataset version, and registry path are also recorded in
`assignment.yaml`.

5. Run the generated configs under `outputs/adapter_experiments/...`.
6. Validate finished jobs with `tbx validate`.
7. Upload finished jobs with `tbx import`.

Phase order comes from the manifest. If you omit `phase1`, then `phase2`,
`phase3`, and `phase4` still work as progressive 1%, 10%, and 100% slices. The
helper automatically excludes earlier phases so contributors do not rerun tasks
that were already assigned in previous phases.

See `adapter_experiments/README.md` for the full manager/contributor workflow,
manifest format, and output layout.

## Generate A Model-Matrix Runner Manifest

If you want to compose a benchmark coordination manifest with a model or agent
matrix, render a runner manifest from the repo-root `.env` first:

```bash
uv run python adapter_experiments/scripts/generate_model_coordination_manifest.py   --experiment-name batch1-runners   --task-set-registry-path adapter_experiments/registries/batch1_all.json   --task-set-lock-path adapter_experiments/registries/batch1_all.lock.json   --output outputs/adapter_experiments/batch1/runners.generated.yaml   --contributor local
```

The script writes the requested output path, resolves provider keys from `.env`,
and assigns runner adapters round robin if you repeat `--contributor`. A
contributor can then compose that runner manifest with the benchmark assignment
manifest:

```bash
uv run python adapter_experiments/scripts/coordination.py contributor   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --phase phase2   --contributor Haowei   --runner-manifest outputs/adapter_experiments/batch1/runners.generated.yaml   --runner-contributor local
```

The generated manifest can also be used by itself as a standalone inline-agent
coordination manifest because it already contains `job_template` and
`adapters[*].agent`.
