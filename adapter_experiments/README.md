# Adapter Experiments

This workspace keeps unified task-set manifests, benchmark coordination manifests, optional runner manifests, and helper scripts together. Checked-in outputs stop at the unified task set. Each contributor materializes phase registries, assignment files, benchmark-scoped registries, and job configs locally under `outputs/`.

Please keep harbor at `../harbor`. Check out to commit 9ee6790376583608f541c133d1af2dae47b8fc32.

## Layout

- `scripts/sample_registry.py`: sample one unified task set and write its lock
- `scripts/subset_registry.py`: low-level deterministic subset helper
- `scripts/coordination.py`: manager/contributor helper for phased runs
- `scripts/generate_model_coordination_manifest.py`: render a runner manifest from `.env`
- `manifests/`: checked-in task-set and coordination manifests
- `registries/`: checked-in unified task sets and small examples
- `outputs/adapter_experiments/...`: ignored local phase registries, locks,
  assignment files, benchmark-scoped registries, and contributor configs
  generated from a coordination manifest

## Coordinated Flow

### Contributor Pipeline

1. Fill the repo-root `.env`.

   Required for uploading metadata and results:
   - `SUPABASE_URL`
   - `SUPABASE_SECRET_KEY`

   Required for executing the generated jobs:
   - environment keys such as `DAYTONA_API_KEY`
   - provider keys such as `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and any
     provider-specific keys referenced by the runner manifest

2. Inspect your assignment:

```bash
uv run python adapter_experiments/scripts/coordination.py show   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --contributor Haowei
```

3. Materialize your local phase outputs.

   For a benchmark manifest plus runner manifest:

```bash
uv run python adapter_experiments/scripts/coordination.py contributor   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --phase phase2   --contributor Haowei   --runner-manifest outputs/adapter_experiments/batch1/runners.generated.yaml   --runner-contributor local
```

   If your coordination manifest already defines inline `job_template` and
   `adapters[*].agent`, omit `--runner-manifest` and `--runner-contributor`.

4. Review the generated files.

   - `outputs/adapter_experiments/<experiment>/phases/<phase>/registry.json`:
     phase dataset registry to upload
   - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/assignment.yaml`:
     local metadata record containing dataset name/version, assigned adapters,
     and runner-manifest provenance
   - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/registries/*.json`:
     benchmark-scoped registries generated when using `--runner-manifest`
   - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/*.yaml`:
     job configs to run

5. Upload the phase dataset metadata from your local phase registry before
   running jobs:

```bash
uv run tbx upload-dataset   -n adapter-experiments-batch1-phase2   -v 1.0   --registry-path outputs/adapter_experiments/batch1/phases/phase2/registry.json
```

6. Run the generated configs:

```bash
uv run python scripts/run_job.py   -c outputs/adapter_experiments/batch1/contributors/Haowei/phase2/gaia__codex__gpt-5.4.yaml
```

7. Validate and import finished jobs as usual:

```bash
uv run tbx validate   --job-path jobs/gaia__codex__gpt-5.4__batch1__phase2

uv run tbx import   --job-path jobs/gaia__codex__gpt-5.4__batch1__phase2
```

Contributors upload both phase dataset metadata and finished trials. They do
not recreate the schema.


### Manager Pipeline

1. Fill the repo-root `.env`.

   Required for infrastructure bootstrap:
   - `SUPABASE_ACCESS_TOKEN`
   - `SUPABASE_PROJECT_REF`
   - `SUPABASE_URL`
   - `SUPABASE_SECRET_KEY`

   If you plan to generate a runner manifest, also include the provider keys
   referenced by `scripts/generate_model_coordination_manifest.py`.

2. Bootstrap the Supabase schema and storage bucket once:

```bash
uv run python scripts/setup_supabase.py
```

3. Generate the unified task set once:

```bash
uv run python adapter_experiments/scripts/sample_registry.py   --manifest adapter_experiments/manifests/batch1_all.yaml
```

4. Create the coordination manifests.

   Benchmark coordination manifest:
   - assigns benchmark adapters to contributors
   - example checked-in file: `adapter_experiments/manifests/batch1_coordination.yaml`

   Optional runner manifest:
   - defines model or agent configs used to generate job YAMLs
   - can be rendered from `.env` with:

```bash
uv run python adapter_experiments/scripts/generate_model_coordination_manifest.py   --experiment-name batch1-runners   --task-set-registry-path adapter_experiments/registries/batch1_all.json   --task-set-lock-path adapter_experiments/registries/batch1_all.lock.json   --output outputs/adapter_experiments/batch1/runners.generated.yaml   --contributor local
```

5. Inspect the plan and materialize the phase registry when a phase opens:

```bash
uv run python adapter_experiments/scripts/coordination.py show   --manifest adapter_experiments/manifests/batch1_coordination.yaml

uv run python adapter_experiments/scripts/coordination.py phase   --manifest adapter_experiments/manifests/batch1_coordination.yaml   --phase phase2
```

6. Share the benchmark manifest path, runner manifest path if any, and phase
   name with contributors.

The manager owns schema bootstrap and manifest definition. Contributors do not
need to rerun `scripts/setup_supabase.py`.

## Progressive Phases

The coordination helper uses manifest order to define the exclusion chain.

- If you ignore `phase1`, simply omit it from the manifest.
- `phase2` can be the 1% slice.
- `phase3` can be the cumulative 10% slice minus everything already in
  `phase2`.
- `phase4` can be the remaining 100% slice minus the previous phases.

When you materialize `phase3` or `phase4`, the helper regenerates the earlier
phase locks locally and passes them as exclusions automatically. That keeps the
selection deterministic and avoids rerunning earlier tasks.

## Coordination Manifest Format

```yaml
experiment:
  name: test-example
  output_root: ../../outputs/adapter_experiments/test-example

task_set:
  registry_path: ../registries/test-example.json
  lock_path: ../registries/test-example.lock.json

phases:
  phase2:
    percent: 1
    dataset_name: test-example-phase2
    dataset_version: "1.0"
  phase3:
    percent: 10
    dataset_name: test-example-phase3
    dataset_version: "1.0"
  phase4:
    percent: 100
    dataset_name: test-example-phase4
    dataset_version: "1.0"

job_template:
  jobs_dir: jobs
  n_attempts: 1
  timeout_multiplier: 1.0
  orchestrator:
    type: local
    n_concurrent_trials: 1
    quiet: false
    retry:
      max_retries: 3
      exclude_exceptions:
        - BadRequestError
        - RateLimitError
        - AgentTimeoutError
        - VerifierTimeoutError
        - RewardFileNotFoundError
      wait_multiplier: 1.0
      min_wait_sec: 1.0
      max_wait_sec: 60.0
  environment:
    type: daytona
    force_build: false
    delete: true

adapters:
  terminus-2__minimax-m2.5:
    agent:
      name: terminus-2
      model_name: openai/MiniMax-M2.5
  terminus-2__gemini-3-flash-preview:
    agent:
      name: terminus-2
      model_name: openai/gemini-3-flash-preview
    job_overrides:
      orchestrator:
        n_concurrent_trials: 5

contributors:
  alice:
    adapters:
      - terminus-2__minimax-m2.5
  bob:
    adapters:
      - terminus-2__gemini-3-flash-preview
```

Notes:

- `task_set.registry_path` and `task_set.lock_path` should point to the unified
  registry generated by `sample_registry.py`.
- The YAML block above shows the inline-agent coordination format.
- Inline-agent manifests include `job_template` and `adapters[*].agent`, so
  `coordination.py contributor` writes one job config per assigned adapter.
- Benchmark-only manifests can omit `job_template` and `adapters[*].agent` and
  instead keep benchmark metadata plus `metadata.source_datasets`; use those
  manifests with `coordination.py contributor --runner-manifest <runner-manifest>`.
- `job_template` contains the common job config fields. `job_name`, `agents`,
  and `datasets` are generated automatically.
- `assignment.yaml` is generated for every contributor run and records dataset
  metadata, assigned adapters, and runner-manifest provenance.
- Each adapter should be assigned to exactly one contributor.
- The generated files land in:
  - `outputs/adapter_experiments/<experiment>/phases/<phase>/registry.json`
  - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/assignment.yaml`
  - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/registries/*.json` when using `--runner-manifest`
  - `outputs/adapter_experiments/<experiment>/contributors/<contributor>/<phase>/*.yaml`
- See `adapter_experiments/manifests/batch1_coordination.yaml` for a checked-in
  benchmark-assignment manifest.

## Low-Level Subset Helper

If you need to build a subset manually, `subset_registry.py` is still available:

```bash
uv run python adapter_experiments/scripts/subset_registry.py   --source-registry adapter_experiments/registries/batch1_all.json   --source-lock adapter_experiments/registries/batch1_all.lock.json   --output-registry outputs/adapter_experiments/manual-phase2.json   --percent 1   --dataset-name adapter-experiments-manual-phase2
```

The coordination helper uses the same deterministic selection logic internally,
so most phased runs should use `coordination.py` instead of calling
`subset_registry.py` directly.
