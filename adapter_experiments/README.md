# Adapter Experiments

This workspace keeps adapter-only sampling manifests, helper scripts, and
generated subset registries together instead of mixing them into the repo-wide
`scripts/` directory.

## Layout

- `scripts/`: runnable entrypoints for adapter experiment helpers
- `manifests/`: checked-in sampling specs
- `registries/`: generated registry JSON and lock files

## Sample a registry

Create or copy a manifest under `adapter_experiments/manifests/`, then run:

```bash
uv run python adapter_experiments/scripts/sample_registry.py \
  --manifest adapter_experiments/manifests/example.yaml
```

Use `--dry-run` to validate the manifest and inspect the resolved sample
without writing files.

## Create progressive subsets

Use `subset_registry.py` to split a flattened full registry into deterministic
phase registries while keeping reproducibility in the generated lock files:

```bash
uv run python adapter_experiments/scripts/subset_registry.py \
  --source-registry adapter_experiments/registries/0321_batch1_full.json \
  --source-lock adapter_experiments/registries/0321_batch1_full.lock.json \
  --output-registry adapter_experiments/registries/batch1_subsets/phase2.json \
  --percent 1 \
  --dataset-name adapter-experiments-0321-batch1-phase2
```

To build later phases without reusing earlier tasks, pass earlier lock files
with `--exclude-lock`:

```bash
uv run python adapter_experiments/scripts/subset_registry.py \
  --source-registry adapter_experiments/registries/0321_batch1_full.json \
  --source-lock adapter_experiments/registries/0321_batch1_full.lock.json \
  --exclude-lock adapter_experiments/registries/batch1_subsets/phase2.lock.json \
  --output-registry adapter_experiments/registries/batch1_subsets/phase3.json \
  --percent 10 \
  --dataset-name adapter-experiments-0321-batch1-phase3
```

Typical progressive setup:

- `phase2`: `--percent 1`
- `phase3`: `--percent 10 --exclude-lock phase2.lock.json`
- `phase4`: `--percent 100 --exclude-lock phase2.lock.json --exclude-lock phase3.lock.json`

Notes:

- Selection is deterministic from the full registry lock file, using the stored
  per-source sample ranks instead of resampling.
- Percentages are applied per source dataset, with `ceil` rounding by default.
- `phase3` is the cumulative 10% slice minus `phase2`.
- `phase4` is the remaining full set after subtracting both `phase2` and
  `phase3`.
- Use `--dry-run` to inspect counts before writing output files.

## Manifest format

```yaml
source_registry: ../../../harbor/registry.json
seed: 20260312
output:
  dataset_name: adapter-experiments-sample
  dataset_version: "1.0"
  description: Reproducible adapter experiment subset.
  registry_path: ../registries/adapter-experiments-sample.json
sources:
  - name: terminal-bench
    version: "2.0"
    sample_size: 8
  - name: aider-polyglot
    version: "1.0"
    sample_size: 12
```

Notes:

- `source_registry` and `output.registry_path` are resolved relative to the
  manifest file.
- `sources` are explicit `name@version` selectors with explicit sample sizes.
- The generated registry contains one synthetic dataset entry that flattens all
  selected tasks.
- A `.lock.json` file is written next to the registry with the resolved sample,
  task fingerprints, and source-registry digest.
