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
