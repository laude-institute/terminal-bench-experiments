# Generated Registries

Checked-in adapter-experiment registries live here only when they are the
canonical unified task sets or small reviewable examples.

Contributor-local phase outputs should be generated under
`outputs/adapter_experiments/` with `adapter_experiments/scripts/coordination.py`
so the repo does not accumulate per-phase, per-contributor registry churn.

Registry `.json` outputs in this directory remain visible to git on purpose.
Reproducibility `.lock.json` files stay ignored to avoid lockfile churn from
local sampling runs.
