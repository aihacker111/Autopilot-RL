# LOTUS v5 — P0/P1 training

v5 is a separate controller and training entry point. Existing v4 checkpoints and
training scripts are retained. Its algorithms have engineering tests and short
training checks; a long multi-seed run is still required to establish whether it
outperforms v4. Camera/CARLA transfer (P2) is unchanged.

## Start

Python 3.12+ and Node 18+ are required. Install a PyTorch build matching the target
CUDA runtime before the other dependencies. `requirements-v5.txt` describes the
supported range; `requirements-v5-tested.txt` records the local CPU test versions.
Every run records its actual versions, source hashes, full map-data hash,
realism-prior hash, optional NPC-policy hash, config and split manifest.

From the repository root:

```bash
python -m pip install -r requirements-v5.txt
python -m pytest -q tests
python rl/v5_train.py --config configs/v5_smoke.yaml
python rl/v5_train.py --config configs/v5_debug.yaml
python rl/v5_train.py --config configs/v5_full.yaml
```

Equivalent pipeline dispatch:

```bash
python pipeline.py --stages v5 --config configs/v5_full.yaml
```

The full preset starts with 5,000,000 transitions, 4 Node workers x 4 environments,
latent 320 / hidden 512, 3 bootstrap models, 8-step planning, 8-step burn-in,
128 candidates and 3 futures. This is a starting configuration for a CUDA server,
not a measured VRAM guarantee. Profile the debug run before allocating a long run.
The simulator and teacher also need substantial CPU capacity. `device: auto`
selects CUDA when available and otherwise CPU; a full CPU run will be slow.
The replay uses roughly 0.7 GB for 250,000 transitions, plus maps, ensembles,
optimizer tensors, activation memory and checkpoint serialization overhead.
Keep space for both `latest.pt` and `best.pt`, each including replay and simulation.

Use `--out` to isolate experiments, and `--device cpu` when explicitly testing CPU:

```bash
python rl/v5_train.py --config configs/v5_full.yaml --seed 2027 --out rl/runs/lotus_v5_seed2027
```

Existing run directories are protected from accidental reuse. Defaults live in
`v5_full.yaml`; debug/smoke/baseline files override only their differences. A run's
resolved configuration is saved as `config.yaml`.

## Resume

```bash
python rl/v5_train.py --config configs/v5_full.yaml --resume rl/runs/lotus_v5_full/latest.pt
```

A checkpoint includes world model, GRU/context encoder, policy, target Q networks,
bootstrap ensembles, both optimizers, main/counterfactual replay, curriculum archive,
pending/mastered scenarios, Python/NumPy/Torch/CUDA RNGs, Node RNGs, the exact live
simulation graph, online GRU state, previous executed commands, MPPI warm start,
update credit and stage counters. Snapshot writes use atomic replacement.

Resume refuses changed source, runtime versions, schemas, geographic splits,
map data, NPC weights, priors or structural settings. Changing the stop budget,
device, output path and reporting/checkpoint intervals is permitted. Bitwise
reproducibility is tested on the same CPU runtime; it is not promised across
hardware or CPU/CUDA transitions. Checkpoints use trusted local pickle/V8 data.

After an interruption, resume the last **committed** checkpoint. An interruption
can occur midway through an RPC or optimizer step, so the handler does not pretend
that a partially completed iteration is an exact checkpoint. Very early interrupts
before the first checkpoint require a fresh run directory.

## Evaluate and select

```bash
python rl/v5_train.py --evaluate rl/runs/lotus_v5_full/best.pt --matrix --split validation
# Use test only after choosing the design using validation:
python rl/v5_train.py --evaluate rl/runs/lotus_v5_full/best.pt --matrix --split test
```

The matrix covers prior/full planner, each with speed/lateral shields on/off.
Selection uses only validation. Eligibility requires zero observed person contacts
and the configured contact/km and red-light bounds; eligible models rank by route
completion, progress, red lights, shield exposure and jerk. If none are eligible,
the selected checkpoint is the best available under the fallback safety ranking,
not a declaration that it is safe. Examine event counts, km exposure, groups and
multiple seeds before drawing conclusions from a low event rate.

Reports contain completion, progress, crash, hit, **all** contacts/km, person
contacts, red lights, timeouts, speed/lateral shield interventions/km, mean absolute
jerk, decision latency, collision Brier score and ECE. Unknown future labels are
censored, including at timeouts. Calibration fields are null when no valid labels
are available. Decision latency is amortized per active environment, not a bound
on single-car worst-case latency. Exceeding the evaluation decision cap is an error,
not a scored safe episode.

HCM is divided into deterministic 8 km spatial cells. Training/validation/test
window centres have a halo at boundaries to keep the loaded map windows apart.
`split_seed` is independent of training seed so seed comparisons use the same
geographic suite. Generated maps are separated by map and traffic seeds. The
manifest and its hash are saved; test is never used by the training loop.

Long routes in this simulator still continue through **independent map windows**.
They are not a continuous geographic 20 km drive. GRU state is reset and replay
sequences are cut at every window boundary; the v5 value target does not bootstrap
into an unrelated map. Timeouts retain terminal observations for bootstrapping.

## What P0/P1 implements

- Absolute-generation ring replay, valid sequence sampling, stale-contact removal,
  episode-boundary masks and censored 1/2/3-second contact labels.
- Separate requested/executed commands and shield corrections. Dynamics/reward
  train on the executed action. v5 smoothing uses executed commands; shield
  intervention is taught separately through a policy correction loss.
- Counterfactual branches preserve the live environment/RNG/episode statistics,
  use the same learned NPC brain when enabled, and do not invent absorbing padding.
- DeepSets + GRU posterior with sequence burn-in. Visible object IDs keep stable
  neighbor ordering where possible; the recurrent belief retains history through
  occlusion. This is not an explicit calibrated multi-object tracking filter.
- Traffic context is predicted from observation/action history. True genomes
  supervise the context head during training, but are not controller inputs at
  inference. Imagination can explicitly condition on a proposed scenario genome.
- Bootstrapped dynamics/contact ensembles, multiple behavior hypotheses,
  mean/CVaR risk scoring, disagreement penalties and extra planning iterations
  when estimated contact disagreement is high. Disagreement is a heuristic;
  calibration is reported rather than assumed.
- Full-planner labels distill into the prior, weighted by estimated uncertainty;
  labels during warm-up and strong shield intervention are excluded. Teacher
  DAgger and shield correction losses remain separate.
- Curriculum pre-screens in imagination and with the prior, then measures success
  using the **full planner with the configured shield setting**. Beta-smoothed
  success estimates receive additional trials when intervals are too wide.
- Curriculum scoring combines intermediate difficulty, improvement since the
  previous check and descriptor novelty. Old elites are rechecked; mastered
  scenarios get a small replay mass. Teacher-failed candidates enter a bounded
  retry queue instead of being permanently labelled impossible.
- Default ordinary traffic remains the majority of new maps. Evolved scenarios
  have a bounded pool probability, realism filtering and teacher-solvability checks.
- Seed-aware HCM caching, robust Node worker lifecycle, fixed pipeline error
  handling and cached phase-2 gate rechecking.
- Recurrent JSON prior export with Python/JavaScript parity, executed-action
  feedback and reset hooks in the browser. Full world-model planning remains Python.

`calib/priors_vn.json` explicitly contains **placeholder** realism priors. Real
trajectory calibration cannot be completed without real trajectory data. To use
measured data, run the existing `calib/convert_tracks.py` and
`calib/fit_from_tracks.py` tools as appropriate, then set `priors` to the measured
JSON. Its hash is recorded and changing it requires a new experiment.
The emergency lateral guard is a local geometric heuristic, not a safety proof.

## Baselines and ablations

```bash
python rl/v5_train.py --config configs/v5_baseline.yaml
python rl/v5_sweep.py --config configs/v5_full.yaml --steps 1000000 --seeds 2026,2027,2028 --variants full,no_memory,no_curriculum,no_ensemble,no_distillation,no_counterfactual
```

The sweep runs sequentially in isolated output directories, keeps the same data
budget and split manifest, and reports validation means/std across seeds. Compare
wall-clock cost as well as transitions; ensembles/planning have different costs.
`v5_baseline.yaml` is a v5-runtime ablation with memory, extra ensemble and new
curriculum/distillation features disabled. The original v4 remains a separate
baseline; use the same independent v5 evaluation protocol when comparing it.

## Outputs and monitoring

The v5 training command now starts the monitor automatically and prints dashboard
and simulator links immediately. For the HCM preset the simulator link is
`http://127.0.0.1:8765/sim?driver=latest&map=hcm`. The page uses the latest **policy
prior** snapshot and updates as snapshots are exported; it is a separate live
simulation, not a stream of the training worker's full-MPPI decisions.

Use `--port 9000` to change the port, `--port 0` to disable HTTP, or
`--keep-monitor` to keep serving after training finishes. `--bind 127.0.0.1`
restricts listening to localhost. On a remote server, forward the port:

```bash
ssh -L 8765:127.0.0.1:8765 user@YOUR_SERVER
```

The initial prior is exported as soon as the controller is ready, before waiting
for the first `export_every` milestone. Console output is saved in `train.log`.
A pinned source-hash migration allows checkpoints from the preceding v5 entry
point to resume with this observability update; unrelated code changes still fail
strict validation.


`latest.pt`: exact resumable training state. `best.pt`: validation-selected state.
`policy.json` / `best_policy.json`: recurrent prior only. `metadata.json`,
`config.yaml`, `splits.json`: provenance. `metrics.jsonl`, `status.json`: progress.
`validation_*.json`: episode-level validation and calibration reports.
`curriculum.json`: current archive/pending/mastered scenarios and generation log.
`result.json`: completion state and best validation; test remains unevaluated.

```bash
python monitor/server.py --wm lotus_v5_full --bind 127.0.0.1
```

The shared simulator can load v5 priors, with execution feedback and episode resets.
Use the JSONL logs/evaluation artifacts for detailed v5 metrics; the legacy dashboard
was originally designed around v2-v4 charts. No long training job is started by
installation or tests.
