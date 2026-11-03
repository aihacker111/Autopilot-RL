# LOTUS v6 training

V6 corrects three learning contracts in v5. It is a new experiment, with checkpoint schema 6. These changes have smoke/integration coverage; improved driving quality must still be demonstrated by a longer run and held-out evaluation.

## What changed

- **Multi-step world model:** reward, requested-action Q, collision, physics, kinematics and action-projection heads are trained along ensemble imagined trajectories. A smaller posterior objective retains one-step supervision. Reward/value losses now backpropagate into multi-step dynamics, instead of only fitting observation-encoded states.
- **Actions and shielding:** dynamics/reward/kinematics use the executed action. Q uses the requested action and a shield-mode bit; its target is physical reward minus `intervention_cost * mean((requested - executed)^2)`. Actor, TD targets and planner tail use that same Q contract. A learned projector estimates executed actions inside planning. The real simulator shield remains authoritative. Counterfactual branches are explicitly unshielded and excluded from actor updates.
- **Protected teacher replay:** teacher-driven contiguous sequences have separate storage. Online/CF samples cannot overwrite them. Default batches contain 50% demonstrations, 10% CF and 40% online data when all sources are available. Gaps truncate recurrence and censor future labels. Teacher cloning requires a known contact-free one-second target, with immediate-contact and correction-based quality weighting.
- **Handover based on validation:** teacher probability starts at 1. Both the planner and prior must pass fixed validation gates twice before teacher probability drops by 0.25. Regression raises teacher probability again; the floor is 0.10. No unconditional switch at 10k steps.
- **Diagnostics and checkpoints:** interval-averaged losses, model errors by horizon, Q targets, projection error, collision label statistics, teacher/demo fractions and measured latency. Checkpoints include demo buffers, handover, optimizers, recurrent state, environment and RNGs.

## Run

From the repository root, using the Python environment with PyTorch, NumPy, PyYAML and Node available:

```bash
python train_v6.py --config configs/v6_smoke.yaml
python train_v6.py --config configs/v6_debug.yaml
python train_v6.py --config configs/v6_full.yaml
```

Use a fresh output directory for each experiment (`--out rl/runs/lotus_v6_seed2 --seed 2`). The full configuration targets **5,000,000 environment transitions**, inherits the large v5 model and HCM simulator, saves every 10k transitions and validates every 20k. It disables evolutionary scenario changes initially so core learning can be assessed on a stable distribution. Debug uses a smaller model and 100k transitions; smoke uses generated maps and 48 transitions to check plumbing, not driving ability.

The embedded monitor prints localhost and network dashboard/simulator links, default port 8765. The `/sim?driver=latest&map=hcm` link loads the exported **policy prior**, refreshed every 2k transitions; it is a separate rollout, not a live replay of the training worker or full planner. Fixed validation reports contain full-planner behavior. Use `--port 0` to disable the server, `--bind 127.0.0.1` for local access or `--keep-monitor` to retain it after training exits. On a remote host, forward port 8765 with SSH before using the localhost link.

## Gates and choosing a checkpoint

Before collecting data, the full configuration evaluates the defensive teacher on 12 fixed validation episodes. It stops and saves a checkpoint if completion is below 50% or contacts/km exceeds 1.0. Inspect `teacher_baseline.json`; a poor teacher requires better demonstrations or scenarios before a long run.

Handover gates require at least 12 prior episodes, completion at least 60%, contacts/km at most 0.25, red-light fraction at most 0.02 and projection MAE at most 0.10. Planner results must also meet these limits. Two consecutive passes reduce teacher use. If no handover occurs by 100k transitions, training stops with a checkpoint and explanation. These are initial engineering thresholds, not statistically established safety guarantees; inspect exposure (`km`, counts), results across seeds and intervention dependence before changing them. Simply raising the step budget does not fix a failed gate.

`best.pt` / `best_policy.json` are selected using the fixed validation ranking. `latest.pt` is the resumable state. `validation_<steps>.json` contains both controller reports and handover decisions. Training never selects a checkpoint using the test split.

```bash
python train_v6.py --config configs/v6_full.yaml --resume rl/runs/lotus_v6_full/latest.pt
python train_v6.py --evaluate rl/runs/lotus_v6_full/best.pt --split validation --matrix
# Final held-out assessment after choosing settings/checkpoint:
python train_v6.py --evaluate rl/runs/lotus_v6_full/best.pt --split test --matrix
```

The matrix runs prior/planner × shield on/off. Compare completion, contacts/person contacts, red lights, intervention fraction, stationary fraction, progress, jerk, latency and projection MAE. Loss decreasing alone does not establish better driving. `initial_q_minus_return` excludes timeouts/stage truncations; with a planner it also includes mismatch between future planner actions and the prior policy valued by Q. Collision calibration excludes unknown/censored targets.

## V5 compatibility

V5 files and entry point remain available. A v5 checkpoint cannot be exactly resumed under v6 because action semantics, critic and replay schema changed. To initialize a **new** v6 experiment from matching-size v5/v6 weights:

```bash
python train_v6.py --config configs/v6_full.yaml --out rl/runs/lotus_v6_from_v5 --init-from /absolute/path/to/v5/latest.pt
```

Model dimensions must match. Incompatible v5 critic input layers are freshly initialized; compatible critic layers and encoder/dynamics/policy weights can transfer. Optimizers, replay, teacher schedule, steps and RNG are reset. `initialization.json` records loaded/skipped tensors. This is not an exact continuation and a poorly trained prior may be worse than a fresh start. Exact `--resume` accepts v6 only and checks configuration, source, data and runtime provenance.

## References

- [TD-MPC2 official implementation](https://github.com/nicklashansen/tdmpc2): imagined rollout supervision and random two-Q reduction.
- [Efficient Online Reinforcement Learning with Offline Data (RLPD)](https://proceedings.mlr.press/v202/ball23a.html): mixing protected demonstration/offline data with online experience.

The recurrent encoder, discrete multiple-choice dynamics, projection model, intervention objective and handover gates are project-specific extensions; this is not an exact reproduction of either method.
