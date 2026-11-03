"""Sequential, equal-budget multi-seed ablations. No test-set model selection.
python rl/v5_sweep.py --config configs/v5_full.yaml --steps 1000000 --seeds 2026,2027,2028 --variants full,no_memory,no_curriculum
"""
import argparse
from pathlib import Path
import numpy as np
from v5_train import load_config, train, atomic_json, ROOT

VARIANTS = {
    'full': {},
    'no_memory': dict(memory=False),
    'no_curriculum': dict(evo_every=0),
    'no_ensemble': dict(ensemble=1, uncertainty_beta=0., uncertainty_cost=0.),
    'no_distillation': dict(planner_coef=0.),
    'no_counterfactual': dict(counterfactual_probability=0., counterfactual_fraction=0.),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/v5_full.yaml'))
    p.add_argument('--steps', type=int, default=1000000); p.add_argument('--seeds', default='2026,2027,2028')
    p.add_argument('--variants', default='full,no_memory,no_curriculum,no_ensemble,no_distillation,no_counterfactual')
    p.add_argument('--out', default='rl/runs/v5_ablation')
    a = p.parse_args(); base = load_config(a.config); results = []
    for variant in a.variants.split(','):
        if variant not in VARIANTS: raise ValueError(f'Unknown variant: {variant}')
        for seed in map(int, a.seeds.split(',')):
            out = str(Path(a.out) / f'{variant}_seed{seed}')
            cfg = dict(base, **VARIANTS[variant], seed=seed, steps=a.steps, out=out)
            result = train(cfg)
            results.append(dict(variant=variant, seed=seed, validation=result['best_validation']))
            atomic_json(ROOT / a.out / 'runs.json', results)
    summary = {}
    for variant in a.variants.split(','):
        rows = [r['validation'] for r in results if r['variant'] == variant and r['validation']]
        summary[variant] = {k: dict(mean=float(np.mean([r[k] for r in rows])), std=float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0., seeds=len(rows))
                            for k in ('ok', 'contact_per_km', 'red', 'shield_per_km', 'route_progress')} if rows else {'validation_missing': True}
    atomic_json(ROOT / a.out / 'summary.json', summary)


if __name__ == '__main__': main()
