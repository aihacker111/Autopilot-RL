from pathlib import Path
import yaml
from v5_train import ROOT


def load_config(path):
    cfg = yaml.safe_load((ROOT / 'configs/v5_full.yaml').read_text())['v5']
    defaults = yaml.safe_load((ROOT / 'configs/v6_full.yaml').read_text())['v6']
    custom = yaml.safe_load(Path(path).read_text())['v6']
    unknown = set(custom) - set(cfg) - set(defaults)
    if unknown: raise ValueError(f'Unknown v6 settings: {sorted(unknown)}')
    cfg.update(defaults); cfg.update(custom)
    if not cfg['generated_maps'] and not (ROOT/'web/hcm/index.json').is_file(): raise FileNotFoundError('HCM map missing')
    if cfg['latent'] % 8: raise ValueError('latent must be divisible by 8')
    if cfg['ensemble'] > cfg['futures']: raise ValueError('futures must cover the dynamics ensemble')
    for k in ('workers','envs','batch','horizon','samples','ensemble','eval_batch','checkpoint_every','export_every','log_every'):
        if cfg[k] < 1: raise ValueError(f'{k} must be positive')
    if cfg['burn_in'] < 0 or cfg['counterfactual_length'] < cfg['horizon'] + cfg['burn_in']: raise ValueError('Invalid burn-in/counterfactual length')
    if cfg['demo_fraction'] < 0 or cfg['counterfactual_fraction'] < 0 or cfg['demo_fraction'] + cfg['counterfactual_fraction'] >= 1: raise ValueError('Reserve a positive online fraction')
    if cfg['handover_required_passes'] < 1 or not 0 < cfg['handover_decrement'] <= 1: raise ValueError('Invalid handover schedule')
    if not 0 <= cfg['teacher_probability_floor'] <= 1: raise ValueError('Invalid teacher floor')
    if cfg['intervention_cost'] < 0 or cfg['posterior_coef'] < 0: raise ValueError('Invalid loss coefficient')
    return cfg
