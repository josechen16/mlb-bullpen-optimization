#!/usr/bin/env python3
"""Predict six MLB plate-appearance outcome probabilities from a saved XGBoost model.

Run from your project root:
  python bullpen_predict.py --batter 605141 --pitcher 669203 --stand R --throws R \
      --date 2025-07-01 --inning 8 --outs 1 --first 1 --second 0 --third 0 --lead -1

IDs above are illustrative: replace with actual MLBAM IDs.
Requires the three metadata/history/workload files and xgb_matchup_model.json in --model-dir.
Only use trusted feature_history.pkl files produced by your own training script.
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# The training script pickled History when run as __main__. Provide compatible class
# definitions so trusted local model state can be restored without importing trainer.
def zero_counts():
    return np.zeros(6, dtype=np.int64)

class History:
    def __init__(self):
        self.counts = defaultdict(zero_counts)
        self.n = defaultdict(int)
        self.recent = defaultdict(deque)

    def snapshot(self, key, prior, strength):
        n = self.n.get(key, 0)
        counts = self.counts.get(key)
        if counts is None:
            counts = zero_counts()
        return (counts + strength * prior) / (n + strength), n

    def recent_snapshot(self, key, date, days, prior, strength):
        hist = self.recent.get(key, ())
        start = date - pd.Timedelta(days=days)
        counts = zero_counts()
        n = 0
        for dt, idx in reversed(hist):
            dt = pd.Timestamp(dt)
            if dt < start:
                break
            if dt < date:
                counts[idx] += 1
                n += 1
        return (counts + strength * prior) / (n + strength), n


def get_workload(df, pitcher, date):
    fields = ['pitches_last_1d', 'pitches_last_3d', 'pitches_last_7d',
              'pitches_last_14d', 'appearances_last_7d', 'appearances_last_14d',
              'days_since_last_outing', 'days_rest', 'back_to_back']
    default = dict.fromkeys(fields, 0)
    default.update(days_since_last_outing=30, days_rest=30)
    subset = df[(df['pitcher'] == pitcher) & (df['game_date'] < date)]
    if subset.empty:
        return default
    day = subset['game_date']
    result = {}
    for d in (1, 3, 7, 14):
        result[f'pitches_last_{d}d'] = int(subset.loc[day >= date - pd.Timedelta(days=d), 'pitches'].sum())
    for d in (7, 14):
        result[f'appearances_last_{d}d'] = int((day >= date - pd.Timedelta(days=d)).sum())
    rest = min((date - day.max()).days, 30)
    result.update(days_since_last_outing=rest, days_rest=min(max(rest-1, 0), 30), back_to_back=int(rest == 1))
    return result


def make_features(args, metadata, state, workload):
    date = pd.Timestamp(args.date).normalize()
    season = date.year
    batter, pitcher = args.batter, args.pitcher
    stand, throw = args.stand.upper(), args.throws.upper()
    classes = metadata['classes']
    prior = np.array([metadata['priors'][c] for c in classes], dtype=float)
    smooth = metadata['smoothing']
    bk = ('b', batter, season)
    pk = ('p', pitcher, season)
    bhand = ('bh', batter, season, throw)
    phand = ('ph', pitcher, season, stand)
    h2h = ('h2h', batter, pitcher)
    groups = [
        ('batter_season', state.snapshot(bk, prior, smooth['season'])),
        ('pitcher_season', state.snapshot(pk, prior, smooth['season'])),
    ]
    for d in (7, 14, 30):
        groups.append((f'batter_recent_{d}d', state.recent_snapshot(bk, date, d, prior, smooth['recent'])))
    for d in (7, 14, 30):
        groups.append((f'pitcher_recent_{d}d', state.recent_snapshot(pk, date, d, prior, smooth['recent'])))
    groups += [
        ('batter_vs_pitch_hand', state.snapshot(bhand, prior, smooth['platoon'])),
        ('pitcher_vs_batter_hand', state.snapshot(phand, prior, smooth['platoon'])),
        ('batter_pitcher_h2h', state.snapshot(h2h, prior, smooth['h2h'])),
    ]
    values = {}
    for prefix, (rates, n) in groups:
        for j, cls in enumerate(classes):
            values[f'{prefix}_p_{cls}'] = float(rates[j])
        values[f'{prefix}_pa'] = int(n)
    for cls in classes:
        values[f'matchup_batter_minus_pitcher_{cls}'] = (values[f'batter_vs_pitch_hand_p_{cls}'] - values[f'pitcher_vs_batter_hand_p_{cls}'])
        values[f'recent_batter_minus_season_{cls}'] = (values[f'batter_recent_14d_p_{cls}'] - values[f'batter_season_p_{cls}'])
        values[f'recent_pitcher_minus_season_{cls}'] = (values[f'pitcher_recent_14d_p_{cls}'] - values[f'pitcher_season_p_{cls}'])
    values.update(same_side=int(stand == throw), batter_is_left=int(stand == 'L'),
                  pitcher_is_left=int(throw == 'L'), inning=args.inning,
                  outs_when_up=args.outs, pitch_team_lead=args.lead,
                  runner_on_1b=args.first, runner_on_2b=args.second,
                  runner_on_3b=args.third, num_runners=args.first + args.second + args.third)
    values.update(get_workload(workload, pitcher, date))
    names = metadata['feature_names']
    missing = set(names) - set(values)
    if missing:
        raise ValueError(f'Missing trained features: {sorted(missing)}')
    return pd.DataFrame([[values[n] for n in names]], columns=names, dtype='float32'), values


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', default='results/bullpen_quick')
    p.add_argument('--batter', type=int, required=True, help='Batter MLBAM ID')
    p.add_argument('--pitcher', type=int, required=True, help='Pitcher MLBAM ID')
    p.add_argument('--stand', choices=['L', 'R'], required=True, help='Batter handedness')
    p.add_argument('--throws', choices=['L', 'R'], required=True, help='Pitcher throwing hand')
    p.add_argument('--date', required=True, help='YYYY-MM-DD; must be after historical cutoff')
    p.add_argument('--inning', type=int, required=True)
    p.add_argument('--outs', type=int, choices=[0, 1, 2], required=True)
    p.add_argument('--first', type=int, choices=[0, 1], default=0)
    p.add_argument('--second', type=int, choices=[0, 1], default=0)
    p.add_argument('--third', type=int, choices=[0, 1], default=0)
    p.add_argument('--lead', type=int, required=True, help='Pitching team score minus batting team score')
    args = p.parse_args()
    folder = Path(args.model_dir)
    metadata = json.loads((folder / 'model_metadata.json').read_text(encoding='utf-8'))
    cutoff = pd.Timestamp(metadata['final_fit_through']).normalize()
    target_date = pd.Timestamp(args.date).normalize()
    if target_date <= cutoff:
        p.error(f'Prediction date must be after {cutoff.date()}; saved state includes outcomes through that date (leakage prevention).')
    if target_date.year != cutoff.year:
        p.error('Prediction in another season requires refreshing historical state; do not silently reset season features.')
    if (target_date - cutoff).days > 30:
        print('WARNING: Historical state is over 30 days old; recent-form and fatigue features may be stale.', file=sys.stderr)
    # SECURITY: pickle must be generated by your own trusted training run.
    with (folder / metadata['history_path']).open('rb') as f:
        state = pickle.load(f)
    workload = pd.read_parquet(folder / metadata['workload_history_path'])
    workload['game_date'] = pd.to_datetime(workload['game_date']).dt.normalize()
    X, values = make_features(args, metadata, state, workload)
    model = XGBClassifier()
    model.load_model(str(folder / metadata['model_path']))
    if model.n_features_in_ != X.shape[1]:
        raise ValueError(f'Model expects {model.n_features_in_} features; metadata specifies {X.shape[1]}.')
    probabilities = model.predict_proba(X)[0]
    if len(probabilities) != len(metadata['classes']):
        raise ValueError('Model class count does not match metadata.')
    print(f'\nMatchup: batter {args.batter} vs pitcher {args.pitcher} | {args.date}')
    print(f'Game: inning {args.inning}, outs {args.outs}, bases {args.first}{args.second}{args.third}, pitching lead {args.lead:+d}')
    print(f'Historical data through: {cutoff.date()} | {len(X.columns)} features')
    print(f'Prior PA counts: batter season {values["batter_season_pa"]}, pitcher season {values["pitcher_season_pa"]}, H2H {values["batter_pitcher_h2h_pa"]}')
    print('\nPA outcome probabilities:')
    for cls, prob in zip(metadata['classes'], probabilities):
        print(f'  {cls:<12} {prob:7.2%}')
    print(f'  {"TOTAL":<12} {sum(probabilities):7.2%}')
    print('\nNote: Predictions are model estimates, not guaranteed calibrated probabilities.')

if __name__ == '__main__':
    main()
