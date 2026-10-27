#!/usr/bin/env python3
"""Train-only MLB PA probability model. Python 3.9+.

Usage: python bullpen_train_final.py
Outputs in results/bullpen_quick/: model, metadata, history state, validation/test diagnostics.
Quick (~20-minute target, machine/network dependent) training-only version.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import json
import pickle
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pybaseball import statcast, cache
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

DATA_DIR = Path('data')
OUT_DIR = Path('results/bullpen_quick')
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Lightweight 2025 sample. This does not cover the full season.
SEASONS = [(2025, '2025-04-01', '2025-09-04')]
TRAIN_END = '2025-07-04'
VAL_START, VAL_END = '2025-07-05', '2025-08-04'
TEST_START, TEST_END = '2025-08-05', '2025-09-04'
# The final model is refit through this date. Do not interpret final-model fit as test performance.
FINAL_TRAIN_END = TEST_END
RANDOM_STATE = 42
MAX_ROUNDS = 350
EARLY_STOPPING_ROUNDS = 25
MIN_HISTORY_PA = 0  # Do not discard new players: smoothing + history count handle sparse samples.
SMOOTHING = {'season': 80., 'recent': 40., 'platoon': 60., 'h2h': 100.}
OUTCOMES = ['1B', '2B_3B', 'BB_HBP', 'HR', 'K', 'OTHER_OUT']
OUTCOME_INDEX = {k: i for i, k in enumerate(OUTCOMES)}
TARGETS = ['K', 'BB_HBP', '1B', '2B_3B', 'HR', 'OTHER_OUT']
# External league priors are conservative initialization only; not calculated from future data.
PRIOR = {'1B': .145, '2B_3B': .052, 'BB_HBP': .095, 'HR': .030, 'K': .225, 'OTHER_OUT': .453}
PRIOR_VEC = np.array([PRIOR[x] for x in OUTCOMES], dtype=np.float64)

EVENT_MAP = {
    'single':'1B', 'double':'2B_3B', 'triple':'2B_3B', 'home_run':'HR',
    'strikeout':'K', 'strikeout_double_play':'K',
    'walk':'BB_HBP', 'intent_walk':'BB_HBP', 'hit_by_pitch':'BB_HBP',
    'field_out':'OTHER_OUT', 'force_out':'OTHER_OUT',
    'grounded_into_double_play':'OTHER_OUT', 'fielders_choice':'OTHER_OUT',
    'fielders_choice_out':'OTHER_OUT', 'double_play':'OTHER_OUT',
    'triple_play':'OTHER_OUT', 'sac_fly':'OTHER_OUT', 'sac_bunt':'OTHER_OUT',
    'other_out':'OTHER_OUT',
}


def fetch_data():
    cache.enable()
    frames = []
    for season, start, end in SEASONS:
        path = DATA_DIR / ('statcast_regular_%s_%s_%s.parquet' % (season, start, end))
        if path.exists():
            print('Using cached:', path, flush=True)
            raw = pd.read_parquet(path)
        else:
            print('Downloading:', start, end, flush=True)
            raw = statcast(start_dt=start, end_dt=end, verbose=True, parallel=False)
            if raw.empty:
                raise RuntimeError('No Statcast rows downloaded for %s' % season)
            raw.to_parquet(path, index=False)
        if 'game_type' in raw:
            raw = raw.loc[raw.game_type.eq('R')].copy()
        needed = ['game_date','game_type','events','game_pk','at_bat_number','pitch_number',
                  'batter','pitcher','stand','p_throws','on_1b','on_2b','on_3b',
                  'fld_score','bat_score','inning','outs_when_up']
        raw = raw[[c for c in needed if c in raw.columns]].copy()
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    raw['game_date'] = pd.to_datetime(raw.game_date).dt.normalize()
    raw = raw.loc[raw.game_date.le(pd.Timestamp(FINAL_TRAIN_END))].copy()
    return raw


def make_pa(raw):
    # A PA has a final-pitch event; some nonstandard events are intentionally excluded.
    pa = raw.loc[raw.events.isin(EVENT_MAP)].copy()
    pa['outcome'] = pa.events.map(EVENT_MAP)
    pa = pa.sort_values(['game_date','game_pk','at_bat_number','pitch_number'])
    pa = pa.drop_duplicates(['game_pk','at_bat_number'], keep='last').reset_index(drop=True)
    pa['season'] = pa.game_date.dt.year.astype(int)
    pa['same_side'] = pa.stand.eq(pa.p_throws).astype(int)
    for b in (1,2,3):
        pa['runner_on_%db' % b] = pa['on_%db' % b].notna().astype(int)
    pa['num_runners'] = pa[['runner_on_1b','runner_on_2b','runner_on_3b']].sum(axis=1)
    pa['pitch_team_lead'] = pd.to_numeric(pa.fld_score, errors='coerce') - pd.to_numeric(pa.bat_score, errors='coerce')
    # The 'inning' and 'outs_when_up' fields describe the situation before the PA.
    return pa


def workload_table(raw):
    # One pitch per Statcast row. Deduplicate pitch identity to avoid accidental double counting.
    pitches = raw.drop_duplicates(['game_pk','at_bat_number','pitch_number'])
    daily = pitches.groupby(['pitcher','game_date']).size().rename('pitches').reset_index()
    # Each date's workload features use strictly earlier dates (not current-game pitches).
    out = {}
    for pitcher, group in daily.groupby('pitcher', sort=False):
        dates = group.game_date.tolist()
        counts = group.pitches.to_numpy()
        for j, date in enumerate(dates):
            previous = [(dates[k], int(counts[k])) for k in range(j) if dates[k] < date]
            record = {}
            for d in (1,3,7,14):
                record['pitches_last_%dd' % d] = sum(n for day,n in previous if day >= date-pd.Timedelta(days=d))
            record['appearances_last_7d'] = sum(1 for day,n in previous if day >= date-pd.Timedelta(days=7))
            record['appearances_last_14d'] = sum(1 for day,n in previous if day >= date-pd.Timedelta(days=14))
            if previous:
                rest = (date-previous[-1][0]).days
                record['days_since_last_outing'] = min(rest, 30)
                record['days_rest'] = min(max(rest-1, 0), 30)  # complete days off
                record['back_to_back'] = int(rest == 1)
            else:
                record['days_since_last_outing'] = 30
                record['days_rest'] = 30
                record['back_to_back'] = 0
            out[(pitcher, date)] = record
    return out


def zero_counts():
    return np.zeros(6, dtype=np.int64)


class History:
    """Chronological state for future prediction; updated AFTER producing each PA's features."""
    def __init__(self):
        self.counts = defaultdict(zero_counts)
        self.n = defaultdict(int)
        self.recent = defaultdict(deque)  # only batter/pitcher; store (date, class)

    def snapshot(self, key, prior, strength):
        n = self.n[key]
        return (self.counts[key] + strength*prior)/(n+strength), n

    def recent_snapshot(self, key, date, days, prior):
        hist = self.recent[key]
        start = date-pd.Timedelta(days=days)
        # Store 30 days at most; the current PA is not in the deque yet.
        counts = np.zeros(6, dtype=np.int64)
        n = 0
        for dt, idx in reversed(hist):
            if dt < start:
                break
            if dt < date:  # strictly previous calendar days; avoids same-day game order leakage
                counts[idx] += 1
                n += 1
        return (counts + SMOOTHING['recent']*prior)/(n+SMOOTHING['recent']), n

    def update(self, keys, date, idx):
        for key in keys:
            self.counts[key][idx] += 1
            self.n[key] += 1
        for key in keys[:2]:
            hist = self.recent[key]
            hist.append((date, idx))
            cutoff = date-pd.Timedelta(days=31)
            while hist and hist[0][0] < cutoff:
                hist.popleft()


def feature_names():
    names = []
    for prefix in ('batter_season','pitcher_season','batter_recent_7d','batter_recent_14d',
                   'batter_recent_30d','pitcher_recent_7d','pitcher_recent_14d','pitcher_recent_30d',
                   'batter_vs_pitch_hand','pitcher_vs_batter_hand','batter_pitcher_h2h'):
        names.extend([prefix+'_p_'+c for c in OUTCOMES])
        names.append(prefix+'_pa')
    names += ['same_side','batter_is_left','pitcher_is_left','inning','outs_when_up',
              'pitch_team_lead','runner_on_1b','runner_on_2b','runner_on_3b','num_runners',
              'pitches_last_1d','pitches_last_3d','pitches_last_7d','pitches_last_14d',
              'appearances_last_7d','appearances_last_14d','days_since_last_outing',
              'days_rest','back_to_back']
    return names

FEATURES = feature_names()


def build_features(pa, workload):
    state = History()
    records = []
    print('Building leakage-safe season, recent, platoon, H2H and fatigue features...', flush=True)
    # Do not reset H2H at season boundary; season features do reset.
    cols = ['game_date','season','batter','pitcher','stand','p_throws','outcome',
            'same_side','inning','outs_when_up','pitch_team_lead',
            'runner_on_1b','runner_on_2b','runner_on_3b','num_runners']
    for i, row in enumerate(pa[cols].itertuples(index=False, name=None)):
        (date, season, batter, pitcher, stand, throw, outcome, same_side, inning, outs,
         lead, r1, r2, r3, runners) = row
        # String prefixes prevent identifier collisions.
        bk = ('b',int(batter),int(season))
        pk = ('p',int(pitcher),int(season))
        bhand = ('bh',int(batter),int(season),str(throw))
        phand = ('ph',int(pitcher),int(season),str(stand))
        h2h = ('h2h',int(batter),int(pitcher))
        keys = (bk, pk, bhand, phand, h2h)
        values = {}
        groups = [
            ('batter_season', state.snapshot(bk, PRIOR_VEC, SMOOTHING['season'])),
            ('pitcher_season', state.snapshot(pk, PRIOR_VEC, SMOOTHING['season'])),
        ]
        for days in (7,14,30):
            groups.append(('batter_recent_%dd' % days, state.recent_snapshot(bk,date,days,PRIOR_VEC)))
        for days in (7,14,30):
            groups.append(('pitcher_recent_%dd' % days, state.recent_snapshot(pk,date,days,PRIOR_VEC)))
        groups.extend([
            ('batter_vs_pitch_hand', state.snapshot(bhand,PRIOR_VEC,SMOOTHING['platoon'])),
            ('pitcher_vs_batter_hand', state.snapshot(phand,PRIOR_VEC,SMOOTHING['platoon'])),
            ('batter_pitcher_h2h', state.snapshot(h2h,PRIOR_VEC,SMOOTHING['h2h'])),
        ])
        for prefix,(rates,n) in groups:
            for j,cls in enumerate(OUTCOMES):
                values[prefix+'_p_'+cls] = float(rates[j])
            values[prefix+'_pa'] = n
        values.update({
            'same_side': same_side, 'batter_is_left': int(stand=='L'),
            'pitcher_is_left': int(throw=='L'), 'inning':inning,'outs_when_up':outs,
            'pitch_team_lead':lead,'runner_on_1b':r1,'runner_on_2b':r2,
            'runner_on_3b':r3,'num_runners':runners,
        })
        values.update(workload.get((pitcher,date), {
            'pitches_last_1d':0,'pitches_last_3d':0,'pitches_last_7d':0,
            'pitches_last_14d':0,'appearances_last_7d':0,'appearances_last_14d':0,
            'days_since_last_outing':30,'days_rest':30,'back_to_back':0,
        }))
        records.append(values)
        state.update(keys,date,OUTCOME_INDEX[outcome])
        if (i+1)%50000 == 0:
            print('  Features built: {:,} / {:,}'.format(i+1,len(pa)),flush=True)
    X = pd.DataFrame.from_records(records,columns=FEATURES)
    X = X.apply(pd.to_numeric,errors='coerce').astype('float32')
    X = X.replace([np.inf,-np.inf],np.nan)
    # XGBoost handles NaN natively; never impute using future test statistics.
    return X,state


def brier(y,p):
    onehot = np.eye(len(OUTCOMES))[y]
    return float(np.mean(np.sum((p-onehot)**2,axis=1)))


def score(y,p):
    return {'log_loss':float(log_loss(y,p,labels=np.arange(len(OUTCOMES)))),
            'brier_score':brier(y,p),
            'accuracy':float(accuracy_score(y,np.argmax(p,axis=1))),
            'n_pa':int(len(y))}


def model_params(rounds,early_stop=None):
    args = dict(objective='multi:softprob',num_class=len(OUTCOMES),n_estimators=int(rounds),
                learning_rate=0.06,max_depth=3,min_child_weight=12,
                subsample=0.85,colsample_bytree=0.85,reg_lambda=5.0,reg_alpha=0.2,
                tree_method='hist',eval_metric='mlogloss',random_state=RANDOM_STATE,
                n_jobs=2)
    if early_stop is not None:
        args['early_stopping_rounds'] = early_stop
    return args


def calibration_plot(y,probs,path):
    fig,ax = plt.subplots(figsize=(8,7))
    for k,cls in enumerate(OUTCOMES):
        pred=probs[:,k]
        true=(y==k).astype(float)
        xs,ys=[],[]
        for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            mask=(pred>=lo)&(pred<(hi if hi<1 else 1.000001))
            if mask.sum()>=30:
                xs.append(pred[mask].mean());ys.append(true[mask].mean())
        if xs: ax.plot(xs,ys,marker='o',label=cls)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='Predicted probability',ylabel='Observed frequency',title='Held-out test calibration')
    ax.legend();fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def main():
    raw=fetch_data()
    pa=make_pa(raw)
    if len(pa)<5000:
        raise RuntimeError('Too few PAs; verify full-season downloads and cached files.')
    print('Retained {:,} PA'.format(len(pa)),flush=True)
    workload=workload_table(raw)
    X,state=build_features(pa,workload)
    y=pa.outcome.map(OUTCOME_INDEX).to_numpy(dtype=np.int32)
    dates=pa.game_date
    train=dates.le(pd.Timestamp(TRAIN_END)).to_numpy()
    val=dates.between(pd.Timestamp(VAL_START),pd.Timestamp(VAL_END)).to_numpy()
    test=dates.between(pd.Timestamp(TEST_START),pd.Timestamp(TEST_END)).to_numpy()
    final=dates.le(pd.Timestamp(FINAL_TRAIN_END)).to_numpy()
    for name,mask in [('train',train),('validation',val),('test',test)]:
        print(name, int(mask.sum()),flush=True)
        if mask.sum()<500: raise RuntimeError('Insufficient '+name+' PAs; check dates.')
    if not (set(y[train])==set(range(6))): raise RuntimeError('Missing classes in training')

    print('Training up to %d rounds, early stopping on validation log loss...' % MAX_ROUNDS,flush=True)
    tuning=XGBClassifier(**model_params(MAX_ROUNDS,EARLY_STOPPING_ROUNDS))
    tuning.fit(X.loc[train],y[train],eval_set=[(X.loc[val],y[val])],verbose=100)
    best_rounds=int(tuning.best_iteration)+1
    print('Best rounds:',best_rounds,flush=True)
    pval=tuning.predict_proba(X.loc[val])
    ptest=tuning.predict_proba(X.loc[test])
    train_freq=np.bincount(y[train],minlength=6).astype(float)
    train_freq/=train_freq.sum()
    rows=[]
    for split,mask,probs in [('validation',val,pval),('test',test,ptest)]:
        for name,p in [('XGBoost',probs),('Class-frequency baseline',np.tile(train_freq,(int(mask.sum()),1)))]:
            rows.append({'split':split,'model':name,**score(y[mask],p)})
    metrics=pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR/'evaluation_metrics.csv',index=False)
    print(metrics.to_string(index=False),flush=True)
    test_out=pa.loc[test,['game_date','game_pk','batter','pitcher','stand','p_throws','outcome']].copy()
    for k,cls in enumerate(OUTCOMES): test_out['prob_'+cls]=ptest[:,k]
    test_out['predicted_outcome']=[OUTCOMES[i] for i in np.argmax(ptest,axis=1)]
    test_out.to_csv(OUT_DIR/'heldout_test_predictions.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,8))
    ConfusionMatrixDisplay(confusion_matrix(y[test],np.argmax(ptest,axis=1),labels=np.arange(6)),display_labels=OUTCOMES).plot(ax=ax,xticks_rotation=40)
    fig.tight_layout();fig.savefig(OUT_DIR/'heldout_confusion_matrix.png',dpi=170);plt.close(fig)
    calibration_plot(y[test],ptest,OUT_DIR/'heldout_calibration.png')

    # The final artifact uses all data through FINAL_TRAIN_END; evaluation above is ONLY for tuning model.
    print('Refitting deployment model on all historical data, %d rounds...' % best_rounds,flush=True)
    final_model=XGBClassifier(**model_params(best_rounds))
    final_model.fit(X.loc[final],y[final],verbose=False)
    final_model.save_model(str(OUT_DIR/'xgb_matchup_model.json'))
    pd.DataFrame({'feature':FEATURES,'importance':final_model.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(OUT_DIR/'feature_importance.csv',index=False)
    metadata={
        'model_path':'xgb_matchup_model.json',
        'history_path':'feature_history.pkl',
        'workload_history_path':'pitcher_daily_workload.parquet',
        'feature_names':FEATURES,'classes':OUTCOMES,'priors':PRIOR,
        'smoothing':SMOOTHING,'season_ranges':SEASONS,
        'train_end_for_evaluation':TRAIN_END,'validation_window':[VAL_START,VAL_END],
        'heldout_test_window':[TEST_START,TEST_END],
        'final_fit_through':FINAL_TRAIN_END,'best_n_estimators':best_rounds,
        'n_final_pa':int(final.sum()),'n_features':len(FEATURES),
        'notes':'Final model refit includes held-out test dates. Metrics refer ONLY to the pre-refit tuning model. Feature state is after final labeled PA. Prediction code must compute identical features from pre-PA information and update state only after an observed PA.',
    }
    (OUT_DIR/'model_metadata.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
    # WARNING: pickle is for trusted locally generated files only. State enables separate inference code.
    with (OUT_DIR/'feature_history.pkl').open('wb') as f: pickle.dump(state,f,pickle.HIGHEST_PROTOCOL)
    # Needed by the separate inference script to calculate future rolling workload.
    raw.drop_duplicates(['game_pk','at_bat_number','pitch_number']).groupby(
        ['pitcher','game_date']).size().rename('pitches').reset_index().to_parquet(
        OUT_DIR/'pitcher_daily_workload.parquet',index=False)
    print('\nSaved final model:',(OUT_DIR/'xgb_matchup_model.json').resolve())
    print('Saved metadata:',(OUT_DIR/'model_metadata.json').resolve())
    print('Saved feature history:',(OUT_DIR/'feature_history.pkl').resolve())
    print('Saved workload history:',(OUT_DIR/'pitcher_daily_workload.parquet').resolve())
    print('Final model has %d features, trained on %d PAs.' % (len(FEATURES),final.sum()))

if __name__=='__main__': main()
