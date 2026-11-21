#!/usr/bin/env python3
"""Fit the approved development models and score frozen predictions.

Development commands have fixed, allowlisted inputs. Final access is disabled
until a separate input protocol is frozen, and always requires an explicit flag.
No engine or Maia inference is performed by this model/scoring module.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'freeze/approved_models.json'
INPUT_PROTOCOL = ROOT / 'freeze/final_input_protocol.json'
DEV = 'development_with_multipv.csv'
MAIA = 'notebook_runs/maia_bounded_four_archive_20261117_v1/maia_bounds_features.csv'
REPLIES = 'notebook_runs/opponent_reply_targets_development_v1/reply_targets.csv'
FOLDS = 'notebook_runs/final_optuna_development_20261117_v1/outer_fold_assignments.csv'
IDENTITY = ['source_pgn', 'game_index', 'ply', 'site']
PACKAGES = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'joblib',
            'threadpoolctl', 'python-dateutil', 'pytz', 'tzdata', 'six', 'chess']
FULL = ['eval_before_cp', 'initial_damage', 'player_elo', 'opponent_elo',
        'move_number', 'player_clock_before_move', 'opponent_clock_before_move',
        'initial_time_seconds', 'increment_seconds', 'material_balance_cp',
        'number_of_pieces', 'hanging_pieces', 'opponent_hanging_pieces',
        'king_attackers', 'opponent_king_attackers', 'reasonable_candidate_moves',
        'eval_gap_best_second_cp', 'in_check', 'gives_check', 'is_capture',
        'is_promotion', 'game_phase', 'multipv_good_reply_count_10pp',
        'multipv_best_second_gap_pp']
PROB = 'maia_preserving_reply_probability'
META = ['row_id'] + IDENTITY + ['win_before', 'event', 'time_control',
                              'multipv_lines_returned']
DEVELOPMENT_HASHES = {
    DEV: 'aca96c1cdcd20ea654dbca181acf69aa5d9396a89fc2bbb95532ee88189653e6',
    MAIA: '0bab47adb3dc063365abc34437c17fb16aa796bfdb48f010daf2a2e64ca4dd1d',
    REPLIES: 'c7dec063d2398b50e0883027cd0caf0089c3b766d29346fa409260853a8f1783',
    FOLDS: '98f0613732bfcca0d02632674152e531d58470eadfb82db05525621ec0bd340b',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert,
                                    allow_nan=False) + '\n')


def clean(value):
    """Convert undefined numerical results to JSON null, not fabricated zeros."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_config():
    c = json.loads(CONFIG.read_text())
    expected = dict(objective='regression_l1', n_estimators=240, learning_rate=.04,
                    num_leaves=11, min_child_samples=90, reg_lambda=7, verbosity=-1)
    if (c['features']['full'] != FULL or c['features']['maia'] != PROB or
        c['features']['reply_control'] != ['initial_damage', 'opponent_elo'] or
        c['regression_params'] != expected or
        c['classifier_params'] != dict(expected, objective='binary') or
        c['horizons'] != [2, 4, 6, 10] or c['seed'] != 1701 or c['jobs'] != 2 or
        c['reply_probability_cutoff'] != .5 or c['recovery_ratio_cutoff'] != .75 or
        c['bootstrap_repeats'] != 2000 or
        c['primary_archives'] != ['2017-04','2017-06','2017-07','2018-02','2019-02'] or
        c['secondary_archives'] != ['2017-04','2017-06','2017-07','2018-02'] or
        c['eligibility'] != dict(win_before_min=10,win_before_max=90,initial_damage_strictly_above=20) or
        c['comparisons'] != ['primary_regression','raw_maia_vs_majority','reply_control_vs_maia','ten_ply_control_vs_maia']):
        raise ValueError('The approved model/feature contract changed.')
    return c


def versions():
    native=sorted((Path(lightgbm.__file__).parent/'lib').glob('*lightgbm*'))
    return dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
                lightgbm_native_sha256={p.name:digest(p) for p in native if p.is_file()},
                packages={p: metadata.version(p) for p in PACKAGES})


def row_ids(frame):
    """Stable identities, independent of CSV ordering and local row indices."""
    if frame[IDENTITY].isna().any().any():
        raise ValueError('Missing row identity.')
    for column in ['game_index','ply']:
        values=pd.to_numeric(frame[column],errors='raise')
        if not (np.isfinite(values)&values.eq(values.astype(int))&values.gt(0)).all():
            raise ValueError('Nonpositive or noninteger row identity.')
    if frame.duplicated(IDENTITY[:3]).any() or frame.duplicated(['site', 'ply']).any():
        raise ValueError('Duplicate mistake or game/ply identity.')
    return frame[IDENTITY].apply(lambda r: hashlib.sha256(json.dumps(
        [str(r.source_pgn), int(r.game_index), int(r.ply), str(r.site)],
        separators=(',', ':')).encode()).hexdigest(), axis=1)


def read_development():
    """No input override: these are the only data files this loader can open."""
    c = load_config()
    for name, expected in DEVELOPMENT_HASHES.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Development source changed: ' + name)
    sources = pd.read_csv(ROOT / DEV, usecols=['source_pgn']).source_pgn
    allowed = {'lichess_db_standard_rated_' + m + '.pgn.zst'
               for m in c['primary_archives']}
    if sources.isna().any() or set(sources) != allowed:
        raise ValueError('Only the five canonical development archives are allowed.')
    # Read feature inputs separately from horizon outcomes.
    x = pd.read_csv(ROOT / DEV, usecols=list(dict.fromkeys(META[1:] + FULL)))
    x['row_id'] = row_ids(x)
    if len(x) != 23167 or x.site.nunique() != 6512:
        raise ValueError('Unexpected canonical development population.')
    x['row_index'] = np.arange(len(x))
    maia = pd.read_csv(ROOT / MAIA, usecols=IDENTITY + [PROB])
    x = x.merge(maia, on=IDENTITY, how='left', validate='one_to_one', sort=False)
    check_features(x.drop(columns='row_index'), require_maia=False)
    gaps = x.multipv_best_second_gap_pp.isna()
    if gaps.sum() != 76 or not x.loc[gaps, 'multipv_lines_returned'].eq(1).all():
        raise ValueError('Changed one-legal-reply gap semantics.')
    ycols = ['win_after'] + [f'{prefix}_{h}ply' for h in c['horizons']
                           for prefix in ['change_in_damage', 'win_after']]
    y = pd.read_csv(ROOT / DEV, usecols=ycols)
    y.insert(0, 'row_id', x.row_id.to_numpy())
    for h in c['horizons']:
        if not np.allclose(y[f'change_in_damage_{h}ply'],
                           y.win_after-y[f'win_after_{h}ply'], atol=1e-9):
            raise ValueError('Horizon target equation changed.')
    if not np.allclose(x.initial_damage, x.win_before-y.win_after, atol=1e-9):
        raise ValueError('Damage perspective/identity changed.')
    replies = pd.read_csv(ROOT / REPLIES, usecols=IDENTITY + ['status', 'win_after_actual_reply'])
    target = x[IDENTITY + ['row_id', 'win_before', 'initial_damage']].merge(
        replies, on=IDENTITY, how='left', validate='one_to_one', sort=False)
    valid = target.status.eq('ok') & np.isfinite(target.win_after_actual_reply)
    y['actual_preserved'] = np.where(valid,
        ((target.win_before-target.win_after_actual_reply)/target.initial_damage >= .75).astype(float),
        np.nan)
    four = secondary_mask(x, c)
    if (four.sum() != 15953 or x.loc[four, 'site'].nunique() != 4485 or
        y.loc[four, 'actual_preserved'].notna().sum() != 15851 or
        x.loc[four, PROB].isna().any() or x.loc[~four, PROB].notna().any()):
        raise ValueError('Changed secondary training eligibility.')
    return x, y


def secondary_mask(x, c):
    return x.source_pgn.isin(['lichess_db_standard_rated_' + m + '.pgn.zst'
                             for m in c['secondary_archives']])


def check_features(x, require_maia=True):
    allowed = set(FULL + [PROB] + META)
    if set(x.columns)-allowed or not set(FULL + META).issubset(x.columns):
        raise ValueError('Feature schema contains forbidden or missing columns.')
    if x.row_id.duplicated().any() or x.row_id.isna().any():
        raise ValueError('Invalid feature row identities.')
    if not np.array_equal(x.row_id.to_numpy(), row_ids(x).to_numpy()):
        raise ValueError('Row IDs disagree with chess identities.')
    if not (x.initial_damage.gt(20) & x.win_before.between(10, 90)).all():
        raise ValueError('Primary >20 / 10–90 eligibility changed.')
    numeric = x[[f for f in FULL if f != 'game_phase']].to_numpy(float)
    if np.isinf(numeric).any():
        raise ValueError('Infinite feature value.')
    if require_maia and (PROB not in x or not x[PROB].between(0, 1).all()):
        raise ValueError('Maia probability is missing or out of range.')
    if PROB in x and not x.loc[x[PROB].notna(), PROB].between(0, 1).all():
        raise ValueError('Invalid saved Maia probability.')
    if not x.loc[x.multipv_lines_returned.eq(1), 'multipv_best_second_gap_pp'].isna().all():
        raise ValueError('A one-legal-reply gap must remain missing.')
    if not x.multipv_good_reply_count_10pp.between(0, 5).all():
        raise ValueError('Invalid MultiPV good-reply count.')


def model_specs(c):
    specs = {}
    for h in c['horizons']:
        specs[f'primary_{h}'] = dict(task='primary', horizon=h, features=FULL,
                                    cohort='five', classifier=False)
    specs['ten_control'] = dict(task='ten', horizon=10, features=FULL,
                                cohort='four', classifier=False)
    specs['ten_maia'] = dict(task='ten', horizon=10, features=FULL+[PROB],
                             cohort='four', classifier=False)
    for name, extra in [('reply_control', []), ('reply_maia', [PROB])]:
        specs[name] = dict(task='reply', horizon=1, features=['initial_damage', 'opponent_elo']+extra,
                           cohort='four_valid_replies', classifier=True)
    return specs


def make_model(spec, c):
    features = spec['features']
    if set(features)-set(FULL+[PROB]):
        raise ValueError('Predictor allowlist violation.')
    if spec['classifier']:
        return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('model', LGBMClassifier(**c['classifier_params'],
                                    random_state=c['seed'], n_jobs=c['jobs']))])
    numeric = [f for f in features if f != 'game_phase']
    transforms = [('numeric', SimpleImputer(strategy='median', add_indicator=True), numeric),
                  ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['game_phase'])]
    return Pipeline([('prep', ColumnTransformer(transforms)),
        ('model', LGBMRegressor(**c['regression_params'], random_state=c['seed'], n_jobs=c['jobs']))])


def fit_models(x, y, c):
    """Training receives explicitly selected development rows; X never contains Y."""
    if not np.array_equal(x.row_id, y.row_id):
        raise ValueError('Training labels are misaligned.')
    allowed={'lichess_db_standard_rated_'+month+'.pgn.zst' for month in c['primary_archives']}
    if not x.source_pgn.isin(allowed).all():
        raise ValueError('Fitting is restricted to development archives.')
    check_features(x.drop(columns=['row_index'], errors='ignore'), require_maia=False)
    specs = model_specs(c)
    models, fitting = {}, {}
    four = secondary_mask(x, c)
    for name, spec in specs.items():
        mask = pd.Series(True, index=x.index) if spec['cohort']=='five' else four.copy()
        target = 'actual_preserved' if spec['classifier'] else f'change_in_damage_{spec["horizon"]}ply'
        if spec['classifier']:
            mask &= y.actual_preserved.notna()
        if not mask.any() or not np.isfinite(y.loc[mask, target]).all():
            raise ValueError('Invalid training target for ' + name)
        if spec['cohort'] != 'five' and not x.loc[mask, PROB].between(0, 1).all():
            raise ValueError('Secondary model lacks pre-reply Maia features.')
        model = make_model(spec, c)
        model.fit(x.loc[mask, spec['features']], y.loc[mask, target])
        models[name] = model
        prep = model.named_steps['imputer'] if spec['classifier'] else model.named_steps['prep'].named_transformers_['numeric']
        fitting[name] = dict(rows=int(mask.sum()), games=int(x.loc[mask, 'site'].nunique()),
            row_ids_sha256=hashlib.sha256('\n'.join(sorted(x.loc[mask, 'row_id'])).encode()).hexdigest(),
            numeric_medians=clean(prep.statistics_.tolist()),
            missing_indicator_features=prep.indicator_.features_.tolist(),
            effective_estimator_params=model.named_steps['model'].get_params())
    constants = dict(primary_mean={}, primary_recovered_majority={},
        ten_recovered_majority=bool((1+y.loc[four, 'change_in_damage_10ply']/x.loc[four, 'initial_damage'] < .75).mean() >= .5),
        reply_prevalence=float(
        y.loc[four & y.actual_preserved.notna(), 'actual_preserved'].mean()))
    for h in c['horizons']:
        target = y[f'change_in_damage_{h}ply']
        constants['primary_mean'][str(h)] = float(target.mean())
        constants['primary_recovered_majority'][str(h)] = bool((1+target/x.initial_damage < .75).mean() >= .5)
    return models, constants, fitting


def predict_features(models, constants, x, c, allow_missing_maia=False):
    """Pure inference on a strict feature-only table; label columns are rejected."""
    check_features(x, require_maia=not allow_missing_maia)
    out = []
    specs = model_specs(c)
    for name, spec in specs.items():
        model = models[name]
        frame=x if spec['task']=='primary' else x[x[PROB].notna()]
        if frame.empty:continue
        values = (model.predict_proba(frame[spec['features']])[:, 1] if spec['classifier']
                  else model.predict(frame[spec['features']]))
        out.append(pd.DataFrame(dict(row_id=frame.row_id.to_numpy(), task=spec['task'],
                                    horizon=spec['horizon'], model=name, prediction=values)))
    for h in c['horizons']:
        for name, value in [('no_change', 0.), ('training_mean', constants['primary_mean'][str(h)])]:
            out.append(pd.DataFrame(dict(row_id=x.row_id.to_numpy(), task='primary', horizon=h,
                                        model=name, prediction=value)))
    frame=x[x[PROB].notna()]
    for name, values in [('raw_maia', frame[PROB].to_numpy()), ('training_majority', constants['reply_prevalence'])]:
        out.append(pd.DataFrame(dict(row_id=frame.row_id.to_numpy(), task='reply', horizon=1,
                                    model=name, prediction=values)))
    result = pd.concat(out, ignore_index=True)
    if not np.isfinite(result.prediction).all():
        raise ValueError('Nonfinite predictions.')
    return result


def binary_metrics(y, score, cutoff=.5):
    y, score = np.asarray(y, bool), np.asarray(score, float)
    predicted = score >= cutoff
    tn, fp = int((~y & ~predicted).sum()), int((~y & predicted).sum())
    fn, tp = int((y & ~predicted).sum()), int((y & predicted).sum())
    div = lambda a, b: a/b if b else np.nan
    return dict(n=len(y), accuracy=div(tp+tn, len(y)),
        balanced_accuracy=.5*(div(tp,tp+fn)+div(tn,tn+fp)), precision=div(tp,tp+fp),
        recall=div(tp,tp+fn), other_class_recall=div(tn,tn+fp), tn=tn, fp=fp, fn=fn, tp=tp,
        roc_auc=float(roc_auc_score(y, score)) if len(np.unique(y))==2 else np.nan)


def regression_metrics(actual, prediction, damage, majority):
    y, p = np.asarray(actual, float), np.asarray(prediction, float)
    recovered = 1+y/np.asarray(damage) < .75
    ratio = 1+p/np.asarray(damage)
    # Strict ratio <.75 corresponds to score >-.75, including boundary ties.
    classification = binary_metrics(recovered, -ratio, np.nextafter(-.75, np.inf))
    return dict(mae=float(np.abs(y-p).mean()),
        spearman=float(spearmanr(y,p).statistic) if np.ptp(y) and np.ptp(p) else np.nan,
        recovered_only_mae=float(np.abs(y[recovered]-p[recovered]).mean()) if recovered.any() else np.nan,
        majority_accuracy=float(np.mean(recovered==majority)), **classification)


def paired_intervals(frame, predictions, labels, c, repeats):
    """Paired, row-weighted whole-game bootstrap; no model refitting."""
    rows = []
    merged = predictions.merge(labels, on='row_id', validate='many_to_one').merge(
        frame[['row_id','site','initial_damage']], on='row_id', validate='many_to_one')
    for threshold in [20,25,30]:
        subset = merged[merged.initial_damage > threshold]
        for (task,h), part in subset.groupby(['task','horizon']):
            target = 'actual_preserved' if task=='reply' else f'change_in_damage_{h}ply'
            wide = part.pivot(index='row_id', columns='model', values='prediction').sort_index()
            ref = part.drop_duplicates('row_id').set_index('row_id').loc[wide.index]
            valid = np.isfinite(ref[target].to_numpy(float))
            if task!='reply':
                valid &= np.isfinite(ref[[f'change_in_damage_{v}ply' for v in c['horizons']]]).all(axis=1)
            wide, ref = wide.loc[valid], ref.loc[valid]
            if ref.empty:
                continue
            y = ref[target].to_numpy(float)
            pairs = ([('reply_maia','reply_control'), ('raw_maia','training_majority')] if task=='reply'
                     else [('ten_maia','ten_control')] if task=='ten'
                     else [(f'primary_{h}', 'no_change'), (f'primary_{h}', 'training_mean')])
            for model,baseline in pairs:
                a, b = wide[model].to_numpy(), wide[baseline].to_numpy()
                if task=='reply':
                    stats = np.column_stack([(a-y)**2,(b-y)**2,
                        ((a>=.5)==y).astype(float),((b>=.5)==y).astype(float),
                        ((a>=.5)&(y==1)).astype(float),((b>=.5)&(y==1)).astype(float),
                        ((a<.5)&(y==0)).astype(float),((b<.5)&(y==0)).astype(float),y,1-y])
                    def gains(totals, sizes):
                        divide = lambda n,d: np.divide(n,d,out=np.full(np.shape(n),np.nan),where=d!=0)
                        ba=.5*(divide(totals[...,4],totals[...,8])+divide(totals[...,6],totals[...,9]))
                        bb=.5*(divide(totals[...,5],totals[...,8])+divide(totals[...,7],totals[...,9]))
                        return np.stack([(totals[...,1]-totals[...,0])/sizes,
                            100*(totals[...,2]-totals[...,3])/sizes,100*(ba-bb)],axis=-1)
                    names=['brier_improvement','accuracy_gain_pp','balanced_accuracy_gain_pp']
                else:
                    stats=np.column_stack([np.abs(y-a),np.abs(y-b)])
                    def gains(totals,sizes):
                        absolute=(totals[...,1]-totals[...,0])/sizes
                        percent=np.divide(100*(totals[...,1]-totals[...,0]),totals[...,1],
                            out=np.full(np.shape(absolute),np.nan),where=totals[...,1]!=0)
                        return np.stack([absolute,percent],axis=-1)
                    names=['mae_improvement_pp','mae_improvement_percent']
                codes,games=pd.factorize(ref.site,sort=True)
                sums=np.zeros((len(games),stats.shape[1]));np.add.at(sums,codes,stats)
                sizes=np.bincount(codes)
                rng=np.random.default_rng(c['seed']);draws=[]
                for start in range(0,repeats,100):
                    sample=rng.integers(0,len(games),(min(100,repeats-start),len(games)))
                    draws.append(gains(sums[sample].sum(axis=1),sizes[sample].sum(axis=1)))
                draws=np.vstack(draws);estimate=gains(sums.sum(axis=0),sizes.sum())
                for j,name in enumerate(names):
                    usable=draws[np.isfinite(draws[:,j]),j]
                    low,high=np.quantile(usable,[.025,.975]) if len(usable) else (np.nan,np.nan)
                    rows.append(dict(task=task,horizon=int(h),model=model,baseline=baseline,metric=name,
                        threshold=threshold,n=len(ref),games=len(games),estimate=estimate[j],
                        lower95=low,upper95=high,valid_replicates=len(usable),repeats=repeats))
    return pd.DataFrame(rows)


def score_predictions(x, labels, predictions, constants, c, repeats=None):
    if labels.row_id.duplicated().any() or set(x.row_id)!=set(labels.row_id):
        raise ValueError('Labels must match the feature population one-to-one.')
    expected_labels={'row_id','win_after','actual_preserved'}|{
        f'{prefix}_{h}ply' for h in c['horizons'] for prefix in ['win_after','change_in_damage']}
    if set(labels.columns)!=expected_labels:
        raise ValueError('Prepared label schema differs from the frozen target contract.')
    aligned=labels.set_index('row_id').loc[x.row_id]
    if not np.allclose(x.initial_damage.to_numpy(),x.win_before.to_numpy()-aligned.win_after.to_numpy(),atol=1e-9):
        raise ValueError('Labels disagree with initial damage or evaluation perspective.')
    for h in c['horizons']:
        if not np.allclose(aligned[f'change_in_damage_{h}ply'],
            aligned.win_after-aligned[f'win_after_{h}ply'],atol=1e-9,equal_nan=True):
            raise ValueError('Incorrect signed change-in-damage target.')
    chances=aligned[['win_after']+[f'win_after_{h}ply' for h in c['horizons']]].to_numpy(float)
    if np.isinf(chances).any() or ((chances[np.isfinite(chances)]<0)|(chances[np.isfinite(chances)]>100)).any():
        raise ValueError('Invalid target winning chance.')
    if predictions.duplicated(['row_id','task','horizon','model']).any():
        raise ValueError('Duplicate model predictions.')
    if set(predictions.row_id)!=set(x.row_id):
        raise ValueError('Prediction coverage differs from feature rows.')
    if not np.isfinite(predictions.prediction).all():raise ValueError('Invalid saved prediction.')
    expected={(s['task'],s['horizon'],name) for name,s in model_specs(c).items()}
    expected|={('primary',h,m) for h in c['horizons'] for m in ['no_change','training_mean']}
    expected|={('reply',1,m) for m in ['raw_maia','training_majority']}
    groups=predictions.groupby(['task','horizon','model'])
    if set(groups.groups)!=expected or any(set(g.row_id)!=set(
        x.row_id if key[0]=='primary' else x.loc[x[PROB].notna(),'row_id']) for key,g in groups):
        raise ValueError('Incomplete prediction grid or an unapproved model.')
    known=labels.actual_preserved.dropna()
    if not known.isin([0,1]).all():raise ValueError('Invalid immediate binary labels.')
    merged=predictions.merge(labels,on='row_id',validate='many_to_one').merge(
        x[['row_id','site','initial_damage']],on='row_id',validate='many_to_one')
    metrics=[]
    for threshold in [20,25,30]:
        for (task,h,model),part in merged[merged.initial_damage.gt(threshold)].groupby(['task','horizon','model']):
            target='actual_preserved' if task=='reply' else f'change_in_damage_{h}ply'
            valid=np.isfinite(part[target].to_numpy(float))
            if task!='reply':
                valid &= np.isfinite(part[[f'change_in_damage_{v}ply' for v in c['horizons']]]).all(axis=1)
            block=part.loc[valid]
            if block.empty:
                continue
            if task=='reply':
                scores=block.prediction.to_numpy(float);y=block[target].to_numpy(float)
                if not ((scores>=0)&(scores<=1)).all():raise ValueError('Invalid probability.')
                m=binary_metrics(y,scores);m['brier']=float(np.mean((scores-y)**2))
            else:
                m=regression_metrics(block[target],block.prediction,block.initial_damage,
                    constants['ten_recovered_majority'] if task=='ten' else
                    constants['primary_recovered_majority'][str(h)])
                if model in ['no_change','training_mean']:m['spearman']=np.nan
            metrics.append(dict(task=task,horizon=int(h),model=model,threshold=threshold,
                games=int(block.site.nunique()),excluded_missing_target=int((~valid).sum()),**m))
    ci=paired_intervals(x,predictions,labels,c,repeats or c['bootstrap_repeats'])
    return pd.DataFrame(metrics),ci


def contexts(x):
    frame=x[['row_id','site','player_elo','opponent_elo','time_control','event','game_phase']].copy()
    for column in ['player_elo','opponent_elo']:
        frame[column+'_band']=pd.cut(frame[column],[-np.inf,1600,2000,np.inf],right=False,
            labels=['<1600','1600–<2000','>=2000']).astype('string').fillna('Missing')
    def speed(event):
        value=str(event).casefold()
        for marker,label in [('ultrabullet','UltraBullet'),('bullet','Bullet'),('blitz','Blitz'),
                             ('rapid','Rapid'),('classical','Classical'),('correspondence','Correspondence')]:
            if marker in value:return label
        return 'Standard/unspecified'
    frame['speed_label']=frame.event.map(speed)
    return frame


def distributions(x, cohort):
    frame=contexts(x);rows=[]
    for field in ['player_elo_band','opponent_elo_band','time_control','speed_label','game_phase']:
        for value,part in frame.groupby(field,dropna=False,observed=True):
            rows.append(dict(cohort=cohort,variable=field,group=str(value),n=len(part),
                             games=part.site.nunique(),share=len(part)/len(frame)))
    return pd.DataFrame(rows)


def subgroup_metrics(x, labels, predictions, constants, c):
    context=contexts(x)
    data=predictions.merge(labels,on='row_id',validate='many_to_one').merge(
        x[['row_id','initial_damage']],on='row_id',validate='many_to_one').merge(
        context,on='row_id',validate='many_to_one')
    rows=[]
    for (task,h,model),part in data.groupby(['task','horizon','model']):
        target='actual_preserved' if task=='reply' else f'change_in_damage_{h}ply'
        valid=np.isfinite(part[target].to_numpy(float))
        if task!='reply':
            valid &= np.isfinite(part[[f'change_in_damage_{v}ply' for v in c['horizons']]]).all(axis=1)
        part=part.loc[valid]
        rating='opponent_elo_band' if task=='reply' else 'player_elo_band'
        for field in [rating,'time_control','speed_label','game_phase']:
            for value,block in part.groupby(field,dropna=False,observed=True):
                if task=='reply':
                    result=binary_metrics(block[target],block.prediction)
                    result['brier']=float(np.mean((block.prediction-block[target])**2))
                else:
                    majority=(constants['ten_recovered_majority'] if task=='ten'
                              else constants['primary_recovered_majority'][str(h)])
                    result=regression_metrics(block[target],block.prediction,block.initial_damage,majority)
                    if model in ['no_change','training_mean']:result['spearman']=np.nan
                rows.append(dict(task=task,horizon=h,model=model,variable=field,group=str(value),
                    games=block.site.nunique(),small_group=len(block)<100,**result))
    return pd.DataFrame(rows)


def fit_development(out):
    out=Path(out)
    if out.exists():raise FileExistsError('Use a new fit output directory.')
    c=load_config();x,y=read_development()
    print('Predictors:',json.dumps({k:v['features'] for k,v in model_specs(c).items()}),flush=True)
    models,constants,fitting=fit_models(x,y,c)
    out.mkdir(parents=True)
    artifacts={}
    for name,model in models.items():
        path=out/(name+'.joblib');joblib.dump(model,path,compress=3)
        artifacts[path.name]=digest(path)
    x[['row_id']+IDENTITY].to_csv(out/'training_membership.csv.gz',index=False)
    artifacts['training_membership.csv.gz']=digest(out/'training_membership.csv.gz')
    distribution=pd.concat([distributions(x,'development_five'),
        distributions(x.loc[secondary_mask(x,c)],'development_four')],ignore_index=True)
    distribution.to_csv(out/'development_distributions.csv',index=False)
    artifacts['development_distributions.csv']=digest(out/'development_distributions.csv')
    json_sources=[CONFIG,ROOT/'scripts/frozen_chess_evaluation.py',ROOT/'tune_chess.py']
    manifest=dict(schema_version=1,status='fitted_on_development_only',configuration=c,
        configuration_sha256=digest(CONFIG),source_sha256=DEVELOPMENT_HASHES,
        code_sha256={str(p.relative_to(ROOT)):digest(p) for p in json_sources},
        source_reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        versions=versions(),model_specs=model_specs(c),training=fitting,constants=constants,
        artifacts=artifacts,test_access=False,final_test_executed=False)
    write_json(out/'manifest.json',clean(manifest))
    for name in models:
        loaded=joblib.load(out/(name+'.joblib'))
        spec=model_specs(c)[name];mask=secondary_mask(x,c) if spec['cohort']!='five' else pd.Series(True,index=x.index)
        sample=x.loc[mask,spec['features']].iloc[:50]
        a=models[name].predict_proba(sample) if spec['classifier'] else models[name].predict(sample)
        b=loaded.predict_proba(sample) if spec['classifier'] else loaded.predict(sample)
        if not np.array_equal(a,b):raise AssertionError('Serialization changed '+name)
    print('Fitted and round-trip checked eight approved estimators:',out,flush=True)


def load_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['configuration_sha256']!=digest(CONFIG) or manifest['versions']!=versions():
        raise ValueError('Model configuration/environment differs from the fitted bundle.')
    for name,value in manifest['code_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('Fitted source code changed: '+name)
    for name,value in manifest['artifacts'].items():
        if Path(name).name!=name or digest(directory/name)!=value:
            raise ValueError('Artifact integrity mismatch: '+name)
    specs=model_specs(load_config())
    if manifest['model_specs']!=specs:raise ValueError('Bundle model roster changed.')
    return {name:joblib.load(directory/(name+'.joblib')) for name in specs},manifest


def final_access_guard(allow_final_test, input_protocol):
    """Must run before reading, statting or hashing any supplied final input."""
    if not allow_final_test:
        raise PermissionError('Final-test access requires explicit user authorization; development validation only.')
    required=['game_ranges_or_sampling','rating_filters','time_control_filters',
              'core_engine_and_cache_protocol','intermediate_lineage']
    if input_protocol.get('status')!='frozen' or any(input_protocol.get(k) is None for k in required):
        raise PermissionError('Final access disabled: exact extraction/input protocol is unresolved.')


def evaluate_prepared(bundle_dir, input_manifest, out, allow_final_test=False):
    """Future one-time scoring of prepared, separate feature and label tables."""
    protocol=json.loads(INPUT_PROTOCOL.read_text())
    final_access_guard(allow_final_test,protocol)
    _evaluate_prepared_files(bundle_dir,input_manifest,out,digest(INPUT_PROTOCOL))


def _evaluate_prepared_files(bundle_dir,input_manifest,out,protocol_hash,development_validation=False):
    """Shared file pipeline; development validation uses synthetic/development sources."""
    models,bundle=load_bundle(bundle_dir);c=bundle['configuration']
    contract=json.loads(Path(input_manifest).read_text())
    if contract['input_protocol_sha256']!=protocol_hash:raise ValueError('Input protocol hash mismatch.')
    out=Path(out)
    if out.exists():raise FileExistsError('Evaluation output already exists; do not rerun final scoring.')
    # Exclusive marker in the fitted bundle prevents a second run to a different output directory.
    marker=Path(bundle_dir)/('DEVELOPMENT_VALIDATION_STARTED.json' if development_validation
                            else 'FINAL_EVALUATION_STARTED.json')
    with marker.open('x') as handle:json.dump(dict(input_manifest=str(input_manifest),output=str(out)),handle)
    out.mkdir(parents=True)
    base=Path(input_manifest).resolve().parent
    feature_path=base/contract['features']['path']
    if digest(feature_path)!=contract['features']['sha256']:raise ValueError('Feature hash mismatch.')
    header=set(pd.read_csv(feature_path,nrows=0).columns)
    if header!=set(FULL+[PROB]+META):raise ValueError('Prepared features must contain the exact feature/metadata schema.')
    x=pd.read_csv(feature_path);check_features(x)
    allowed=({'lichess_db_standard_rated_'+m+'.pgn.zst' for m in c['primary_archives']}
             if development_validation else {'lichess_db_standard_rated_2025-03.pgn.zst'})
    if not x.source_pgn.isin(allowed).all():
        raise ValueError('Unexpected final archive.')
    members=pd.read_csv(Path(bundle_dir)/'training_membership.csv.gz')
    if set(x.site)&set(members.site):raise ValueError('Training/test game overlap.')
    predictions=predict_features(models,bundle['constants'],x,c)
    predictions.to_csv(out/'predictions_before_labels.csv.gz',index=False)
    # Only after predictions have been persisted may any outcome file be opened.
    label_path=base/contract['labels']['path']
    if digest(label_path)!=contract['labels']['sha256']:raise ValueError('Label hash mismatch.')
    labels=pd.read_csv(label_path)
    metrics,intervals=score_predictions(x,labels,predictions,bundle['constants'],c)
    metrics.to_csv(out/'metrics.csv',index=False);intervals.to_csv(out/'paired_game_intervals.csv',index=False)
    subgroup_metrics(x,labels,predictions,bundle['constants'],c).to_csv(out/'subgroups.csv',index=False)
    pd.concat([pd.read_csv(Path(bundle_dir)/'development_distributions.csv'),distributions(
        x,'development_validation' if development_validation else 'final_temporal')],
              ignore_index=True).to_csv(out/'development_vs_final_distributions.csv',index=False)
    write_json(out/'completion.json',dict(status='complete',one_time=True,
        development_validation=development_validation,
        input_protocol_sha256=protocol_hash,input_manifest_sha256=digest(input_manifest),
        bundle_manifest_sha256=digest(Path(bundle_dir)/'manifest.json'),
        versions=versions(),predictions_sha256=digest(out/'predictions_before_labels.csv.gz')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('fit-development');fit.add_argument('--out',type=Path,required=True)
    evaluate=sub.add_parser('evaluate-prepared')
    evaluate.add_argument('--bundle',type=Path,required=True)
    evaluate.add_argument('--input-manifest',type=Path,required=True)
    evaluate.add_argument('--out',type=Path,required=True)
    evaluate.add_argument('--allow-final-test',action='store_true')
    args=p.parse_args()
    if args.command=='fit-development':fit_development(args.out)
    else:evaluate_prepared(args.bundle,args.input_manifest,args.out,args.allow_final_test)


if __name__=='__main__':
    main()
