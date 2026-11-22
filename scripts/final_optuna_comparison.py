#!/usr/bin/env python3
"""One precommitted nested Optuna comparison, restricted to canonical development.

Prepare (validates and freezes protocol before any model fits):
  python scripts/final_optuna_comparison.py --out NEW_DIRECTORY --prepare-only
Run/resume the exact prepared protocol:
  python scripts/final_optuna_comparison.py --out NEW_DIRECTORY --resume

No input override is offered. Only the named development table and previously
validated development reply targets are read. Existing source artifacts are read
only. Completed outer-fold/task units are checkpointed without replacement.
"""
import argparse
import importlib.metadata
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tune_chess import EXPECTED_ARCHIVES, SETS, pipeline
from scripts.tune_recovery_threshold import checked_development, checked_folds, digest, players
from scripts.evaluate_recovery_classifier import (checked_six_ply_label,
    confusion_metrics, select_probability_cutoff, validate_features)
from scripts.analyze_opponent_reply_targets import load_validated

DATA = ROOT / 'development_with_multipv.csv'
REPLIES = ROOT / 'notebook_runs/opponent_reply_targets_development_v1/reply_targets.csv'
HORIZONS = (2, 4, 6, 10)
REPLY_TARGETS = ('opponent_reply_regret_pp', 'missed_opportunity_damage_pp')
SEED = 1701
TRIALS = 12
REPEATS = 2000
JOBS = 2
FULL = list(SETS['full_multipv'])
MODEL_FEATURES = {
    'fixed_lightgbm': FULL, 'optuna_lightgbm': FULL, 'optuna_xgboost': FULL,
    'decision_tree': FULL, 'random_forest': FULL,
    'engine_only': list(SETS['evaluation']),
    'engine_ratings': list(SETS['evaluation_ratings']),
}
CONSTANTS = ('no_change', 'training_mean', 'training_median')
TREE = dict(max_depth=5, min_samples_leaf=90)
FOREST = dict(n_estimators=150, min_samples_leaf=40, max_features=.8)
CUTOFFS = tuple(np.round(np.arange(0, 1.001, .01), 2))
REGRET_BINS = (-np.inf, 0, 2, 5, 10, 20, np.inf)
REGRET_LABELS = ('<0', '0–<2', '2–<5', '5–<10', '10–<20', '>=20')
LGBM_SPACE = dict(n_estimators=[120, 240, 400], learning_rate=[.02, .04, .07],
                  num_leaves=[7, 11, 15, 31], min_child_samples=[40, 90, 160],
                  reg_lambda=[1, 7, 15])
XGB_SPACE = dict(n_estimators=[120, 240, 400], learning_rate=[.02, .04, .07],
                 max_depth=[2, 3, 4, 5], min_child_weight=[3, 5, 10, 20],
                 reg_lambda=[1, 7, 15])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()
                              if isinstance(x, np.generic) else str(x)) + '\n')


def purge(frame, train, valid):
    held = players(frame.iloc[valid])
    candidate = frame.iloc[train]
    bad = (candidate.white.astype(str).str.strip().str.casefold().isin(held) |
           candidate.black.astype(str).str.strip().str.casefold().isin(held))
    bad_games = set(candidate.loc[bad, 'site'])
    train = train[~frame.iloc[train].site.isin(bad_games).to_numpy()]
    if not len(train) or not players(frame.iloc[train]).isdisjoint(held):
        raise ValueError('Player purge failed.')
    return train


def verify_fold(frame, train, valid, player):
    if not len(train) or not len(valid) or len(np.intersect1d(train, valid)):
        raise ValueError('Empty or overlapping fold.')
    if not set(frame.iloc[train].site).isdisjoint(frame.iloc[valid].site):
        raise ValueError('Game leakage.')
    if player and not players(frame.iloc[train]).isdisjoint(players(frame.iloc[valid])):
        raise ValueError('Player leakage.')
    return dict(train_rows=len(train), valid_rows=len(valid),
                train_games=frame.iloc[train].site.nunique(),
                valid_games=frame.iloc[valid].site.nunique(),
                game_overlap=0, player_overlap=0 if player else None)


def reuse_outer_map(frame, protocol):
    source = ROOT / f'notebook_runs/recovery_classifier_nested_{protocol}_v1'
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['data_sha256'] != digest(DATA) or manifest['seed'] != SEED:
        raise ValueError('Saved fold provenance changed.')
    saved = pd.read_csv(source / 'outer_predictions.csv',
                        usecols=['row_index', 'game', 'outer_fold']).drop_duplicates()
    if len(saved) != len(frame) or saved.row_index.duplicated().any():
        raise ValueError('Saved outer folds do not cover development exactly once.')
    saved = saved.set_index('row_index').reindex(frame.index)
    if saved.isna().any().any() or not np.array_equal(saved.game, frame.site):
        raise ValueError('Saved row identities do not match development.')
    mapping = saved.outer_fold.to_numpy(int)
    if set(mapping) != {1, 2, 3}:
        raise ValueError('Unexpected outer fold IDs.')
    if saved.groupby('game').outer_fold.nunique().gt(1).any():
        raise ValueError('Saved outer folds split a game.')
    # Independently reconstruct the historical protocol and verify its assignments.
    for fold, (train, valid) in enumerate(checked_folds(
            frame, frame.site, 3, SEED + 1, protocol == 'player'), 1):
        if not np.array_equal(np.flatnonzero(mapping == fold), valid):
            raise ValueError('Historical fold reconstruction differs from saved rows.')
        verify_fold(frame, train, valid, protocol == 'player')
    return mapping, source


def load_inputs():
    frame, _ = checked_development(DATA)
    validate_features(frame, 'full_multipv')
    checked_six_ply_label(frame)
    for h in HORIZONS:
        y = frame[f'change_in_damage_{h}ply'].to_numpy(float)
        if (not np.isfinite(y).all() or
                not np.allclose(y, frame.win_after - frame[f'win_after_{h}ply'], atol=1e-9) or
                not np.allclose(frame[f'persistence_ratio_{h}ply'],
                                1 + y / frame.initial_damage, atol=1e-9)):
            raise ValueError(f'Target formula failed at {h} plies.')
    joined, reply_manifest = load_validated(DATA, REPLIES)
    if reply_manifest['smoke_only'] or len(joined) != len(frame):
        raise ValueError('Full canonical development reply extraction required.')
    keys = ['source_pgn', 'game_index', 'ply']
    extra = joined[keys + ['status'] + list(REPLY_TARGETS)]
    frame = frame.merge(extra, on=keys, how='left', validate='one_to_one', sort=False)
    if not frame.status.eq('ok').sum() == 23023:
        raise ValueError('Reply eligibility differs from validated extraction.')
    for features in MODEL_FEATURES.values():
        if not set(features).issubset(FULL) or any(
                x in features for x in list(REPLY_TARGETS) + ['status', 'result', 'site']):
            raise ValueError('Automated predictor leakage check failed.')
    return frame, reply_manifest


def task_specs():
    return ([('regression', str(h), f'change_in_damage_{h}ply') for h in HORIZONS] +
            [('classification', '6', 'recovered_6ply')] +
            [('reply', target, target) for target in REPLY_TARGETS])


def make_protocol(frame, reply_manifest):
    base = json.loads((ROOT / 'config.json').read_text())['base_params']
    sources = [DATA, REPLIES, REPLIES.parent / 'manifest.json', ROOT / 'config.json',
               ROOT / 'my_tuning/selected.json', ROOT / 'chess_hyperparameters.ipynb',
               ROOT / 'AGENTS.md', Path(__file__).resolve(), ROOT / 'tune_chess.py',
               ROOT / 'scripts/tune_recovery_threshold.py',
               ROOT / 'scripts/evaluate_recovery_classifier.py',
               ROOT / 'scripts/analyze_opponent_reply_targets.py',
               ROOT / 'scripts/extract_opponent_reply_targets.py']
    maps, fold_rows, audits = {}, [], []
    for protocol in ('game', 'player'):
        mapping, source = reuse_outer_map(frame, protocol)
        maps[protocol] = mapping
        sources += [source / 'manifest.json', source / 'outer_predictions.csv']
        fold_rows.append(frame[['source_pgn', 'game_index', 'ply', 'site']].assign(
            row_index=frame.index, protocol=protocol, outer_fold=mapping))
        for fold in (1, 2, 3):
            valid = np.flatnonzero(mapping == fold)
            train = np.flatnonzero(mapping != fold)
            if protocol == 'player':
                train = purge(frame, train, valid)
            for scope in ('all', 'reply'):
                tr, va = train.copy(), valid.copy()
                if scope == 'reply':
                    eligible = frame.status.eq('ok').to_numpy()
                    tr, va = tr[eligible[tr]], va[eligible[va]]
                audit = verify_fold(frame, tr, va, protocol == 'player')
                audits.append(dict(protocol=protocol, scope=scope, outer_fold=fold,
                                   level='outer', inner_fold=0, **audit))
                sub = frame.iloc[tr]
                inner = checked_folds(sub, sub.site, 3, SEED, protocol == 'player')
                seen = np.zeros(len(sub), int)
                for k, (it, iv) in enumerate(inner, 1):
                    seen[iv] += 1
                    audit = verify_fold(sub, it, iv, protocol == 'player')
                    audits.append(dict(protocol=protocol, scope=scope, outer_fold=fold,
                                       level='inner', inner_fold=k, **audit))
                if not (seen == 1).all():
                    raise ValueError('Inner validation coverage is not exactly one.')
    if not np.array_equal(maps['game'], maps['player']):
        raise ValueError('Player and game protocols must hold out the same games.')
    metadata = dict(
        kind='Precommitted final Optuna comparison: development only',
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_archives=sorted(EXPECTED_ARCHIVES), rows=len(frame), games=frame.site.nunique(),
        input_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        versions={x: importlib.metadata.version(x) for x in
                  ('numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost',
                   'optuna', 'matplotlib', 'chess')},
        python=sys.version, seed=SEED, jobs=JOBS, outer_folds=3, inner_folds=3,
        optuna_trials_per_algorithm_per_outer_fold_per_target=TRIALS,
        sampler='Seeded sequential TPE; 5 startup trials; no pruning or early stopping',
        search_spaces={'optuna_lightgbm': LGBM_SPACE, 'optuna_xgboost': XGB_SPACE},
        fixed_params=base, decision_tree_params=TREE, random_forest_params=FOREST,
        predictors_by_model={**MODEL_FEATURES, **{x: [] for x in CONSTANTS},
                             'training_majority': []},
        feature_selection='None. Full allowlist fixed before fitting.',
        preprocessing='Median imputation with missing indicators and phase one-hot encoding; '
                      'fit separately inside every training fold, cached across trials only.',
        regression_selection='Minimize pooled inner OOF MAE; L1 boosting objectives.',
        classification_selection='Six-ply label ratio <0.75, fixed exploratory definition. '
                                 'Maximize inner OOF average precision, then choose probability '
                                 'cutoff for balanced accuracy from selected inner OOF scores.',
        probability_cutoffs=list(CUTOFFS), default_probability_cutoff=.5,
        cutoff_tie_break='Nearest to .5, then lower cutoff',
        tasks=task_specs(), protocols=['game', 'player'], bootstrap_repeats=REPEATS,
        bootstrap='Paired games sampled with replacement; row-weighted pooled metrics; '
                  'percentile 95% intervals, conditional on fitted folds (no refitting).',
        robustness='Evaluate fixed OOF predictions at initial_damage >25 and >30; '
                   'do not retune or select models from these slices.',
        regret_bands=dict(edges=['-inf', 0, 2, 5, 10, 20, 'inf'], labels=REGRET_LABELS,
                          closed='left; negative values retained'),
        reply_eligible=int(frame.status.eq('ok').sum()),
        reply_exclusions=frame.loc[~frame.status.eq('ok'), 'status'].value_counts().to_dict(),
        reply_engine_manifest=reply_manifest,
        split_reuse='Canonical saved recovery-classifier outer assignments reused exactly for '
                    'both protocols. Reply tasks restrict those same folds to valid rows; '
                    'older reply-only folds used a different eligible game universe. '
                    'Training purge precedes reply filtering. Inner folds are recreated '
                    'within each eligible outer-training split using seed 1701.',
        fixed_reference='config.json base_params and full_multipv, matching notebook fixed '
                        'curve/threshold model. my_tuning/selected.json is preserved; its '
                        'horizon-specific predictor choices are not substituted.',
        limitations=['Development-only, exploratory; historical feature/design decisions used '
                     'these archives. Outer folds never guide choices in this run.',
                     'The historical primary engine version is unrecorded; preserve its '
                     'existing targets. Existing MultiPV extractor maps mates to 100000 cp, '
                     'whereas core/reply targets use 1000 cp; preserve existing features and '
                     'record the discrepancy without redesigning them here.',
                     'PGN results cannot identify abandonment/disconnection. Reply FEN '
                     'reconstruction omits prior repetition history.',
                     '12 trials per study are a finite search, not proof of optimal tuning.',
                     'Bootstrap intervals do not include training or hyperparameter-selection '
                     'uncertainty; many comparisons are unadjusted.'])
    return metadata, pd.concat(fold_rows, ignore_index=True), pd.DataFrame(audits)


def prepared_views(frame, train, valid, inner):
    result = {}
    for key in ('full_multipv', 'evaluation', 'evaluation_ratings'):
        features = list(SETS[key])
        if not set(features).issubset(FULL):
            raise ValueError('Predictor allowlist failure.')
        prep = pipeline(features, {}, SEED, JOBS).named_steps['prep']
        outer = (prep.fit_transform(frame.iloc[train][features]),
                 prep.transform(frame.iloc[valid][features]))
        views = []
        if key == 'full_multipv':
            sub = frame.iloc[train]
            for it, iv in inner:
                transformer = clone(prep)
                views.append((it, iv, transformer.fit_transform(sub.iloc[it][features]),
                              transformer.transform(sub.iloc[iv][features])))
        result[key] = (outer, views)
    return result


def estimator(name, classification, params):
    common = dict(random_state=SEED)
    if name == 'decision_tree':
        return (DecisionTreeClassifier if classification else DecisionTreeRegressor)(
            **TREE, **common)
    if name == 'random_forest':
        return (RandomForestClassifier if classification else RandomForestRegressor)(
            **FOREST, **common, n_jobs=JOBS)
    p = {k: v for k, v in params.items() if k not in ('objective', 'verbosity')}
    if name == 'optuna_xgboost':
        return (XGBClassifier if classification else XGBRegressor)(
            **p, **common, n_jobs=JOBS, tree_method='hist', verbosity=0,
            objective='binary:logistic' if classification else 'reg:absoluteerror',
            eval_metric='logloss' if classification else 'mae')
    return (LGBMClassifier if classification else LGBMRegressor)(
        **p, **common, n_jobs=JOBS, verbosity=-1,
        objective='binary' if classification else 'regression_l1')


def predict(model, x, classification):
    p = model.predict_proba(x)[:, 1] if classification else model.predict(x)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite prediction.')
    return p


def inner_predictions(name, params, target, views, classification):
    prediction = np.full(len(target), np.nan)
    for it, iv, xt, xv in views:
        model = estimator(name, classification, params)
        model.fit(xt, target[it])
        prediction[iv] = predict(model, xv, classification)
    if not np.isfinite(prediction).all():
        raise ValueError('Incomplete inner OOF predictions.')
    return prediction


def tune(name, target, views, classification, base, unit):
    space = LGBM_SPACE if name == 'optuna_lightgbm' else XGB_SPACE
    seed = {k: base[k] for k in space} if name == 'optuna_lightgbm' else dict(
        n_estimators=240, learning_rate=.04, max_depth=3, min_child_weight=10, reg_lambda=7)
    study = optuna.create_study(direction='maximize' if classification else 'minimize',
                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5),
                pruner=optuna.pruners.NopPruner())
    study.enqueue_trial(seed)
    scores = {}

    def objective(trial):
        params = {k: trial.suggest_categorical(k, choices) for k, choices in space.items()}
        p = inner_predictions(name, params, target, views, classification)
        scores[trial.number] = p
        per_fold = [float(average_precision_score(target[iv], p[iv]) if classification
                          else np.abs(target[iv] - p[iv]).mean()) for _, iv, _, _ in views]
        trial.set_user_attr('inner_fold_scores', per_fold)
        return float(average_precision_score(target, p) if classification
                     else np.abs(target - p).mean())

    study.optimize(objective, n_trials=TRIALS, n_jobs=1)
    history = study.trials_dataframe()
    history.to_csv(unit / f'{name}_trials.csv', index=False)
    if len(study.trials) != TRIALS or any(t.state != optuna.trial.TrialState.COMPLETE
                                        for t in study.trials):
        raise ValueError('Incomplete Optuna budget.')
    return study.best_params, scores[study.best_trial.number], study.best_value


def run_unit(frame, protocol, mapping, fold, task, target_id, column, base, unit):
    classification = task == 'classification'
    valid, train = np.flatnonzero(mapping == fold), np.flatnonzero(mapping != fold)
    if protocol == 'player':
        train = purge(frame, train, valid)
    if task == 'reply':
        eligible = frame.status.eq('ok').to_numpy()
        train, valid = train[eligible[train]], valid[eligible[valid]]
    verify_fold(frame, train, valid, protocol == 'player')
    sub = frame.iloc[train]
    inner = checked_folds(sub, sub.site, 3, SEED, protocol == 'player')
    inner_map = []
    for k, (it, iv) in enumerate(inner, 1):
        verify_fold(sub, it, iv, protocol == 'player')
        for role, positions in [('train', it), ('valid', iv)]:
            inner_map.append(pd.DataFrame(dict(row_index=frame.index[train[positions]],
                                               inner_fold=k, role=role)))
    pd.concat(inner_map).to_csv(unit / 'inner_assignments.csv.gz', index=False)
    views = prepared_views(frame, train, valid, inner)
    y = ((frame.persistence_ratio_6ply.to_numpy(float) < .75).astype(int)
         if classification else frame[column].to_numpy(float))
    yt = y[train]
    rows, selections, thresholds, inner_rows = [], [], [], []
    models = ['training_majority'] if classification else list(CONSTANTS)
    models += list(MODEL_FEATURES)
    for name in models:
        params, inner_score, chosen_cutoff = {}, None, np.nan
        if name in CONSTANTS or name == 'training_majority':
            value = {'no_change': 0., 'training_mean': float(yt.mean()),
                     'training_median': float(np.median(yt)),
                     'training_majority': float(yt.mean() >= .5)}[name]
            p = np.full(len(valid), value)
        else:
            features = MODEL_FEATURES[name]
            fs = ('evaluation' if name == 'engine_only' else
                  'evaluation_ratings' if name == 'engine_ratings' else 'full_multipv')
            (xt, xv), iviews = views[fs]
            params = base.copy()
            if name.startswith('optuna_'):
                params, inner_score, objective = tune(name, yt, iviews, classification, base, unit)
            elif classification:
                if fs != 'full_multipv':
                    iviews = []
                    for it, iv in inner:
                        prep = pipeline(features, {}, SEED, JOBS).named_steps['prep']
                        iviews.append((it, iv, prep.fit_transform(sub.iloc[it][features]),
                                       prep.transform(sub.iloc[iv][features])))
                inner_score = inner_predictions(name, params, yt, iviews, True)
                objective = average_precision_score(yt, inner_score)
            else:
                objective = np.nan
            if classification:
                chosen_cutoff, table = select_probability_cutoff(
                    yt, inner_score, CUTOFFS, 'balanced_accuracy')
                thresholds.append(table.assign(model=name))
                inner_rows.append(pd.DataFrame(dict(row_index=frame.index[train], model=name,
                                                     actual=yt, score=inner_score)))
            elif inner_score is not None:
                inner_rows.append(pd.DataFrame(dict(row_index=frame.index[train], model=name,
                                                     actual=yt, prediction=inner_score)))
            model = estimator(name, classification, params)
            model.fit(xt, yt)
            p = predict(model, xv, classification)
            effective = model.get_params()
            selections.append(dict(model=name, selected_params=json.dumps(effective, sort_keys=True),
                                   inner_objective=float(objective), cutoff=chosen_cutoff,
                                   features=json.dumps(features)))
        strategies = [('majority', .5)] if name == 'training_majority' else (
            [('default_0.5', .5), ('inner_balanced_accuracy', chosen_cutoff)]
            if classification else [('regression', np.nan)])
        for strategy, cutoff in strategies:
            rows.append(pd.DataFrame(dict(
                row_index=frame.index[valid], game=frame.iloc[valid].site.to_numpy(),
                protocol=protocol, outer_fold=fold, task=task, target=target_id,
                model=name, strategy=strategy, actual=y[valid], prediction=p,
                predicted_recovered=(p >= cutoff) if classification else False,
                decision_cutoff=cutoff, initial_damage=frame.iloc[valid].initial_damage.to_numpy(),
                majority_prediction=float(yt.mean() >= .5) if classification else np.nan)))
        print(f'{protocol} outer {fold} {task} {target_id}: {name} complete', flush=True)
    pd.concat(rows).to_csv(unit / 'outer_predictions.csv.gz', index=False)
    pd.DataFrame(selections).to_csv(unit / 'selections.csv', index=False)
    if thresholds:
        pd.concat(thresholds).to_csv(unit / 'inner_threshold_scores.csv', index=False)
    if inner_rows:
        pd.concat(inner_rows).to_csv(unit / 'selected_inner_predictions.csv.gz', index=False)
    write_json(unit / 'complete.json', dict(protocol=protocol, outer_fold=fold,
               task=task, target=target_id, train_rows=len(train), valid_rows=len(valid),
               completed_utc=datetime.now(timezone.utc).isoformat()))


def reg_metrics(part):
    y, p = part.actual.to_numpy(float), part.prediction.to_numpy(float)
    # Fold-specific constants are still baselines; pooled fold artifacts are not rankings.
    constant = part.model.iloc[0] in CONSTANTS
    corr = (float(spearmanr(y, p).statistic) if not constant and len(y) > 1
            and np.ptp(y) and np.ptp(p) else np.nan)
    recovered = (1 + y / part.initial_damage.to_numpy(float) < .75
                 if part.task.iloc[0] == 'regression' else np.zeros(len(part), bool))
    return dict(n=len(part), games=part.game.nunique(), mae=float(np.abs(y - p).mean()),
                spearman=corr, recovered_n=int(recovered.sum()),
                recovered_only_mae_diagnostic=float(np.abs(y[recovered] - p[recovered]).mean())
                if recovered.any() else np.nan)


def classification_metrics(part):
    y, score = part.actual.to_numpy(bool), part.prediction.to_numpy(float)
    result = confusion_metrics(y, part.predicted_recovered.to_numpy(bool))
    result.update(games=part.game.nunique(), recovery_prevalence=float(y.mean()),
                  ap_reference_prevalence=float(y.mean()),
                  majority_baseline_accuracy=float((y == part.majority_prediction.to_numpy(bool)).mean()),
                  roc_auc=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
                  average_precision=float(average_precision_score(y, score)))
    if part.model.iloc[0] == 'training_majority':
        result.update(roc_auc=np.nan, average_precision=np.nan)
    return result


def metric_tables(oof):
    summaries, folds, slices = [], [], []
    keys = ['protocol', 'task', 'target', 'model', 'strategy']
    for key, part in oof.groupby(keys, sort=True):
        tags = dict(zip(keys, key))
        metric = classification_metrics if tags['task'] == 'classification' else reg_metrics
        summaries.append(dict(**tags, **metric(part)))
        for fold, p in part.groupby('outer_fold'):
            folds.append(dict(**tags, outer_fold=fold, **metric(p)))
        for threshold in (25, 30):
            p = part[part.initial_damage > threshold]
            if len(p):
                slices.append(dict(**tags, initial_damage_threshold=threshold, **metric(p)))
    return pd.DataFrame(summaries), pd.DataFrame(folds), pd.DataFrame(slices)


def bootstrap_regression(oof):
    records = []
    reg = oof[oof.task != 'classification']
    for (protocol, task, target), part in reg.groupby(['protocol', 'task', 'target']):
        for threshold in (20, 25, 30):
            p = part[part.initial_damage > threshold]
            wide = p.pivot(index='row_index', columns='model', values='prediction').sort_index()
            reference = p.drop_duplicates('row_index').set_index('row_index').loc[wide.index]
            if wide.isna().any().any():
                raise ValueError('Models have different eligible outer rows.')
            errors = np.abs(wide.to_numpy() - reference.actual.to_numpy(float)[:, None])
            game_code, games = pd.factorize(reference.game, sort=True)
            sums = np.zeros((len(games), errors.shape[1]))
            sizes = np.bincount(game_code)
            np.add.at(sums, game_code, errors)
            rng = np.random.default_rng(SEED)
            draws = np.empty((REPEATS, errors.shape[1]))
            for start in range(0, REPEATS, 50):
                count = min(50, REPEATS - start)
                indices = rng.integers(0, len(games), (count, len(games)))
                draws[start:start+count] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
            point = errors.mean(axis=0)
            comparators = [x for x in ('no_change', 'training_mean', 'training_median',
                           'fixed_lightgbm', 'optuna_lightgbm', 'optuna_xgboost') if x in wide]
            for i, name in enumerate(wide.columns):
                low, high = np.quantile(draws[:, i], [.025, .975])
                records.append(dict(protocol=protocol, task=task, target=target, model=name,
                    comparator='', metric='mae', estimate=point[i], lower95=low, upper95=high,
                    initial_damage_threshold=threshold, n=len(wide), games=len(games)))
                for comparator in comparators:
                    j = wide.columns.get_loc(comparator)
                    gain = draws[:, j] - draws[:, i]
                    for metric, values, estimate in (
                        ('mae_improvement_pp', gain, point[j] - point[i]),
                        ('mae_improvement_percent', 100 * gain / draws[:, j],
                         100 * (point[j] - point[i]) / point[j])):
                        low, high = np.quantile(values, [.025, .975])
                        records.append(dict(protocol=protocol, task=task, target=target,
                            model=name, comparator=comparator, metric=metric, estimate=estimate,
                            lower95=low, upper95=high, initial_damage_threshold=threshold,
                            n=len(wide), games=len(games)))
        print(f'{protocol} {task} {target}: paired cluster intervals complete', flush=True)
    return pd.DataFrame(records)


def classification_intervals(oof):
    """Cluster intervals for decisions and paired accuracy/balanced-accuracy gains."""
    records = []
    for protocol, part in oof[oof.task == 'classification'].groupby('protocol'):
        blocks = {(name, strategy): block.sort_values('row_index') for (name, strategy), block
                  in part.groupby(['model', 'strategy'])}
        anchor = next(iter(blocks.values()))
        code, games = pd.factorize(anchor.game, sort=True)
        labels = anchor.actual.to_numpy(bool)
        counts = []
        for key, block in blocks.items():
            if not np.array_equal(block.row_index, anchor.row_index):
                raise ValueError('Unpaired classifier rows.')
            pred = block.predicted_recovered.to_numpy(bool)
            cells = np.column_stack([~labels & ~pred, ~labels & pred,
                                     labels & ~pred, labels & pred]).astype(int)
            aggregate = np.zeros((len(games), 4))
            np.add.at(aggregate, code, cells)
            counts.append(aggregate)
        cube = np.stack(counts, axis=1)
        draws = np.empty((REPEATS, len(blocks), 2))
        rng = np.random.default_rng(SEED)
        for k in range(REPEATS):
            c = cube[rng.integers(0, len(games), len(games))].sum(axis=0)
            tn, fp, fn, tp = c.T
            draws[k, :, 0] = (tn + tp) / c.sum(axis=1)
            draws[k, :, 1] = .5 * (tp / (tp + fn) + tn / (tn + fp))
        keys = list(blocks)
        for j, (name, strategy) in enumerate(keys):
            point = classification_metrics(blocks[(name, strategy)])
            for m, metric in enumerate(('accuracy', 'balanced_accuracy')):
                low, high = np.quantile(draws[:, j, m], [.025, .975])
                records.append(dict(protocol=protocol, model=name, strategy=strategy,
                    comparator='', metric=metric, estimate=point[metric], lower95=low, upper95=high))
                for comparator in [('training_majority', 'majority'), ('fixed_lightgbm', strategy)]:
                    if comparator not in blocks:
                        continue
                    k = keys.index(comparator)
                    delta = draws[:, j, m] - draws[:, k, m]
                    low, high = np.quantile(delta, [.025, .975])
                    other = classification_metrics(blocks[comparator])[metric]
                    records.append(dict(protocol=protocol, model=name, strategy=strategy,
                        comparator=comparator[0], metric=metric + '_gain',
                        estimate=point[metric] - other, lower95=low, upper95=high))
    return pd.DataFrame(records)


def additional_outputs(frame, oof, out):
    context = frame[['player_elo', 'time_control', 'game_phase']].copy()
    context['elo_band'] = pd.cut(context.player_elo, [-np.inf, 1600, 2000, np.inf],
                                right=False, labels=['<1600', '1600–<2000', '>=2000'])
    context['row_index'] = frame.index
    enriched = oof.merge(context[['row_index', 'elo_band', 'time_control', 'game_phase']],
                         on='row_index', validate='many_to_one')
    rows = []
    for field in ('elo_band', 'time_control', 'game_phase'):
        for keys, block in enriched.groupby(['protocol', 'task', 'target', 'model', 'strategy', field],
                                            observed=True, dropna=False):
            tags = dict(zip(['protocol', 'task', 'target', 'model', 'strategy', 'group'], keys))
            metric = classification_metrics if tags['task'] == 'classification' else reg_metrics
            rows.append(dict(**tags, group_type=field, **metric(block)))
    pd.DataFrame(rows).to_csv(out / 'elo_time_phase_diagnostics.csv', index=False)
    valid = frame[frame.status.eq('ok')].copy()
    valid['reply_regret_band'] = pd.cut(valid.opponent_reply_regret_pp, REGRET_BINS,
                                        right=False, labels=REGRET_LABELS)
    rows = []
    for label, block in valid.groupby('reply_regret_band', observed=False):
        persistent = block.persistence_ratio_6ply >= .75
        rows.append(dict(reply_regret_band=str(label), n=len(block), games=block.site.nunique(),
                         still_persistent_6ply=int(persistent.sum()),
                         persistent_share=float(persistent.mean())))
    pd.DataFrame(rows).to_csv(out / 'reply_regret_bands_persistence_6ply.csv', index=False)
    curve_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, protocol in zip(axes, ('game', 'player')):
        part = oof[(oof.task == 'classification') & (oof.protocol == protocol) &
                   (oof.strategy == 'default_0.5')]
        for name, block in part.groupby('model'):
            precision, recall, cutoffs = precision_recall_curve(block.actual, block.prediction)
            curve_rows.append(pd.DataFrame(dict(protocol=protocol, model=name,
                precision=precision, recall=recall, outer_score_cutoff=np.r_[cutoffs, np.nan])))
            if name in ('fixed_lightgbm', 'optuna_lightgbm', 'optuna_xgboost'):
                ap = average_precision_score(block.actual, block.prediction)
                ax.plot(recall, precision, label=f'{name} AP={ap:.3f}')
        prevalence = part.drop_duplicates('row_index').actual.mean()
        ax.axhline(prevalence, color='gray', linestyle='--', label=f'Prevalence {prevalence:.3f}')
        ax.set(title=f'{protocol}: development-only, 6-ply recovery',
               xlabel='Recovery recall', ylabel='Recovery precision', xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(out / 'precision_recall_curves.png', dpi=170)
    plt.close(fig)
    pd.concat(curve_rows).to_csv(out / 'precision_recall_curve_points.csv', index=False)


def summarize(out, frame):
    units = sorted((out / 'units').iterdir())
    if len(units) != 42 or not all((u / 'complete.json').exists() for u in units):
        raise ValueError('All 42 outer-fold/task units must finish before reporting.')
    oof = pd.concat([pd.read_csv(u / 'outer_predictions.csv.gz') for u in units], ignore_index=True)
    expected = dict(regression=len(frame), classification=len(frame), reply=23023)
    for key, block in oof.groupby(['protocol', 'task', 'target', 'model', 'strategy']):
        if len(block) != expected[key[1]] or block.row_index.duplicated().any():
            raise ValueError(f'Invalid outer prediction coverage: {key}')
    oof.to_csv(out / 'outer_predictions.csv.gz', index=False)
    summary, folds, robustness = metric_tables(oof)
    intervals = bootstrap_regression(oof)
    intervals.to_csv(out / 'paired_game_bootstrap_intervals.csv', index=False)
    classification_intervals(oof).to_csv(out / 'classification_game_bootstrap_intervals.csv', index=False)
    gain = intervals[(intervals.comparator == 'no_change') &
                     (intervals.initial_damage_threshold == 20)].pivot(
        index=['protocol', 'task', 'target', 'model'], columns='metric',
        values=['estimate', 'lower95', 'upper95'])
    gain.columns = ['_'.join(x) for x in gain.columns]
    summary = summary.merge(gain.reset_index(), how='left', on=['protocol', 'task', 'target', 'model'])
    summary.to_csv(out / 'compact_summary.csv', index=False)
    folds.to_csv(out / 'fold_metrics.csv', index=False)
    robustness.to_csv(out / 'robustness_gt25_gt30.csv', index=False)
    selections, history, thresholds = [], [], []
    for u in units:
        tags = json.loads((u / 'complete.json').read_text())
        tags = {k: tags[k] for k in ('protocol', 'outer_fold', 'task', 'target')}
        selections.append(pd.read_csv(u / 'selections.csv').assign(**tags))
        for name in ('optuna_lightgbm', 'optuna_xgboost'):
            history.append(pd.read_csv(u / f'{name}_trials.csv').assign(model=name, **tags))
        if (u / 'inner_threshold_scores.csv').exists():
            thresholds.append(pd.read_csv(u / 'inner_threshold_scores.csv').assign(**tags))
    pd.concat(selections).to_csv(out / 'chosen_hyperparameters_thresholds.csv', index=False)
    pd.concat(history).to_csv(out / 'optuna_trial_history.csv', index=False)
    pd.concat(thresholds).to_csv(out / 'inner_threshold_scores.csv', index=False)
    additional_outputs(frame, oof, out)
    write_json(out / 'completed.json', dict(status='complete', development_only=True,
                completed_utc=datetime.now(timezone.utc).isoformat(),
                units=len(units), trials=len(pd.concat(history)),
                source_files_unchanged=True, sealed_test_accessed=False))
    print('Completed development-only comparison:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    frame, reply_manifest = load_inputs()
    out = args.out.resolve()
    if out.exists():
        if not args.resume:
            raise FileExistsError('Use a new output directory or --resume the exact protocol.')
        metadata = json.loads((out / 'protocol.json').read_text())
        for source, expected in metadata['input_sha256'].items():
            if digest(ROOT / source) != expected:
                raise ValueError(f'Frozen input/code changed: {source}')
        if metadata['versions'] != {x: importlib.metadata.version(x) for x in metadata['versions']}:
            raise ValueError('Environment changed after protocol preparation.')
    else:
        metadata, assignments, audits = make_protocol(frame, reply_manifest)
        out.mkdir(parents=True, exist_ok=False)
        (out / 'units').mkdir()
        write_json(out / 'protocol.json', metadata)
        assignments.to_csv(out / 'outer_fold_assignments.csv', index=False)
        audits.to_csv(out / 'split_validation.csv', index=False)
        write_json(out / 'predictor_allowlists.json', metadata['predictors_by_model'])
        print('Prepared and verified protocol:', out, flush=True)
    if args.prepare_only:
        return
    if (out / 'completed.json').exists():
        print('Already complete; refusing to replace results.')
        return
    if not args.summarize_only:
        assignments = pd.read_csv(out / 'outer_fold_assignments.csv')
        for protocol in ('game', 'player'):
            mapping = assignments[assignments.protocol == protocol].sort_values('row_index').outer_fold.to_numpy()
            for fold in (1, 2, 3):
                for task, target, column in task_specs():
                    unit = out / 'units' / f'{protocol}_fold{fold}_{task}_{target}'
                    if (unit / 'complete.json').exists():
                        continue
                    unit.mkdir(exist_ok=True)
                    start = time.monotonic()
                    run_unit(frame, protocol, mapping, fold, task, target, column,
                             metadata['fixed_params'], unit)
                    print(f'Unit finished in {time.monotonic()-start:.1f}s', flush=True)
    for source, expected in metadata['input_sha256'].items():
        if digest(ROOT / source) != expected:
            raise ValueError(f'Input modified during run: {source}')
    summarize(out, frame)


if __name__ == '__main__':
    main()
