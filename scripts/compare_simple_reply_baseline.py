#!/usr/bin/env python3
"""Pre-reply damage/Elo controls with and without saved Maia probabilities.

Development only, original game folds and player-purged sensitivity check.
No Optuna, feature selection, engine evaluation, or cutoff selection.
"""
import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_bounded_maia_regression import base_frame, split, FOUR
from scripts.report_maia_reply_classification import scores
from scripts.tune_recovery_threshold import digest, players

SOURCE = ROOT/'notebook_runs/maia_reply_subgroups_development_20261118_v1'
CLASSIFICATION = ROOT/'notebook_runs/maia_reply_classification_development_20261117_v1'
MAIA = ROOT/'notebook_runs/maia_bounded_four_archive_20261117_v1'
FOLDS = ROOT/'notebook_runs/final_optuna_development_20261117_v1/outer_fold_assignments.csv'
DATA = ROOT/'development_with_multipv.csv'
FEATURE = 'maia_preserving_reply_probability'
KEYS = ['row_index', 'source_pgn', 'game_index', 'ply', 'site']
SIMPLE = ['initial_damage', 'opponent_elo']
ALLOWLISTS = {'simple_lightgbm': SIMPLE, 'simple_plus_maia_lightgbm': SIMPLE+[FEATURE],
             'simple_logistic': SIMPLE, 'simple_plus_maia_logistic': SIMPLE+[FEATURE],
             'raw_maia': [FEATURE], 'training_majority_prevalence': []}
CONTRASTS = [('simple_plus_maia_lightgbm', 'simple_lightgbm'),
             ('simple_plus_maia_logistic', 'simple_logistic'),
             ('raw_maia', 'simple_lightgbm'),
             ('simple_plus_maia_lightgbm', 'raw_maia'),
             ('simple_lightgbm', 'training_majority_prevalence')]
SUBGROUPS = ['reply_maker_elo', 'archive_speed_label', 'archive', 'game_phase', 'pre_reply_clock']
SEED, JOBS, REPEATS, CUTOFF = 1701, 2, 2000, .5


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x:
                    x.item() if isinstance(x, np.generic) else str(x))+'\n')


def check_predictors(allowlists):
    expected = {'simple_lightgbm': ['initial_damage', 'opponent_elo'],
        'simple_plus_maia_lightgbm': ['initial_damage', 'opponent_elo', FEATURE],
        'simple_logistic': ['initial_damage', 'opponent_elo'],
        'simple_plus_maia_logistic': ['initial_damage', 'opponent_elo', FEATURE],
        'raw_maia': [FEATURE], 'training_majority_prevalence': []}
    if allowlists != expected:
        raise ValueError('Predictor lists differ from the prespecified pre-reply controls.')


def load_frame():
    frame = base_frame()
    source = pd.read_csv(SOURCE/'classified_replies_with_subgroups.csv.gz',
        usecols=KEYS+['actual_preserved', FEATURE]+[s for s in SUBGROUPS if s != 'game_phase'])
    if len(source) != 15851 or source.row_index.duplicated().any() or set(source.source_pgn) != FOUR:
        raise ValueError('Source classification population changed.')
    before = frame.row_index.tolist()
    frame = frame.merge(source, on=KEYS, how='left', sort=False, validate='one_to_one')
    valid = frame.actual_preserved.notna()
    if frame.row_index.tolist() != before or valid.sum() != 15851 or len(frame) != 15953:
        raise ValueError('Different row universe or missing-reply eligibility.')
    if not frame.loc[valid, FEATURE].between(0, 1).all():
        raise ValueError('Missing/out-of-range saved Maia probability.')
    return frame


def prepare(out):
    if out.exists():
        raise FileExistsError('Use a new output directory.')
    check_predictors(ALLOWLISTS)
    source_protocol = json.loads((SOURCE/'protocol.json').read_text())
    if set(source_protocol['source_archives']) != FOUR or source_protocol['cutoff'] != CUTOFF:
        raise ValueError('Unexpected source scope or cutoff.')
    artifacts = json.loads((SOURCE/'artifact_manifest.json').read_text())
    for name in ['classified_replies_with_subgroups.csv.gz', 'protocol.json', 'validation.json']:
        if digest(SOURCE/name) != artifacts['artifact_sha256'][name]:
            raise ValueError('Saved probabilities/labels changed.')
    for name in ['development_with_multipv.csv', 'scripts/test_bounded_maia_regression.py',
                 'scripts/tune_recovery_threshold.py', 'tune_chess.py', 'config.json', 'my_tuning/selected.json']:
        old = json.loads((MAIA/'regression_protocol.json').read_text())
        if digest(ROOT/name) != old['source_sha256'][name]:
            raise ValueError(f'Frozen source changed: {name}')
    frame = load_frame()
    valid = frame.actual_preserved.notna().to_numpy()
    maps = pd.read_csv(FOLDS)
    checks, membership = [], []
    for mode in ['game', 'player']:
        for fold in [1, 2, 3]:
            tr0, va0 = split(frame, maps[maps.protocol.eq(mode)], fold, mode == 'player')
            train, held = tr0[valid[tr0]], va0[valid[va0]]
            checks.append(dict(protocol=mode, outer_fold=fold, train_rows=len(train), heldout_rows=len(held),
                train_games=frame.iloc[train].site.nunique(), heldout_games=frame.iloc[held].site.nunique(),
                game_overlap=0, player_overlap=0 if mode == 'player' else None))
            for role, index in [('train', train), ('validation', held)]:
                membership.append(frame.iloc[index][KEYS].assign(protocol=mode, outer_fold=fold, role=role))
    fixed = dict(json.loads((MAIA/'regression_protocol.json').read_text())['fixed_params'])
    fixed['objective'] = 'binary'
    paths = [DATA, FOLDS, SOURCE/'classified_replies_with_subgroups.csv.gz', SOURCE/'protocol.json',
        SOURCE/'artifact_manifest.json', SOURCE/'validation.json',
        CLASSIFICATION/'evaluation_predictions.csv.gz', CLASSIFICATION/'classification_metrics.csv',
        MAIA/'regression_protocol.json', ROOT/'config.json', ROOT/'my_tuning/selected.json',
        ROOT/'tune_chess.py', ROOT/'scripts/test_bounded_maia_regression.py',
        ROOT/'scripts/tune_recovery_threshold.py', ROOT/'scripts/report_maia_reply_classification.py',
        Path(__file__).resolve()]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_archives=sorted(FOUR),
        kind='Stronger simple pre-reply classification baseline; development only',
        label='Saved immediate remaining-damage ratio >=0.75, original mistake-maker perspective.',
        primary_probability_metric='Brier score', cutoff=CUTOFF,
        primary_contrast='simple_plus_maia_lightgbm versus simple_lightgbm',
        secondary_check='Same two-feature logistic regression with and without Maia.',
        allowlists=ALLOWLISTS, lightgbm_fixed_params=fixed,
        logistic_params=dict(C=1., solver='lbfgs', max_iter=2000, class_weight=None),
        seed=SEED, jobs=JOBS, bootstrap_repeats=REPEATS,
        preprocessing='Median imputer with missing indicators inside training folds. '
            'Logistic additionally uses training-fold StandardScaler. No feature selection, '
            'calibration, class weighting, threshold selection or hyperparameter search.',
        controls='Only initial_damage (pp) and opponent_elo (reply-maker rating). '
            'No actual reply, future horizon, game result, post-reply evaluation or identity is a predictor.',
        reference='Fixed shallow LightGBM classifier provides nonlinear effects and interaction '
            'of the two controls. Same fixed settings for the augmented classifier. '
            'Binary objective is used only in this separate experiment; frozen regressors remain unchanged.',
        folds='Reuse original three game-fold validation assignments, restricted to four earlier '
            'archives. Player check purges whole training games sharing held-out players before '
            'filtering the missing-label rows, as in the existing four-archive protocol.',
        eligibility='15,851 labeled replies /4,472 games for all methods. 102 missing replies excluded.',
        intervals='Paired percentile game-cluster bootstrap of fixed held-out predictions, 2000 '
            'draws, seed 1701. Positive Brier/log-loss improvement = baseline minus augmented; '
            'positive accuracy/balanced-accuracy gains = augmented minus baseline. '
            'Conditional on fitted models; no bootstrap refits or multiplicity adjustment.',
        robustness='>25 and >30 subsets of the same primary >20 predictions; no refits.',
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm']})
    out.mkdir(parents=True)
    save_json(out/'protocol.json', protocol)
    save_json(out/'predictor_allowlists.json', ALLOWLISTS)
    pd.DataFrame(checks).to_csv(out/'split_validation.csv', index=False)
    pd.concat(membership, ignore_index=True).to_csv(out/'fold_membership.csv.gz', index=False)
    frame.loc[valid, KEYS].to_csv(out/'eligible_rows.csv', index=False)
    print('Prepared fixed damage/Elo baselines, with and without Maia; no search.', flush=True)


def make_model(name, protocol):
    stages = [('imputer', SimpleImputer(strategy='median', add_indicator=True))]
    if name.endswith('logistic'):
        stages += [('scaler', StandardScaler()),
                   ('model', LogisticRegression(**protocol['logistic_params'], random_state=SEED))]
    elif name.endswith('lightgbm'):
        stages += [('model', LGBMClassifier(**protocol['lightgbm_fixed_params'], random_state=SEED, n_jobs=JOBS))]
    else:
        raise ValueError('Unknown learned model.')
    return Pipeline(stages)


def metric_rows(part):
    y = part.actual_preserved.to_numpy(int)
    probability = part.probability.to_numpy(float)
    result = scores(part, probability >= CUTOFF, probability)
    result.update(brier=float(brier_score_loss(y, probability)),
                  log_loss=float(log_loss(y, probability, labels=[0, 1])))
    result['non_preserving_recall'] = result['tn']/(result['tn']+result['fp'])
    return result


def values_from_totals(totals):
    """Last axis per method: squared-error sum, log-loss sum, TN,FP,FN,TP."""
    n = totals[..., 2:].sum(axis=-1)
    preserved = totals[..., 5]/(totals[..., 4]+totals[..., 5])
    other = totals[..., 2]/(totals[..., 2]+totals[..., 3])
    return np.stack([totals[..., 0]/n, totals[..., 1]/n,
        100*(totals[..., 2]+totals[..., 5])/n,
        50*(preserved+other), 100*preserved, 100*other], axis=-1)


METRICS = ['brier_improvement', 'log_loss_improvement', 'accuracy_gain_pp',
           'balanced_accuracy_gain_pp', 'preserving_recall_gain_pp', 'non_preserving_recall_gain_pp']


def cluster_intervals(part, repeats=REPEATS, seed=SEED, verify_expanded=False):
    if part.duplicated(['method', 'row_index']).any():
        raise ValueError('Duplicate OOF predictions.')
    wide = part.pivot(index='row_index', columns='method', values='probability').sort_index()
    reference = part.drop_duplicates('row_index').set_index('row_index').loc[wide.index]
    for _, block in part.groupby('method'):
        aligned = block.set_index('row_index').sort_index()
        if not aligned[['site', 'outer_fold', 'actual_preserved']].equals(
                reference[['site', 'outer_fold', 'actual_preserved']]):
            raise ValueError('Different rows, games, folds or labels across models.')
    if wide.isna().any().any():
        raise ValueError('Incomplete matched probabilities.')
    y = reference.actual_preserved.to_numpy(int)[:, None]
    probabilities = wide.to_numpy(float)
    prediction = probabilities >= CUTOFF
    clipped = np.clip(probabilities, 1e-15, 1-1e-15)
    observations = np.stack([(probabilities-y)**2, -(y*np.log(clipped)+(1-y)*np.log1p(-clipped)),
        (y == 0)&~prediction, (y == 0)&prediction,
        (y == 1)&~prediction, (y == 1)&prediction], axis=-1).astype(float)
    codes, games = pd.factorize(reference.site, sort=True)
    totals = np.zeros((len(games), probabilities.shape[1], observations.shape[-1]))
    np.add.at(totals, codes, observations)
    point = values_from_totals(totals.sum(axis=0))
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, probabilities.shape[1], len(METRICS)))
    for start in range(0, repeats, 50):
        n = min(50, repeats-start)
        index = rng.integers(0, len(games), size=(n, len(games)))
        draws[start:start+n] = values_from_totals(totals[index].sum(axis=1))
        if start == 0 and verify_expanded:
            for i in range(5):
                multiplicity = np.bincount(index[i], minlength=len(games))
                positions = np.repeat(np.arange(len(codes)), multiplicity[codes])
                expected = values_from_totals(observations[positions].sum(axis=0))
                np.testing.assert_allclose(draws[i], expected, atol=1e-10)
    rows, saved = [], []
    signs = np.array([-1, -1, 1, 1, 1, 1])
    for augmented, baseline in CONTRASTS:
        a, b = wide.columns.get_loc(augmented), wide.columns.get_loc(baseline)
        gains = (draws[:, a]-draws[:, b])*signs
        estimates = (point[a]-point[b])*signs
        for i, metric in enumerate(METRICS):
            lo, hi = np.quantile(gains[:, i], [.025, .975])
            rows.append(dict(augmented=augmented, baseline=baseline, metric=metric,
                estimate=estimates[i], lower95=lo, upper95=hi, n=len(wide), games=len(games),
                bootstrap_repeats=repeats))
            if augmented == 'simple_plus_maia_lightgbm' and baseline == 'simple_lightgbm':
                saved.append(pd.DataFrame(dict(replicate=np.arange(repeats)+1, metric=metric, gain=gains[:, i])))
    return rows, pd.concat(saved, ignore_index=True)


def summarize(out, frame, oof):
    summary, folds, robust, intervals, primary_draws = [], [], [], [], []
    eligible = set(frame.loc[frame.actual_preserved.notna(), 'row_index'])
    for (mode, method), part in oof.groupby(['protocol', 'method']):
        if part.row_index.duplicated().any() or set(part.row_index) != eligible:
            raise ValueError('Different OOF row coverage.')
        summary.append(dict(protocol=mode, method=method, **metric_rows(part)))
        for fold, block in part.groupby('outer_fold'):
            folds.append(dict(protocol=mode, method=method, outer_fold=fold, **metric_rows(block)))
        for threshold in [25, 30]:
            robust.append(dict(protocol=mode, method=method, initial_damage_threshold=threshold,
                               **metric_rows(part[part.initial_damage > threshold])))
    for mode, part in oof.groupby('protocol'):
        for threshold in [20, 25, 30]:
            rows, draws = cluster_intervals(part[part.initial_damage > threshold],
                                            verify_expanded=threshold == 20)
            intervals.extend(dict(protocol=mode, initial_damage_threshold=threshold, **row) for row in rows)
            if threshold == 20:
                primary_draws.append(draws.assign(protocol=mode))
    summary, ci = pd.DataFrame(summary), pd.DataFrame(intervals)
    summary.to_csv(out/'classification_metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(robust).to_csv(out/'robustness_metrics.csv', index=False)
    ci.to_csv(out/'paired_game_bootstrap_intervals.csv', index=False)
    primary = ci[ci.augmented.eq('simple_plus_maia_lightgbm')&ci.baseline.eq('simple_lightgbm')]
    primary[primary.initial_damage_threshold.eq(20)].to_csv(out/'primary_incremental_value.csv', index=False)
    pd.concat(primary_draws, ignore_index=True).to_csv(out/'primary_bootstrap_replicates.csv.gz', index=False)
    metadata = frame[['row_index']+SUBGROUPS]
    joined = oof.merge(metadata, on='row_index', validate='many_to_one')
    subgroup = []
    for field in SUBGROUPS:
        for key, part in joined.groupby(['protocol', 'method', field], observed=True, dropna=False):
            subgroup.append(dict(protocol=key[0], method=key[1], subgroup_type=field,
                                 subgroup=str(key[2]), **metric_rows(part)))
    pd.DataFrame(subgroup).to_csv(out/'subgroup_metrics.csv', index=False)
    return summary, ci


def validate(out, frame, oof, protocol):
    membership = pd.read_csv(out/'fold_membership.csv.gz')
    indexed = frame.set_index('row_index')
    for (mode, fold), part in membership.groupby(['protocol', 'outer_fold']):
        train = indexed.loc[part[part.role.eq('train')].row_index]
        held = indexed.loc[part[part.role.eq('validation')].row_index]
        if not set(train.site).isdisjoint(held.site):
            raise ValueError('Game leakage.')
        if mode == 'player' and not players(train).isdisjoint(players(held)):
            raise ValueError('Player leakage.')
        for method, predictions in oof[oof.protocol.eq(mode)&oof.outer_fold.eq(fold)].groupby('method'):
            if set(predictions.row_index) != set(held.index):
                raise ValueError('Different held-out rows.')
            np.testing.assert_array_equal(predictions.actual_preserved, held.loc[predictions.row_index, 'actual_preserved'])
            if method == 'training_majority_prevalence':
                np.testing.assert_allclose(predictions.probability, train.actual_preserved.astype(int).mean())
            if method == 'raw_maia':
                np.testing.assert_allclose(predictions.probability, held.loc[predictions.row_index, FEATURE], atol=1e-15)
    old = pd.read_csv(CLASSIFICATION/'evaluation_predictions.csv.gz')
    raw = oof[oof.method.eq('raw_maia')].merge(old, on=KEYS+['protocol', 'outer_fold'],
        validate='one_to_one', suffixes=('_new', '_old'))
    if len(raw) != 2*15851:
        raise ValueError('Raw reference row coverage changed.')
    np.testing.assert_array_equal(raw.actual_preserved_new, raw.actual_preserved_old)
    np.testing.assert_allclose(raw.probability, raw[FEATURE], atol=1e-15)
    draws = pd.read_csv(out/'primary_bootstrap_replicates.csv.gz')
    primary = pd.read_csv(out/'primary_incremental_value.csv')
    for row in primary.itertuples(index=False):
        values = draws[draws.protocol.eq(row.protocol)&draws.metric.eq(row.metric)].gain
        np.testing.assert_allclose([row.lower95, row.upper95], values.quantile([.025, .975]), atol=1e-10)
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Frozen source modified: {name}')
    save_json(out/'validation.json', dict(status='passed', rows=15851, games=4472,
        all_models_same_rows_folds_and_labels=True, game_and_player_disjointness_verified=True,
        raw_maia_reference_reproduced=True, training_only_majority_verified=True,
        primary_bootstrap_draws_independently_checked_by_raw_row_expansion=True,
        primary_interval_quantiles_verified=True, exact_pre_reply_predictor_lists_verified=True,
        original_source_hashes_unchanged=True, new_engine_calls=0, new_optuna_trials=0))


def report(out, summary, ci):
    compact = summary[['protocol', 'method', 'n', 'games', 'accuracy', 'balanced_accuracy',
        'precision', 'recall', 'non_preserving_recall', 'brier', 'roc_auc', 'average_precision',
        'tn', 'fp', 'fn', 'tp']].copy()
    compact.to_csv(out/'compact_summary.csv', index=False)
    order = ['training_majority_prevalence', 'simple_lightgbm', 'raw_maia',
             'simple_plus_maia_lightgbm', 'simple_logistic', 'simple_plus_maia_logistic']
    lines = ['# Stronger pre-reply baseline and incremental Maia value — development only', '',
        'Target: the actual immediate opponent reply preserves at least 75% of the initial '
        'mistake damage, from the original mistake-maker perspective. This is the existing '
        'project-specific event, not a prediction of 10-ply persistence or of whether the '
        'opponent noticed a mistake.', '',
        '**15,851 valid replies /4,472 games**, from April, June and July 2017 and February '
        '2018. Every method uses the identical held-out rows and the original game folds. '
        'The fixed **0.5 cutoff** is used for every classifier. The 102 missing actual replies '
        'remain excluded rather than labeled as negative.', '',
        '## Comparison', '',
        '| Protocol | Model | Accuracy | Balanced accuracy | Precision | Preserving recall | Non-preserving recall | Brier | ROC-AUC |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode in ['game', 'player']:
        for method in order:
            row = summary[summary.protocol.eq(mode)&summary.method.eq(method)].iloc[0]
            lines.append(f'| {mode} | {method} | {100*row.accuracy:.2f}% | {100*row.balanced_accuracy:.2f}% | '
                f'{100*row.precision:.2f}% | {100*row.recall:.2f}% | {100*row.non_preserving_recall:.2f}% | '
                f'{row.brier:.6f} | {row.roc_auc:.4f} |')
    lines += ['', '## Does Maia add value beyond damage and Elo?', '',
        'Primary contrast: **the same fixed LightGBM classifier with initial_damage and '
        'opponent_elo, versus those same two inputs plus the saved Maia probability**. '
        'Primary probability metric is Brier score; accuracy and balanced accuracy use 0.5. '
        'Positive improvements/gains favor adding Maia.', '',
        '| Protocol | Metric | Improvement | Paired game-cluster 95% interval |',
        '| --- | --- | ---: | --- |']
    interpretations = []
    for mode in ['game', 'player']:
        for metric in ['brier_improvement', 'accuracy_gain_pp', 'balanced_accuracy_gain_pp']:
            row = ci[ci.protocol.eq(mode)&ci.initial_damage_threshold.eq(20)&
                ci.augmented.eq('simple_plus_maia_lightgbm')&ci.baseline.eq('simple_lightgbm')&ci.metric.eq(metric)].iloc[0]
            lines.append(f'| {mode} | {metric} | {row.estimate:.6f} | [{row.lower95:.6f}, {row.upper95:.6f}] |')
            if metric == 'brier_improvement':
                state = ('supports incremental probability-prediction value' if row.lower95 > 0 else
                    'favors the simple baseline' if row.upper95 < 0 else 'includes zero; incremental value is uncertain')
                interpretations.append(f'The {mode} Brier interval {state}.')
    lines += ['', ' '.join(interpretations), '',
        'Logistic regression with the same two controls, and its three-input augmented version, '
        'are transparent sensitivity checks. Their paired intervals and all raw Maia-versus-control '
        'contrasts are exported. No learner or cutoff was selected from these outer results.', '',
        '## Design and leakage checks', '',
        '- Simple predictors: **initial_damage** in winning-chance percentage points and '
        '**opponent_elo**, the reply-maker rating. Both are known before the opponent replies.',
        '- Augmented predictors: exactly those same inputs plus '
        '**maia_preserving_reply_probability**, the unchanged saved midpoint. No observed reply '
        'identity, actual reply evaluation, future move, horizon label or result is a predictor.',
        '- The LightGBM control captures nonlinear damage/Elo effects and interactions with fixed '
        'settings: binary objective, 240 trees, learning rate .04, 11 leaves, minimum child '
        'size 90, L2 penalty 7, seed 1701, two threads. Identical settings apply after adding Maia.',
        '- Logistic regression uses C=1, lbfgs, maximum 2000 iterations, and no class weighting. '
        'Its standardization and every model’s median imputation/missing indicators fit within '
        'each training fold. No feature selection, recalibration or hyperparameter search.',
        '- Existing game folds are retained. The player check removes whole training games '
        'involving any held-out player, before filtering missing labels. Game and player overlap '
        'checks, memberships and preprocessing statistics are saved.',
        '- Training-majority/prevalence probabilities use training labels only. Raw Maia '
        'probabilities reproduce the preceding report. The external Maia policy itself is not '
        'trained on these local folds; its training membership is not independently audited.', '',
        '## Interpretation limits and files', '',
        'This directly tests incremental prediction beyond these **two specified controls**. '
        'It does not prove that Maia adds information beyond every possible pre-reply feature, '
        'or establish a causal effect. It also separates the incremental-value question '
        'from the weaker comparison against majority prediction.', '',
        'Intervals use 2,000 paired resamples of whole games, seed 1701, retaining all mistakes '
        'within sampled games. They condition on fitted OOF predictions without bootstrap '
        'refitting and are unadjusted for repeated development comparisons. The first five '
        'draws were independently recomputed from expanded raw rows, and saved draw quantiles '
        'reproduce primary interval endpoints. These remain exploratory development results.', '',
        'The >25/>30 initial-damage checks use subsets of the primary >20 OOF predictions, '
        'without refits. Rating, speed, archive, phase and clock subgroup metrics are exported. '
        'No subgroup-specific cutoff or model was selected.', '',
        '- compact_summary.csv and classification_metrics.csv: metrics and raw confusion counts.',
        '- primary_incremental_value.csv: requested paired increments beyond the nonlinear control.',
        '- paired_game_bootstrap_intervals.csv: all prespecified contrasts and robustness intervals.',
        '- outer_predictions.csv.gz and fold_predictions/: all matched OOF probabilities.',
        '- preprocessing_by_fold.json: learned preprocessing and logistic coefficients.',
        '- protocol.json, predictor_allowlists.json, fold_membership.csv.gz, validation.json: provenance and checks.',
        '- COMMANDS.md: exact reproduction commands.', '',
        '**Development-only. No March 2025 access, new engine evaluations, policy inference or '
        'Optuna. Original frozen models/configurations and old results are unchanged.**', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def run(out):
    protocol = json.loads((out/'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Source changed: {name}')
    if (out/'completed.json').exists():
        print('Already complete; no refitting.')
        return
    check_predictors(protocol['allowlists'])
    frame = load_frame()
    valid = frame.actual_preserved.notna().to_numpy()
    maps = pd.read_csv(FOLDS)
    folder = out/'fold_predictions'; folder.mkdir(exist_ok=True)
    prep_path = out/'preprocessing_by_fold.json'
    preprocessing = json.loads(prep_path.read_text()) if prep_path.exists() else []
    for mode in ['game', 'player']:
        for fold in [1, 2, 3]:
            path = folder/f'{mode}_fold{fold}.csv'
            if path.exists():
                continue
            tr0, va0 = split(frame, maps[maps.protocol.eq(mode)], fold, mode == 'player')
            train, held = tr0[valid[tr0]], va0[valid[va0]]
            y = frame.iloc[train].actual_preserved.astype(int)
            predictions = {}
            for name in ['simple_lightgbm', 'simple_plus_maia_lightgbm', 'simple_logistic', 'simple_plus_maia_logistic']:
                columns = ALLOWLISTS[name]
                model = make_model(name, protocol)
                model.fit(frame.iloc[train][columns], y)
                predictions[name] = model.predict_proba(frame.iloc[held][columns])[:, 1]
                stats = dict(protocol=mode, outer_fold=fold, method=name, predictors=columns,
                    train_rows=len(train), imputer_statistics=model.named_steps['imputer'].statistics_.tolist(),
                    missing_indicator_columns=model.named_steps['imputer'].indicator_.features_.tolist())
                if name.endswith('logistic'):
                    stats.update(scaler_mean=model.named_steps['scaler'].mean_.tolist(),
                        scaler_scale=model.named_steps['scaler'].scale_.tolist(),
                        coefficients=model.named_steps['model'].coef_.tolist(),
                        intercept=model.named_steps['model'].intercept_.tolist())
                preprocessing.append(stats)
            predictions['raw_maia'] = frame.iloc[held][FEATURE].to_numpy()
            predictions['training_majority_prevalence'] = np.full(len(held), y.mean())
            blocks = []
            for method, probability in predictions.items():
                if not np.isfinite(probability).all() or ((probability < 0)|(probability > 1)).any():
                    raise ValueError('Invalid prediction.')
                part = frame.iloc[held][KEYS+['initial_damage']].copy()
                part['actual_preserved'] = frame.iloc[held].actual_preserved.astype(bool).to_numpy()
                part['protocol'], part['outer_fold'], part['method'] = mode, fold, method
                part['probability'] = probability
                part['predicted_preserved'] = probability >= CUTOFF
                blocks.append(part)
            pd.concat(blocks, ignore_index=True).to_csv(path, index=False)
            save_json(prep_path, preprocessing)
            print(f'{mode} fold {fold}: two controls, their Maia additions and raw references complete.', flush=True)
    oof = pd.concat([pd.read_csv(path) for path in sorted(folder.glob('*.csv'))], ignore_index=True)
    oof.to_csv(out/'outer_predictions.csv.gz', index=False)
    summary, ci = summarize(out, frame, oof)
    validate(out, frame, oof, protocol)
    report(out, summary, ci)
    save_json(out/'completed.json', dict(status='complete', primary_contrast=protocol['primary_contrast'],
        new_engine_calls=0, new_optuna_trials=0, completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Stronger-baseline comparison and independent checks complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.out.resolve()
    if not output.exists():
        prepare(output)
    elif not args.resume:
        raise FileExistsError('Use a new directory or --resume.')
    if not args.prepare_only:
        run(output)
