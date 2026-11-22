#!/usr/bin/env python3
"""Nested development-only comparison of six-ply recovery classifiers.

Inner game-grouped folds select LightGBM classifier hyperparameters by average
precision, then select probability cutoffs by accuracy and balanced accuracy.
The same outer folds evaluate the classifier, fixed/tuned regression decisions,
and a training-majority baseline. No class weights are used.

Example:
  python3 scripts/evaluate_recovery_classifier.py --out notebook_runs/classifier_game_run
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tune_chess import EXPECTED_ARCHIVES, SETS, pipeline  # noqa: E402
from scripts.tune_recovery_threshold import (  # noqa: E402
    DEFAULT_CUTOFFS, LABEL_CUTOFF, SAFE_PREDICTORS, checked_development,
    checked_folds, digest, nested_thresholds,
)

CLASSIFIER_CANDIDATES = (
    dict(n_estimators=120, learning_rate=0.04, num_leaves=7, min_child_samples=160, reg_lambda=7),
    dict(n_estimators=240, learning_rate=0.04, num_leaves=11, min_child_samples=90, reg_lambda=7),
    dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=90, reg_lambda=7),
    dict(n_estimators=240, learning_rate=0.04, num_leaves=15, min_child_samples=40, reg_lambda=1),
)
PROBABILITY_CUTOFFS = tuple(round(x / 100, 2) for x in range(15, 81, 5))
STRATEGIES = (
    'training_majority', 'regression_fixed_0.75', 'regression_tuned_accuracy',
    'regression_tuned_balanced_accuracy', 'classifier_0.5',
    'classifier_tuned_accuracy', 'classifier_tuned_balanced_accuracy',
)
COMPARISONS = (
    ('classifier_0.5', 'regression_fixed_0.75'),
    ('classifier_tuned_accuracy', 'regression_tuned_accuracy'),
    ('classifier_tuned_balanced_accuracy', 'regression_tuned_balanced_accuracy'),
    ('classifier_tuned_balanced_accuracy', 'training_majority'),
)
BOOTSTRAP_METRICS = (
    'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy',
    'recovered_precision', 'recovered_recall', 'recovered_f1',
    'specificity', 'false_positive_rate',
)


def checked_six_ply_label(frame):
    required = {'persistence_ratio_6ply', 'change_in_damage_6ply', 'initial_damage'}
    if required - set(frame.columns):
        raise ValueError(f'Missing label-construction columns: {required - set(frame.columns)}')
    ratio = pd.to_numeric(frame['persistence_ratio_6ply'], errors='raise').to_numpy(dtype=float)
    change = pd.to_numeric(frame['change_in_damage_6ply'], errors='raise').to_numpy(dtype=float)
    initial = pd.to_numeric(frame['initial_damage'], errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(ratio).all() or not np.isfinite(change).all()
            or not np.isfinite(initial).all() or (initial <= 20).any()
            or not np.allclose(ratio, 1 + change / initial, atol=1e-9)):
        raise ValueError('Six-ply persistence ratio disagrees with the regression target.')
    return ratio < LABEL_CUTOFF


def validate_features(frame, feature_set):
    if feature_set not in SETS:
        raise ValueError(f'Unknown feature set: {feature_set}')
    features = list(SETS[feature_set])
    if not features or len(features) != len(set(features)) or set(features) - SAFE_PREDICTORS:
        raise ValueError('Predictor allowlist rejected a feature.')
    if set(features) - set(frame.columns):
        raise ValueError(f'Missing predictors: {sorted(set(features) - set(frame.columns))}')
    return features


def classifier_pipeline(features, params, regression_params, seed, jobs):
    if params.get('class_weight') is not None or 'is_unbalance' in params or 'scale_pos_weight' in params:
        raise ValueError('This initial classifier study excludes class weighting.')
    prep = clone(pipeline(features, regression_params, seed, jobs).named_steps['prep'])
    prep.set_output(transform='pandas')
    return Pipeline([
        ('prep', prep),
        ('model', LGBMClassifier(objective='binary', class_weight=None, verbosity=-1,
                                 random_state=seed, n_jobs=jobs, **params)),
    ])


def fit_probability(frame, train, valid, label, features, params, regression_params, seed, jobs):
    if set(features) - SAFE_PREDICTORS or set(features) - set(frame.columns):
        raise ValueError('Unsafe or absent predictors.')
    if np.unique(label[train]).size != 2:
        raise ValueError('Both recovery classes are required within every training fold.')
    model = classifier_pipeline(features, params, regression_params, seed, jobs)
    model.fit(frame.iloc[train][features], label[train].astype(int))
    probability = model.predict_proba(frame.iloc[valid][features])[:, 1]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid recovery probability scores.')
    return probability


def confusion_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    if actual.ndim != 1 or actual.shape != predicted.shape or not len(actual):
        raise ValueError('Aligned nonempty labels are required.')
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tp = int(np.sum(actual & predicted))
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return dict(n=len(actual), tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn + tp) / len(actual),
                balanced_accuracy=(recall + specificity) / 2,
                recovered_precision=precision, recovered_recall=recall,
                recovered_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                specificity=specificity,
                false_positive_rate=fp / (tn + fp) if tn + fp else np.nan,
                actual_recovered=fn + tp, predicted_recovered=fp + tp)


def score_metrics(actual, predicted, score, ranking=True):
    result = confusion_metrics(actual, predicted)
    if ranking and np.unique(actual).size == 2:
        result['roc_auc'] = roc_auc_score(actual, score)
        result['average_precision'] = average_precision_score(actual, score)
    else:
        result['roc_auc'] = np.nan
        result['average_precision'] = np.nan
    return result


def select_probability_cutoff(actual, probability, candidates, metric):
    records = []
    for cutoff in candidates:
        row = dict(probability_cutoff=float(cutoff),
                   **confusion_metrics(actual, probability >= cutoff))
        records.append(row)
    table = pd.DataFrame(records)
    if table[metric].isna().all():
        raise ValueError(f'No valid {metric} values for threshold selection.')
    table['distance_from_half'] = abs(table['probability_cutoff'] - .5)
    chosen = table.sort_values([metric, 'distance_from_half', 'probability_cutoff'],
                               ascending=[False, True, True], kind='stable').iloc[0]
    return float(chosen['probability_cutoff']), table


def nested_classifier(frame, groups, features, regression_params, candidates, cutoffs,
                      outer_folds, inner_folds, seed, jobs, player_disjoint=False):
    label = checked_six_ply_label(frame)
    rows, chosen_rows, candidate_rows, threshold_rows = [], [], [], []
    outer = checked_folds(frame, groups, outer_folds, seed + 1, player_disjoint)
    for fold_id, (train, valid) in enumerate(outer, 1):
        training, train_groups = frame.iloc[train], groups.iloc[train]
        inner = checked_folds(training, train_groups, inner_folds, seed, player_disjoint)
        best = None
        for candidate_id, params in enumerate(candidates):
            inner_score = np.full(len(train), np.nan)
            for inner_train, inner_valid in inner:
                inner_score[inner_valid] = fit_probability(
                    training, inner_train, inner_valid, label[train], features,
                    params, regression_params, seed, jobs)
            if not np.isfinite(inner_score).all():
                raise ValueError('Inner OOF classifier scores did not cover training mistakes.')
            ap = average_precision_score(label[train], inner_score)
            auc = roc_auc_score(label[train], inner_score)
            candidate_rows.append(dict(outer_fold=fold_id, candidate_id=candidate_id,
                                       inner_average_precision=ap, inner_roc_auc=auc,
                                       params=json.dumps(params, sort_keys=True)))
            if best is None or (ap, auc, -candidate_id) > (best[0], best[1], -best[2]):
                best = (ap, auc, candidate_id, params, inner_score)
        ap, auc, candidate_id, params, inner_score = best
        cutoff_accuracy, accuracy_table = select_probability_cutoff(
            label[train], inner_score, cutoffs, 'accuracy')
        cutoff_balanced, balanced_table = select_probability_cutoff(
            label[train], inner_score, cutoffs, 'balanced_accuracy')
        assert accuracy_table['probability_cutoff'].equals(balanced_table['probability_cutoff'])
        threshold_rows.append(accuracy_table.assign(outer_fold=fold_id, candidate_id=candidate_id))
        probability = fit_probability(frame, train, valid, label, features,
                                      params, regression_params, seed, jobs)
        for strategy, cutoff in [('classifier_0.5', .5),
                                 ('classifier_tuned_accuracy', cutoff_accuracy),
                                 ('classifier_tuned_balanced_accuracy', cutoff_balanced)]:
            rows.append(pd.DataFrame({
                'row_index': frame.index[valid], 'game': groups.iloc[valid].to_numpy(),
                'outer_fold': fold_id, 'strategy': strategy,
                'actual_recovered': label[valid],
                'predicted_recovered': probability >= cutoff,
                'score': probability, 'decision_cutoff': cutoff,
            }))
        chosen_rows.append(dict(outer_fold=fold_id, candidate_id=candidate_id,
                                selected_params=json.dumps(params, sort_keys=True),
                                inner_average_precision=ap, inner_roc_auc=auc,
                                selected_accuracy_cutoff=cutoff_accuracy,
                                selected_balanced_accuracy_cutoff=cutoff_balanced,
                                training_mistakes=len(train), training_games=train_groups.nunique(),
                                validation_mistakes=len(valid), validation_games=groups.iloc[valid].nunique()))
        print(f'Classifier outer fold {fold_id}: candidate={candidate_id}, '
              f'AP={ap:.4f}, accuracy cutoff={cutoff_accuracy:.2f}, '
              f'balanced cutoff={cutoff_balanced:.2f}', flush=True)
    oof = pd.concat(rows, ignore_index=True)
    for _, part in oof.groupby('strategy'):
        if len(part) != len(frame) or part.duplicated('row_index').any():
            raise ValueError('Every classifier strategy needs one held-out prediction per mistake.')
    return (oof, pd.DataFrame(chosen_rows), pd.DataFrame(candidate_rows),
            pd.concat(threshold_rows, ignore_index=True))


def combine_comparators(regression_oof, classifier_oof, frame):
    expected = checked_six_ply_label(frame)
    reference = classifier_oof[classifier_oof['strategy'] == 'classifier_0.5'].sort_values('row_index')
    reg_reference = regression_oof[regression_oof['strategy'] == 'fixed_0.75'].sort_values('row_index')
    locations = frame.index.get_indexer(reference['row_index'])
    if ((locations < 0).any()
            or not np.array_equal(reference['row_index'], reg_reference['row_index'])
            or not np.array_equal(reference['game'], reg_reference['game'])
            or not np.array_equal(reference['outer_fold'], reg_reference['outer_fold'])
            or not np.array_equal(reference['actual_recovered'], reg_reference['actual_recovered'])
            or not np.array_equal(reference['actual_recovered'], expected[locations])):
        raise ValueError('Classifier and regression outer games/labels do not match.')
    names = {'training_majority': 'training_majority',
             'fixed_0.75': 'regression_fixed_0.75',
             'tuned_accuracy': 'regression_tuned_accuracy',
             'tuned_balanced_accuracy': 'regression_tuned_balanced_accuracy'}
    converted = regression_oof[['row_index', 'game', 'outer_fold', 'strategy',
                                'actual_recovered', 'predicted_recovered',
                                'decision_cutoff', 'predicted_ratio']].copy()
    converted['strategy'] = converted['strategy'].map(names)
    converted['score'] = -converted['predicted_ratio']
    converted.loc[converted['strategy'] == 'training_majority', 'score'] = np.nan
    converted = converted.drop(columns='predicted_ratio')
    combined = pd.concat([converted, classifier_oof], ignore_index=True)
    if set(combined['strategy']) != set(STRATEGIES):
        raise ValueError('Comparator strategies differ from the expected list.')
    for _, part in combined.groupby('strategy'):
        if len(part) != len(frame) or part.duplicated('row_index').any():
            raise ValueError('Every strategy must score the same outer mistakes once.')
    return combined


def summary_table(oof):
    rows = []
    for strategy in STRATEGIES:
        part = oof[oof['strategy'] == strategy]
        rows.append(dict(strategy=strategy, games=part['game'].nunique(),
                         **score_metrics(part['actual_recovered'].to_numpy(bool),
                                         part['predicted_recovered'].to_numpy(bool),
                                         part['score'].to_numpy(float),
                                         ranking=strategy != 'training_majority')))
    return pd.DataFrame(rows)


def bootstrap_paired(oof, repeats, seed):
    reference = oof[oof['strategy'] == 'classifier_0.5'].sort_values('row_index')
    games = reference['game'].to_numpy()
    labels = reference['actual_recovered'].to_numpy(bool)
    wide_predictions, wide_scores = {}, {}
    for strategy in STRATEGIES:
        part = oof[oof['strategy'] == strategy].sort_values('row_index')
        if not np.array_equal(part['game'], games) or not np.array_equal(part['actual_recovered'], labels):
            raise ValueError('Paired bootstrap requires identical game and label order.')
        wide_predictions[strategy] = part['predicted_recovered'].to_numpy(bool)
        wide_scores[strategy] = part['score'].to_numpy(float)
    game_indices = pd.Series(np.arange(len(games))).groupby(games, sort=True).apply(
        lambda values: values.to_numpy()).tolist()
    rng = np.random.default_rng(seed)
    comparisons = [(left, right) for left, right in COMPARISONS]
    results = {(left, right, metric): [] for left, right in comparisons for metric in BOOTSTRAP_METRICS}
    for iteration in range(repeats):
        sample = np.concatenate([game_indices[i] for i in rng.integers(0, len(game_indices), len(game_indices))])
        y = labels[sample]
        needed = set(x for pair in comparisons for x in pair)
        scores = {name: score_metrics(y, wide_predictions[name][sample], wide_scores[name][sample],
                                     ranking=name != 'training_majority') for name in needed}
        for left, right in comparisons:
            for metric in BOOTSTRAP_METRICS:
                results[(left, right, metric)].append(scores[left][metric] - scores[right][metric])
        if (iteration + 1) % 100 == 0:
            print(f'Game-cluster bootstrap {iteration + 1}/{repeats}', flush=True)
    point = summary_table(oof).set_index('strategy')
    rows = []
    for (left, right, metric), values in results.items():
        valid = np.asarray(values, float)
        valid = valid[np.isfinite(valid)]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(model=left, comparator=right, metric=metric,
                         difference=float(point.loc[left, metric] - point.loc[right, metric]),
                         lower95=float(low), upper95=float(high),
                         bootstrap_repeats=repeats, games=len(game_indices)))
    return pd.DataFrame(rows)


def plots(oof, summary, out):
    chosen = ['regression_fixed_0.75', 'classifier_0.5']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name in chosen:
        part = oof[oof['strategy'] == name]
        y = part['actual_recovered'].to_numpy(bool)
        score = part['score'].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        label = 'Regressor score' if name.startswith('regression') else 'Classifier probability'
        axes[0].plot(fpr, tpr, label=f'{label} (AUC {roc_auc_score(y, score):.3f})')
        axes[1].plot(recall, precision,
                     label=f'{label} (AP {average_precision_score(y, score):.3f})')
    prevalence = oof[oof['strategy'] == chosen[0]]['actual_recovered'].mean()
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    axes[1].axhline(prevalence, linestyle='--', color='gray', label=f'Prevalence {prevalence:.3f}')
    axes[0].set(xlabel='False-positive rate', ylabel='Recovered recall', title='ROC')
    axes[1].set(xlabel='Recovered recall', ylabel='Recovered precision', title='Precision–recall')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / 'roc_and_precision_recall.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    names = ['training_majority', 'regression_fixed_0.75', 'regression_tuned_balanced_accuracy',
             'classifier_0.5', 'classifier_tuned_accuracy',
             'classifier_tuned_balanced_accuracy', 'regression_tuned_accuracy']
    for ax, name in zip(axes.flat, names):
        item = summary.set_index('strategy').loc[name]
        matrix = np.array([[item['tn'], item['fp']], [item['fn'], item['tp']]], dtype=int)
        totals = matrix.sum(axis=1, keepdims=True)
        fraction = np.divide(matrix, totals, out=np.zeros_like(matrix, float), where=totals > 0)
        ax.imshow(fraction, cmap='Blues', vmin=0, vmax=1)
        ax.set(title=name.replace('_', ' ') + f'\naccuracy {item["accuracy"]:.1%}',
               xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Persistent', 'Recovered'], yticklabels=['Persistent', 'Recovered'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:,}\n{fraction[i, j]:.0%}', ha='center', va='center',
                        fontsize=8, color='white' if fraction[i, j] > .55 else 'black')
    axes.flat[-1].axis('off')
    fig.supxlabel('Predicted class')
    fig.supylabel('Actual class')
    fig.suptitle('Outer-fold confusion matrices; recovered is the exploratory positive class')
    fig.savefig(out / 'outer_confusion_matrices.png', dpi=160)
    plt.close(fig)


def run(args):
    if args.out.resolve().exists():
        raise FileExistsError(f'Refusing to overwrite existing run: {args.out}')
    if args.input.name != 'development_with_multipv.csv':
        raise ValueError('Only the canonical development table is accepted.')
    frame, groups = checked_development(args.input.resolve())
    labels = checked_six_ply_label(frame)
    features = validate_features(frame, args.feature_set)
    config = json.loads(args.config.read_text())
    regression_params = json.loads(args.params_json) if args.params_json else dict(config['base_params'])
    if regression_params.get('objective') != 'regression_l1':
        raise ValueError('Regression comparator must use the existing L1 objective.')
    classifier, selections, candidate_scores, threshold_scores = nested_classifier(
        frame, groups, features, regression_params, CLASSIFIER_CANDIDATES,
        PROBABILITY_CUTOFFS, args.outer_folds, args.inner_folds, args.seed,
        args.jobs, args.player_disjoint)
    regression, regression_selections, regression_grid = nested_thresholds(
        frame, groups, features, regression_params, 6, DEFAULT_CUTOFFS,
        args.outer_folds, args.inner_folds, args.seed, args.jobs, args.player_disjoint)
    oof = combine_comparators(regression, classifier, frame)
    summary = summary_table(oof)
    intervals = bootstrap_paired(oof, args.bootstrap_repeats, args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        kind='exploratory six-ply recovery classifier, nested development evaluation',
        created_utc=datetime.now(timezone.utc).isoformat(),
        data=str(args.input.resolve()), data_sha256=digest(args.input.resolve()),
        source_archives=sorted(EXPECTED_ARCHIVES), mistakes=len(frame), games=groups.nunique(),
        recovered=int(labels.sum()), persistent=int((~labels).sum()),
        label_definition='persistence_ratio_6ply < 0.75 (exploratory)',
        horizon=6, primary_regression_target='change_in_damage_6ply',
        feature_set=args.feature_set, predictors=features,
        classifier_candidates=list(CLASSIFIER_CANDIDATES), class_weight=None,
        classifier_selection_metric='inner out-of-fold average precision',
        classifier_cutoffs=list(PROBABILITY_CUTOFFS),
        regression_params=regression_params, regression_cutoffs=list(DEFAULT_CUTOFFS),
        seed=args.seed, jobs=args.jobs, outer_folds=args.outer_folds,
        inner_folds=args.inner_folds, player_disjoint=args.player_disjoint,
        bootstrap_repeats=args.bootstrap_repeats,
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'matplotlib']},
        code_sha256={name: digest(ROOT / name) for name in
                     ['scripts/evaluate_recovery_classifier.py',
                      'scripts/tune_recovery_threshold.py', 'tune_chess.py', 'config.json']},
        command_arguments={name: str(value) if isinstance(value, Path) else value
                           for name, value in vars(args).items()},
        note='Five development archives only. Outer evaluation is held out from inner selection. '
             'Prior feature/design decisions used development data, so these are exploratory results. '
             'PGN terminal-result targets cannot identify abandonment/disconnection.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    oof.to_csv(out / 'outer_predictions.csv', index=False)
    summary.to_csv(out / 'outer_summary.csv', index=False)
    intervals.to_csv(out / 'paired_game_bootstrap_intervals.csv', index=False)
    selections.to_csv(out / 'selected_classifier_by_fold.csv', index=False)
    candidate_scores.to_csv(out / 'inner_candidate_scores.csv', index=False)
    threshold_scores.to_csv(out / 'inner_classifier_threshold_scores.csv', index=False)
    regression_selections.to_csv(out / 'selected_regression_thresholds_by_fold.csv', index=False)
    regression_grid.to_csv(out / 'inner_regression_threshold_scores.csv', index=False)
    plots(oof, summary, out)
    print(summary[['strategy', 'roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy',
                   'recovered_precision', 'recovered_recall', 'recovered_f1', 'specificity', 'fp']]
          .to_string(index=False))
    print('Saved:', out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'development_with_multipv.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--params-json', help='Fixed existing L1 regression parameters as JSON')
    parser.add_argument('--feature-set', default='full_multipv', choices=sorted(SETS))
    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--bootstrap-repeats', type=int, default=500)
    parser.add_argument('--player-disjoint', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.bootstrap_repeats < 1:
        parser.error('Use >=2 folds and >=1 bootstrap repetition.')
    run(args)


if __name__ == '__main__':
    main()
