#!/usr/bin/env python3
"""Development-only decision-threshold study for the fixed chess regressor.

The prediction is a persistence ratio, not a recovery probability. The true
recovered label always means actual persistence ratio < 0.75. Only the cutoff
applied to the *predicted* ratio is tuned. Each outer game's cutoff is chosen
using inner game-grouped predictions from outer-training games.

Example:
  python3 scripts/tune_recovery_threshold.py --out notebook_runs/threshold_example
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Save figures without a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tune_chess import EXPECTED_ARCHIVES, SETS, get_folds, pipeline, read_dev  # noqa: E402

LABEL_CUTOFF = 0.75
DEFAULT_CUTOFFS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                   0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
SAFE_PREDICTORS = frozenset((
    'eval_before_cp', 'initial_damage', 'player_elo', 'opponent_elo',
    'move_number', 'player_clock_before_move', 'opponent_clock_before_move',
    'initial_time_seconds', 'increment_seconds', 'material_balance_cp',
    'number_of_pieces', 'hanging_pieces', 'opponent_hanging_pieces',
    'king_attackers', 'opponent_king_attackers', 'reasonable_candidate_moves',
    'eval_gap_best_second_cp', 'in_check', 'gives_check', 'is_capture',
    'is_promotion', 'game_phase', 'multipv_good_reply_count_10pp',
    'multipv_best_second_gap_pp',
))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_development(path):
    if path.name != 'development_with_multipv.csv':
        raise ValueError('Only development_with_multipv.csv is accepted.')
    source = pd.read_csv(path, usecols=['source_pgn'])['source_pgn']
    if source.isna().any() or set(source.astype(str).map(lambda s: Path(s).name)) != EXPECTED_ARCHIVES:
        raise ValueError('Source archives are not exactly the five development months.')
    frame, groups = read_dev(path)
    groups = groups.astype(str).str.strip()
    if len(frame) != 23167 or groups.nunique() != 6512 or groups.isin(['', '?', '-']).any():
        raise ValueError('Development game identities/counts differ from the canonical table.')
    identities = frame[['source_pgn', 'game_index']].assign(game=groups).drop_duplicates()
    if (identities.groupby(['source_pgn', 'game_index'])['game'].nunique().gt(1).any()
            or identities.groupby('game').size().gt(1).any()):
        raise ValueError('Unstable or reused game identity.')
    missing_gap = frame['multipv_best_second_gap_pp'].isna()
    if missing_gap.sum() != 76 or frame.loc[missing_gap, 'multipv_lines_returned'].ne(1).any():
        raise ValueError('Documented one-reply MultiPV missingness changed.')
    return frame, groups


def players(frame):
    names = frame[['white', 'black']].astype('string').apply(lambda s: s.str.strip().str.casefold())
    if names.isna().any().any() or names.isin(['', '?', '-', 'anonymous']).any().any():
        raise ValueError('Verified player identifiers are required for player-disjoint folds.')
    return set(names.to_numpy().ravel())


def checked_folds(frame, groups, count, seed, player_disjoint=False):
    result = []
    for train, valid in get_folds(frame, groups, count, seed):
        if player_disjoint:
            held = players(frame.iloc[valid])
            candidate = frame.iloc[train]
            bad = (candidate['white'].astype(str).str.strip().str.casefold().isin(held) |
                   candidate['black'].astype(str).str.strip().str.casefold().isin(held))
            bad_games = set(groups.iloc[train].iloc[np.flatnonzero(bad.to_numpy())])
            train = train[~groups.iloc[train].isin(bad_games).to_numpy()]
            if not len(train) or not players(frame.iloc[train]).isdisjoint(held):
                raise ValueError('Player-disjoint training games could not be verified.')
        if groups.iloc[train].nunique() < count or not set(groups.iloc[train]).isdisjoint(groups.iloc[valid]):
            raise ValueError('Insufficient or overlapping training games.')
        result.append((train, valid))
    return result


def score_labels(actual, predicted):
    actual, predicted = np.asarray(actual, bool), np.asarray(predicted, bool)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError('Aligned nonempty class vectors are required.')
    tn = int(np.sum(~actual & ~predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tp = int(np.sum(actual & predicted))
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return dict(n=len(actual), tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn + tp) / len(actual),
                balanced_accuracy=(recall + specificity) / 2,
                recovered_precision=tp / (tp + fp) if tp + fp else np.nan,
                recovered_recall=recall, persistent_recall=specificity,
                predicted_recovered=int(predicted.sum()), actual_recovered=int(actual.sum()))


def threshold_table(actual, predicted_ratio, candidates):
    if not np.isfinite(predicted_ratio).all():
        raise ValueError('Predicted persistence ratios must be finite.')
    return pd.DataFrame([{'decision_cutoff': float(c), **score_labels(actual, predicted_ratio < c)}
                         for c in candidates])


def select_cutoff(table, metric):
    valid = table[np.isfinite(table[metric])]
    if valid.empty:
        raise ValueError(f'No valid {metric} threshold scores.')
    best = valid.sort_values([metric, 'distance_from_default', 'decision_cutoff'],
                             ascending=[False, True, True], kind='stable').iloc[0]
    return float(best['decision_cutoff'])


def fit_predict(frame, train, valid, features, params, seed, jobs, horizon):
    if not features or len(features) != len(set(features)) or set(features) - SAFE_PREDICTORS:
        raise ValueError('Unapproved or duplicate predictor columns.')
    if set(features) - set(frame.columns):
        raise ValueError('Missing predictor columns.')
    target = f'change_in_damage_{horizon}ply'
    model = pipeline(features, params, seed, jobs)
    model.named_steps['prep'].set_output(transform='pandas')
    model.fit(frame.iloc[train][features], frame.iloc[train][target])
    prediction = model.predict(frame.iloc[valid][features])
    return 1 + prediction / frame.iloc[valid]['initial_damage'].to_numpy()


def nested_thresholds(frame, groups, features, params, horizon, candidates,
                      outer_folds, inner_folds, seed, jobs, player_disjoint=False):
    y = frame[f'change_in_damage_{horizon}ply'].to_numpy(dtype=float)
    damage = frame['initial_damage'].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(damage).all() or (damage <= 20).any():
        raise ValueError('Invalid development targets or initial damage.')
    label = 1 + y / damage < LABEL_CUTOFF
    predictions, selections, grid_scores = [], [], []
    outer = checked_folds(frame, groups, outer_folds, seed + 1, player_disjoint)
    for fold_id, (train, valid) in enumerate(outer, 1):
        training, train_groups = frame.iloc[train], groups.iloc[train]
        inner = checked_folds(training, train_groups, inner_folds, seed, player_disjoint)
        inner_ratio = np.full(len(train), np.nan)
        for inner_train, inner_valid in inner:
            inner_ratio[inner_valid] = fit_predict(
                training, inner_train, inner_valid, features, params, seed, jobs, horizon)
        if not np.isfinite(inner_ratio).all():
            raise ValueError('Inner predictions must cover every outer-training mistake once.')
        table = threshold_table(label[train], inner_ratio, candidates)
        table['outer_fold'] = fold_id
        table['distance_from_default'] = abs(table['decision_cutoff'] - LABEL_CUTOFF)
        grid_scores.append(table)
        chosen = {'fixed_0.75': LABEL_CUTOFF,
                  'tuned_accuracy': select_cutoff(table, 'accuracy'),
                  'tuned_balanced_accuracy': select_cutoff(table, 'balanced_accuracy')}
        outer_ratio = fit_predict(frame, train, valid, features, params, seed, jobs, horizon)
        majority = bool(np.mean(label[train]) >= .5)
        for strategy in ['training_majority', *chosen]:
            cutoff = chosen.get(strategy, np.nan)
            predicted = np.full(len(valid), majority) if strategy == 'training_majority' else outer_ratio < cutoff
            predictions.append(pd.DataFrame({
                'row_index': frame.index[valid], 'game': groups.iloc[valid].to_numpy(),
                'outer_fold': fold_id, 'horizon': horizon, 'strategy': strategy,
                'decision_cutoff': cutoff, 'actual_recovered': label[valid],
                'predicted_recovered': predicted, 'predicted_ratio': outer_ratio,
                'actual_change': y[valid], 'initial_damage': damage[valid],
            }))
        selections.append(dict(outer_fold=fold_id, n_training=len(train), n_validation=len(valid),
                               training_games=train_groups.nunique(), validation_games=groups.iloc[valid].nunique(),
                               majority_recovered=majority, **chosen))
        print(f'Outer fold {fold_id}: selected accuracy={chosen["tuned_accuracy"]:.2f}, '
              f'balanced accuracy={chosen["tuned_balanced_accuracy"]:.2f}; '
              f'train={len(train)}, validation={len(valid)}', flush=True)
    oof = pd.concat(predictions, ignore_index=True)
    for _, part in oof.groupby('strategy'):
        if len(part) != len(frame) or part.duplicated('row_index').any():
            raise ValueError('Outer validation predictions must cover each mistake once.')
    return oof, pd.DataFrame(selections), pd.concat(grid_scores, ignore_index=True)


def summarize(oof):
    rows = []
    for strategy, part in oof.groupby('strategy', sort=False):
        row = {'strategy': strategy, 'games': part['game'].nunique(),
               **score_labels(part['actual_recovered'], part['predicted_recovered'])}
        if strategy != 'training_majority':
            score = -part['predicted_ratio'].to_numpy()
            row['roc_auc'] = roc_auc_score(part['actual_recovered'], score)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_intervals(oof, repeats, seed):
    wide = oof.pivot(index='row_index', columns='strategy', values='predicted_recovered')
    reference = oof[oof['strategy'] == 'fixed_0.75'].set_index('row_index')
    wide = wide.join(reference[['game', 'actual_recovered']], validate='one_to_one')
    strategies = [c for c in wide if c not in ['game', 'actual_recovered']]
    actual = wide['actual_recovered'].astype(bool)
    correct = pd.DataFrame({name: (actual == wide[name].astype(bool)).astype(int)
                            for name in strategies}, index=wide.index)
    correct['game'] = wide['game']
    correct['n'] = 1
    by_game = correct.groupby('game', sort=True).sum()
    values = by_game[strategies].to_numpy()
    sizes = by_game['n'].to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty((repeats, len(strategies)))
    for i in range(repeats):
        chosen = rng.integers(0, len(by_game), len(by_game))
        draws[i] = values[chosen].sum(axis=0) / sizes[chosen].sum()
    point = correct[strategies].mean().to_numpy()
    rows = []
    for j, name in enumerate(strategies):
        row = dict(strategy=name, games=len(by_game), bootstrap_repeats=repeats,
                   accuracy=point[j], accuracy_low95=np.quantile(draws[:, j], .025),
                   accuracy_high95=np.quantile(draws[:, j], .975))
        for reference_name in ['training_majority', 'fixed_0.75']:
            k = strategies.index(reference_name)
            delta = draws[:, j] - draws[:, k]
            row[f'gain_vs_{reference_name}_pp'] = 100 * (point[j] - point[k])
            row[f'gain_vs_{reference_name}_low95_pp'] = 100 * np.quantile(delta, .025)
            row[f'gain_vs_{reference_name}_high95_pp'] = 100 * np.quantile(delta, .975)
        rows.append(row)
    return pd.DataFrame(rows)


def plots(grid, summary, out):
    mean = grid.groupby('decision_cutoff')[['accuracy', 'balanced_accuracy', 'recovered_recall']].mean()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for col, name in [('accuracy', 'Inner accuracy'), ('balanced_accuracy', 'Inner balanced accuracy'),
                      ('recovered_recall', 'Inner recovered recall')]:
        ax.plot(mean.index, mean[col], marker='.', label=name)
    ax.axvline(LABEL_CUTOFF, color='gray', linestyle='--', label='Current decision cutoff 0.75')
    ax.set(xlabel='Decision cutoff on predicted persistence ratio', ylabel='Inner validation score',
           title='Threshold selection on outer-training games only', ylim=(0, 1))
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / 'inner_threshold_sweep.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(summary), figsize=(14, 4), constrained_layout=True)
    for ax, (_, row) in zip(axes, summary.iterrows()):
        matrix = np.array([[row['tn'], row['fp']], [row['fn'], row['tp']]], dtype=int)
        totals = matrix.sum(axis=1, keepdims=True)
        share = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
        ax.imshow(share, vmin=0, vmax=1, cmap='Blues')
        ax.set(title=f'{row["strategy"]}\naccuracy {row["accuracy"]:.1%}',
               xlabel='Predicted class', ylabel='Actual class', xticks=[0, 1], yticks=[0, 1],
               xticklabels=['Persistent', 'Recovered'], yticklabels=['Persistent', 'Recovered'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:,}\n{share[i, j]:.1%}', ha='center', va='center',
                        color='white' if share[i, j] > .55 else 'black')
    fig.suptitle('Outer development validation; actual recovered means persistence ratio <0.75')
    fig.savefig(out / 'outer_confusion_matrices.png', dpi=160)
    plt.close(fig)


def run(args):
    if args.out.resolve().exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {args.out}')
    path = args.input.resolve()
    frame, groups = checked_development(path)
    config = json.loads(args.config.read_text())
    params = json.loads(args.params_json) if args.params_json else dict(config['base_params'])
    if params.get('objective') != 'regression_l1':
        raise ValueError('Keep the fixed regression L1 objective for this threshold study.')
    if args.feature_set not in SETS:
        raise ValueError('Unknown feature set.')
    features = SETS[args.feature_set]
    if set(features) - SAFE_PREDICTORS:
        raise ValueError('Predictor whitelist rejected a source feature.')
    candidates = [float(x) for x in args.candidates.split(',')]
    if (not candidates or len(candidates) != len(set(candidates))
            or LABEL_CUTOFF not in candidates or not np.isfinite(candidates).all()):
        raise ValueError('Provide unique finite decision cutoffs including 0.75.')
    oof, selected, grid = nested_thresholds(
        frame, groups, features, params, args.horizon, candidates,
        args.outer_folds, args.inner_folds, args.seed, args.jobs, args.player_disjoint)
    summary = summarize(oof)
    intervals = cluster_intervals(oof, args.bootstrap_repeats, args.seed)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        kind='development nested decision-threshold study; exploratory',
        data=str(path), data_sha256=digest(path), script_sha256=digest(Path(__file__)),
        tune_chess_sha256=digest(ROOT / 'tune_chess.py'), config_sha256=digest(args.config),
        source_archives=sorted(EXPECTED_ARCHIVES), mistakes=len(frame), games=groups.nunique(),
        horizon=args.horizon, feature_set=args.feature_set, predictors=features,
        params=params, candidates=candidates, label_cutoff=LABEL_CUTOFF,
        seed=args.seed, outer_folds=args.outer_folds, inner_folds=args.inner_folds,
        player_disjoint=args.player_disjoint, bootstrap_repeats=args.bootstrap_repeats,
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'matplotlib']},
        target='change_in_damage; true recovered if 1 + change_in_damage / initial_damage < 0.75',
        note='Cutoffs are on predicted persistence ratio, not recovery probability. '
             'All tuning uses development training games; outer validation is also development. '
             'PGN terminal-result targets cannot identify abandonment/disconnection.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    oof.to_csv(out / 'outer_predictions.csv', index=False)
    selected.to_csv(out / 'selected_thresholds_by_fold.csv', index=False)
    grid.to_csv(out / 'inner_threshold_sweep.csv', index=False)
    summary.to_csv(out / 'outer_summary.csv', index=False)
    intervals.to_csv(out / 'paired_game_cluster_intervals.csv', index=False)
    plots(grid, summary, out)
    print(summary[['strategy', 'accuracy', 'balanced_accuracy', 'recovered_recall',
                   'predicted_recovered', 'roc_auc']].to_string(index=False))
    print('Saved:', out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'development_with_multipv.csv')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--params-json', help='Optional fixed L1 regressor parameters as JSON')
    parser.add_argument('--feature-set', default='full_multipv', choices=sorted(SETS))
    parser.add_argument('--horizon', type=int, default=6, choices=[2, 4, 6, 10])
    parser.add_argument('--candidates', default=','.join(map(str, DEFAULT_CUTOFFS)))
    parser.add_argument('--outer-folds', type=int, default=3)
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--player-disjoint', action='store_true')
    parser.add_argument('--bootstrap-repeats', type=int, default=1000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.outer_folds < 2 or args.inner_folds < 2 or args.bootstrap_repeats < 1:
        parser.error('Use >=2 folds and >=1 bootstrap repetition.')
    run(args)


if __name__ == '__main__':
    main()
