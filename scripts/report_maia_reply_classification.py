#!/usr/bin/env python3
"""Classification from saved Maia reply probabilities at a fixed 0.5 cutoff."""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_bounded_maia_regression import base_frame, split, FOUR

MAIA = ROOT/'notebook_runs/maia_bounded_four_archive_20261117_v1'
PILOT = ROOT/'notebook_runs/maia_budget_simulation_20261117_v1/uniform_predictions_392.csv'
TARGETS = ROOT/'notebook_runs/opponent_reply_targets_development_v1/reply_targets.csv'
FOLDS = ROOT/'notebook_runs/final_optuna_development_20261117_v1/outer_fold_assignments.csv'
DATA = ROOT/'development_with_multipv.csv'
PROBABILITY = 'maia_preserving_reply_probability'
KEYS = ['row_index', 'source_pgn', 'game_index', 'ply', 'site']
SEED, REPEATS, CUTOFF = 1701, 2000, .5


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x:
                    x.item() if isinstance(x, np.generic) else str(x))+'\n')


def scores(frame, prediction, probability=None):
    y, p = frame.actual_preserved.astype(bool).to_numpy(), np.asarray(prediction, bool)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[False, True]).ravel()
    result = dict(n=len(y), games=frame.site.nunique(), actual_positive=int(y.sum()),
        prevalence=float(y.mean()), correct=int((y == p).sum()),
        accuracy=float(accuracy_score(y, p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
        precision=float(precision_score(y, p, zero_division=0)),
        recall=float(recall_score(y, p, zero_division=0)), f1=float(f1_score(y, p, zero_division=0)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    result.update(roc_auc=float(roc_auc_score(y, probability)) if probability is not None else np.nan,
                  average_precision=float(average_precision_score(y, probability)) if probability is not None else np.nan)
    return result


def paired_accuracy_ci(frame):
    # The same sampled games are used for Maia and the training-fold baseline.
    y = frame.actual_preserved.to_numpy(bool)
    values = np.column_stack([(frame.maia_prediction.to_numpy(bool) == y).astype(float),
                              (frame.majority_prediction.to_numpy(bool) == y).astype(float)])
    codes, games = pd.factorize(frame.site, sort=True)
    sums = np.zeros((len(games), 2)); np.add.at(sums, codes, values)
    sizes = np.bincount(codes); rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, REPEATS, 100):
        n = min(100, REPEATS-start)
        sample = rng.integers(0, len(games), (n, len(games)))
        rates = sums[sample].sum(axis=1)/sizes[sample].sum(axis=1)[:, None]
        draws.extend(rates[:, 0]-rates[:, 1])
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(accuracy_gain_pp=100*(values[:, 0]-values[:, 1]).mean(),
                lower95_pp=100*lo, upper95_pp=100*hi, bootstrap_repeats=REPEATS)


def main(out):
    if out.exists():
        raise FileExistsError('Use a new output directory.')
    protocol = json.loads((MAIA/'protocol.json').read_text())
    if set(protocol['source_archives']) != FOUR:
        raise ValueError('Only the four earlier development archives are allowed.')
    ready = json.loads((MAIA/'features_ready.json').read_text())
    if sha(MAIA/'maia_bounds_features.csv') != ready['feature_sha256']:
        raise ValueError('Saved probability feature changed.')
    source_paths = [DATA, TARGETS, FOLDS, MAIA/'maia_bounds_features.csv',
        MAIA/'features_ready.json', MAIA/'pre_reply_inputs.csv', MAIA/'protocol.json', PILOT,
        ROOT/'scripts/test_bounded_maia_regression.py', ROOT/'scripts/tune_recovery_threshold.py',
        ROOT/'tune_chess.py', ROOT/'config.json', ROOT/'my_tuning/selected.json', Path(__file__).resolve()]
    fingerprints = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    frame = base_frame()
    probabilities = pd.read_csv(MAIA/'maia_bounds_features.csv',
        usecols=KEYS+[PROBABILITY, 'maia_probability_lower', 'maia_probability_upper'])
    frame = frame.merge(probabilities, on=KEYS, how='left', validate='one_to_one', sort=False)
    targets = pd.read_csv(TARGETS, usecols=KEYS[1:]+['status', 'win_after_actual_reply'])
    frame = frame.merge(targets, on=KEYS[1:], how='left', validate='one_to_one', sort=False)
    if len(frame) != 15953 or frame.site.nunique() != 4485 or not frame[PROBABILITY].between(0, 1).all():
        raise ValueError('Unexpected feature/row coverage.')
    valid = frame.status.eq('ok') & np.isfinite(frame.win_after_actual_reply)
    frame['actual_preserved'] = ((frame.win_before-frame.win_after_actual_reply)/frame.initial_damage >= .75).where(valid)
    if valid.sum() != 15851:
        raise ValueError('Unexpected valid-reply universe.')
    pilot = pd.read_csv(PILOT)
    if len(pilot) != 392 or not set(pilot.row_index).issubset(set(frame.loc[valid, 'row_index'])):
        raise ValueError('Unexpected exact-pilot rows.')
    indexed = frame.set_index('row_index')
    if not np.array_equal(pilot.actual_preserved, indexed.loc[pilot.row_index, 'actual_preserved'].astype(bool)):
        raise ValueError('Pilot and full-population labels disagree.')
    maps = pd.read_csv(FOLDS)
    rows, audits = [], []
    for mode in ['game', 'player']:
        for fold in [1, 2, 3]:
            train, validation = split(frame, maps[maps.protocol.eq(mode)], fold, mode == 'player')
            train = train[valid.iloc[train].to_numpy()]
            validation = validation[valid.iloc[validation].to_numpy()]
            train_prevalence = float(frame.iloc[train].actual_preserved.astype(bool).mean())
            majority = train_prevalence >= .5  # Fixed tie rule, never derived from held-out labels.
            part = frame.iloc[validation][KEYS+[PROBABILITY, 'maia_probability_lower',
                'maia_probability_upper', 'actual_preserved']].copy()
            part['protocol'], part['outer_fold'] = mode, fold
            part['majority_prediction'] = majority
            part['maia_prediction'] = part[PROBABILITY] >= CUTOFF
            rows.append(part)
            audits.append(dict(protocol=mode, outer_fold=fold, training_rows=len(train),
                heldout_rows=len(validation), training_prevalence=train_prevalence,
                majority_predicts_preserved=bool(majority), game_overlap=0,
                player_overlap=0 if mode == 'player' else None))
    full = pd.concat(rows, ignore_index=True)
    metrics, folds, intervals = [], [], []
    for mode, group in full.groupby('protocol'):
        if group.row_index.duplicated().any() or len(group) != 15851:
            raise ValueError('Repeated or incomplete evaluation rows.')
        for scope in ['full_four_archives', 'exact_392_pilot']:
            part = group.copy()
            if scope == 'exact_392_pilot':
                part = part[part.row_index.isin(pilot.row_index)].copy()
                part[PROBABILITY] = part.row_index.map(pilot.set_index('row_index').probability_any_preserving_reply)
                part['maia_prediction'] = part[PROBABILITY] >= CUTOFF
            methods = {'maia_saved_probability': (part.maia_prediction, part[PROBABILITY]),
                       'training_fold_majority': (part.majority_prediction, None)}
            if scope == 'full_four_archives':
                methods.update(maia_lower_endpoint=(part.maia_probability_lower >= CUTOFF, part.maia_probability_lower),
                               maia_upper_endpoint=(part.maia_probability_upper >= CUTOFF, part.maia_probability_upper))
            for method, (prediction, probability) in methods.items():
                metrics.append(dict(scope=scope, protocol=mode, method=method, cutoff=CUTOFF,
                                    **scores(part, prediction, probability)))
                # Compute the same quantities per original held-out fold.
                part_method = part.assign(predicted=np.asarray(prediction))
                for fold, block in part_method.groupby('outer_fold'):
                    p = block[PROBABILITY] if method == 'maia_saved_probability' else None
                    folds.append(dict(scope=scope, protocol=mode, method=method, outer_fold=fold,
                                      **scores(block, block.predicted, p)))
            intervals.append(dict(scope=scope, protocol=mode, **paired_accuracy_ci(part)))
    metrics = pd.DataFrame(metrics)
    out.mkdir(parents=True)
    metrics.to_csv(out/'classification_metrics.csv', index=False)
    pd.DataFrame(folds).to_csv(out/'fold_metrics.csv', index=False)
    pd.DataFrame(audits).to_csv(out/'majority_baseline_by_fold.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_accuracy_intervals.csv', index=False)
    full.to_csv(out/'evaluation_predictions.csv.gz', index=False)
    confusion = []
    for row in metrics.itertuples(index=False):
        confusion.extend([dict(scope=row.scope, protocol=row.protocol, method=row.method,
            actual='not_preserved', predicted_not_preserved=row.tn, predicted_preserved=row.fp),
            dict(scope=row.scope, protocol=row.protocol, method=row.method,
            actual='preserved', predicted_not_preserved=row.fn, predicted_preserved=row.tp)])
    pd.DataFrame(confusion).to_csv(out/'confusion_matrices.csv', index=False)
    main_row = metrics[metrics.scope.eq('full_four_archives')&metrics.protocol.eq('game')&
                       metrics.method.eq('maia_saved_probability')].iloc[0]
    base = metrics[metrics.scope.eq('full_four_archives')&metrics.protocol.eq('game')&
                   metrics.method.eq('training_fold_majority')].iloc[0]
    exact = metrics[metrics.scope.eq('exact_392_pilot')&metrics.protocol.eq('game')&
                    metrics.method.eq('maia_saved_probability')].iloc[0]
    interval = pd.DataFrame(intervals).query("scope == 'full_four_archives' and protocol == 'game'").iloc[0]
    report = f'''# Maia immediate-reply classification — development only

Positive means **the actual opponent reply preserves at least 75% of the original
mistake damage immediately after that reply**, using the original mistake-maker's
winning chances. The label remains (win_before − win_after_actual_reply) /
initial_damage >= 0.75. This project-specific threshold is exploratory. This
classification target differs from 10-ply recovery and from the within-10-pp
regression tolerance metric. It does not mean the opponent noticed a mistake.

Predict positive when the saved Maia probability is **>=0.5**. The probability
cutoff was fixed for this report; no cutoff selection, fitting or calibration
was performed. Features were already generated before actual replies were used
to evaluate them. March 2025 was not accessed.

## Main comparison: full four-archive development scope

{int(main_row.n):,} valid replies / {int(main_row.games):,} games from April, June
and July 2017 and February 2018. Of 15,953 mistakes, **102 missing/invalid actual
replies are excluded from all metrics**, never treated as a negative event.
Probabilities are the existing midpoints from scoring 97.5% of Maia's policy mass.

| Method | Accuracy | Precision (preserved) | Recall (preserved) | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved Maia probability | {100*main_row.accuracy:.2f}% | {100*main_row.precision:.2f}% | {100*main_row.recall:.2f}% | {main_row.f1:.4f} | {100*main_row.balanced_accuracy:.2f}% |
| Training-fold majority | {100*base.accuracy:.2f}% | {100*base.precision:.2f}% | {100*base.recall:.2f}% | {base.f1:.4f} | {100*base.balanced_accuracy:.2f}% |

The majority class is selected from eligible training games only, separately for
each original outer fold. Every training-fold majority was "preserved".
Accuracy gain: **{interval.accuracy_gain_pp:.2f} percentage points**, paired
game-cluster 95% interval **[{interval.lower95_pp:.2f}, {interval.upper95_pp:.2f}]**
(2,000 game resamples, seed 1701, fixed predictions).

Confusion matrix for saved Maia probabilities; rows are actual labels:

| Actual | Predicted not preserved | Predicted preserved |
| --- | ---: | ---: |
| Not preserved | {int(main_row.tn):,} | {int(main_row.fp):,} |
| Preserved | {int(main_row.fn):,} | {int(main_row.tp):,} |

Maia finds **{int(main_row.tp):,}/{int(main_row.actual_positive):,}** preserving
replies, at a cost of **{int(main_row.fp):,} false positives**. It misses
{int(main_row.fn):,} preserving replies. ROC-AUC {main_row.roc_auc:.4f}; average
precision {main_row.average_precision:.4f}, against prevalence {main_row.prevalence:.4f}.

## Sensitivity and exact pilot

The saved probability bounds supply lower/upper endpoint classifiers, exported
in classification_metrics.csv. They are approximation sensitivity checks, not
confidence intervals. No endpoint replaces the primary midpoint retrospectively.

The same exact-score 392-row / 364-game pilot gives accuracy
**{100*exact.accuracy:.2f}%**, precision **{100*exact.precision:.2f}%**, recall
**{100*exact.recall:.2f}%**; its majority accuracy is 57.91%. Its confusion counts
are TN={int(exact.tn)}, FP={int(exact.fp)}, FN={int(exact.fn)}, TP={int(exact.tp)}.
Eight other four-archive pilot rows have missing replies and are excluded.
The pilot and full population overlap and are not independent confirmations.

The game and player protocols have identical raw Maia probabilities, labels and
predictions because Maia is fixed and was not trained on these local folds.
Player purging affects training-majority selection; it does not establish that
players in the external Maia training data are disjoint. Both baseline protocols
are saved, with training prevalences and validation-row counts per fold. The
four archives predate the documented Maia training window; membership is not
independently audited. Bullet/classical routing and lack of live-clock input
retain the earlier model's limitations.

## Outputs and reproduction

- classification_metrics.csv: full/pilot and midpoint/endpoint results.
- confusion_matrices.csv: raw confusion counts with named actual/predicted axes.
- majority_baseline_by_fold.csv: training-only majority choices and game checks.
- evaluation_predictions.csv.gz: full four-archive labels, saved probabilities,
  fixed-cutoff predictions, original folds and majority predictions.
- paired_accuracy_intervals.csv: paired game-cluster accuracy-gain intervals.

Run scripts/report_maia_reply_classification.py --out NEW_DIRECTORY using
/private/tmp/chess-final-optuna-env/bin/python with writable PYTHONPYCACHEPREFIX
and MPLCONFIGDIR. Source hashes and exact settings are in provenance.json.

No new model fits, engine evaluations, Maia inference or Optuna trials. Original
datasets, results and frozen configurations were preserved.
**Development-only; these are not final-test results.**
'''
    (out/'REPORT.md').write_text(report)
    for name, expected in fingerprints.items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Source changed: {name}')
    save_json(out/'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_archives=sorted(FOUR), cutoff=CUTOFF, positive_label='immediate persistence ratio >=0.75',
        source_sha256=fingerprints, source_files_unchanged=True, validation_rows=15851,
        missing_reply_rows=102, same_rows_for_model_and_baseline=True,
        majority_fitted_on_training_folds_only=True, new_model_fits=0,
        new_engine_calls=0, new_optuna_trials=0))
    print(metrics.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
