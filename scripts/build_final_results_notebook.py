#!/usr/bin/env python3
"""Build a self-contained report from the completed evaluation, without inference.

Only this builder reads the saved artifacts. The generated notebook embeds tables
and curve coordinates and can execute without any repository data or model files.
Existing evaluation artifacts and notebooks are never modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'final_evaluation/march_2025_once_v1'
CURVE = ROOT / 'notebook_runs/learning_curves_20261110T224023Z_31ae5f8e'
FIVE = '2017-04, 2017-06, 2017-07, 2018-02, 2019-02'
FOUR = '2017-04, 2017-06, 2017-07, 2018-02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame):
    return clean(frame.to_dict('records'))


def collect():
    protected = {}

    def track(path):
        protected[path] = sha(path)
        return path

    def load_json(path):
        return json.loads(track(path).read_text())

    def read(path, **kwargs):
        return pd.read_csv(track(path), float_precision='round_trip', **kwargs)

    index = load_json(RUN / 'artifact_hashes.json')
    for name, expected in index.items():
        assert sha(track(RUN / name)) == expected, 'Changed completed artifact: ' + name
    provenance = load_json(RUN / 'run_provenance.json')
    completion = load_json(RUN / 'completion.json')
    assert completion['status'] == 'complete' and completion['one_time']
    assert provenance['test_metrics_computed'] and not provenance['models_retrained']
    cfg = load_json(RUN / 'frozen_model_config.json')
    tables = {key: read(RUN / name) for key, name in {
        'metrics': 'metrics.csv', 'intervals': 'paired_game_intervals.csv',
        'subgroups': 'subgroups.csv', 'distributions': 'development_vs_final_distributions.csv',
        'development_primary': 'development_primary_context.csv',
        'development_reply': 'development_immediate_context.csv',
        'development_ten': 'development_ten_ply_context.csv',
    }.items()}
    assert len(tables['metrics']) == 54 and len(tables['intervals']) == 72
    assert len(tables['subgroups']) == 612
    assert tables['intervals'].valid_replicates.eq(2000).all()
    input_dir = ROOT / 'test_data/march_2025_enriched_v1'
    manifest_path = input_dir / 'input_manifest.json'
    assert sha(track(manifest_path)) == provenance['input_manifest_sha256']
    manifest = load_json(manifest_path)
    for kind in ['features', 'labels']:
        assert sha(track(input_dir / manifest[kind]['path'])) == manifest[kind]['sha256']
    predictions = read(RUN / 'predictions_before_labels.csv.gz')
    assert sha(RUN / 'predictions_before_labels.csv.gz') == completion['predictions_sha256']
    labels = read(input_dir / manifest['labels']['path'])
    features = read(input_dir / manifest['features']['path'],
                    usecols=['row_id', 'initial_damage', 'site'])
    assert len(features) == 3380 and features.site.nunique() == 955
    assert features.row_id.is_unique and labels.row_id.is_unique
    assert len(predictions) == 18 * 3380
    assert not predictions.duplicated(['task', 'horizon', 'model', 'row_id']).any()
    joined = predictions.merge(labels, on='row_id', validate='many_to_one').merge(
        features, on='row_id', validate='many_to_one')
    assert len(joined) == len(predictions)

    # Reproduce existing metrics and derive descriptive score-based summaries only.
    extra, roc, calibration = [], {}, []
    checked = 0

    def close(actual, expected):
        if pd.isna(actual) and pd.isna(expected):
            return
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    for item in tables['metrics'].itertuples():
        block = joined[(joined.task == item.task) & (joined.horizon == item.horizon) &
                       (joined.model == item.model) & (joined.initial_damage > item.threshold)]
        target = 'actual_preserved' if item.task == 'reply' else f'change_in_damage_{item.horizon}ply'
        block = block[np.isfinite(block[target])]
        assert len(block) == item.n and block.site.nunique() == item.games
        y, p = block[target].to_numpy(float), block.prediction.to_numpy(float)
        if item.task == 'reply':
            actual, score, decision = y.astype(bool), p, p >= .5
            close(np.mean((y-p)**2), item.brier)
        else:
            close(np.mean(np.abs(y-p)), item.mae)
            corr = spearmanr(y, p).statistic if np.ptp(y) and np.ptp(p) else np.nan
            close(corr, item.spearman)
            actual = 1 + y / block.initial_damage.to_numpy() < .75
            ratio = 1 + p / block.initial_damage.to_numpy()
            score, decision = -ratio, ratio < .75
            close(np.mean(np.abs(y[actual]-p[actual])), item.recovered_only_mae)
        tp, tn = int((actual & decision).sum()), int((~actual & ~decision).sum())
        fp, fn = int((~actual & decision).sum()), int((actual & ~decision).sum())
        assert [tn, fp, fn, tp] == [item.tn, item.fp, item.fn, item.tp]
        close((tn + tp) / len(actual), item.accuracy)
        close(tp / (tp+fp) if tp+fp else np.nan, item.precision)
        close(tp / (tp+fn), item.recall)
        close(tn / (tn+fp), item.other_class_recall)
        close(.5 * (tp/(tp+fn) + tn/(tn+fp)), item.balanced_accuracy)
        close(roc_auc_score(actual, score), item.roc_auc)
        extra.append(dict(task=item.task, horizon=item.horizon, model=item.model,
                          threshold=item.threshold,
                          f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
                          average_precision=average_precision_score(actual, score),
                          log_loss=log_loss(actual, p, labels=[False, True])
                          if item.task == 'reply' else np.nan))
        if item.task == 'reply' and item.threshold == 20:
            fpr, tpr, _ = roc_curve(actual, score, drop_intermediate=False)
            roc[item.model] = dict(fpr=fpr.tolist(), tpr=tpr.tolist(), auc=item.roc_auc)
            bins = np.minimum((p*10).astype(int), 9)
            assert np.all((p >= 0) & (p <= 1))
            for b in range(10):
                chosen = bins == b
                calibration.append(dict(model=item.model, bin=b, lower=b/10, upper=(b+1)/10,
                    n=int(chosen.sum()), mean_probability=float(p[chosen].mean()) if chosen.any() else None,
                    observed_preserving=float(actual[chosen].mean()) if chosen.any() else None))
        checked += 1
    tables['metrics'] = tables['metrics'].merge(pd.DataFrame(extra),
        on=['task', 'horizon', 'model', 'threshold'], validate='one_to_one')

    # This earlier run records boosting iterations, not changing training sizes.
    curve_manifest = load_json(CURVE / 'manifest.json')
    assert curve_manifest['source_archives'] == [
        'lichess_db_standard_rated_' + month + '.pgn.zst' for month in cfg['primary_archives']]
    assert curve_manifest['rows'] == 23167 and curve_manifest['games'] == 6512
    assert curve_manifest['learning_curves']['params'] == cfg['regression_params']
    assert curve_manifest['learning_curves']['features'] == cfg['features']['full']
    curve_summary = read(CURVE / 'training_validation_summary.csv')
    curve_folds = read(CURVE / 'training_validation_by_fold.csv')
    assert set(curve_summary.horizon) == {6}
    assert curve_summary.iteration.max() == 240
    for row in curve_summary.itertuples():
        fold = curve_folds[(curve_folds.iteration == row.iteration) & (curve_folds.split == row.split)]
        assert len(fold) == 3
        for metric in ['mae', 'accuracy_exploratory', 'no_change_mae', 'majority_accuracy']:
            close(np.average(fold[metric], weights=fold.n), getattr(row, metric))
    tables['boosting'] = curve_summary
    payload = dict(tables={k: records(v) for k, v in tables.items()},
        roc=roc, calibration=clean(calibration), config=cfg,
        metadata=dict(executed='2026-11-21', report_date='2026-11-22',
            execution_commit=provenance['execution_commit'],
            frozen_source_commit=provenance['frozen_source_commit'], versions=completion['versions'],
            predictions_sha256=completion['predictions_sha256'],
            verified_metric_rows=checked, intervals=72, subgroup_rows=612,
            curve_source=str(CURVE.relative_to(ROOT)),
            training_size_curve='Not available in saved artifacts; no new training performed',
            curve_data_sha256=curve_manifest['data_sha256']),
        sources=[dict(path=str(p.relative_to(ROOT)), sha256=h) for p, h in protected.items()])
    return clean(payload), protected


def notebook(payload):
    cells = []

    def md(source):
        cells.append(dict(cell_type='markdown', metadata={}, source=textwrap.dedent(source).strip()+'\n'))

    def code(source, hidden=False):
        cells.append(dict(cell_type='code', execution_count=None, outputs=[],
            metadata={'jupyter': {'source_hidden': True}} if hidden else {},
            source=textwrap.dedent(source).strip()+'\n'))

    md('''
    # Final results: immediate replies and later damage persistence

    The frozen human-move policy predicts whether an opponent's immediate reply preserves
    a mistake's damage. Predicting the change in damage after 2–10 plies produces much smaller
    gains over no-change. Subsequent moves by both players can change that damage; this
    predictive pattern does not establish a causal explanation.

    The March 2025 evaluation was completed once on **21 November 2026**. This notebook
    presents its saved results and compatible development validation results. All tables,
    ROC coordinates and calibration bins are embedded: running the notebook needs Python,
    NumPy, pandas, Matplotlib and IPython, but no data files, models or output directories.

    [Cohorts and targets](#1.-Cohorts-and-targets) · [Development versus test](#2.-Development-versus-test)
    · [Immediate reply](#3.-Immediate-opponent-reply) · [Later play](#4.-Later-damage-persistence)
    · [Diagnostics](#5.-Recovered-versus-persistent-diagnostics) · [Robustness](#6.-Robustness-and-subgroups)
    · [Development curves](#7.-Development-training-diagnostics) · [Provenance](#8.-Limitations-and-provenance)
    ''')
    code('''
    import json
    from io import BytesIO
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from IPython.display import HTML, Image, Markdown, display

    plt.rcParams.update({'figure.dpi': 110, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    COLORS = {'no_change': '#888888', 'training_mean': '#bd6f29', 'lightgbm': '#176b9b',
              'raw_maia': '#137d69', 'reply_control': '#bd6f29', 'reply_maia': '#6653a2',
              'training_majority': '#888888'}
    NAMES = {'no_change': 'No-change', 'training_mean': 'Training-mean',
             'training_majority': 'Training-majority / prevalence', 'raw_maia': 'Raw Maia-2',
             'reply_control': 'LightGBM: damage + opponent Elo',
             'reply_maia': 'LightGBM: damage + opponent Elo + Maia-2',
             'ten_control': 'LightGBM: F24', 'ten_maia': 'LightGBM: F24 + Maia-2',
             **{f'primary_{h}': 'Frozen LightGBM' for h in [2, 4, 6, 10]}}
    FIVE = '2017-04, 2017-06, 2017-07, 2018-02, 2019-02'
    FOUR = '2017-04, 2017-06, 2017-07, 2018-02'
    TEST = '2025-03 temporal holdout; completed one-time frozen evaluation'
    DEV = 'Game-grouped development out-of-fold validation; exploratory'

    def figure(fig):
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=145, bbox_inches='tight', facecolor='white')
        display(Image(data=buffer.getvalue()))
        plt.close(fig)

    def table(frame, title, archives, protocol=TEST, collapsed=False):
        note = f'<b>{title}</b><br>Training/reference archives: {archives}<br>Protocol: {protocol}'
        body = frame.to_html(index=False, na_rep='—', float_format=lambda x: f'{x:,.6f}', border=0)
        body = '<div style="overflow-x:auto; max-width:100%">'+body+'</div>'
        if collapsed:
            body = '<details><summary>Show complete table</summary>'+body+'</details>'
        display(HTML('<p>'+note+'</p>'+body))

    def labelled(frame):
        result = frame.copy()
        if 'model' in result:
            result['model'] = result.model.map(NAMES)
        return result.rename(columns={'horizon': 'Plies', 'model': 'Method', 'threshold': 'Damage > (pp)',
            'n': 'Rows', 'games': 'Games', 'mae': 'MAE (pp)', 'spearman': 'Spearman',
            'accuracy': 'Accuracy', 'balanced_accuracy': 'Balanced accuracy', 'roc_auc': 'ROC-AUC',
            'precision': 'Precision (+)', 'recall': 'Recall (+)', 'other_class_recall': 'Recall (−)',
            'brier': 'Brier loss', 'log_loss': 'Log loss', 'f1': 'F1 (+)',
            'average_precision': 'Average precision (+)', 'majority_accuracy': 'Majority accuracy',
            'recovered_only_mae': 'Recovered-only MAE (diagnostic)'})
    ''')
    code('# Embedded result snapshot; no file reads or writes.\nDATA = json.loads(r\'\'\'\n' +
         json.dumps(payload, ensure_ascii=True, separators=(',', ':'), allow_nan=False) +
         '\n\'\'\')\nT = {name: pd.DataFrame(rows) for name, rows in DATA["tables"].items()}\n'
         'M = T["metrics"]\nPRIMARY = M[(M.task == "primary") & (M.threshold == 20)]\n'
         'REPLY = M[(M.task == "reply") & (M.threshold == 20)]\n'
         'TEN = M[(M.task == "ten") & (M.threshold == 20)]\n'
         'CI = T["intervals"]\n', hidden=True)

    md('''
    ## 1. Cohorts and targets

    Primary models used all **five development months**. Secondary comparisons used the
    **four earlier months**, excluding February 2019. Raw Maia-2 is externally pretrained;
    the four months define its local reference cohort, not its policy training data.
    Development preprocessing was fitted inside training folds. Final models were fitted
    on their complete designated development cohorts before test access.

    Eligibility: `10 <= win_before <= 90` and `initial_damage >20` winning-chance percentage
    points (pp). The supplied March extract is a partial cohort, not the entire archive.

    - **Regression:** `change_in_damage_h = win_after - win_after_h`, from the original
      mistake-maker's perspective. Negative means recovery; positive means further damage.
      MAE is the reported regression loss; Spearman measures ranking.
    - **Immediate reply:** positive means the actual reply retains at least 75% of initial
      damage. Decisions use probability **≥0.5**. Missing replies are excluded.
    - **Later recovery diagnostic:** positive means ratio **<0.75**, where
      `ratio = 1 + change_in_damage / initial_damage`. This is an exploratory label derived
      from regression, not a separately trained classifier. ROC-AUC uses **negative predicted
      ratio**, rather than thresholded predictions.

    Accuracy, recalls, precision, F1 and Brier are shown on the 0–1 scale. Log loss uses
    natural logarithms. Brier and log loss require probabilities; they do not apply to
    regression ranking scores. Constant-baseline Spearman is undefined, shown as —.
    ''')
    code('''
    cohorts = pd.DataFrame([
        ['Primary development', FIVE, 23167, 6512, DEV],
        ['Secondary 10-ply development', FOUR, 15953, 4485, DEV],
        ['Immediate-reply development', FOUR, 15851, 4472, DEV],
        ['Final regression', '2025-03 supplied extract', 3380, 955, TEST],
        ['Final immediate reply', '2025-03 supplied extract', 3356, 951, TEST],
    ], columns=['Use', 'Archive set', 'Rows', 'Games', 'Validation protocol'])
    table(cohorts, 'Cohorts', 'Five or four development months, as listed',
          'Development validation and final temporal holdout, as listed')
    display(Markdown('Development excludes 510 overlapping mistakes. Immediate scoring excludes '
                     '**102 development** and **24 test** rows with missing replies.'))
    ''')
    md('''
    ## 2. Development versus test

    These comparisons use compatible saved **held-out development predictions**, not
    in-sample training scores. The final model fit uses more data than each development
    fold fit, and the cohorts differ by year, ratings and time controls. Differences here
    are descriptive; they are not paired estimates of a generalization gap.
    ''')
    code('''
    dp = T['development_primary'].copy()
    dp['horizon'] = pd.to_numeric(dp.target)
    dp['method'] = dp.model.replace({'fixed_lightgbm': 'Frozen LightGBM',
                                    'no_change': 'No-change', 'training_mean': 'Training-mean'})
    comparison = []
    for d in dp.itertuples():
        key = f'primary_{d.horizon}' if d.model == 'fixed_lightgbm' else d.model
        t = PRIMARY[(PRIMARY.horizon == d.horizon) & (PRIMARY.model == key)].iloc[0]
        comparison.append({'Plies': d.horizon, 'Method': d.method, 'Development MAE': d.mae,
                           'Test MAE': t.mae, 'Development Spearman': d.spearman,
                           'Test Spearman': t.spearman})
    table(pd.DataFrame(comparison), 'Primary development versus test', FIVE, DEV+' versus '+TEST)
    dr = T['development_reply'].copy()
    method_map = {'raw_maia': 'raw_maia', 'simple_lightgbm': 'reply_control',
                  'simple_plus_maia_lightgbm': 'reply_maia',
                  'training_majority_prevalence': 'training_majority'}
    reply_comparison = []
    for d in dr.itertuples():
        model = method_map[d.method]
        t = REPLY[REPLY.model == model].iloc[0]
        for metric in ['accuracy', 'balanced_accuracy', 'roc_auc', 'brier', 'log_loss']:
            reply_comparison.append({'Method': NAMES[model], 'Metric': metric,
                'Development': getattr(d, metric), 'Test': t[metric]})
    table(pd.DataFrame(reply_comparison), 'Immediate reply: development versus test', FOUR,
          DEV+' versus '+TEST)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for cohort, frame, style in [('Development OOF', dp[dp.model == 'fixed_lightgbm'], '--'),
                                ('March test', PRIMARY[PRIMARY.model.str.startswith('primary_')], '-')]:
        frame = frame.sort_values('horizon')
        axes[0].plot(frame.horizon, frame.mae, style+'o', label=cohort)
        axes[1].plot(frame.horizon, frame.spearman, style+'o', label=cohort)
    axes[0].set(title='Primary MAE · five archives', ylabel='MAE (pp)', xlabel='Subsequent plies')
    axes[1].set(title='Primary ranking · five archives', ylabel='Spearman', xlabel='Subsequent plies')
    for ax in axes[:2]:
        ax.set_xticks([2, 4, 6, 10]); ax.grid(alpha=.2); ax.legend()
    x = np.arange(3)
    plot_methods = ['raw_maia', 'reply_control', 'reply_maia']
    dev_auc = [dr.loc[dr.method == next(k for k, v in method_map.items() if v == m), 'roc_auc'].iloc[0]
               for m in plot_methods]
    test_auc = [REPLY.loc[REPLY.model == m, 'roc_auc'].iloc[0] for m in plot_methods]
    axes[2].bar(x-.18, dev_auc, .36, label='Development OOF', color='#83a8bb')
    axes[2].bar(x+.18, test_auc, .36, label='March test', color='#176b9b')
    axes[2].set(xticks=x, xticklabels=['Raw Maia-2', 'Damage/Elo', '+ Maia-2'], ylim=(0, 1),
                ylabel='ROC-AUC', title='Immediate reply · four archives')
    axes[2].axhline(.5, color='gray', ls=':', lw=1); axes[2].legend()
    fig.suptitle('Held-out development results and the frozen March 2025 test', y=1.04)
    fig.tight_layout(); figure(fig)
    ''')
    code('''
    distribution = T['distributions']
    cohort_names = {'development_five': 'Five development months',
                    'development_four': 'Four development months', 'final_temporal': 'March test'}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    definitions = [('player_elo_band', ['<1600', '1600–<2000', '>=2000'], 'Mistake-maker Elo'),
                   ('opponent_elo_band', ['<1600', '1600–<2000', '>=2000'], 'Reply-maker Elo'),
                   ('speed_label', ['UltraBullet', 'Bullet', 'Blitz', 'Rapid', 'Classical'], 'Time control')]
    for ax, (variable, preferred, title) in zip(axes, definitions):
        frame = distribution[distribution.variable == variable]
        groups = [g for g in preferred if g in set(frame.group)]
        groups += sorted(set(frame.group)-set(groups))
        for i, (cohort, label) in enumerate(cohort_names.items()):
            part = frame[frame.cohort == cohort].set_index('group').reindex(groups)
            ax.bar(np.arange(len(groups))+(i-1)*.25, part.share.fillna(0)*100, .25, label=label)
        ax.set(title=title, ylabel='Share of eligible mistake rows (%)',
               xticks=np.arange(len(groups)), xticklabels=groups)
        ax.tick_params(axis='x', rotation=25); ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Development versus test composition · descriptive cohorts', y=1.04)
    fig.tight_layout(); figure(fig)
    table(distribution, 'All rating, speed, phase and exact time-control distributions',
          FIVE+'; '+FOUR+'; March 2025', 'Descriptive eligible-row cohorts; no model validation', collapsed=True)
    ''')
    md('''
    ## 3. Immediate opponent reply

    The four methods use identical **3,356 replies / 951 games**. The control uses initial
    damage and opponent Elo; its augmented version adds only the frozen Maia probability.
    Raw Maia-2 was not locally fitted. ROC curves use continuous scores. The training
    majority probability stays fixed at **0.5581351334300675**.
    ''')
    code('''
    columns = ['model', 'n', 'games', 'accuracy', 'balanced_accuracy', 'roc_auc', 'brier',
               'log_loss', 'precision', 'recall', 'other_class_recall', 'f1', 'average_precision']
    table(labelled(REPLY[columns]), 'Immediate reply · positive = preserving ≥75% damage', FOUR)
    table(labelled(REPLY[['model', 'tn', 'fp', 'fn', 'tp']]),
          'Confusion counts · negative = non-preserving; positive = preserving', FOUR)
    reply_ci = CI[(CI.task == 'reply') & (CI.threshold == 20)].copy()
    reply_ci['model'] = reply_ci.model.map(NAMES)
    reply_ci['baseline'] = reply_ci.baseline.map(NAMES)
    table(reply_ci[['model', 'baseline', 'metric', 'estimate', 'lower95', 'upper95', 'n', 'games']],
          'Paired immediate-reply improvements · positive favors Maia', FOUR)

    fig, ax = plt.subplots(figsize=(7, 5.3))
    for model in ['raw_maia', 'reply_control', 'reply_maia', 'training_majority']:
        curve = DATA['roc'][model]
        ax.plot(curve['fpr'], curve['tpr'], color=COLORS[model], lw=2,
                label=f"{NAMES[model]} (AUC {curve['auc']:.4f})")
    ax.set(xlabel='False-positive rate · non-preserving replies',
           ylabel='True-positive rate · preserving replies', xlim=(0, 1), ylim=(0, 1.01),
           title='Immediate-reply ROC · March 2025 test\\nFour-archive local reference; 3,356 replies / 951 games')
    ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=.2); figure(fig)
    ''')
    md('''
    ### Probability calibration

    Reliability uses **10 fixed equal-width bins** on [0,1]. A point compares the mean saved
    probability with the observed preserving frequency in its bin; marker area reflects
    bin size. The diagonal indicates agreement. Empty bins are omitted. These are descriptive
    test summaries with no recalibration, threshold selection or fitting. Small bins are
    unstable, and the plot has no game-cluster uncertainty bands.
    ''')
    code('''
    calibration = pd.DataFrame(DATA['calibration'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect agreement')
    for model in ['raw_maia', 'reply_control', 'reply_maia']:
        points = calibration[(calibration.model == model) & (calibration.n > 0)]
        axes[0].plot(points.mean_probability, points.observed_preserving, color=COLORS[model],
                     lw=1.5, label=NAMES[model])
        axes[0].scatter(points.mean_probability, points.observed_preserving,
                        s=15+100*points.n/3356, color=COLORS[model])
        axes[1].step((points.lower+points.upper)/2, points.n, where='mid',
                    color=COLORS[model], label=NAMES[model])
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted preserving probability',
                ylabel='Observed preserving frequency', title='Probability reliability')
    axes[1].set(xlim=(0, 1), xlabel='Probability bin midpoint', ylabel='Replies per bin',
                title='Probability distribution')
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('March 2025 immediate replies · four-archive local reference', y=1.04)
    fig.tight_layout(); figure(fig)
    table(calibration, 'Calibration bin counts and means', FOUR, collapsed=True)
    ''')
    md('''
    ## 4. Later damage persistence

    ### Primary regression: five development archives

    All horizons use **3,380 mistakes / 955 games** in the final test. No-change predicts zero;
    training-mean uses a constant learned only from development. The frozen LightGBM loss
    is mean absolute error. Lower MAE is better; higher Spearman is better.
    ''')
    code('''
    table(labelled(PRIMARY[['horizon', 'model', 'n', 'games', 'mae', 'spearman', 'recovered_only_mae']]),
          'Primary regression · MAE loss and ranking', FIVE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for model, label in [('no_change', 'No-change'), ('training_mean', 'Training-mean'),
                          ('lightgbm', 'Frozen LightGBM')]:
        part = PRIMARY[PRIMARY.model.str.startswith('primary_') if model == 'lightgbm'
                       else PRIMARY.model == model].sort_values('horizon')
        axes[0].plot(part.horizon, part.mae, 'o-', color=COLORS[model], label=label)
    axes[0].set(title='Regression loss across horizons', ylabel='MAE (pp)', xlabel='Subsequent plies')
    axes[0].legend()
    part = PRIMARY[PRIMARY.model.str.startswith('primary_')].sort_values('horizon')
    axes[1].plot(part.horizon, part.spearman, 'o-', color=COLORS['lightgbm'])
    axes[1].set(title='Weak ranking at every horizon', ylabel='Spearman', xlabel='Subsequent plies',
                ylim=(0, max(.2, part.spearman.max()*1.2)))
    for ax in axes: ax.set_xticks([2, 4, 6, 10]); ax.grid(alpha=.2)
    fig.suptitle('March 2025 temporal test · five-archive frozen training', y=1.04)
    fig.tight_layout(); figure(fig)
    primary_ci = CI[(CI.task == 'primary') & (CI.threshold == 20)]
    table(primary_ci[['horizon', 'baseline', 'metric', 'estimate', 'lower95', 'upper95']],
          'Primary regression improvements · positive favors LightGBM', FIVE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, metric, xlabel in zip(axes, ['mae_improvement_pp', 'mae_improvement_percent'],
                                ['MAE reduction (pp)', 'Relative MAE reduction (%)']):
        for offset, baseline, color in [(-.10, 'no_change', '#176b9b'), (.10, 'training_mean', '#bd6f29')]:
            part = primary_ci[(primary_ci.metric == metric) &
                              (primary_ci.baseline == baseline)].sort_values('horizon')
            errors = np.array([part.estimate-part.lower95, part.upper95-part.estimate])
            ax.errorbar(part.estimate, np.arange(4)+offset, xerr=errors, fmt='o', capsize=4,
                        color=color, label='Versus '+NAMES[baseline])
        ax.axvline(0, color='gray', ls='--'); ax.set_yticks(np.arange(4), ['2', '4', '6', '10'])
        ax.set(xlabel=xlabel, ylabel='Subsequent plies'); ax.invert_yaxis(); ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Primary MAE improvements · paired game-cluster 95% intervals\\n'
                 'March 2025 test; five-archive training', y=1.08)
    fig.tight_layout(); figure(fig)
    ''')
    md('''
    Against no-change, the 2- and 4-ply intervals include zero. Gains at 6 and 10 plies
    are positive but modest. The larger gains against training-mean do not imply strong
    prediction. Intervals use **2,000 paired whole-game resamples, seed 1701**, with fixed
    predictions and no bootstrap refitting or multiplicity adjustment.

    ### Matched 10-ply comparison: four development archives

    These models use identical F24 predictors, settings and training rows; one adds only
    the frozen Maia probability. Their difference isolates that input within this specified
    model comparison. Comparing either with the five-archive primary model does not isolate
    Maia, because the training cohorts differ.
    ''')
    code('''
    table(labelled(TEN[['model', 'n', 'games', 'mae', 'spearman', 'recovered_only_mae']]),
          'Matched 10-ply test regression', FOUR)
    ten_ci = CI[(CI.task == 'ten') & (CI.threshold == 20)]
    table(ten_ci[['metric', 'estimate', 'lower95', 'upper95', 'n', 'games']],
          'Matched 10-ply Maia increment · positive favors adding Maia', FOUR)
    rows = []
    for d in T['development_ten'].itertuples():
        model = 'ten_control' if d.variant == 'fixed_lightgbm' else 'ten_maia'
        t = TEN[TEN.model == model].iloc[0]
        rows.append({'Method': NAMES[model], 'Development MAE': d.mae, 'Test MAE': t.mae,
                     'Development Spearman': d.spearman, 'Test Spearman': t.spearman})
    table(pd.DataFrame(rows), 'Matched 10-ply development versus test', FOUR, DEV+' versus '+TEST)
    ''')
    md('''
    ## 5. Recovered versus persistent diagnostics

    Positive means **recovered (<0.75 ratio)**; negative means persistent (≥0.75).
    These labels are derived from regression, not separate classifiers. Accuracy close to
    majority accuracy can coexist with very low recovered recall. AUC and average precision
    use continuous negative predicted ratio; this score is not a probability.
    ''')
    code('''
    classification = ['horizon', 'model', 'accuracy', 'majority_accuracy', 'balanced_accuracy',
                      'roc_auc', 'precision', 'recall', 'other_class_recall', 'f1', 'average_precision']
    table(labelled(PRIMARY[classification]), 'Primary regression-derived recovery · all horizons and baselines', FIVE)
    table(labelled(PRIMARY[['horizon', 'model', 'tn', 'fp', 'fn', 'tp']]),
          'Primary recovery confusion counts · positive = recovered', FIVE, collapsed=True)
    table(labelled(TEN[classification]), 'Matched 10-ply regression-derived recovery', FOUR)
    ''')
    md('''
    ## 6. Robustness and subgroups

    The >25 and >30 damage subsets reuse the same frozen >20 predictions, with no refits.
    All saved rows are retained below, including unfavorable results. Subgroup estimates
    describe this cohort and are not causal effects or independent confirmations.

    **UltraBullet is a small-sample limitation:** the test cohort contains only 40 mistakes
    from 9 games. Classical also has only 2 mistakes from 1 game. These groups do not
    support reliable performance findings, and some speeds are outside the Maia domain.
    ''')
    code('''
    for task, title, archives in [('primary', 'Primary regression', FIVE),
                                  ('reply', 'Immediate reply', FOUR), ('ten', 'Matched 10-ply', FOUR)]:
        frame = M[(M.task == task) & (M.threshold > 20)]
        table(labelled(frame), title+' · complete >25/>30 robustness metrics', archives, collapsed=True)
    table(CI, 'All 72 paired intervals · >20/>25/>30',
          'Five months for primary; four months for immediate and matched 10-ply', collapsed=True)
    for task, title, archives in [('primary', 'Primary regression', FIVE),
                                  ('reply', 'Immediate reply', FOUR), ('ten', 'Matched 10-ply', FOUR)]:
        table(labelled(T['subgroups'][T['subgroups'].task == task]),
              title+' · all saved Elo, speed and phase subgroups', archives, collapsed=True)
    ''')
    md('''
    ## 7. Development training diagnostics

    ### Saved LightGBM boosting-iteration curve

    This **earlier development-only** run uses the same F24 feature list and fixed 240-tree
    settings as the primary regressor, at **6 plies only**. It uses three game folds across
    all five development months. Lines are mistake-weighted training and held-out fold
    averages. Training occurrences repeat across folds and are not unique sample counts.
    This is a historical diagnostic, not a test-set iteration sweep or a new model selection.
    ''')
    code('''
    boosting = T['boosting']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for split, label, color in [('training', 'Training folds', '#bd6f29'),
                                 ('validation', 'Held-out development folds', '#176b9b')]:
        part = boosting[boosting.split == split].sort_values('iteration')
        axes[0].plot(part.iteration, part.mae, color=color, label=label)
        axes[1].plot(part.iteration, part.accuracy_exploratory, color=color, label=label)
    baseline = boosting[boosting.split == 'validation'].iloc[0]
    axes[0].axhline(baseline.no_change_mae, color='gray', ls='--', label='No-change')
    axes[1].axhline(baseline.majority_accuracy, color='gray', ls='--', label='Training-majority')
    axes[0].set(ylabel='MAE loss (pp)', title='Regression loss')
    axes[1].set(ylabel='Accuracy · exploratory recovered label', title='Regression-derived accuracy')
    for ax in axes: ax.set_xlabel('Boosting iteration'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Earlier 6-ply LightGBM development diagnostic · five archives', y=1.04)
    fig.tight_layout(); figure(fig)
    table(boosting, 'Saved boosting-iteration metrics · not the final test', FIVE,
          'Earlier three-fold game-grouped development; training and held-out folds', collapsed=True)
    ''')
    md('''
    ### Training-size learning curve

    **Not available in the saved artifacts.** The historical files named “learning curves”
    vary boosting iteration, not the number of training games. No training-size curve is
    fabricated or fitted for this report. The final results show no obvious performance
    collapse relative to development, but do not by themselves diagnose underfitting or
    rule out overfitting.
    ''')
    md('''
    ## 8. Limitations and provenance

    - The supplied March sample is not a representative random sample of the whole month.
      Original full-PGN sampling ranges, upstream settings and core Stockfish binary identity
      were not independently logged. Newly generated features and immediate labels used
      the frozen Stockfish 19 depth-18 procedure; earlier core/horizon values were preserved.
    - Core targets use the original ±1000 cp mate conversion, while the legacy MultiPV
      convention uses ±100000 cp. These frozen conventions were retained.
    - Maia uses a fixed 97.5% policy-mass prefix and midpoint residual bound, without
      renormalization. Numerical integration bounds are not behavioral confidence intervals.
      Documented Maia training ends in November 2023; individual membership is unaudited.
    - Post-end targets follow the frozen PGN-result rule. Results alone cannot distinguish
      abandonment or disconnection. Final validation is temporal, not player-disjoint.
    - Confidence intervals condition on fitted predictions and cluster by game, not player;
      they are not adjusted for multiple comparisons. No new intervals were invented for
      metrics lacking saved intervals. Calibration and added F1/AP/log-loss summaries are
      descriptive reporting from saved predictions, not extra model-selection criteria.

    The four frozen comparisons, all robustness thresholds and saved subgroups appear here.
    Separate horizon classifiers, tuned cutoffs, continuous Maia-weighted features and
    clock/agreement variants remain development-only in the development notebook.

    Further detail: [final report](final_evaluation/march_2025_once_v1/REPORT.md),
    [complete supplement](final_evaluation/march_2025_once_v1/SUPPLEMENT.md),
    [methods](METHODOLOGY.md),
    [frozen evaluation plan](MARCH_2025_EVALUATION_PLAN.md),
    [development notebook](chess_development_results.ipynb).
    These links are optional references; notebook execution does not open them.
    ''')
    code('''
    meta = DATA['metadata']
    display(Markdown(f"**Evaluation commit:** `{meta['execution_commit']}`\\n\\n"
                     f"**Frozen source/configuration commit:** `{meta['frozen_source_commit']}`\\n\\n"
                     f"**Saved prediction SHA-256:** `{meta['predictions_sha256']}`"))
    versions = pd.DataFrame(list(meta['versions']['packages'].items()), columns=['Package', 'Evaluation version'])
    display(HTML(versions.to_html(index=False, border=0)))
    table(pd.DataFrame(DATA['sources']), 'Embedded snapshot source hashes',
          'Development and completed March evaluation artifacts', 'Reporting provenance only', collapsed=True)
    assert len(M) == 54 and len(CI) == 72 and len(T['subgroups']) == 612
    assert CI.valid_replicates.eq(2000).all()
    assert REPLY.n.eq(3356).all() and PRIMARY.n.eq(3380).all() and TEN.n.eq(3380).all()
    for row in M.itertuples():
        assert row.tn + row.fp + row.fn + row.tp == row.n
        assert abs((row.tn + row.tp)/row.n - row.accuracy) < 1e-12
    for model in REPLY.model:
        counts = pd.DataFrame(DATA['calibration'])
        assert counts.loc[counts.model == model, 'n'].sum() == 3356
        curve = DATA['roc'][model]
        auc = np.trapezoid(curve['tpr'], curve['fpr'])
        assert abs(auc - REPLY.loc[REPLY.model == model, 'roc_auc'].iloc[0]) < 1e-12
    display(Markdown('**Snapshot checks passed:** 54 metric rows, 72 paired intervals, '
                     '612 subgroup rows; ROC areas, calibration counts and confusion totals agree. '
                     'No model inference or training is performed by this notebook.'))
    ''')
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.9.6'), final_results=dict(
            execution_commit=payload['metadata']['execution_commit'], self_contained=True,
            final_evaluation_status='completed', completed_on=payload['metadata']['executed'],
            frozen_source_commit=payload['metadata']['frozen_source_commit'],
            evaluation_rerun=False, source='Saved completed evaluation and development summaries'),
        project_status=dict(updated_on=payload['metadata']['report_date'],
            final_evaluation_status='completed', final_evaluation_completed_on=payload['metadata']['executed'],
            frozen_source_commit=payload['metadata']['frozen_source_commit'],
            execution_commit=payload['metadata']['execution_commit'],
            completion_record='final_evaluation/march_2025_once_v1/completion.json',
            execution_record='final_evaluation/march_2025_once_v1/run_provenance.json')),
        nbformat=4, nbformat_minor=5)


def execute(nb):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    shell = InteractiveShell.instance()
    count = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        count += 1
        with capture_output() as captured:
            result = shell.run_cell(cell['source'], store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Notebook cell {count} failed:\n{captured.stdout}\n{captured.stderr}')
        outputs = []
        for name, content in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if content:
                outputs.append(dict(output_type='stream', name=name, text=content))
        for rich in captured.outputs:
            outputs.append(dict(output_type='display_data', data=rich.data, metadata=rich.metadata))
        cell['execution_count'], cell['outputs'] = count, outputs
        print(f'Executed reporting cell {count}', flush=True)
    nb['metadata']['execution'] = dict(status='complete', code_cells=count,
        method='Sequential IPython; saved embedded aggregates only; no inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'chess_final_results.ipynb')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Refusing to overwrite existing notebook: '+str(args.out))
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/chess-final-results-mpl')
    payload, protected = collect()
    nb = notebook(payload)
    # Stable cell IDs support notebook format 4.5 without external nbformat tooling.
    for i, cell in enumerate(nb['cells']):
        cell['id'] = f'final-results-{i:02d}'
    if args.execute:
        execute(nb)
    for path, expected in protected.items():
        assert sha(path) == expected, 'Source changed during reporting: '+str(path)
    with args.out.open('x') as handle:
        json.dump(nb, handle, indent=1, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(f'Saved {args.out}; all {len(protected)} source artifacts unchanged.')


if __name__ == '__main__':
    main()
