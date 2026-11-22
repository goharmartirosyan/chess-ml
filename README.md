# Predicting Human Exploitation of Chess Mistakes Using Machine Learning

> To what extent can machine learning predict whether a human chess player will capitalize on an opponent’s serious mistake, both immediately and over subsequent moves?

This project studies human play after Stockfish identifies a serious mistake: a loss of more than 20 winning-chance percentage points, with the pre-mistake winning chance between 10% and 90%. Immediate exploitation is measured by whether the actual opponent reply preserves at least 75% of that damage. Longer-term persistence is measured by the change in damage after 2, 4, 6 and 10 plies (half-moves), from the original mistake-maker’s perspective.

## Main models and results

The final evaluation was **completed once on 21 November 2026**, using a separate March 2025 sample after models, features and thresholds were frozen. The [execution record](final_evaluation/run_provenance.json) documents completion; status flags in `freeze/` describe the historical pre-test stage.

| Model or comparison | What it predicts | Final-test result |
| --- | --- | --- |
| **Primary LightGBM regressors** — five development archives | Change in damage at 2, 4, 6 and 10 plies, using engine, rating, clock, position and MultiPV features | MAE reductions over no-change: **0.21%, 0.93%, 1.54% and 2.36%**, respectively. Spearman correlations: **0.096–0.140**. Training-mean is also reported as a baseline. |
| **Raw Maia-2 probability** — externally pretrained human-move policy | Probability that the immediate reply preserves at least 75% of initial damage, with a fixed 0.5 classification cutoff | **73.69% accuracy**, versus **55.07%** for the training-majority baseline; **72.87% balanced accuracy**, **0.8102 ROC-AUC**, **0.179704 Brier score**. |
| **Immediate-reply LightGBM, with and without Maia-2** — four development archives | The same immediate event, using initial damage and opponent Elo, then adding only the Maia-2 probability | Accuracy increased from **56.70% to 73.39%**; Brier score improved from **0.243971 to 0.176765**. |
| **10-ply LightGBM, with and without Maia-2** — four development archives | Change in damage at 10 plies, using the same 24 baseline features, then adding only Maia-2 | MAE decreased from **24.0661 to 23.6892** percentage points; Spearman increased from **0.0784 to 0.1464**. |

Immediate replies are meaningfully predictable, while longer-term damage changes remain weakly predictable. Primary improvements at 2 and 4 plies have confidence intervals containing zero; improvements at 6 and 10 plies are positive but small. These are predictive associations, not causal explanations. UltraBullet results are a small-sample limitation, not a finding. Complete metrics, paired game-cluster confidence intervals and limitations are in the [final report](final_evaluation/REPORT.md).

## Two results notebooks

- [chess_development_results.ipynb](chess_development_results.ipynb): development validation, baseline and feature comparisons, and exploratory analyses. Validation groups mistakes by game; player-disjoint checks are reported separately.
- [chess_final_results.ipynb](chess_final_results.ipynb): the completed March 2025 test results, development-versus-test comparisons, metrics, confidence intervals, ROC and calibration plots. Both results notebooks contain embedded tables and can run without external datasets. The final notebook does not rerun model predictions.

## Datasets

The **five development archives** are April, June and July 2017, February 2018 and February 2019. The **four earlier archives** exclude February 2019 and are used for the local Maia-2 comparisons. Raw Maia-2 itself was not trained on these local CSVs.

The [exported datasets](dataset_exports/) are:

| CSV | Rows / games | Used for |
| --- | ---: | --- |
| [train_primary_five_archives.csv](dataset_exports/train_primary_five_archives.csv) | 23,167 / 6,512 | All four primary LightGBM regressors and their no-change/training-mean baseline reference cohort. |
| [train_ten_ply_four_archives.csv](dataset_exports/train_ten_ply_four_archives.csv) | 15,953 / 4,485 | Matched 10-ply LightGBM comparison, with and without the Maia-2 probability. |
| [train_immediate_reply_four_archives.csv](dataset_exports/train_immediate_reply_four_archives.csv) | 15,851 / 4,472 | Immediate-reply LightGBM comparison, with and without Maia-2, and the majority/prevalence reference. Excludes 102 missing development replies. |
| [test_features.csv](dataset_exports/test_features.csv) | 3,380 / 955 | Frozen March 2025 inputs for all final comparisons. |


Training CSVs contain identifiers, approved predictors and targets. The [frozen model configuration](freeze/approved_models.json) specifies the predictors; target and identifier columns are not model inputs.

## Code and execution records

Final training and evaluation commands are documented in [notebook_runs/COMMANDS.md](notebook_runs/COMMANDS.md).

- [scripts/frozen_chess_evaluation.py](scripts/frozen_chess_evaluation.py): fits the eight final LightGBM pipelines and evaluates the four frozen comparisons.
- [scripts/build_final_results_notebook.py](scripts/build_final_results_notebook.py): builds the final notebook from saved results.

[Model settings](freeze/approved_models.json), [dependency versions](freeze/requirements-modeling.txt) and [important SHA-256 hashes](freeze/final_hashes.json) identify the final implementation and artifacts. The hash record contains no fitted weights. [METHODOLOGY.md](METHODOLOGY.md) describes the methods; the results notebooks contain the development and final-test evidence.
