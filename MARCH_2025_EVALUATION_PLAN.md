# March 2025 evaluation plan

**Updated 2026-11-21.** Approved models are fitted. Feature/label preparation and structural checks are complete. Final evaluation awaits the user-created freeze commit and explicit authorization.

## Data and target

Train on **2017-04, 2017-06, 2017-07, 2018-02 and 2019-02** for the primary comparison. The three secondary comparisons use the **four earlier archives, excluding 2019-02**. Fit all preprocessing on the corresponding development training data. Evaluate all listed comparisons together once on the sealed **2025-03** archive, with identical eligible test rows within each comparison.

The test cohort is **3,380 eligible mistakes / 955 games** from the five supplied CSVs of the March 2025 partial extract. Freeze their hashes and prepared tables; perform no new sampling. The original extraction binary and full-PGN sampling are unrecorded limitations, detailed in the appendix.

Keep `10 <= win_before <= 90`, `initial_damage >20` percentage points and Stockfish **depth 18**. Primary target: `change_in_damage_h = win_after - win_after_h`, at **2/4/6/10 subsequent plies**, always from the original mistake-maker's perspective. Retain the existing terminal-target rules. Missing actual replies are excluded from immediate-reply scoring.

## Only final comparisons

| Comparison | Models / predictors | Metrics |
| --- | --- | --- |
| **Primary: persistence at 2/4/6/10 plies** | Original frozen **LightGBM**, existing 24 features including MultiPV, versus **no-change** and **training-mean** baselines; five archives | MAE, Spearman, absolute/percent MAE improvement and paired 95% intervals |
| **Secondary: immediate reply** | **Raw Maia-2** versus **training-majority/prevalence** baseline; fixed cutoff **0.5**, four-archive training prevalence | Accuracy, balanced accuracy, ROC-AUC, Brier, both class recalls |
| **Secondary: immediate Maia increment** | **LightGBM with initial damage + opponent Elo**, versus the same model **+ Maia probability**; four archives, cutoff **0.5** | Same immediate metrics; paired Brier improvement |
| **Secondary: ten-ply Maia increment** | **LightGBM with existing 24 features**, versus the same model **+ Maia probability**; both trained on four archives | MAE, Spearman and paired MAE improvement |

No additional model, feature variant or tuned cutoff enters this final evaluation.

## Maia-2 and reporting

Use **Maia-2 0.11.0**, the recorded Blitz/Rapid checkpoints, existing rating/model routing and **97.5% policy-mass midpoint** calculation. Preserve its pre-reply inputs and settings. The immediate event is retention of **at least 75%** of initial damage. The [official training documentation](https://github.com/CSSLab/maia2#training) specifies **2018-05–2023-11**; March 2025 is outside that documented range. Individual training membership is not independently audited.

ROC-AUC uses continuous scores. Use **2,000 paired whole-game bootstrap samples**, seed **1701**. Keep required regression-derived recovery diagnostics, >25/>30 robustness and subgroup reporting as specified in the appendix. After freezing, report the test rating/time-control distributions alongside development data. Report every prespecified result; treat UltraBullet as a small-sample limitation.

## Before the one-time run

Eight model/preprocessing pipelines are fitted on their full declared development cohorts. Their hashes, learned preprocessing and training-only baselines are in [the fitted-model record](freeze/fitted_models.json); source and environment versions are in [the code manifest](freeze/code_manifest.json). Validation reproduced all eight saved model predictions, passed 15 synthetic tests, and exactly replayed 64 Maia policy positions.

[The input protocol](freeze/final_input_protocol.json) freezes exact artifact reuse, selection, enrichment and label construction. The user confirmed unchanged extractor filters; preparation applies the stricter >20 rule. [The readiness audit](freeze/test_readiness_audit.json) verifies schemas, alignment, target equations, missing replies, feature bounds, game overlap, versions and hashes. Record the user-created freeze commit, then obtain explicit authorization for the one-time evaluation. [Technical appendix](MARCH_2025_EVALUATION_APPENDIX.md). **No fitted research-model predictions, final test or commit was run.**
