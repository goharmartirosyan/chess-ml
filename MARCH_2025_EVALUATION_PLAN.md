## Data and target

Train on **2017-04, 2017-06, 2017-07, 2018-02 and 2019-02** for the primary comparison. The three secondary comparisons use the **four earlier archives, excluding 2019-02**. Fit all preprocessing on the corresponding development training data. Evaluate all listed comparisons together once on the sealed **2025-03** archive, with identical eligible test rows within each comparison.

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

Fit and hash only the listed models/preprocessing; lock the exact test-input/extraction protocol and evaluator. Current repository reference: `0c0185aa2e21daa12e1d89f882f29db0f5ec6a02`; untracked code is identified by appendix hashes. The final evaluator and fitted-artifact versions remain pending. Date/record the completed freeze, then obtain the user's explicit authorization for the one-time evaluation. **This revision changes documentation only; March 2025 remains unopened.**
