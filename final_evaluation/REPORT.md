# March 2025 final evaluation

Frozen source/configuration commit: `bbe496325c83b8695a52cb1b93896ed4e8a34768`. Execution commit: `729c8abd011c9dd2868c2fea2cc514f69823a474`. The latter contains the same evaluation source/configuration and the committed pre-test verification records.

## Main finding

Immediate damage preservation was substantially predictable from the frozen human-move policy, while longer-play damage changes remained weakly predictable. This is a predictive finding, not evidence that players consciously noticed a mistake or that any feature caused the outcome.

Raw Maia-2 achieved **73.69% accuracy, 72.87% balanced accuracy and 0.8102 ROC-AUC** on the immediate task, against 55.07% accuracy for the training-majority baseline. Adding Maia to the damage/Elo LightGBM improved Brier score by **0.067206** (paired 95% interval **[0.060561, 0.073612]**).

For the primary regression, LightGBM reduced no-change MAE by **0.21%, 0.93%, 1.54% and 2.36%** at 2/4/6/10 plies. The 2- and 4-ply intervals include zero; the 6- and 10-ply intervals are above zero. Spearman correlations were only **0.096–0.140**. The matched four-archive ten-ply Maia increment was positive but modest: **0.376914 pp MAE** (95% interval **[0.202856, 0.557825]**); augmented Spearman remained **0.1464**.

The complete ordered feature list, learner settings, frozen baseline constants and unmodified method snapshots are in [METHODOLOGY.md](METHODOLOGY.md). The `freeze/` directory remains a pre-test snapshot; current run authorization/completion is recorded in this result directory.

## Cohorts, targets and fixed protocol

| Use | Archive set | Rows / games | Protocol |
| --- | --- | --- | --- |
| Primary training | 2017-04, 2017-06, 2017-07, 2018-02, 2019-02 | 23,167 / 6,512 | Frozen full-development fit |
| Ten-ply secondary training | 2017-04, 2017-06, 2017-07, 2018-02 | 15,953 / 4,485 | Frozen full-development fit |
| Immediate classifier/prevalence training | 2017-04, 2017-06, 2017-07, 2018-02 | 15,851 / 4,472 | Valid actual replies; 102 missing excluded |
| Final primary and ten-ply evaluation | 2025-03 supplied partial extract | 3,380 / 955 | March 2025 temporal held-out; one-time frozen evaluation |
| Final immediate evaluation | 2025-03 supplied partial extract | 3,356 / 951 | Same temporal holdout; 24 missing replies excluded |

Raw Maia uses released external Blitz/Rapid policies; it was not fitted on the four local archives. Those months define the local classifier training and prevalence reference. The documented external policy training range ends in November 2023, before March 2025; individual training membership is not independently audited.

Selection stayed `10 <= win_before <= 90` and `initial_damage >20` pp. The primary target is `win_after - win_after_h`, normalized to the original mistake-maker: negative means recovery and positive means further damage. The immediate positive label is `(win_before - win_after_actual_reply) / initial_damage >=0.75`. Every immediate classifier uses `p >=0.5`. Missing replies were never relabeled negative.

The evaluator saved all predictions before opening labels for scoring. Earlier preparation/integrity checks constructed and checked labels without comparing them with model predictions. Numeric imputation, missing indicators and phase encoding were fixed in the serialized training pipelines. There was no development/test game overlap.

## 1. Primary persistence regression

Local training archives: **2017-04, 2017-06, 2017-07, 2018-02, 2019-02**. Validation: **March 2025 temporal held-out; one-time frozen evaluation**. All rows below use **3,380 mistakes / 955 games**, threshold >20. MAE units are winning-chance percentage points; smaller is better. Spearman is higher-is-better and undefined for constant baselines.

| Plies | Method | MAE (pp) | Spearman |
| --- | --- | --- | --- |
| 2 | No-change | 13.1401 | — |
| 2 | Training-mean | 15.2092 | — |
| 2 | Frozen LightGBM | 13.1121 | 0.1400 |
| 4 | No-change | 17.6036 | — |
| 4 | Training-mean | 19.5465 | — |
| 4 | Frozen LightGBM | 17.4403 | 0.1320 |
| 6 | No-change | 20.6187 | — |
| 6 | Training-mean | 22.6039 | — |
| 6 | Frozen LightGBM | 20.3010 | 0.0961 |
| 10 | No-change | 24.4218 | — |
| 10 | Training-mean | 26.4267 | — |
| 10 | Frozen LightGBM | 23.8443 | 0.1190 |

Paired improvement is comparator MAE minus model MAE; positive values favor LightGBM. Each interval uses the same whole-game resamples for both arms.

| Plies | Comparator | MAE improvement (pp) | 95% interval (pp) | Relative gain | 95% relative interval |
| --- | --- | --- | --- | --- | --- |
| 2 | No-change | 0.027944 | [-0.026046, 0.080649] | 0.213% | [-0.196, 0.607]% |
| 2 | Training-mean | 2.097090 | [1.922497, 2.276189] | 13.788% | [12.433, 15.235]% |
| 4 | No-change | 0.163311 | [-0.027115, 0.369149] | 0.928% | [-0.159, 2.050]% |
| 4 | Training-mean | 2.106192 | [1.822970, 2.402081] | 10.775% | [9.320, 12.274]% |
| 6 | No-change | 0.317700 | [0.078549, 0.574370] | 1.541% | [0.382, 2.731]% |
| 6 | Training-mean | 2.302893 | [1.993822, 2.621621] | 10.188% | [8.801, 11.620]% |
| 10 | No-change | 0.577432 | [0.243856, 0.930068] | 2.364% | [1.002, 3.743]% |
| 10 | Training-mean | 2.582397 | [2.190138, 2.988805] | 9.772% | [8.281, 11.303]% |

The stronger no-change comparison is the main interpretation: the absolute improvements are small, and no clear improvement is established at 2 or 4 plies. Beating the training-mean baseline at every horizon does not by itself imply strong longer-play prediction.

## 2. Immediate reply: raw Maia and the damage/Elo increment

Local training/reference archives: **2017-04, 2017-06, 2017-07, 2018-02**; raw Maia is externally pretrained. Validation: **March 2025 temporal held-out; one-time frozen evaluation**. All methods use **3,356 replies / 951 games**, threshold >20, cutoff 0.5. Positive class: preserving >=75% of initial damage.

| Method | Accuracy | Balanced accuracy | ROC-AUC | Brier | Precision | Preserving recall | Non-preserving recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Training-majority/prevalence | 55.07% | 50.00% | 0.5000 | 0.247490 | 55.07% | 100.00% | 0.00% |
| Raw Maia-2 | 73.69% | 72.87% | 0.8102 | 0.179704 | 73.80% | 80.95% | 64.79% |
| LightGBM: damage + opponent Elo | 56.70% | 53.87% | 0.5692 | 0.243971 | 57.51% | 81.82% | 25.93% |
| LightGBM: damage + opponent Elo + Maia | 73.39% | 72.56% | 0.8100 | 0.176765 | 73.53% | 80.74% | 64.39% |

The majority classifier and its probability use only training labels: its fixed preserving probability is 0.5581351334300675. It predicts preserving for every test row. Brier and ROC-AUC use continuous probabilities, not thresholded decisions.

| Method | TN: correctly non-preserving | FP: incorrectly preserving | FN: missed preserving | TP: correctly preserving |
| --- | --- | --- | --- | --- |
| Training-majority/prevalence | 0 | 1,508 | 0 | 1,848 |
| Raw Maia-2 | 977 | 531 | 352 | 1,496 |
| LightGBM: damage + opponent Elo | 391 | 1,117 | 336 | 1,512 |
| LightGBM: damage + opponent Elo + Maia | 971 | 537 | 356 | 1,492 |

There are 1,848 preserving and 1,508 non-preserving valid replies. The 24 missing replies are excluded from every immediate arm, leaving the paired populations identical.

| Prespecified contrast | Metric | Improvement | Paired 95% interval |
| --- | --- | --- | --- |
| Raw Maia-2 vs Training-majority/prevalence | Brier reduction | 0.067786 | [0.059061, 0.076163] |
| Raw Maia-2 vs Training-majority/prevalence | Accuracy gain (percentage points) | 18.623361 | [16.397366, 20.783356] |
| Raw Maia-2 vs Training-majority/prevalence | Balanced-accuracy gain (percentage points) | 22.870090 | [21.246213, 24.369028] |
| LightGBM: damage + opponent Elo + Maia vs LightGBM: damage + opponent Elo | Brier reduction | 0.067206 | [0.060561, 0.073612] |
| LightGBM: damage + opponent Elo + Maia vs LightGBM: damage + opponent Elo | Accuracy gain (percentage points) | 16.686532 | [14.703209, 18.675440] |
| LightGBM: damage + opponent Elo + Maia vs LightGBM: damage + opponent Elo | Balanced-accuracy gain (percentage points) | 18.689644 | [16.790962, 20.592071] |

Both comparisons show clear gains on their prespecified contrasts. The augmented LightGBM comparison establishes incremental prediction beyond exactly initial damage and opponent Elo; it does not establish incremental value beyond every possible pre-reply variable or a causal effect.

## 3. Matched ten-ply Maia increment

Both models trained on exactly **2017-04, 2017-06, 2017-07, 2018-02**, with the same settings and preprocessing; the augmented model adds only the saved Maia probability to F24. Validation: **March 2025 temporal held-out; one-time frozen evaluation**, **3,380 mistakes / 955 games**, threshold >20.

| Method | MAE (pp) | Spearman |
| --- | --- | --- |
| LightGBM: F24 | 24.0661 | 0.0784 |
| LightGBM: F24 + Maia | 23.6892 | 0.1464 |

MAE reduction: **0.376914 pp**, paired 95% interval **[0.202856, 0.557825]**; relative reduction **1.566%**, interval **[0.852, 2.294]%**. The increment is positive, but absolute error remains large and correlation weak. The five-archive primary ten-ply model and these four-archive models have different training cohorts; their differences cannot be attributed solely to Maia.

## 4. Prespecified later recovery diagnostics

Primary regressors trained on **2017-04, 2017-06, 2017-07, 2018-02, 2019-02**; validation **March 2025 temporal held-out; one-time frozen evaluation**, **3,380 mistakes / 955 games**, threshold >20. These are regression-derived labels, not separately trained classifiers. Recovered means actual ratio <0.75; prediction uses the same strict threshold, and ROC-AUC uses negative predicted ratio as a continuous ranking score. The threshold is project-specific and exploratory.

| Plies | Accuracy | Training-majority accuracy | Balanced accuracy | ROC-AUC | Recovered precision | Recovered recall | Non-recovered recall | Recovered-only MAE (diagnostic) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 72.10% | 72.10% | 50.36% | 0.6051 | 50.00% | 1.17% | 99.55% | 33.4487 |
| 4 | 66.78% | 66.51% | 50.70% | 0.5901 | 62.16% | 2.03% | 99.38% | 36.6831 |
| 6 | 64.47% | 63.99% | 51.05% | 0.5719 | 63.33% | 3.12% | 98.98% | 39.1527 |
| 10 | 62.99% | 61.95% | 52.02% | 0.5677 | 64.23% | 6.14% | 97.90% | 43.4410 |

Recovery recall is low under the frozen regression-derived threshold. Accuracy near the majority baseline must not be interpreted as strong recovery classification. Full baseline and matched-ten-ply diagnostic metrics and confusion counts are retained in the labelled CSV and supplement. No separate horizon classifier or tuned cutoff was introduced.

## 5. Test versus development distributions

Archive sets: five-development and four-development cohorts listed above, compared with the **2025-03 temporal holdout**. These are descriptive distributions of eligible mistake rows, not performance estimates or the full archive populations. The four-archive distribution includes all 15,953 feature rows, while immediate models fit only the 15,851 rows with valid replies. Per-group game counts can overlap across rating bands and phases.

### Mistake-maker rating

| Group | Five-development n (%) | Four-development n (%) | March test n (%) |
| --- | --- | --- | --- |
| 1600–<2000 | 9,202 (39.72%) | 6,806 (42.66%) | 1,199 (35.47%) |
| <1600 | 11,881 (51.28%) | 7,549 (47.32%) | 1,639 (48.49%) |
| >=2000 | 2,084 (9.00%) | 1,598 (10.02%) | 542 (16.04%) |

### Reply-maker rating

| Group | Five-development n (%) | Four-development n (%) | March test n (%) |
| --- | --- | --- | --- |
| 1600–<2000 | 9,304 (40.16%) | 6,900 (43.25%) | 1,200 (35.50%) |
| <1600 | 11,596 (50.05%) | 7,321 (45.89%) | 1,631 (48.25%) |
| >=2000 | 2,267 (9.79%) | 1,732 (10.86%) | 549 (16.24%) |

### Recorded speed

| Group | Five-development n (%) | Four-development n (%) | March test n (%) |
| --- | --- | --- | --- |
| Blitz | 10,775 (46.51%) | 7,261 (45.51%) | 1,771 (52.40%) |
| Bullet | 7,337 (31.67%) | 4,854 (30.43%) | 1,037 (30.68%) |
| Classical | 3,175 (13.70%) | 2,900 (18.18%) | 2 (0.06%) |
| Correspondence | 69 (0.30%) | 51 (0.32%) | 0 (0.00%) |
| Rapid | 879 (3.79%) | 0 (0.00%) | 530 (15.68%) |
| Standard/unspecified | 613 (2.65%) | 613 (3.84%) | 0 (0.00%) |
| UltraBullet | 319 (1.38%) | 274 (1.72%) | 40 (1.18%) |

### Game phase

| Group | Five-development n (%) | Four-development n (%) | March test n (%) |
| --- | --- | --- | --- |
| endgame | 2,597 (11.21%) | 1,806 (11.32%) | 325 (9.62%) |
| middlegame | 5,055 (21.82%) | 3,507 (21.98%) | 741 (21.92%) |
| opening | 15,515 (66.97%) | 10,640 (66.70%) | 2,314 (68.46%) |

The test cohort has more >=2000-rated mistake-makers (16.04% versus 9.00% in five-archive development), more Rapid (15.68% versus 3.79%) and very little Classical (2 mistakes from one game). These are cohort differences, not a basis for changing the frozen models. UltraBullet contains only **40 mistakes / 9 games** and is a small-sample/out-of-domain limitation, not a finding. No UltraBullet-specific cutoff or model was selected. Every exact recorded time-control distribution is included in the supplement and distribution CSV.

## 6. Robustness, subgroups and development context

The >25 and >30 checks reuse subsets of the saved >20 predictions without refits. They contain 2,568 / 880 and 1,947 / 808 mistakes/games respectively; immediate scoring uses 2,547 / 876 and 1,927 / 800 after missing-reply exclusions. Every arm, interval and diagnostic is preserved in [SUPPLEMENT.md](SUPPLEMENT.md) and the labelled CSV files; no favorable threshold was substituted for >20.

All **612 prespecified subgroup rows** are exported, including Elo, exact time control, recorded speed and phase, with sample/game counts and sparse-group flags. These are descriptive fixed-model diagnostics. Groups with fewer than 100 mistakes are flagged by the frozen code; small game counts can also make groups unreliable. Subgroups have no dedicated refits, cutoffs or multiplicity-adjusted inferential claims.


| Primary horizon | Development no-change MAE | Development frozen-LGBM MAE | March no-change MAE | March frozen-LGBM MAE |
| --- | --- | --- | --- | --- |
| 2 | 12.5766 | 12.5589 | 13.1401 | 13.1121 |
| 4 | 16.9359 | 16.8435 | 17.6036 | 17.4403 |
| 6 | 19.8345 | 19.5792 | 20.6187 | 20.3010 |
| 10 | 23.6354 | 23.0491 | 24.4218 | 23.8443 |

Protocol for the preceding table: game-grouped development OOF versus the single March temporal holdout; all primary models use the five-development archive feature/cohort specification. This is contextual reporting, not another prespecified inferential contrast.


## Reproduction, verification and artifacts

The recorded command was executed once and returned exit code 0. The bundle’s exclusive final-run marker remains present. Raw evaluator outputs are preserved unchanged, including all unfavorable results. Reporting and independent arithmetic checks read the saved predictions only; they never loaded or invoked a model.

Independent checks reproduced all 54 exported metric rows (including continuous-score ROC-AUC via ranks) and verified the first five game-bootstrap draws for every prespecified contrast by explicitly expanding sampled rows. Maximum numerical difference was 5.11e-15. Native/package versions and run hashes are in `completion.json` and `run_provenance.json`. The original evaluator log, including its feature-name warnings, is retained as `execution.log`; the frozen schema/identity/hash checks passed.

| Artifact | Contents |
| --- | --- |
| predictions_before_labels.csv.gz | All saved OOF-independent final predictions, before scoring labels |
| metrics.csv / labelled_metrics.csv | All 54 metric rows at >20/>25/>30; labelled copy adds archive/protocol/target metadata |
| paired_game_intervals.csv / labelled_paired_game_intervals.csv | All 72 prespecified interval rows, signs, populations and valid-draw counts |
| subgroups.csv / labelled_subgroups.csv | All 612 subgroup metric rows and sparse-group flags |
| development_vs_final_distributions.csv | All rating/time-control/speed/phase cohort distributions |
| development_*_context.csv | Approved-method saved development OOF context only |
| SUPPLEMENT.md | Complete robustness, interval, classification, time-control and subgroup tables |
| bootstrap_first5_expanded_checks.csv / verification.json | Independent arithmetic and clustered-resampling checks |
| completion.json / run_provenance.json / artifact_hashes.json | Frozen versions, source commits, authorization, execution and output checksums |
| execution.log / build_report.py | Preserved evaluator log and saved-output-only report generator |
