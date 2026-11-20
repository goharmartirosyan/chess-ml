# March 2025 evaluation plan 


| Training cohort | Archives | Existing development sample |
| --- | --- | --- |
| Primary | 2017-04, 2017-06, 2017-07, 2018-02, 2019-02 | 23,167 mistakes / 6,512 games |
| Matched ten-ply secondary | First four months above, excluding 2019-02 | 15,953 mistakes / 4,485 games |
| Immediate-reply secondary | Same four earlier months, valid actual replies only | 15,851 replies / 4,472 games; 102 missing replies excluded |

Canonical table: `development_with_multipv.csv`, recorded SHA-256 `aca96c1cdcd20ea654dbca181acf69aa5d9396a89fc2bbb95532ee88189653e6`. Its 510 overlapping observations were removed from 23,677 original rows. Do not train on the original duplicated table or add observations of unverified provenance.

Mistake key: `(source_pgn, game_index, ply)`; verified game key: `site`. Precommit duplicate resolution and cross-cohort game-identity checks. No final-test game may enter development training. Keep missing actual replies eligible for feature generation/later horizons where the existing target rules permit; exclude them only from immediate-reply scoring. All models within a comparison share the same eligible rows.

## C. Targets, engines and feature timing

```text
win(cp) = 100 / (1 + exp(-0.00368208 * cp))
initial_damage = win_before - win_after
remaining_damage_h = win_before - win_after_h
change_in_damage_h = win_after - win_after_h
persistence_ratio_h = 1 + change_in_damage_h / initial_damage
```

Selection: `10 <= win_before <= 90 AND initial_damage >20` percentage points. Evaluate >25 and >30 as subsets of fixed >20 predictions, without refitting. All scores retain the original mistake-maker perspective; ratios are unclipped. Categories: mostly recovered <0.25; partially persistent 0.25–<0.75; strongly persistent 0.75–1.0; worsened >1.0.

Primary engine depth is 18 at all positions. The recorded reply/feature engine is Stockfish 19, one thread, 64 MiB hash; binary SHA-256 `8eed61129d1493c5d1f2fd9323f0c54c47ac49319911fbde18c6b9c87e8b13c5`. Verify the actual binary before an authorized run. The historical core-target engine version is unrecorded. Final extraction must explicitly lock the core engine and cache policy and report any engine-version difference from historical targets.

Core/reply mate conversion is ±1000 cp. The existing historical MultiPV extractor uses ±100000 cp: preserve/version that feature contract for the existing model; never silently change its interpretation. Any proposed correction requires explicit development-side resolution and a new freeze before test preprocessing.

MultiPV: up to five opponent replies, depth 18, after the mistake and before the actual reply. `multipv_good_reply_count_10pp` counts lines within 10 winning-chance points of the opponent's best line. `multipv_best_second_gap_pp` is the best–second gap; one legal reply means missing, not zero. Keep `multipv_lines_returned` as metadata. The canonical development table has 76 valid missing gaps.

Preserve the implemented terminal-target rules in `prepare_chess_modeling.py`: applicable outcomes construct post-end horizon targets; unresolved targets have recorded exclusions. Horizon/terminal flags are target metadata, not predictors. PGN result alone does not identify abandonment/disconnection. Do not use PGN outcomes to invent a missing actual reply. FEN-only candidate analysis cannot restore prior repetition history.

Generate known-at-mistake features for every eligible position before joining labels. Predictor matrices exclude actual replies, later moves/evaluations, results, targets, future-mistake flags and completion flags. Identifier/FEN/UCI fields are reconstruction metadata, not learned-model predictors.

## D. Exact final-model recipes

### D1. Original persistence LightGBM and matched ten-ply pair

The following ordered 24-feature list (`F24`) is used for the primary model and the four-archive ten-ply reference:

```text
eval_before_cp
initial_damage
player_elo
opponent_elo
move_number
player_clock_before_move
opponent_clock_before_move
initial_time_seconds
increment_seconds
material_balance_cp
number_of_pieces
hanging_pieces
opponent_hanging_pieces
king_attackers
opponent_king_attackers
reasonable_candidate_moves
eval_gap_best_second_cp
in_check
gives_check
is_capture
is_promotion
game_phase
multipv_good_reply_count_10pp
multipv_best_second_gap_pp
```

The four-archive ten-ply augmented model uses exactly `F24 + [maia_preserving_reply_probability]`. Both ten-ply arms use identical four-archive training rows. Adding Maia must not simultaneously change the training cohort, preprocessing or hyperparameters. Do not add four-archive Maia regressors at 2/4/6 plies to the final evaluation.

Fixed regression settings:

```json
{
  "objective": "regression_l1",
  "n_estimators": 240,
  "learning_rate": 0.04,
  "num_leaves": 11,
  "min_child_samples": 90,
  "reg_lambda": 7,
  "verbosity": -1
}
```

Use `random_state=1701`, `n_jobs=2`, LightGBM 4.6.0. Fit numeric median imputation with missing indicators and phase one-hot encoding on development training rows; ignore unknown phase categories as in the saved pipeline. Save the final ordered schema. At test time, transform and predict only.

Primary baselines: no-change predicts 0; mean predicts the per-horizon mean target from all five development archives. Neither uses test labels. No other regression baseline or estimator is an evaluation arm in this revision.

### D2. Immediate-reply models

The label is preserving if `(win_before - W_after_actual_reply) / initial_damage >=0.75`. This exploratory project-specific event is separate from later persistence and does not measure human awareness.

- Raw Maia: the bounded probability in Section E, preserving if `p >=0.5`.
- Majority/prevalence baseline: preserving probability from valid labeled replies in the four development archives, with its class decision at 0.5. Use this probability, not the hard majority decision, for Brier and ROC-AUC.
- Control LightGBM: exactly `initial_damage`, `opponent_elo` (reply-maker rating).
- Augmented control: those same two inputs plus `maia_preserving_reply_probability`.

Both classifiers use the fixed LightGBM settings above with `objective="binary"`, `random_state=1701`, `n_jobs=2`; no class weighting, recalibration, search or feature selection. Fit median imputation/missing indicators on the same valid four-archive development rows. Both decisions use `p >=0.5`. No logistic-regression sensitivity arm is included.

## E. Frozen Maia-2 policy and integration settings

Use Maia-2 **0.11.0**, Torch **2.8.0**, CPU, `eval()` and inference mode, seed **1701**, four Torch threads. Policy inputs: post-mistake FEN, opponent/reply-maker Elo as active rating, original mistake-maker Elo as other rating. Missing ratings default to 1500 with a metadata flag; retain released rating-bin preprocessing, including collapsed bins below 1100 and at/above 2000. Live clocks are not Maia inputs.

| Checkpoint | Recorded SHA-256 |
| --- | --- |
| `blitz_model.pt` | `5090d5d0d49dc29787c08d13febbed5b2da81a18c2ddcd8a90070cd3b43c44b2` |
| `rapid_model.pt` | `65aae8465eed5e65df66a24ea7370715579f9e5435098d06fe18bdb1e267e997` |

Checkpoint identities come from saved manifests; they were not reloaded for this revision. Before execution, verify them and the installed package-file hashes recorded in the policy protocol.

Routing preserves the ordered existing implementation: Event Blitz → Blitz; Rapid → Rapid; Bullet/UltraBullet → Blitz; Classical/Correspondence → Rapid. Otherwise use Blitz when `initial_time_seconds + 40*increment_seconds <480`, Rapid otherwise; missing time settings default to effective 600 seconds. Retain domain flags and original speed labels.

Use full-precision legal-move masked softmax, without temperature fitting or move sampling. Sort descending probability, UCI ascending for ties; score the smallest prefix reaching **97.5%** policy mass at depth 18. A hypothetical reply qualifies when `(win_before - W_candidate_reply) / initial_damage >=0.75`. If `L` is the qualifying scored mass and `R` the unscored mass, the feature is **`L + R/2`**, with numerical bounds `[L, L+R]`. Do not renormalize selected mass, treat the residual as bad, use the observed reply to select candidates, or add cached branches outside the prefix. Terminal candidate boards use exact 0/50/100 chances. These bounds concern numerical integration, not human-behavior or engine uncertainty.

The [official Maia-2 training documentation](https://github.com/CSSLab/maia2#training) and [legacy training configuration](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training.yaml), checked 2026-11-21, specify 2018-05 through 2023-11. March 2025 is later than this documented range; the four earlier development archives precede it. This date comparison is not an independent game/player-membership audit. Keep the recorded checkpoints; do not switch to a newer model. Treat UltraBullet as a sparse, out-of-domain limitation, with no subgroup-specific cutoff.

## F. Evaluation, metrics and required reporting

Fit each listed model/preprocessor once on its full declared development cohort, then serialize and hash it before test access. The final run is temporal held-out evaluation, not another outer CV fold. Development selection used game-grouped folds and separate player-purged validation; no preprocessing or tuning uses test rows. Retain the existing development player-disjoint evidence separately. A prespecified unseen-local-player reporting slice may use fixed models and a predefined membership rule; it must not retrain on test-player identities or imply disjointness from Maia's external training players.

For each of the four approved comparisons, report every result and label the training archive set, target, eligible rows/games, exclusions, feature/model version and cutoff. Primary regression reports MAE and Spearman at 2/4/6/10, with no-change MAE minus model MAE in absolute pp and `100 * gain / no_change_MAE`; percentage gain is undefined when the comparator MAE is zero. Ten-ply secondary reports both arms' MAE/Spearman and paired MAE reduction. Immediate comparisons report accuracy, balanced accuracy, ROC-AUC, Brier, both class recalls and confusion counts; the two-control increment's primary probability contrast is control Brier minus augmented Brier.

ROC-AUC always uses continuous scores. For required regression-derived recovery diagnostics, actual/predicted ratio <0.75 defines recovered and negative predicted ratio supplies the ranking score; it is not a calibrated probability. Report accuracy, majority-baseline accuracy, balanced accuracy, ROC-AUC and recovery precision/recall without introducing a separate classifier or tuning thresholds. Recovered-only MAE is diagnostic. Undefined precision, single-class ROC-AUC and baseline Spearman remain missing with explanations.

Use 2,000 paired percentile game-cluster bootstrap resamples, seed 1701, retaining every mistake in each sampled game and identical draws for compared models. Resample fixed final-test predictions; no bootstrap refitting. Recompute relative gains within each draw and report 95% intervals/valid-draw counts. They condition on fitted models and are unadjusted across repeated comparisons.

Report >25/>30 robustness using fixed primary-selection predictions. Retain rating (<1600 / 1600–<2000 / >=2000), recorded time-control/speed and prepared game-phase summaries, with row/game counts and missing categories. Name the rating perspective; immediate analyses use reply-maker Elo. After freezing and within the authorized final run/reporting, show test rating/time-control distributions beside development distributions; do not use them for model redesign. UltraBullet is a small-sample limitation. Associations/SHAP do not establish causality.

## G. Code identity and final-freeze requirements

Repository reference at revision: `0c0185aa2e21daa12e1d89f882f29db0f5ec6a02`. Relevant research scripts are untracked, so the commit alone is insufficient. The hashes below identify the existing implementations/configurations for the approved comparisons; they are not a substitute for a final evaluator version.

**Final evaluator and final fitted model/preprocessing artifact hashes remain pending.** Existing scripts retain development-only guards. Prepare and validate a separate evaluator on development/synthetic cases, then record its commit/hash, exact invocation, model/preprocessing hashes, deterministic extraction/input protocol, dependency locks and resolved engine/MultiPV compatibility settings before marking the plan frozen. Do not remove guards or pass test rows to tuning code. The self-contained results notebook remains a display snapshot.

Recorded modeling environment: Python 3.9.6; NumPy 2.0.2; pandas 2.3.3; SciPy 1.13.1; scikit-learn 1.6.1; LightGBM 4.6.0; chess 1.11.2. Recorded policy environment: Maia-2 0.11.0; Torch 2.8.0; NumPy 2.5.5; pandas 2.3.3; SciPy 1.18.1; scikit-learn 1.6.1; chess 1.11.2. Preserve separate environment records; validate and lock actual execution environments before test preprocessing.

| Approved-comparison source/configuration | SHA-256 at this revision |
| --- | --- |
| [scripts/extract_bad_moves.py](scripts/extract_bad_moves.py) | `71988820f1bed93fd8438e839f3a68bc0175afae780cccbb5f227ae1be20b5f7` |
| [prepare_chess_modeling.py](prepare_chess_modeling.py) | `332d2d4cdae2bdef9c357d445e0a336e19690078c0845e72c5068538fe8a0436` |
| [extract_post_mistake_multipv.py](extract_post_mistake_multipv.py) | `029d83064252f9dae80d1819d0f83e735db098c758cb2c0ded3b65b1d1c58c17` |
| [scripts/extract_opponent_reply_targets.py](scripts/extract_opponent_reply_targets.py) | `09afc7b273dbebf5449f28712ffd64635edb193f8e76a4ed2f4b2a48167190d6` |
| [tune_chess.py](tune_chess.py) | `0844c653976f14e780265b75a99f699971ef86a33a4e323692ef6e7dac3ec769` |
| [scripts/human_reply_policy.py](scripts/human_reply_policy.py) | `e3930acf4c91030581a9812b2f538c06c06f0874f46a038be774d4449b6fd9f6` |
| [scripts/scale_bounded_maia_feature.py](scripts/scale_bounded_maia_feature.py) | `709c0c85dc3905d09d0ca91bfc32f3382881002339bc3b7d20ab92e3405dd503` |
| [scripts/test_bounded_maia_regression.py](scripts/test_bounded_maia_regression.py) | `4e24bde301707c02dfbf5bddd6a31ebbd753ae9f455895ea0e84f62d1fe8b13a` |
| [scripts/compare_simple_reply_baseline.py](scripts/compare_simple_reply_baseline.py) | `0dfab9cabcda34a224dc9089e1faf43109e5d570f74a83b9916db93a03ca9ebc` |
| [notebook_runs/final_optuna_development_20261117_v1/protocol.json](notebook_runs/final_optuna_development_20261117_v1/protocol.json) | `91b61e96726dfc4872678de894132b62ec2039de9443f9ef6c037d48e2200a11` |
| [notebook_runs/maia_bounded_four_archive_20261117_v1/protocol.json](notebook_runs/maia_bounded_four_archive_20261117_v1/protocol.json) | `135fff0b5c7564f1adc738071dcbd1d96ce6053f079eb2e9d073ba45f797d10a` |
| [notebook_runs/maia_bounded_four_archive_20261117_v1/regression_protocol.json](notebook_runs/maia_bounded_four_archive_20261117_v1/regression_protocol.json) | `fd3023b69146df10ce9d46ac4d6061a88580366016fb9ee1fb3c42452c1407da` |
| [notebook_runs/simple_reply_baseline_development_20261118_v1/protocol.json](notebook_runs/simple_reply_baseline_development_20261118_v1/protocol.json) | `fbbdbcd011a75d2627f48a103d28387f5af711e4583a1ecf9454e484d46fcc3b` |
| [notebook_runs/human_reply_policy_development_20261117_v1/protocol.json](notebook_runs/human_reply_policy_development_20261117_v1/protocol.json) | `4f4ece4c1e7be87b5dfd6ad07b14e3311724bd41f029f912515464d6a7ca7bae` |
