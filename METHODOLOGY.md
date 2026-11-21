# Frozen methodology and execution record

These settings were fixed before the one-time evaluation. The original freeze
records remain unchanged; the copies here make the results self-contained.

Source/configuration freeze commit: `bbe496325c83b8695a52cb1b93896ed4e8a34768`. Execution commit: `729c8abd011c9dd2868c2fea2cc514f69823a474`. Training-artifact source reference: `e5de7253ceb6777b4bcec23a876b3dbcd2108178`.

## Inputs and model recipes

F24 is the following ordered input list. Actual replies, later evaluations,
outcomes and horizon targets are absent from the predictor table.

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

Primary regressors use F24 at 2, 4, 6 and 10 plies and are fitted on all
five development archives. The matched ten-ply comparison uses F24 with and
without the Maia midpoint, with both arms fitted on the four earlier archives.
The immediate control uses initial damage and reply-maker Elo; its augmented
version adds only that same midpoint. Those two classifiers use the four-archive
valid-reply cohort. Raw Maia is an external frozen policy, not a locally fitted
classifier.

Every regression LightGBM uses:

```json
{
  "objective": "regression_l1",
  "n_estimators": 240,
  "learning_rate": 0.04,
  "num_leaves": 11,
  "min_child_samples": 90,
  "reg_lambda": 7,
  "verbosity": -1,
  "random_state": 1701,
  "n_jobs": 2
}
```

The immediate classifiers use the same numerical settings with objective
`binary`, no class weighting and no recalibration. Numeric median imputation and
missing indicators, plus regression phase one-hot encoding, were learned on
development training rows and serialized in each pipeline. No preprocessing was
fitted on test rows. The exact learned medians, transformed schemas and effective
estimator parameters are in `fitted_models_snapshot.json`.

Primary no-change predicts zero. Per-horizon training-mean constants and the
four-archive immediate prevalence are saved in the model bundle:

```json
{
  "primary_mean": {
    "2": -6.263533757820715,
    "4": -8.005349413829395,
    "6": -9.250294115309382,
    "10": -10.734169225797375
  },
  "primary_recovered_majority": {
    "2": false,
    "4": false,
    "6": false,
    "10": false
  },
  "ten_recovered_majority": false,
  "reply_prevalence": 0.5581351334300675
}
```

## Frozen Maia and engine preparation

Maia-2 0.11.0 / Torch 2.8.0 used the recorded Blitz/Rapid checkpoints on CPU,
four Torch threads, seed 1701 and batches of 32. Inputs are the post-mistake
position and the two ratings. Full-precision legal-move probabilities were
sorted by probability descending and UCI ascending; the smallest prefix reaching
97.5% mass was scored. The saved feature is the midpoint between qualified
scored mass and that mass plus residual unscored probability. Unselected scores
were not used and selected probabilities were not renormalized. Numerical bound
width is at most 0.025, independently of actual reply identity.

The pinned Stockfish 19 depth-18 procedure supplied new MultiPV-5 and candidate/
actual-reply scores with one engine thread and 64 MiB hash. Candidate/reply mate
conversion is ±1000 cp; legacy MultiPV conversion stays ±100000 cp. Exact-FEN
read-only reuse and within-batch transposition-table behavior are retained in the
preparation protocol. Original supplied core/horizon evaluations were reused
unchanged, with the original binary/search provenance limitations disclosed in
the main report.

## One-time execution and report scope

Eight serialized estimators and frozen training-only baselines were used. The
recorded evaluator was invoked once, returned exit code zero and left the
exclusive final-run marker in the bundle. It wrote predictions before reading
labels for performance scoring. All >25/>30 and subgroup reports use those same
saved predictions. No model or threshold was selected from test outcomes.

The original files in `freeze/` are pre-test snapshots. Their pre-test execution
or authorization flags were not retroactively edited. Explicit session
authorization and actual completion are recorded in `run_provenance.json` and
`completion.json` here. The report-generation scripts read saved predictions and
results only; no estimator is loaded or called by them.

The unadjusted 95% intervals use 2,000 paired whole-game percentile bootstrap
draws, seed 1701, without refitting. They quantify uncertainty conditional on the
fitted models. Spearman measures ranking of the numeric change target; immediate
ROC-AUC uses continuous preserving probabilities. The recovery diagnostic uses
the strict ratio <0.75 and negative predicted ratio as its continuous score.

## Frozen record copies

- `frozen_model_config.json`: full ordered features, parameters and policy/engine identities.
- `fitted_models_snapshot.json`: pipeline/weight hashes, preprocessing, cohorts and constants.
- `final_input_protocol_snapshot.json`: exact data identities and provenance limitations.
- `evaluation_plan_snapshot.md`: unchanged pre-test plan, preserved as a historical snapshot.
- `recorded_invocation_snapshot.json`: unchanged pre-test invocation record; it does not describe current execution status.
