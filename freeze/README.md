# Evaluation freeze

This directory records the approved models, input contract and software versions
for the one-time March 2025 evaluation. The comparisons and metrics are defined
in the [evaluation plan](../MARCH_2025_EVALUATION_PLAN.md); implementation details
are in the [technical appendix](../MARCH_2025_EVALUATION_APPENDIX.md).

Feature and label preparation is complete for 3,380 mistakes / 955 games.
No final model evaluation has run. Commit the source/configuration freeze before
that evaluation, then obtain separate explicit authorization to run it.

## Why each file is kept

For scoring prepared data, the evaluator directly reads `approved_models.json`
and `final_input_protocol.json`, together with the local fitted-model bundle.
The other files support preparation, reproduction, validation or the release
audit. They are not all required simply to calculate a prediction.

| File | How it is used | Why keep it |
| --- | --- | --- |
| [README.md](README.md) | Documentation | Explains the freeze records, local dependencies and verification commands. |
| [approved_models.json](approved_models.json) | Evaluator and preparation | Fixes the approved models, features, hyperparameters, archives, seeds, cutoffs and engine/Maia settings. Prevents silent changes to the comparison. |
| [fitted_models.json](fitted_models.json) | Readiness audit | Identifies the exact fitted pipelines through hashes, training counts, preprocessing schemas and baseline constants. Contains metadata, not model binaries. |
| [code_manifest.json](code_manifest.json) | Readiness audit | Records source/configuration/document hashes and the source files that must match the freeze commit. |
| [final_input_protocol.json](final_input_protocol.json) | Evaluator and readiness audit | Fixes the eligible test cohort, prepared-file identities, selection rules and provenance limitations. Its frozen status is required before final evaluation. |
| [final_evaluation_invocation.json](final_evaluation_invocation.json) | Release recipe | Records the exact future command, model bundle and input-manifest hashes. It does not execute that command. |
| [requirements-modeling.txt](requirements-modeling.txt) | Environment reproduction and release audit | Lists exact package versions needed to fit and score the approved LightGBM models. |
| [environment-modeling.json](environment-modeling.json) | Environment record and release audit | Adds the recorded Python version, operating system and architecture to the modeling package versions. |
| [modeling_runtime.json](modeling_runtime.json) | Environment validation evidence | Identifies the validated local scoring executable, native LightGBM hash and passing synthetic checks. |
| [requirements-policy.txt](requirements-policy.txt) | Environment reproduction and release audit | Lists exact packages for regenerating the frozen Maia probabilities. The policy environment is not needed to score already-prepared features. |
| [environment-policy.json](environment-policy.json) | Feature preparation and release audit | Lets the feature-generation driver verify the Python/platform/package environment before Maia inference. |
| [engine_identity.json](engine_identity.json) | Supporting provenance | Records the checked Stockfish binary version and checksum. The operational settings also appear in the model and enrichment protocols. |
| [extraction_source_reference.json](extraction_source_reference.json) | Supporting provenance | Identifies the original extractor source and defaults, and distinguishes confirmed filters from unrecorded original run settings. |
| [test_enrichment_protocol.json](test_enrichment_protocol.json) | Readiness audit and preparation provenance | Pins the feature-generation sources, Maia checkpoints, depth, MultiPV settings, policy coverage, batching and cache rules. |
| [test_completion_protocol.json](test_completion_protocol.json) | Preparation completion and release audit | Restricts the completion driver to the recorded export/label job, with evaluation disabled. It is not a model configuration. |
| [test_enrichment_status.json](test_enrichment_status.json) | Preparation completion evidence | Records that feature and label preparation finished and identifies the resulting artifacts by checksum. |
| [test_readiness_audit.json](test_readiness_audit.json) | Current readiness report | Reports prepared-data integrity, unchanged models, environment checks, commit status and remaining release requirements. |
| [validation_summary.json](validation_summary.json) | Development validation evidence | Records synthetic checks, reproduction of saved development model predictions and Maia policy replay. Its older freeze/blocker fields describe the earlier validation stage. |
| [integrity_audit.json](integrity_audit.json) | Historical supporting evidence | Preserves an earlier model/source/environment checksum audit. It is not the current readiness report and is not read by the evaluator. |

The release audit requires several environment and preparation records to be
committed even though the scorer does not read them directly. Supporting evidence
such as `integrity_audit.json` is useful for traceability, but is not an intrinsic
runtime requirement. File references and manifest hashes must be updated if
records are later consolidated or moved.

`validation_summary.json` and `integrity_audit.json` are historical snapshots.
Use `test_readiness_audit.json` for current readiness; none of these reports
contains final-test model performance. Source filenames and raw merge provenance
remain in the ignored local records.

## Local files excluded from Git

The following paths are local dependencies, not repository download links:

| Local path or asset | Role |
| --- | --- |
| Source extracts and their local provenance manifest | Original supplied March data; filenames are kept locally |
| `test_data/march_2025_prepared_v2/` | Combined source rows, eligibility selection and row lineage |
| `test_data/march_2025_enriched_v1/` | Final prepared inputs, labels and preparation checkpoints |
| `frozen_artifacts/development_20261121_v2/` | Eight fitted pipelines, training membership and baseline constants |
| `.venv-final-evaluation/` | Machine-specific scoring environment |
| `freeze_validation/` | Detailed local validation outputs |
| Development CSVs, saved predictions, engine caches and Maia checkpoint files | External dependencies for reproducing development fitting and feature generation; exact paths/hashes are recorded in the manifests |

Keep the local model bundle and data backed up. A fresh clone requires matching
local artifacts before data-dependent commands can run. Package versions and
platform details are recorded here; the virtual environment itself is not
committed.

Within the local enriched directory, `features.csv` is the strict predictor
table, `labels.csv` holds observed targets, and `input_manifest.json` records
both checksums. `test_with_frozen_features.csv` combines them for inspection;
it must not be passed directly to the models. Missing actual replies remain
missing and are excluded only from immediate-reply scoring.

## Verification

Run commands from the repository root. With the recorded modeling dependencies,
the synthetic evaluator and readiness tests require no test dataset:

```sh
python -m unittest tests.test_frozen_chess_evaluation tests.test_audit_sealed_test_readiness -v
```

With the matching local model bundle, prepared data and validated scoring
environment available, audit their integrity without evaluating model performance:

```sh
.venv-final-evaluation/bin/python scripts/audit_sealed_test_readiness.py --prepared test_data/march_2025_enriched_v1
```

The audit checks schemas, row alignment, target equations, legal-reply coverage,
missing values, probability bounds, game overlap, versions and hashes. It uses
read-only Git checks to identify the committed freeze and never stages or commits
files. No model predictions or performance metrics are calculated by this command.

## Data scope and run control

The cohort is the fixed supplied March partial extract. The source defaults and
confirmation of unchanged filters are recorded in
[extraction_source_reference.json](extraction_source_reference.json). Original
extraction command lines, binary identities and full-PGN sampling are not
independently logged; those limitations remain explicit in the input protocol.

The final evaluator requires explicit authorization. An exclusive marker in the
local fitted bundle prevents a second final run, including to another output
directory. The preparation scripts and integrity audit do not invoke evaluation.
