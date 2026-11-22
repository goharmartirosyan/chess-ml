# Final model commands

These entries document the final training and evaluation workflow. They require the matching local data, fitted-model bundle and environment; the repository’s self-contained results notebooks do not require those files. The March 2025 evaluation is complete. Earlier development searches and feature experiments are outside this command record.

## Training on development data

`scripts/frozen_chess_evaluation.py` fits the four primary regressors, two immediate-reply classifiers and two matched 10-ply regressors with the settings in `freeze/approved_models.json`. This CLI example uses a placeholder for a new output directory; it is not another recorded training run.

```sh
.venv-final-evaluation/bin/python scripts/frozen_chess_evaluation.py fit-development \
  --out frozen_artifacts/NEW_DIRECTORY
```

## Completed final evaluation

The following command is a historical execution record, not an instruction to rerun the test. Original paths are preserved even where result files were later copied to shorter public paths.

```text
.venv-final-evaluation/bin/python scripts/frozen_chess_evaluation.py evaluate-prepared \
  --bundle frozen_artifacts/development_20261121_v2 \
  --input-manifest test_data/march_2025_enriched_v1/input_manifest.json \
  --out final_evaluation/march_2025_once_v1 \
  --allow-final-test
```

Execution commit: `729c8abd011c9dd2868c2fea2cc514f69823a474`.

[Model settings](../freeze/approved_models.json) · [Dependency versions](../freeze/requirements-modeling.txt).
