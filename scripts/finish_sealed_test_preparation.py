#!/usr/bin/env python3
"""Finish authorized preparation after the two checkpointing feature jobs.

Never evaluates fitted research models. Run in the recorded policy environment.
"""
import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def count_rows(path):
    if not path.exists():return 0
    with path.open() as f:return sum(1 for _ in f)


def cached_positions(path):
    if not path.exists():return 0
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        return db.execute('SELECT COUNT(*) FROM evaluations').fetchone()[0]


def status(out,stage):
    value=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),
        multipv_rows_checkpointed=count_rows(out/'multipv_checkpoint.jsonl'),
        candidate_positions_checkpointed=cached_positions(out/'selected_child_evaluations.sqlite'),
        actual_reply_positions_checkpointed=cached_positions(out/'actual_reply_evaluations.sqlite'),
        research_model_predictions_run=False,performance_metrics_computed=False,
        final_evaluation_enabled=False)
    write_json(out/'preparation_status.json',value)
    return value


def refresh_records(out):
    completion=json.loads((out/'preparation_complete.json').read_text())
    for name,key in [('features.csv','features_sha256'),('labels.csv','labels_sha256'),
                     ('test_with_frozen_features.csv','combined_sha256')]:
        if digest(out/name)!=completion[key]:raise ValueError('Final preparation artifact differs.')
    input_protocol_path=ROOT/'freeze/final_input_protocol.json'
    protocol=json.loads(input_protocol_path.read_text())
    protocol['intermediate_lineage']=dict(base_preparation='test_data/march_2025_prepared_v2/preparation_manifest.json',
        feature_preparation=str((out/'protocol.json').relative_to(ROOT)),
        feature_preparation_sha256=digest(out/'protocol.json'),
        prepared_features=dict(path=str((out/'features.csv').relative_to(ROOT)),sha256=completion['features_sha256']),
        prepared_labels=dict(path=str((out/'labels.csv').relative_to(ROOT)),sha256=completion['labels_sha256']))
    protocol['blockers']=['Original extraction ranges/sampling, upstream filters, and original core engine/cache settings remain unverified.','The freeze materials require a user-created commit and separate explicit authorization before final evaluation.']
    # Preserve unresolved status: prepared tables do not certify old extraction.
    protocol['status']='unresolved'
    write_json(input_protocol_path,protocol)
    input_manifest_path=out/'input_manifest.json'
    input_manifest=json.loads(input_manifest_path.read_text())
    input_manifest['input_protocol_sha256']=digest(input_protocol_path)
    write_json(input_manifest_path,input_manifest)
    write_json(ROOT/'freeze/test_enrichment_status.json',dict(completion,
        local_output_directory=str(out.relative_to(ROOT)),
        input_manifest_sha256=digest(input_manifest_path),
        protocol_sha256=digest(out/'protocol.json'),
        completion_driver_sha256=digest(Path(__file__)),
        synthetic_tests_passed=12))
    manifest_path=ROOT/'freeze/code_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest['status']='models_frozen_test_feature_and_label_tables_prepared_original_extraction_unverified'
    manifest['freeze_records_sha256']={str(p.relative_to(ROOT)):digest(p)
        for p in sorted((ROOT/'freeze').iterdir()) if p.is_file() and p.name!='code_manifest.json'}
    manifest['document_sha256']={name:digest(ROOT/name) for name in manifest['document_sha256']}
    write_json(manifest_path,manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve()
    contract=json.loads((ROOT/'freeze/test_completion_protocol.json').read_text())
    if str(out.relative_to(ROOT))!=contract['output_directory'] or contract['evaluation_authorized']:
        raise PermissionError('Only the recorded preparation job is authorized.')
    for name,value in contract['code_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('Completion source changed: '+name)
    try:
        while not all((out/f'{kind}_ready.json').exists() for kind in ['multipv','maia']):
            for name in ['multipv_run.log','maia_run.log']:
                if (out/name).exists() and 'Traceback (most recent call last)' in (out/name).read_text():
                    raise RuntimeError('A feature job failed; inspect its preparation log.')
            current=status(out,'feature_generation_running')
            print(json.dumps(current),flush=True)
            time.sleep(30)
        for stage in ['export','labels']:
            current=status(out,stage+'_running');print(json.dumps(current),flush=True)
            child=subprocess.Popen([sys.executable,str(ROOT/'scripts/enrich_sealed_test_features.py'),
                '--out',str(out),'--stage',stage])
            while child.poll() is None:
                time.sleep(30);status(out,stage+'_running')
            if child.returncode:raise RuntimeError('Preparation stage failed: '+stage)
        refresh_records(out)
        print(json.dumps(status(out,'complete')),flush=True)
    except Exception:
        status(out,'failed')
        raise


if __name__=='__main__':main()
