#!/usr/bin/env python3
"""Verify frozen Maia assets and reproduce saved development policy outputs.

Reads only named development artifacts. Performs no engine calls or test access.
Run in the original Maia-2 environment, separately from the modeling environment.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import maia2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import human_reply_policy as core


def validate(out):
    if out.exists():raise FileExistsError('Use a new validation output.')
    source=ROOT/'notebook_runs/maia_bounded_four_archive_20261117_v1'
    original=ROOT/'notebook_runs/human_reply_policy_development_20261117_v1'
    protocol=json.loads((original/'protocol.json').read_text())
    ready=json.loads((source/'features_ready.json').read_text())
    package=Path(maia2.__file__).parent
    checks={}
    for name,value in protocol['package_file_sha256'].items():
        actual=core.digest(package/name)
        if actual!=value:raise ValueError('Installed policy source differs: '+name)
        checks['package/'+name]=actual
    for kind,value in protocol['model_sha256'].items():
        path=original/'model_assets'/f'{kind}_model.pt'
        if core.digest(path)!=value:raise ValueError('Checkpoint differs: '+kind)
        checks[kind+'_checkpoint']=value
    features=pd.read_csv(source/'maia_bounds_features.csv')
    if core.digest(source/'maia_bounds_features.csv')!=ready['feature_sha256']:
        raise ValueError('Saved bounded feature changed.')
    np.testing.assert_allclose(features.maia_preserving_reply_probability,
        (features.maia_probability_lower+features.maia_probability_upper)/2,atol=1e-12)
    if (features.probability_bound_width.max()>.025+1e-10 or
        not features.maia_preserving_reply_probability.between(0,1).all()):
        raise ValueError('Invalid frozen integration bounds.')
    inputs=pd.read_csv(source/'pre_reply_inputs.csv')
    four={f'lichess_db_standard_rated_{m}.pgn.zst' for m in ['2017-04','2017-06','2017-07','2018-02']}
    if set(inputs.source_pgn)!=four or len(inputs)!=15953:
        raise ValueError('Only four-archive development inputs are permitted.')
    # Replay full original 32-position batches, excluding the reused pilot rows.
    # Changing the batch shape can change float32 kernels by a few parts per million.
    pilot_ids=set(pd.read_csv(original/'pre_reply_inputs.csv',usecols=['row_index']).row_index)
    policy_by_row=inputs.set_index('row_index').policy_model.to_dict()
    selected_ids={kind:[] for kind in sorted(inputs.policy_model.unique())}
    saved={}
    with (source/'human_move_probabilities.jsonl').open() as handle:
        for line in handle:
            record=json.loads(line)
            row=record['row_index'];kind=policy_by_row[row]
            if row not in pilot_ids and len(selected_ids[kind])<32:
                selected_ids[kind].append(row);saved[row]=record['move_probabilities']
    selected=pd.concat([inputs.set_index('row_index',drop=False).loc[ids] for ids in selected_ids.values()])
    torch.set_num_threads(4);torch.manual_seed(1701)
    results=[]
    for kind,part in selected.groupby('policy_model'):
        network=core.load_policy(kind,original/'model_assets')
        inferred=core.full_precision_policy(network,part[['root_fen','active_elo','other_elo']])
        for row,actual in zip(part.itertuples(index=False),inferred):
            expected=saved[row.row_index]
            if set(actual)!=set(expected):raise ValueError('Legal policy moves changed.')
            error=max(abs(actual[move]-expected[move]) for move in expected)
            if error>1e-6:raise ValueError(f'Policy inference differs from saved development output: {kind} {row.row_index} {error}')
            results.append(dict(row_index=int(row.row_index),policy=kind,
                                max_absolute_probability_difference=float(error),legal_moves=len(actual)))
    names=['maia2','torch','numpy','pandas','scipy','scikit-learn','chess','joblib',
           'threadpoolctl','einops','gdown','pyzstd','requests','pyyaml','tqdm']
    actual_versions={n:metadata.version(n) for n in names}
    for name in ['maia2','torch','numpy','pandas','scipy','scikit-learn','chess']:
        if actual_versions[name]!=protocol['package_versions'][name]:
            raise ValueError('Policy environment version changed: '+name)
    out.mkdir(parents=True)
    result=dict(status='passed',scope='First original 32-position nonpilot development batch per checkpoint',
        inference_batch_size=32,absolute_probability_tolerance=1e-6,
        python=platform.python_version(),packages=actual_versions,asset_sha256=checks,
        inference_reproductions=results,bounded_rows=len(features),
        maximum_bound_width=float(features.probability_bound_width.max()),
        source_sha256={str(p.relative_to(ROOT)):core.digest(p) for p in [
            source/'pre_reply_inputs.csv',source/'human_move_probabilities.jsonl',
            source/'maia_bounds_features.csv',ROOT/'scripts/human_reply_policy.py',
            Path(__file__).resolve()]},engine_calls=0,test_access=False)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'requirements-policy.txt').write_text('\n'.join(
        name+'=='+version for name,version in sorted(actual_versions.items()))+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    validate(p.parse_args().out)
