#!/usr/bin/env python3
"""Audit prepared tables and frozen artifact integrity without model inference.

Only preparation invariants are checked. No accuracy/error/correlation metric,
test distribution, fitted estimator loading or evaluation marker is produced.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import chess

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ['source_pgn','game_index','ply','site']
META = ['row_id']+IDENTITY+['win_before','event','time_control','multipv_lines_returned']
ARCHIVE = 'lichess_db_standard_rated_2025-03.pgn.zst'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_tables(features,labels,bounds,status,config):
    full=config['features']['full'];prob=config['features']['maia']
    require(set(features)==set(META+full+[prob]),'Unexpected/missing predictor columns.')
    expected_labels=['row_id','win_after','actual_preserved']+[f'{prefix}_{h}ply'
        for h in config['horizons'] for prefix in ['win_after','change_in_damage']]
    require(set(labels)==set(expected_labels),'Unexpected/missing label columns.')
    require(len(features)>0,'No prepared rows.')
    require(not features[IDENTITY+['row_id']].isna().any().any(),'Missing row identity.')
    require(not features.duplicated(IDENTITY[:3]).any(),'Duplicated source/game/ply.')
    require(not features.duplicated(['site','ply']).any(),'Duplicated game/ply.')
    require(features.source_pgn.eq(ARCHIVE).all(),'Unexpected archive identity.')
    for col in ['game_index','ply']:
        values=pd.to_numeric(features[col],errors='raise')
        require((np.isfinite(values)&values.gt(0)&values.eq(values.astype(int))).all(),'Invalid integer identity.')
    row_ids=features[IDENTITY].apply(lambda r:hashlib.sha256(json.dumps(
        [str(r.source_pgn),int(r.game_index),int(r.ply),str(r.site)],separators=(',',':')).encode()).hexdigest(),axis=1)
    require(np.array_equal(row_ids,features.row_id),'Row identity hash differs.')
    require(not features.row_id.duplicated().any(),'Duplicate feature identity.')
    require(not labels.row_id.duplicated().any(),'Duplicate label identity.')
    require(np.array_equal(features.row_id,labels.row_id),'Feature/label order differs.')
    require(features.win_before.between(10,90).all()&features.initial_damage.gt(20).all(),'Fixed eligibility differs.')
    numeric=[c for c in full if c!='game_phase']
    require(not np.isinf(features[numeric].to_numpy(float)).any(),'Infinite numeric feature.')
    lines=features.multipv_lines_returned
    require((lines.between(1,5)&lines.eq(lines.astype(int))).all(),'Invalid MultiPV line count.')
    good=features.multipv_good_reply_count_10pp
    require((good.ge(1)&good.le(lines)&good.eq(good.astype(int))).all(),'Invalid MultiPV good-reply count.')
    gap=features.multipv_best_second_gap_pp
    require(np.array_equal(gap.isna(),lines.eq(1)),'One-legal-reply gap semantics differ.')
    require(gap.dropna().between(0,100).all(),'Invalid MultiPV gap.')
    require(features[prob].between(0,1).all(),'Incomplete/invalid Maia probability.')
    require(not bounds.row_id.duplicated().any(),'Duplicate bound identities.')
    bounds=bounds.set_index('row_id').loc[features.row_id]
    lower=bounds.maia_probability_lower.to_numpy(float)
    upper=bounds.maia_probability_upper.to_numpy(float)
    require((np.isfinite(lower)&np.isfinite(upper)&(lower>=0)&(upper<=1)&(upper>=lower)&
        (upper-lower<=.025+1e-10)).all(),'Invalid frozen integration bounds.')
    require(np.allclose(features[prob],(lower+upper)/2,atol=1e-12,rtol=0),'Maia midpoint differs.')
    require(labels.win_after.between(0,100).all(),'Invalid initial chance label.')
    require(np.allclose(features.initial_damage,features.win_before-labels.win_after,atol=1e-9,rtol=0),
        'Initial-damage identity differs.')
    for h in config['horizons']:
        future=labels[f'win_after_{h}ply'];target=labels[f'change_in_damage_{h}ply']
        require(future.between(0,100).all(),'Invalid horizon chance label.')
        require(np.allclose(target,labels.win_after-future,atol=1e-9,rtol=0),'Horizon target equation differs.')
    actual=labels.actual_preserved
    require(actual.dropna().isin([0,1]).all(),'Immediate labels must be 0/1 or missing.')
    require(not status.row_id.duplicated().any(),'Duplicate reply-status rows.')
    statuses=status.set_index('row_id').loc[features.row_id,'status']
    require(statuses.isin(['ok','missing_actual_reply']).all(),'Unresolved actual-reply reconstruction.')
    require(np.array_equal(actual.isna(),statuses.eq('missing_actual_reply')),'Missing reply labels were changed.')
    return dict(feature_schema=True,label_schema=True,row_alignment=True,row_identity=True,
        fixed_eligibility=True,multipv_semantics=True,maia_bounds_and_midpoint=True,
        target_equations=True,missing_reply_handling=True)


def validate_reply_metadata(features,roots):
    require(not roots.row_id.duplicated().any(),'Duplicate reconstructed root identities.')
    roots=roots.set_index('row_id').loc[features.row_id]
    legal=roots.root_fen.map(lambda fen:chess.Board(fen).legal_moves.count()).to_numpy()
    require((legal>0).all(),'A reconstructed position has no legal reply.')
    require(np.array_equal(features.multipv_lines_returned,np.minimum(5,legal)),
        'MultiPV line coverage differs from the legal reply count.')
    require(np.array_equal(features.multipv_best_second_gap_pp.isna(),legal==1),
        'Missing gap does not correspond to exactly one legal reply.')
    return True


def committed_reference(paths):
    """Read-only: require the actual source/config files to exist at HEAD."""
    try:
        reference=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,stderr=subprocess.DEVNULL).decode().strip()
        for name in paths:
            content=subprocess.check_output(['git','show',reference+':'+name],cwd=ROOT,stderr=subprocess.DEVNULL)
            if hashlib.sha256(content).hexdigest()!=digest(ROOT/name):return None
        return reference
    except subprocess.CalledProcessError:
        return None


def audit(prepared):
    prepared=Path(prepared).resolve()
    protocol=json.loads((ROOT/'freeze/test_enrichment_protocol.json').read_text())
    require(prepared==ROOT/protocol['local_output_directory'],'Unrecorded preparation directory.')
    completion=json.loads((prepared/'preparation_complete.json').read_text())
    require(json.loads((prepared/'preparation_status.json').read_text())['stage']=='complete',
        'Preparation completion process has not finished.')
    for name,key in [('features.csv','features_sha256'),('labels.csv','labels_sha256'),
                     ('test_with_frozen_features.csv','combined_sha256')]:
        require(digest(prepared/name)==completion[key],'Prepared file hash differs: '+name)
    require(not completion['predictions_run'] and not completion['performance_metrics_computed'],
        'Preparation record indicates model evaluation.')
    config=json.loads((ROOT/'freeze/approved_models.json').read_text())
    inputs=json.loads((prepared/'input_manifest.json').read_text())
    for kind in ['features','labels']:
        require(inputs[kind]['path']==kind+'.csv','Unexpected input-manifest path.')
        require(inputs[kind]['sha256']==digest(prepared/(kind+'.csv')),'Input manifest checksum differs.')
    input_protocol=ROOT/'freeze/final_input_protocol.json'
    require(inputs['input_protocol_sha256']==digest(input_protocol),'Input protocol hash differs.')
    # Data validation only; never compare a prediction against these labels.
    features=pd.read_csv(prepared/'features.csv')
    labels=pd.read_csv(prepared/'labels.csv')
    checks=validate_tables(features,labels,pd.read_csv(prepared/'maia_bounds_features.csv'),
        pd.read_csv(prepared/'reply_label_status.csv'),config)
    checks['legal_reply_coverage']=validate_reply_metadata(features,pd.read_csv(
        prepared/'pre_reply_inputs.csv',usecols=['row_id','root_fen']))
    fitted=json.loads((ROOT/'freeze/fitted_models.json').read_text())
    require(platform.python_version()==fitted['versions']['python'],'Modeling Python version differs.')
    for name,value in fitted['versions']['packages'].items():
        require(metadata.version(name)==value,'Modeling dependency version differs: '+name)
    distribution=metadata.distribution('lightgbm')
    for name,value in fitted['versions']['lightgbm_native_sha256'].items():
        require(digest(distribution.locate_file('lightgbm/lib/'+name))==value,'Native LightGBM library differs.')
    bundle=ROOT/fitted['bundle_path']
    require(digest(bundle/'manifest.json')==fitted['bundle_manifest_sha256'],'Fitted manifest differs.')
    for name,value in fitted['artifacts'].items():
        require(digest(bundle/name)==value,'Frozen fitted artifact differs: '+name)
    for name,value in fitted['code_sha256'].items():
        require(digest(ROOT/name)==value,'Frozen model source differs: '+name)
    members=pd.read_csv(bundle/'training_membership.csv.gz',usecols=['site'])
    require(not set(features.site)&set(members.site),'Development/test game overlap.')
    require(not (bundle/'FINAL_EVALUATION_STARTED.json').exists(),'Final evaluation has already started.')
    for name,value in protocol['source_sha256'].items():
        require(digest(ROOT/name)==value,'Enrichment source/cache differs: '+name)
    for name,value in protocol['base_output_sha256'].items():
        require(digest(ROOT/'test_data/march_2025_prepared_v2'/name)==value,'Base preparation output differs: '+name)
    source_protocol=json.loads(input_protocol.read_text())
    roster=source_protocol['game_ranges_or_sampling']
    local_manifest=ROOT/roster['input_roster']
    require(digest(local_manifest)==roster['input_roster_sha256'],'Local source manifest differs.')
    original=json.loads(local_manifest.read_text())
    for item in original['inputs']:
        require(digest(ROOT/item['path'])==item['sha256'],'Original input differs: '+item['path'])
    blockers=list(source_protocol.get('blockers',[]))
    if source_protocol['status']!='frozen' and not blockers:
        blockers.append('The final input protocol remains unresolved.')
    code_manifest=json.loads((ROOT/'freeze/code_manifest.json').read_text())
    release_paths=sorted(set(code_manifest['code_sha256'])|{
        'freeze/approved_models.json','freeze/fitted_models.json','freeze/final_input_protocol.json',
        'freeze/requirements-modeling.txt','freeze/requirements-policy.txt',
        'freeze/environment-modeling.json','freeze/environment-policy.json',
        'freeze/test_enrichment_protocol.json','freeze/test_completion_protocol.json',
        'freeze/final_evaluation_invocation.json',
        'MARCH_2025_EVALUATION_PLAN.md','MARCH_2025_EVALUATION_APPENDIX.md'})
    commit=committed_reference(release_paths)
    if commit is None:blockers.append('The final source/configuration freeze awaits a user-created commit; no commit was made by this audit.')
    result=dict(date='2026-11-21',status='prepared_data_checks_passed',prepared_rows=len(features),
        checks=dict(checks,game_overlap_absent=True,fitted_artifacts_unchanged=True,
            source_hashes_verified=True,input_manifest_verified=True,modeling_environment_verified=True,
            original_inputs_unchanged=True,base_tables_unchanged=True,
            final_evaluation_not_started=True),
        data_ready_for_scoring=True,final_input_recipe_frozen=source_protocol['status']=='frozen',
        committed_freeze_reference=commit,
        final_freeze_ready=source_protocol['status']=='frozen' and not blockers,
        final_evaluation_authorized=False,blockers=blockers,model_predictions_run=False,
        provenance_limitations=source_protocol.get('provenance_limitations',[]),
        performance_metrics_computed=False,distributions_inspected=False,
        prepared_directory=str(prepared.relative_to(ROOT)),input_manifest_sha256=digest(prepared/'input_manifest.json'))
    (ROOT/'freeze/test_readiness_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    manifest_path=ROOT/'freeze/code_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['freeze_records_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'freeze').iterdir())
        if p.is_file() and p.name!='code_manifest.json'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    audit(parser.parse_args().prepared)
