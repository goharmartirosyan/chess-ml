#!/usr/bin/env python3
"""Independent held-out-development reproduction and frozen-bundle checks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import frozen_chess_evaluation as f


def validate(out, bundle=None):
    if out.exists():raise FileExistsError('Use a new validation directory.')
    c=f.load_config();x,y=f.read_development()
    mapping=pd.read_csv(ROOT/f.FOLDS)
    mapping=mapping[mapping.protocol.eq('game')].set_index('row_index').loc[x.row_index]
    if not np.array_equal(mapping.site,x.site):raise ValueError('Fold identity mismatch.')
    train=mapping.outer_fold.ne(1).to_numpy();held=~train
    if set(x.loc[train,'site'])&set(x.loc[held,'site']):raise ValueError('Game overlap.')
    training=x.loc[train].reset_index(drop=True);target=y.loc[train].reset_index(drop=True)
    models,constants,fit=f.fit_models(training,target,c)
    features=x.loc[held].drop(columns='row_index').reset_index(drop=True)
    labels=y.loc[held].reset_index(drop=True)
    predictions=f.predict_features(models,constants,features,c,allow_missing_maia=True)
    identities=x[['row_index','row_id']]
    scored=predictions.merge(identities,on='row_id',validate='many_to_one')
    results=[]
    checks=[
        ('primary','notebook_runs/final_optuna_development_20261117_v1/outer_predictions.csv.gz',
         'model','fixed_lightgbm','prediction'),
        ('ten','notebook_runs/maia_bounded_four_archive_20261117_v1/regression/change_in_damage_10ply/outer_predictions.csv.gz',
         'variant',None,'prediction'),
        ('reply','notebook_runs/simple_reply_baseline_development_20261118_v1/outer_predictions.csv.gz',
         'method',None,'probability')]
    for task,path,model_column,_,value_column in checks:
        old=pd.read_csv(ROOT/path,low_memory=False)
        old=old[old.protocol.eq('game')&old.outer_fold.eq(1)]
        if task=='primary':
            old=old[old.task.eq('regression')&old.model.eq('fixed_lightgbm')].copy()
            old['name']='primary_'+old.target.astype(str)
        elif task=='ten':
            old=old[old.variant.isin(['fixed_lightgbm','plus_maia'])].copy()
            old['name']=old.variant.map({'fixed_lightgbm':'ten_control','plus_maia':'ten_maia'})
        else:
            old=old[old.method.isin(['simple_lightgbm','simple_plus_maia_lightgbm'])].copy()
            old['name']=old.method.map({'simple_lightgbm':'reply_control','simple_plus_maia_lightgbm':'reply_maia'})
        for name,part in old.groupby('name'):
            new=scored[scored.model.eq(name)]
            joined=part[['row_index',value_column]].merge(new[['row_index','prediction']],
                on='row_index',validate='one_to_one',suffixes=('_saved','_new'))
            left='prediction_saved' if value_column=='prediction' else value_column
            right='prediction_new' if value_column=='prediction' else 'prediction'
            error=float(np.abs(joined[left]-joined[right]).max())
            if len(joined)!=len(part) or error>1e-8:
                raise AssertionError(f'Development prediction mismatch: {name}, {error}')
            results.append(dict(model=name,matched_rows=len(joined),max_absolute_difference=error))
    # The reporter operates on held-out games; these are not full-fit training scores.
    metrics,intervals=f.score_predictions(features,labels,predictions,constants,c)
    out.mkdir(parents=True)
    metrics.to_csv(out/'heldout_metrics.csv',index=False)
    intervals.to_csv(out/'heldout_paired_intervals.csv',index=False)
    f.subgroup_metrics(features,labels,predictions,constants,c).to_csv(out/'heldout_subgroups.csv',index=False)
    predictions.to_csv(out/'heldout_predictions_before_labels.csv.gz',index=False)
    if bundle is not None:
        frozen,manifest=f.load_bundle(bundle)
        # Round-trip inference check only; no performance claim on full-fit training rows.
        subset=features[features[f.PROB].notna()].iloc[:20]
        a=f.predict_features(frozen,manifest['constants'],subset,c)
        if not np.isfinite(a.prediction).all():raise ValueError('Invalid frozen inference.')
    result=dict(status='passed',scope='Saved game outer fold 1; development only',
        train_rows=int(train.sum()),heldout_rows=int(held.sum()),
        train_games=x.loc[train,'site'].nunique(),heldout_games=features.site.nunique(),
        game_overlap=0,model_reproductions=results,bootstrap_repeats=c['bootstrap_repeats'],
        configuration_sha256=f.digest(f.CONFIG),versions=f.versions(),test_access=False,
        source_hashes=f.DEVELOPMENT_HASHES)
    f.write_json(out/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle',type=Path)
    a=p.parse_args();validate(a.out,a.bundle)
