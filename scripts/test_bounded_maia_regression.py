#!/usr/bin/env python3
"""Fixed LightGBM vs fixed LightGBM + bounded Maia feature, four archives only.

Evaluate immediate reply regret first. Read/fit the ten-ply target only if the
predeclared regret gate passes. All preprocessing remains inside training folds.
No Optuna or modification to previous datasets/results/configurations.
"""
import argparse
import importlib.metadata
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tune_chess import SETS,EXPECTED_ARCHIVES,pipeline
from scripts.tune_recovery_threshold import digest,players

DATA=ROOT/'development_with_multipv.csv'
TARGETS=ROOT/'notebook_runs/opponent_reply_targets_development_v1/reply_targets.csv'
PRIOR=ROOT/'notebook_runs/final_optuna_development_20261117_v1'
FOUR={f'lichess_db_standard_rated_{x}.pgn.zst' for x in ['2017-04','2017-06','2017-07','2018-02']}
BASE=list(SETS['full_multipv'])
FEATURE='maia_preserving_reply_probability'
VARIANTS={'fixed_lightgbm':BASE,'plus_maia':BASE+[FEATURE],
          'plus_maia_lower':BASE+['maia_probability_lower'],
          'plus_maia_upper':BASE+['maia_probability_upper']}
SEED,JOBS,REPEATS=1701,2,2000


def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def base_frame():
    # Horizon targets deliberately omitted until after the regret gate.
    columns=list(set(BASE+['source_pgn','game_index','ply','site','white','black','win_before','time_control']))
    frame=pd.read_csv(DATA,usecols=columns)
    if (len(frame)!=23167 or frame.site.nunique()!=6512 or
        set(frame.source_pgn.map(lambda x:Path(x).name))!=EXPECTED_ARCHIVES or
        frame.duplicated(['source_pgn','game_index','ply']).any() or
        not (frame.initial_damage.gt(20)&frame.win_before.between(10,90)).all()):
        raise ValueError('Canonical development data changed.')
    frame['row_index']=frame.index
    return frame[frame.source_pgn.map(lambda x:Path(x).name).isin(FOUR)].reset_index(drop=True)


def split(frame,mapping,fold,player):
    mapped=mapping.set_index('row_index').reindex(frame.row_index)
    if mapped.isna().any().any() or not np.array_equal(mapped.site,frame.site):raise ValueError('Outer game map mismatch.')
    valid=np.flatnonzero(mapped.outer_fold.to_numpy()==fold)
    train=np.flatnonzero(mapped.outer_fold.to_numpy()!=fold)
    if player:
        held=players(frame.iloc[valid]);candidate=frame.iloc[train]
        bad=(candidate.white.astype(str).str.strip().str.casefold().isin(held)|
             candidate.black.astype(str).str.strip().str.casefold().isin(held))
        bad_games=set(candidate.loc[bad,'site'])
        train=train[~candidate.site.isin(bad_games).to_numpy()]
        if not players(frame.iloc[train]).isdisjoint(held):raise ValueError('Player leakage.')
    if not set(frame.iloc[train].site).isdisjoint(frame.iloc[valid].site):raise ValueError('Game leakage.')
    return train,valid


def check_predictors(columns):
    allowed=set(BASE+[FEATURE,'maia_probability_lower','maia_probability_upper'])
    if not columns or len(columns)!=len(set(columns)) or set(columns)-allowed:
        raise ValueError('Unapproved model predictors.')


def prepare(out):
    if (out/'regression_protocol.json').exists():raise FileExistsError('Regression protocol already exists.')
    feature_protocol=json.loads((out/'protocol.json').read_text())
    if set(feature_protocol['source_archives'])!=FOUR:raise ValueError('Only four pretraining-date-disjoint archives allowed.')
    frame=base_frame();maps=pd.read_csv(PRIOR/'outer_fold_assignments.csv')
    rows=[]
    for protocol in ['game','player']:
        mapping=maps[maps.protocol.eq(protocol)]
        for fold in [1,2,3]:
            train,valid=split(frame,mapping,fold,protocol=='player')
            rows.append(dict(protocol=protocol,outer_fold=fold,train_rows=len(train),valid_rows=len(valid),
                             train_games=frame.iloc[train].site.nunique(),valid_games=frame.iloc[valid].site.nunique(),
                             game_overlap=0,player_overlap=0 if protocol=='player' else None))
    frozen=json.loads((PRIOR/'protocol.json').read_text())
    if frozen['predictors_by_model']['fixed_lightgbm']!=BASE:raise ValueError('Frozen predictor list changed.')
    for predictors in VARIANTS.values():check_predictors(predictors)
    source_paths=[DATA,TARGETS,PRIOR/'protocol.json',PRIOR/'outer_fold_assignments.csv',
                  ROOT/'config.json',ROOT/'my_tuning/selected.json',ROOT/'tune_chess.py',
                  ROOT/'scripts/tune_recovery_threshold.py',out/'protocol.json',Path(__file__).resolve()]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_archives=sorted(FOUR),
        fixed_params=frozen['fixed_params'],predictors_by_variant=VARIANTS,seed=SEED,jobs=JOBS,
        comparator='Original 24 predictors, refitted only on the same four-archive training games.',
        primary_addition=FEATURE,endpoint_variants='Sensitivity only; not used to select the primary feature.',
        stages=['opponent_reply_regret_pp','change_in_damage_10ply only if gate passes'],
        gate={'game_gain_lower95_strictly_above':0.,'player_point_gain_nonnegative':True},
        timing='Feature bounds generated for every allowed row before parsing regression targets. '
            'Ten-ply targets are not parsed until the gate passes.',
        preprocessing='Original fold-fitted median imputer/missing indicators and phase encoder. No learned feature selection.',
        interval='Paired game-cluster percentile bootstrap, 2000 repeats, fixed OOF predictions. '
            'Positive gain = fixed-LightGBM MAE minus augmented MAE. Endpoint variants are '
            'sensitivity checks, not statistical bounds on regression predictions.',
        confidence_caveat='Development-only, unadjusted and conditional on fitted models. '
            'Ten-ply evaluation is conditionally triggered, not independent confirmation.',
        bootstrap_repeats=REPEATS,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','lightgbm']})
    save_json(out/'regression_protocol.json',protocol)
    pd.DataFrame(rows).to_csv(out/'regression_split_validation.csv',index=False)
    save_json(out/'regression_predictor_allowlists.json',VARIANTS)
    print('Prepared fixed-model protocol and sequential ten-ply gate.',flush=True)


def metrics(part):
    y,p=part.actual.to_numpy(float),part.prediction.to_numpy(float)
    constant=part.variant.iloc[0] in ['zero','training_mean','training_median']
    recovered=(1+y/part.initial_damage.to_numpy(float)<.75
               if part.target.iloc[0]=='change_in_damage_10ply' else np.zeros(len(part),bool))
    return dict(n=len(part),games=part.game.nunique(),mae=float(np.abs(y-p).mean()),
        spearman=float(spearmanr(y,p).statistic) if not constant and np.ptp(y) and np.ptp(p) else np.nan,
        recovered_n_diagnostic=int(recovered.sum()),
        recovered_only_mae_diagnostic=float(np.abs(y[recovered]-p[recovered]).mean()) if recovered.any() else np.nan)


def evaluate_target(out,frame,target,protocol):
    target_dir=out/'regression'/target;target_dir.mkdir(parents=True,exist_ok=True)
    maps=pd.read_csv(PRIOR/'outer_fold_assignments.csv');fold_audits=[]
    y=frame[target].to_numpy(float)
    for mode in ['game','player']:
        for fold in [1,2,3]:
            path=target_dir/f'{mode}_fold{fold}_predictions.csv'
            if path.exists():continue
            train,valid=split(frame,maps[maps.protocol.eq(mode)],fold,mode=='player')
            train=train[np.isfinite(y[train])];valid=valid[np.isfinite(y[valid])]
            if not len(train) or not len(valid):raise ValueError('Empty target split.')
            predictions={}
            for variant,columns in VARIANTS.items():
                check_predictors(columns)
                model=pipeline(columns,protocol['fixed_params'],SEED,JOBS)
                model.fit(frame.iloc[train][columns],y[train])
                predictions[variant]=model.predict(frame.iloc[valid][columns])
            predictions.update(zero=np.zeros(len(valid)),
                training_mean=np.full(len(valid),y[train].mean()),
                training_median=np.full(len(valid),np.median(y[train])))
            blocks=[]
            for name,p in predictions.items():
                if not np.isfinite(p).all():raise ValueError('Invalid model prediction.')
                blocks.append(pd.DataFrame(dict(protocol=mode,outer_fold=fold,target=target,variant=name,
                    row_index=frame.iloc[valid].row_index.to_numpy(),game=frame.iloc[valid].site.to_numpy(),
                    actual=y[valid],prediction=p,initial_damage=frame.iloc[valid].initial_damage.to_numpy())))
            pd.concat(blocks).to_csv(path,index=False)
            fold_audits.append(dict(protocol=mode,outer_fold=fold,train_rows=len(train),valid_rows=len(valid),
                train_games=frame.iloc[train].site.nunique(),valid_games=frame.iloc[valid].site.nunique(),
                game_overlap=0,player_overlap=0 if mode=='player' else None))
            print(f'{target}, {mode} fold {fold}: fixed baseline and three Maia variants complete.',flush=True)
    if fold_audits:pd.DataFrame(fold_audits).to_csv(target_dir/'eligible_split_checks.csv',index=False)
    oof=pd.concat([pd.read_csv(path) for path in sorted(target_dir.glob('*_predictions.csv'))],ignore_index=True)
    oof.to_csv(target_dir/'outer_predictions.csv.gz',index=False)
    summary=[];fold_metrics=[];robust=[];intervals=[]
    for (mode,name),part in oof.groupby(['protocol','variant']):
        if part.row_index.duplicated().any() or len(part)!=np.isfinite(y).sum():raise ValueError('OOF row coverage mismatch.')
        summary.append(dict(protocol=mode,variant=name,target=target,**metrics(part)))
        for fold,block in part.groupby('outer_fold'):
            fold_metrics.append(dict(protocol=mode,variant=name,target=target,outer_fold=fold,**metrics(block)))
        for threshold in [25,30]:
            robust.append(dict(protocol=mode,variant=name,target=target,initial_damage_threshold=threshold,
                               **metrics(part[part.initial_damage>threshold])))
    for mode,part in oof.groupby('protocol'):
        for threshold in [20,25,30]:
            selected=part[part.initial_damage>threshold]
            wide=selected.pivot(index='row_index',columns='variant',values='prediction').sort_index()
            ref=selected.drop_duplicates('row_index').set_index('row_index').loc[wide.index]
            if wide.isna().any().any():raise ValueError('Variants use different validation rows.')
            for _,block in selected.groupby('variant'):
                aligned=block.sort_values('row_index')
                if not np.array_equal(aligned[['row_index','game','outer_fold','actual']],
                    ref.reset_index()[['row_index','game','outer_fold','actual']]):raise ValueError('Unpaired comparisons.')
            losses=np.abs(wide.to_numpy()-ref.actual.to_numpy(float)[:,None])
            ids,games=pd.factorize(ref.game,sort=True)
            sums=np.zeros((len(games),losses.shape[1]));np.add.at(sums,ids,losses);sizes=np.bincount(ids)
            rng=np.random.default_rng(SEED);draws=np.empty((REPEATS,losses.shape[1]))
            for start in range(0,REPEATS,50):
                n=min(50,REPEATS-start);index=rng.integers(0,len(games),(n,len(games)))
                draws[start:start+n]=sums[index].sum(axis=1)/sizes[index].sum(axis=1)[:,None]
            point=losses.mean(axis=0)
            for i,name in enumerate(wide.columns):
                for comparator in ['fixed_lightgbm','zero','training_median']:
                    j=wide.columns.get_loc(comparator);gain=draws[:,j]-draws[:,i]
                    lo,hi=np.quantile(gain,[.025,.975]);relative=100*gain/draws[:,j]
                    rlo,rhi=np.quantile(relative,[.025,.975])
                    intervals.append(dict(protocol=mode,target=target,variant=name,comparator=comparator,
                        initial_damage_threshold=threshold,n=len(ref),games=len(games),
                        mae_improvement_pp=point[j]-point[i],lower95=lo,upper95=hi,
                        improvement_percent=100*(point[j]-point[i])/point[j],
                        percent_lower95=rlo,percent_upper95=rhi,bootstrap_repeats=REPEATS))
    summary=pd.DataFrame(summary);intervals=pd.DataFrame(intervals)
    summary.to_csv(target_dir/'summary.csv',index=False)
    intervals.to_csv(target_dir/'paired_game_bootstrap_intervals.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(target_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(robust).to_csv(target_dir/'robustness_gt25_gt30.csv',index=False)
    metadata=frame[['row_index','player_elo','time_control','game_phase']].copy()
    metadata['elo_band']=pd.cut(metadata.player_elo,[-np.inf,1600,2000,np.inf],right=False,
                                labels=['<1600','1600–<2000','>=2000'])
    joined=oof.merge(metadata,on='row_index',validate='many_to_one');subgroups=[]
    for field in ['elo_band','time_control','game_phase']:
        for key,block in joined.groupby(['protocol','variant',field],observed=True,dropna=False):
            subgroups.append(dict(protocol=key[0],variant=key[1],target=target,group_type=field,group=str(key[2]),**metrics(block)))
    pd.DataFrame(subgroups).to_csv(target_dir/'subgroup_diagnostics.csv',index=False)
    return summary,intervals


def run(out):
    protocol=json.loads((out/'regression_protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Source changed: {name}')
    if (out/'regression_complete.json').exists():print('Regression already complete.');return
    ready=json.loads((out/'features_ready.json').read_text())
    if digest(out/'maia_bounds_features.csv')!=ready['feature_sha256']:raise ValueError('Feature artifact changed.')
    frame=base_frame();features=pd.read_csv(out/'maia_bounds_features.csv')
    if len(features)!=len(frame) or set(features.row_index)!=set(frame.row_index):raise ValueError('Feature coverage differs.')
    keys=['row_index','source_pgn','game_index','ply','site']
    frame=frame.merge(features[keys+[FEATURE,'maia_probability_lower','maia_probability_upper']],
                      on=keys,validate='one_to_one',sort=False)
    if not frame[FEATURE].between(0,1).all():raise ValueError('Missing/out-of-range Maia feature.')
    reply=pd.read_csv(TARGETS,usecols=['source_pgn','game_index','ply','site','status','opponent_reply_regret_pp'])
    frame=frame.merge(reply,on=['source_pgn','game_index','ply','site'],how='left',validate='one_to_one',sort=False)
    if not frame.status.notna().all():raise ValueError('Target identity mismatch.')
    frame.loc[~frame.status.eq('ok'),'opponent_reply_regret_pp']=np.nan
    summary,ci=evaluate_target(out,frame,'opponent_reply_regret_pp',protocol)
    primary=ci[ci.variant.eq('plus_maia')&ci.comparator.eq('fixed_lightgbm')&ci.initial_damage_threshold.eq(20)]
    game=primary[primary.protocol.eq('game')].iloc[0];player=primary[primary.protocol.eq('player')].iloc[0]
    passed=bool(game.lower95>0 and player.mae_improvement_pp>=0)
    gate=dict(passed=passed,regret_game_gain=float(game.mae_improvement_pp),regret_game_lower95=float(game.lower95),
              regret_game_upper95=float(game.upper95),regret_player_gain=float(player.mae_improvement_pp),
              decision='Evaluate ten-ply change in damage.' if passed else 'Skip ten-ply target: primary regret gate did not pass.')
    save_json(out/'ten_ply_gate.json',gate);print(json.dumps(gate,indent=2),flush=True)
    if passed:
        # Only now parse the primary ten-ply target.
        target=pd.read_csv(DATA,usecols=['source_pgn','game_index','ply','change_in_damage_10ply'])
        target=target[target.source_pgn.map(lambda x:Path(x).name).isin(FOUR)]
        frame=frame.merge(target,on=['source_pgn','game_index','ply'],validate='one_to_one',sort=False)
        evaluate_target(out,frame,'change_in_damage_10ply',protocol)
    for name,expected in protocol['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Original source modified: {name}')
    save_json(out/'regression_complete.json',dict(status='complete',ten_ply_evaluated=passed,
        actual_feature_rows=len(frame),source_files_unchanged=True,new_optuna_trials=0,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:prepare(args.out.resolve())
    else:run(args.out.resolve())
