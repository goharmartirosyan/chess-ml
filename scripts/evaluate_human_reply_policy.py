#!/usr/bin/env python3
"""Evaluate saved pre-reply probability masses after prediction completion only."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'development_with_multipv.csv'
TARGETS=ROOT/'notebook_runs/opponent_reply_targets_development_v1/reply_targets.csv'
FOLDS=ROOT/'notebook_runs/final_optuna_development_20261117_v1/outer_fold_assignments.csv'
SEED,REPEATS=1701,2000


def players(frame):
    return set(frame[['white','black']].astype(str).apply(lambda x:x.str.strip().str.casefold()).to_numpy().ravel())


def purge_train(source,train,valid):
    held=players(source.loc[valid]);candidate=source.loc[train]
    bad=(candidate.white.astype(str).str.strip().str.casefold().isin(held)|
         candidate.black.astype(str).str.strip().str.casefold().isin(held))
    games=set(candidate.loc[bad,'site'])
    train=train[~candidate.site.isin(games).to_numpy()]
    if not players(source.loc[train]).isdisjoint(held):raise ValueError('Player leakage in calibration.')
    return train


def fit_calibration(probability,label):
    p=np.asarray(probability,float);y=np.asarray(label,int)
    use=(p>0)&(p<1)
    if len(np.unique(y[use]))!=2:return None
    x=np.log(p[use]/(1-p[use]))[:,None]
    calibrator=LogisticRegression(C=1.,solver='lbfgs',random_state=SEED,max_iter=1000)
    calibrator.fit(x,y[use])
    return calibrator


def calibrated(calibrator,probability):
    p=np.asarray(probability,float).copy()
    use=(p>0)&(p<1)
    if calibrator is not None:
        p[use]=calibrator.predict_proba(np.log(p[use]/(1-p[use]))[:,None])[:,1]
    return p


def scores(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    clipped=np.clip(p,1e-15,1-1e-15)
    return dict(n=len(y),event_prevalence=float(y.mean()),mean_probability=float(p.mean()),
        brier=float(np.mean((p-y)**2)),
        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))),
        roc_auc=float(roc_auc_score(y,p)) if np.unique(y).size==2 and np.ptp(p)>0 else np.nan,
        average_precision=float(average_precision_score(y,p)) if y.sum() else np.nan)


def paired_bootstrap(oof):
    rows=[]
    for (scope,protocol),part in oof.groupby(['scope','protocol']):
        wide=part.pivot(index='row_index',columns='method',values='probability').sort_index()
        ref=part.drop_duplicates('row_index').set_index('row_index').loc[wide.index]
        if wide.isna().any().any():raise ValueError('Unpaired event probability rows.')
        y=ref.actual_preserved.to_numpy(float)
        losses=(wide.to_numpy()-y[:,None])**2
        labels,games=pd.factorize(ref.game,sort=True)
        sums,sizes=np.zeros((len(games),losses.shape[1])),np.bincount(labels)
        np.add.at(sums,labels,losses)
        rng=np.random.default_rng(SEED)
        draws=np.empty((REPEATS,losses.shape[1]))
        for start in range(0,REPEATS,100):
            n=min(100,REPEATS-start);idx=rng.integers(0,len(games),(n,len(games)))
            draws[start:start+n]=sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None]
        point=losses.mean(axis=0)
        for i,name in enumerate(wide.columns):
            lo,hi=np.quantile(draws[:,i],[.025,.975])
            rows.append(dict(scope=scope,protocol=protocol,method=name,comparator='',metric='brier',
                             estimate=point[i],lower95=lo,upper95=hi,n=len(wide),games=len(games)))
            for comparator in ['training_prevalence','uniform_legal','favorite_acceptable_only','raw_policy_mass']:
                j=wide.columns.get_loc(comparator)
                gain=draws[:,j]-draws[:,i];lo,hi=np.quantile(gain,[.025,.975])
                rows.append(dict(scope=scope,protocol=protocol,method=name,comparator=comparator,
                    metric='brier_improvement',estimate=point[j]-point[i],lower95=lo,upper95=hi,
                    n=len(wide),games=len(games)))
    return pd.DataFrame(rows)


def markdown(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |',
           '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def evaluate(out):
    if not (out/'predictions_ready.json').exists():raise ValueError('Finish pre-reply predictions first.')
    p=pd.read_csv(out/'pre_reply_probabilities.csv')
    protocol=json.loads((out/'protocol.json').read_text())
    source=pd.read_csv(DATA,usecols=['source_pgn','game_index','ply','site','white','black','win_before','initial_damage'])
    source['row_index']=source.index
    keys=['source_pgn','game_index','ply']
    targets=pd.read_csv(TARGETS,usecols=keys+['status','actual_reply_uci','win_after_actual_reply'])
    source=source.merge(targets,on=keys,how='left',validate='one_to_one',sort=False).set_index('row_index')
    source['actual_preserved']=((source.win_before-source.win_after_actual_reply)/source.initial_damage>=.75).where(source.status.eq('ok'))
    # Ground truth construction and actual replies remain in this evaluation stage.
    labeled=p.merge(source[['status','actual_reply_uci','win_after_actual_reply','actual_preserved']],
                     left_on='row_index',right_index=True,validate='one_to_one')
    detailed=pd.read_csv(out/'all_legal_reply_scores.csv.gz')
    observed=labeled[labeled.status.eq('ok')]
    for r in observed.itertuples(index=False):
        move=detailed[detailed.row_index.eq(r.row_index)&detailed.reply_uci.eq(r.actual_reply_uci)]
        if len(move)!=1 or bool(move.iloc[0].preserves_075_damage)!=bool(r.actual_preserved):
            raise ValueError('Actual reply label disagrees with independently generated acceptable set.')
        if not np.isclose(move.iloc[0].original_player_chance,r.win_after_actual_reply,atol=1e-9):
            raise ValueError('Observed reply evaluation differs from pinned legacy targets.')
    labeled.to_csv(out/'evaluation_rows.csv',index=False)
    maps=pd.read_csv(FOLDS)
    rows=[];calibrators=[];audits=[]
    for scope in ['all_five_exploratory','predates_documented_pretraining']:
        allowed=~source.source_pgn.map(lambda x:Path(x).name.endswith('2019-02.pgn.zst'))
        if scope=='all_five_exploratory':allowed[:]=True
        eligible=labeled[labeled.status.eq('ok')]
        if scope!='all_five_exploratory':eligible=eligible[~eligible.pretraining_window_overlap]
        for split in ['game','player']:
            mapping=maps[maps.protocol.eq(split)].set_index('row_index').reindex(source.index)
            if mapping.isna().any().any() or not np.array_equal(mapping.site,source.site):raise ValueError('Saved game map changed.')
            for fold in [1,2,3]:
                valid_ids=source.index[mapping.outer_fold.eq(fold)]
                train_ids=source.index[~mapping.outer_fold.eq(fold)]
                if split=='player':train_ids=purge_train(source,train_ids,valid_ids)
                train_ids=train_ids[source.loc[train_ids].status.eq('ok')&allowed.loc[train_ids]]
                valid=eligible[eligible.row_index.isin(valid_ids)]
                train=eligible[eligible.row_index.isin(train_ids)]
                if not set(source.loc[train_ids].site).isdisjoint(valid.site):raise ValueError('Game leakage.')
                probability=float(source.loc[train_ids].actual_preserved.mean())
                cal=fit_calibration(train.probability_any_preserving_reply,train.actual_preserved)
                methods={
                    'raw_policy_mass':valid.probability_any_preserving_reply.to_numpy(),
                    'calibrated_policy_mass':calibrated(cal,valid.probability_any_preserving_reply),
                    'uniform_legal':valid.uniform_legal_acceptable_probability.to_numpy(),
                    'favorite_acceptable_only':valid.probability_favorite_acceptable_only.to_numpy(),
                    'training_prevalence':np.full(len(valid),probability)}
                for name,values in methods.items():
                    rows.append(pd.DataFrame(dict(scope=scope,protocol=split,outer_fold=fold,method=name,
                        row_index=valid.row_index.to_numpy(),game=valid.site.to_numpy(),
                        actual_preserved=valid.actual_preserved.to_numpy(bool),probability=values)))
                calibrators.append(dict(scope=scope,protocol=split,outer_fold=fold,
                    fit_rows=len(train),fit_games=train.site.nunique(),
                    slope=None if cal is None else float(cal.coef_[0,0]),
                    intercept=None if cal is None else float(cal.intercept_[0]),
                    training_prevalence=probability))
                audits.append(dict(scope=scope,protocol=split,outer_fold=fold,
                    prevalence_train_rows=len(train_ids),calibration_train_rows=len(train),
                    valid_rows=len(valid),game_overlap=0,
                    player_overlap=0 if split=='player' else None))
    oof=pd.concat(rows,ignore_index=True)
    oof.to_csv(out/'probability_oof_predictions.csv',index=False)
    pd.DataFrame(calibrators).to_csv(out/'calibration_by_fold.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'evaluation_split_checks.csv',index=False)
    summary=[];fold_metrics=[];calibration_bins=[]
    for key,part in oof.groupby(['scope','protocol','method']):
        tags=dict(zip(['scope','protocol','method'],key))
        summary.append(dict(**tags,games=part.game.nunique(),**scores(part.actual_preserved,part.probability)))
        for fold,f in part.groupby('outer_fold'):
            fold_metrics.append(dict(**tags,outer_fold=fold,games=f.game.nunique(),**scores(f.actual_preserved,f.probability)))
        bins=pd.cut(part.probability,np.linspace(0,1,11),include_lowest=True)
        for interval,block in part.groupby(bins,observed=False):
            calibration_bins.append(dict(**tags,probability_bin=str(interval),n=len(block),
                mean_probability=float(block.probability.mean()),event_frequency=float(block.actual_preserved.mean())))
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'probability_summary.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(out/'probability_fold_metrics.csv',index=False)
    pd.DataFrame(calibration_bins).to_csv(out/'reliability_bins.csv',index=False)
    intervals=paired_bootstrap(oof)
    intervals.to_csv(out/'paired_game_bootstrap_intervals.csv',index=False)
    pd.DataFrame([dict(archive=Path(name).name,n=len(group),
        mean_probability=group.probability_any_preserving_reply.mean(),
        mean_exact_favorite_probability=group.probability_stockfish_exact.mean(),
        labeled_rows=int(group.status.eq('ok').sum()),
        event_frequency=group.loc[group.status.eq('ok'),'actual_preserved'].mean())
        for name,group in labeled.groupby('source_pgn')]).to_csv(out/'by_archive.csv',index=False)
    p.assign(additional_good_mass=p.probability_other_acceptable_replies).sort_values(
        'additional_good_mass',ascending=False).head(10).to_csv(out/'examples_other_good_replies.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    reliability=pd.DataFrame(calibration_bins)
    for ax,scope in zip(axes,['predates_documented_pretraining','all_five_exploratory']):
        for name in ['raw_policy_mass','calibrated_policy_mass','uniform_legal']:
            part=reliability[reliability.scope.eq(scope)&reliability.protocol.eq('game')&reliability.method.eq(name)&reliability.n.gt(0)]
            ax.plot(part.mean_probability,part.event_frequency,'o-',label=name)
        ax.plot([0,1],[0,1],'--',color='gray')
        ax.set(title=scope.replace('_',' '),xlabel='Mean predicted probability',ylabel='Observed preserving-reply frequency',
               xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    fig.savefig(out/'probability_reliability.png',dpi=170);plt.close(fig)
    # Headline raw probabilities are estimates, not assertions of calibration or causality.
    text=['# Pre-reply human probability of any damage-preserving reply — development pilot',
        f'This is the authorized {len(p)}-mistake pilot, sampled equally from the five development months. '
        'March 2025 and frozen primary models/configurations remain untouched. No Optuna or fine-tuning.',
        '## What is estimated',
        'At the position after the original mistake, let A contain **every legal reply** for which '
        '`(win_before - W_after_reply) / initial_damage >= 0.75`. All winning chances use '
        'the original mistake-maker’s perspective. The estimate is '
        '**P(any preserving reply) = sum(Maia-2 probability of m, for m in A)**. '
        'It is not a probability of Stockfish’s single favorite, an exact-next-move prediction task, '
        'or a claim that the opponent noticed a mistake.',
        'Each candidate child position is evaluated at Stockfish 19 depth 18 using the existing '
        'binary, one thread, 64 MB hash, ±1000 cp mate mapping and exact terminal chances. '
        'Every legal reply is scored; no unscored reply is silently treated as bad. Full-precision '
        'Maia logits are masked to legal moves and numerically normalized, including black-side '
        'mirroring, castling and promotions. Maia’s win-probability head is unused.',
        '## Human-move model and provenance',
        'The official [Maia-2](https://github.com/CSSLab/maia2) released Blitz/Rapid weights are '
        'checksum-verified. Policy inputs are the post-mistake FEN, the **opponent’s Elo as active '
        'player**, and the mistake-maker’s Elo as the other player. The network is fixed and runs '
        'locally on CPU. It does not use live clocks. Bullet is routed to Blitz and slower games '
        'to Rapid with explicit domain flags; these are extrapolations.',
        'The documented training window is May 2018–November 2023, so February 2019 may overlap '
        'pretraining. The four earlier archives are reported separately as predating that documented '
        'window. This is not an independent membership audit, nor a guarantee that pretraining '
        'players are disjoint. [Training configuration](https://github.com/CSSLab/maia2/blob/main/maia2/configs/maia2-training.yaml).',
        '## Coverage',
        f'{len(p):,} mistakes, {p.site.nunique():,} games; '
        f'{len(detailed):,} legal reply branches. All have policy probabilities and depth-18 child '
        f'evaluations. {int(labeled.status.ne("ok").sum())} have no recorded actual reply and receive '
        'pre-reply estimates but are excluded from validation. Sampling used no actual reply or '
        'future outcome, and all probabilities were written before reading the actual replies.',
        f'Mean probability of any preserving reply: {p.probability_any_preserving_reply.mean():.3f}. '
        f'Mean probability of Stockfish’s exact favorite: {p.probability_stockfish_exact.mean():.3f}. '
        f'{int(p.acceptable_reply_count.gt(1).sum())}/{len(p)} positions have multiple acceptable replies. '
        f'Mean probability carried by acceptable replies other than the favorite: '
        f'{p.probability_other_acceptable_replies.mean():.3f}. '
        'If the favorite itself fails the 75% rule, its probability is not counted in the acceptable set.',
        '## Evaluation and calibration',
        'The original game-grouped outer folds are reused. Raw-policy probabilities are fixed '
        'external-model scores; outer folds apply to the training-prevalence baseline and the optional '
        'sigmoid calibrator. The calibrator uses only other outer-training sample games. The player '
        'protocol also purges training games containing players from validation games. It does not '
        'make the external Maia training players disjoint. Probability endpoints 0/1 are preserved '
        'because all/no legal replies qualifying makes the event deterministic under this procedure.',
        'For the four-archive diagnostic, calibration and prevalence training also exclude February '
        '2019. The all-five result remains potentially affected by pretrained-model overlap. '
        'Uniform_legal gives equal probability to every legal move; favorite_acceptable_only '
        'counts just the favorite when it qualifies and is deliberately a narrower proxy. '
        'No probability cutoff or calibration method was selected from outer scores.']
    shown=summary[summary.protocol.eq('game')].copy()
    for column in ['event_prevalence','mean_probability','brier','log_loss','roc_auc','average_precision']:
        shown[column]=shown[column].map(lambda x:f'{x:.4f}' if pd.notna(x) else 'undefined')
    text.append(markdown(shown))
    text+=['All game/player summaries, fold-level scores, labels, calibration coefficients, '
        'reliability bins and paired game-cluster intervals are saved as CSVs. Brier/log loss '
        'measure probability quality; lower is better. Intervals use 2,000 resampled games and '
        'condition on fixed predictions without model refitting or multiple-comparison adjustment.',
        '![Reliability](probability_reliability.png)',
        '## Limits and artifacts', '\n'.join('- '+x for x in protocol['limitations']),
        'The 500 rows are month-balanced, so aggregate rates are pilot rates, not estimates of '
        'the full development population. Exact aggregation of model probabilities does not make '
        'them the true behavioral probabilities or make finite-depth Stockfish infallible. '
        'This analysis adds a pre-reply estimate; it does not alter primary persistence targets.',
        'The core output is `pre_reply_probabilities.csv`; `all_legal_reply_scores.csv.gz` records '
        'every candidate move, its human probability, original-player chance and qualification. '
        '`human_move_probabilities.jsonl` and `candidate_child_evaluations.sqlite` checkpoint '
        'inference and candidate scoring. Use a new directory and `--per-archive 0` to scale '
        'the pipeline to all five development archives; existing raw source caches stay read-only.']
    (out/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    evaluate(parser.parse_args().out.resolve())
