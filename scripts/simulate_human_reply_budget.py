#!/usr/bin/env python3
"""Compare the uniform baseline and simulate cheaper probability integration.

Uses saved pilot scores only. No new inference, engine calls or model fitting.
"""
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'notebook_runs/human_reply_policy_development_20261117_v1'
SEED,REPEATS=1701,2000


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def top_selection(probabilities,coverage=None,k=None):
    p=np.asarray(probabilities,float)
    if coverage is not None:
        count=min(len(p),int(np.searchsorted(np.cumsum(p),coverage-1e-12,side='left'))+1)
    else:count=min(len(p),k)
    return count


def probability_bounds(probabilities,qualifies,count):
    p=np.asarray(probabilities,float);good=np.asarray(qualifies,bool)
    covered=float(p[:count].sum())
    lower=float(p[:count][good[:count]].sum())
    residual=max(0.,1-covered)
    upper=min(1.,lower+residual)
    return lower,upper,(lower+upper)/2


def simulate(out):
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Offline uniform comparison and policy-budget simulation; no engine calls.',
        comparison_rows='Same 392 labeled mistakes from the four archives before documented pretraining.',
        bound='L=sum(scored good move probabilities); U=L+unscored probability mass; midpoint=(L+U)/2.',
        feasibility={'maximum_bound_width':.05,'maximum_mean_midpoint_error_vs_exact':.01},
        budget_order=[.95,.99],selection='First feasible coverage; uses exact probability integration error '
            'only, not actual human move/outcome or regression labels.',
        ten_ply_gate='Game-fold regret MAE gain with paired 95% lower bound >0; player-purged '
            'point gain must be nonnegative. Gate chosen before scaled regression results.',
        bootstrap_repeats=REPEATS,seed=SEED)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=pd.read_csv(PILOT/'evaluation_rows.csv')
    details=pd.read_csv(PILOT/'all_legal_reply_scores.csv.gz')
    sample=rows[~rows.pretraining_window_overlap].copy()
    labeled=sample[sample.status.eq('ok')].sort_values('row_index')
    assert len(labeled)==392 and labeled.site.nunique()==364
    counts=details.groupby('row_index').preserves_075_damage.agg(['sum','size'])
    uniform=counts.loc[labeled.row_index,'sum'].to_numpy()/counts.loc[labeled.row_index,'size'].to_numpy()
    maia=labeled.probability_any_preserving_reply.to_numpy(float)
    y=labeled.actual_preserved.to_numpy(bool).astype(float)
    errors=np.column_stack([(uniform-y)**2,(maia-y)**2])
    codes,games=pd.factorize(labeled.site,sort=True)
    sums=np.zeros((len(games),2));np.add.at(sums,codes,errors);sizes=np.bincount(codes)
    rng=np.random.default_rng(SEED);draws=[]
    for start in range(0,REPEATS,100):
        n=min(100,REPEATS-start);idx=rng.integers(0,len(games),(n,len(games)))
        values=sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None]
        draws.extend(values[:,0]-values[:,1])
    low,high=np.quantile(draws,[.025,.975])
    comparison=dict(n=len(y),games=len(games),uniform_legal_brier=errors[:,0].mean(),
        maia_brier=errors[:,1].mean(),maia_brier_improvement=errors[:,0].mean()-errors[:,1].mean(),
        lower95=low,upper95=high,bootstrap_repeats=REPEATS)
    pd.DataFrame([comparison]).to_csv(out/'uniform_comparison_392.csv',index=False)
    db=sqlite3.connect(f'file:{PILOT/"candidate_child_evaluations.sqlite"}?mode=ro',uri=True)
    evaluation={fen:json.loads(record) for fen,record in db.execute('SELECT fen,record FROM evaluations')}
    db.close()
    methods=[(f'top_{k}',None,k) for k in [1,3,5,8,10,15]]+[
        (f'mass_{int(1000*c):04d}',c,None) for c in [.90,.95,.975,.99]]
    individual=[];selected_fens={};full_fens={}
    for row in sample.itertuples(index=False):
        part=details[details.row_index.eq(row.row_index)].sort_values(
            ['policy_probability','reply_uci'],ascending=[False,True],kind='stable')
        p=part.policy_probability.to_numpy(float);q=part.preserves_075_damage.to_numpy(bool)
        exact=float(p[q].sum())
        assert abs(exact-row.probability_any_preserving_reply)<1e-10
        full_fens[row.row_index]=set(part.child_fen)
        for name,coverage,k in methods:
            count=top_selection(p,coverage,k)
            lower,upper,midpoint=probability_bounds(p,q,count)
            assert lower-1e-12<=exact<=upper+1e-12
            individual.append(dict(row_index=row.row_index,method=name,labeled=row.status=='ok',
                evaluated_replies=count,total_replies=len(p),lower_bound=lower,upper_bound=upper,
                bound_width=upper-lower,midpoint=midpoint,exact_probability=exact,
                midpoint_absolute_error=abs(midpoint-exact)))
            selected_fens[(row.row_index,name)]=set(part.iloc[:count].child_fen)
    individual=pd.DataFrame(individual)
    summaries=[]
    for scope,ids in [('all_400',set(sample.row_index)),('labeled_392',set(labeled.row_index))]:
        all_fens=set().union(*(full_fens[i] for i in ids))
        all_engine={f for f in all_fens if evaluation[f]['status']=='engine'}
        all_uncached={f for f in all_engine if not evaluation[f]['legacy_cache']}
        for name,coverage,k in methods:
            selected=set().union(*(selected_fens[(i,name)] for i in ids))
            engine={f for f in selected if evaluation[f]['status']=='engine'}
            uncached={f for f in engine if not evaluation[f]['legacy_cache']}
            part=individual[individual.row_index.isin(ids)&individual.method.eq(name)]
            summaries.append(dict(scope=scope,method=name,coverage=coverage,top_k=k,n=len(part),
                full_nonterminal_calls=len(all_engine),selected_nonterminal_calls=len(engine),
                calls_saved=len(all_engine)-len(engine),calls_saved_percent=100*(1-len(engine)/len(all_engine)),
                full_calls_after_legacy_cache=len(all_uncached),selected_calls_after_legacy_cache=len(uncached),
                legacy_aware_calls_saved=len(all_uncached)-len(uncached),
                mean_evaluated_replies=part.evaluated_replies.mean(),mean_bound_width=part.bound_width.mean(),
                median_bound_width=part.bound_width.median(),p95_bound_width=part.bound_width.quantile(.95),
                max_bound_width=part.bound_width.max(),mean_absolute_error=part.midpoint_absolute_error.mean(),
                max_absolute_error=part.midpoint_absolute_error.max()))
    summary=pd.DataFrame(summaries)
    individual.to_csv(out/'budget_simulation_by_position.csv',index=False)
    summary.to_csv(out/'budget_simulation_summary.csv',index=False)
    chosen=None
    for coverage in protocol['budget_order']:
        name=f'mass_{int(1000*coverage):04d}'
        r=summary[summary.scope.eq('all_400')&summary.method.eq(name)].iloc[0]
        if r.max_bound_width<=.05+1e-12 and r.mean_absolute_error<=.01:
            chosen=dict(coverage=coverage,method=name,maximum_probability_width=1-coverage,
                feature='midpoint of probability bounds',actual_reply_used=False)
            break
    result=dict(scaling_feasible=chosen is not None,chosen=chosen,uniform_comparison=comparison,
                selection_uses_no_observed_human_outcomes=True)
    (out/'budget_decision.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    print(summary[summary.scope.eq('all_400')].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    simulate(parser.parse_args().out.resolve())
