#!/usr/bin/env python3
"""Generate bounded pre-reply Maia features on four non-overlap development months.

No target columns or actual replies are parsed. Highest-policy-probability moves
alone determine the scored prefix; cache availability NEVER changes that prefix.
Run --prepare-only first, then --resume. Original artifacts stay read-only.
"""
import argparse
import importlib.metadata
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import chess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import human_reply_policy as core
from simulate_human_reply_budget import top_selection

ROOT=HERE.parent
PILOT=ROOT/'notebook_runs/human_reply_policy_development_20261117_v1'
SIMULATION=ROOT/'notebook_runs/maia_budget_simulation_20261117_v1'
FOUR={f'lichess_db_standard_rated_{x}.pgn.zst' for x in ('2017-04','2017-06','2017-07','2018-02')}
WORKERS=12
FEATURE='maia_preserving_reply_probability'


def known_inputs():
    full=core.load_known_positions()
    frame=full[full.source_pgn.map(lambda x:Path(x).name).isin(FOUR)].copy()
    frame['row_index']=frame.index
    records=[]
    for row in frame.itertuples(index=False):
        board=chess.Board(row.fen_before)
        original=chess.WHITE if row.side=='White' else chess.BLACK
        if board.turn!=original:raise ValueError('Perspective mismatch.')
        move=chess.Move.from_uci(row.uci)
        if move not in board.legal_moves:raise ValueError('Illegal mistake.')
        board.push(move)
        model,domain=core.choose_policy(row.event,row.initial_time_seconds,row.increment_seconds)
        record=row._asdict()
        record.update(root_fen=board.fen(),active_elo=int(row.opponent_elo) if pd.notna(row.opponent_elo) else 1500,
            other_elo=int(row.player_elo) if pd.notna(row.player_elo) else 1500,
            policy_model=model,policy_domain=domain,legal_replies=board.legal_moves.count())
        if record['legal_replies']<=0:raise ValueError('No reply exists.')
        records.append(record)
    return pd.DataFrame(records)


def prepare(out):
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    summary=pd.read_csv(SIMULATION/'budget_simulation_summary.csv')
    feasible=summary[summary.scope.eq('all_400')&summary.coverage.notna()&
                     summary.max_bound_width.le(.05+1e-12)&summary.mean_absolute_error.le(.01)]
    if feasible.empty:raise ValueError('No simulated budget meets the declared tolerances.')
    chosen=feasible.sort_values('selected_nonterminal_calls').iloc[0]
    frame=known_inputs()
    pilot_protocol=json.loads((PILOT/'protocol.json').read_text())
    sources=[core.DATA,core.LEGACY/'positions.sqlite',core.LEGACY/'manifest.json',
        PILOT/'candidate_child_evaluations.sqlite',PILOT/'human_move_probabilities.jsonl',
        PILOT/'pre_reply_inputs.csv',PILOT/'protocol.json',PILOT/'validation.json',
        SIMULATION/'protocol.json',SIMULATION/'budget_simulation_summary.csv',
        ROOT/'scripts/human_reply_policy.py',ROOT/'scripts/simulate_human_reply_budget.py',
        Path(__file__).resolve(),ROOT/'config.json',ROOT/'my_tuning/selected.json']
    wal=core.LEGACY/'positions.sqlite-wal'
    if wal.exists():sources.append(wal)
    if core.digest(core.ENGINE)!=pilot_protocol['stockfish_sha256']:raise ValueError('Engine binary changed.')
    if {importlib.metadata.version(x) for x in ['maia2']}!={'0.11.0'}:raise ValueError('Maia package changed.')
    for name,expected in pilot_protocol['model_sha256'].items():
        if core.digest(Path(pilot_protocol['model_assets'])/f'{name}_model.pt')!=expected:
            raise ValueError('Model checkpoint changed.')
    out.mkdir(parents=True)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Bounded Maia feature on four development archives predating documented pretraining',
        source_archives=sorted(FOUR),rows=len(frame),games=frame.site.nunique(),coverage=float(chosen.coverage),
        budget_selection='Cheapest simulated mass-coverage policy satisfying the unchanged <=0.05 '
            'maximum bound width and <=0.01 mean oracle midpoint error. The intermediate 97.5% '
            'coverage is eligible; the earlier two-option 95/99 fallback is preserved in the '
            'simulation directory. No human outcome or regression label chooses this budget.',
        pilot_simulation=chosen.to_dict(),models=pilot_protocol['model_assets'],
        model_sha256=pilot_protocol['model_sha256'],
        package_versions=pilot_protocol['package_versions'],stockfish_sha256=pilot_protocol['stockfish_sha256'],
        engine_workers=WORKERS,engine_depth=18,engine_threads=1,engine_hash_mb=64,mate_score_cp=1000,
        target_event=pilot_protocol['event'],feature=FEATURE,
        feature_point='Midpoint L+R/2; L=qualified mass in the selected policy prefix, R=unscored mass.',
        prefix='Sort probabilities descending, UCI ascending as deterministic tie-break; choose '
            'smallest prefix whose mass >= coverage. Ignore ALL unselected scores even on cache hits.',
        leakage='Only known FEN after mistake, original damage/before chance and pre-reply ratings/context. '
            'No actual reply, outcome, regret, horizon label or completion flag is parsed by this program. '
            'Generate every allowed row, including unrecorded replies.',
        regression_gate='Fixed LightGBM + midpoint must have positive game-fold MAE improvement '
            'with paired 95% lower bound >0 for immediate reply regret; player-purged point gain '
            'must be nonnegative. Only then test change_in_damage_10ply. No Optuna.',
        source_sha256={str(p.relative_to(ROOT)):core.digest(p) for p in sources},
        cautions=['The bounds concern numerical integration of a fixed Maia policy under depth-18 '
            'candidate scoring. They are not confidence intervals for actual human behavior or '
            'bounds on engine finite-search error.',
            'The four archives precede the documented training window; model membership is not '
            'independently audited. Maia does not condition on live clocks and some speeds are out of domain.'])
    core.json_write(out/'protocol.json',protocol)
    core.json_write(out/'selected_budget.json',dict(coverage=float(chosen.coverage),method=chosen.method,
        maximum_bound_width=float(chosen.max_bound_width),mean_oracle_error=float(chosen.mean_absolute_error),
        selected_engine_calls=int(chosen.selected_nonterminal_calls),
        full_engine_calls=int(chosen.full_nonterminal_calls),savings_percent=float(chosen.calls_saved_percent)))
    frame.to_csv(out/'pre_reply_inputs.csv',index=False)
    print(f'Prepared {len(frame)} mistakes, {frame.site.nunique()} games, coverage={chosen.coverage}.',flush=True)


def infer(out,protocol):
    import torch
    torch.set_num_threads(4);torch.manual_seed(core.SEED)
    sample=pd.read_csv(out/'pre_reply_inputs.csv').set_index('row_index',drop=False)
    checkpoint=out/'human_move_probabilities.jsonl'
    existing={}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            r=json.loads(line);existing[r['row_index']]=r
    else:
        original=pd.read_csv(PILOT/'pre_reply_inputs.csv').set_index('row_index')
        cached=[]
        for line in (PILOT/'human_move_probabilities.jsonl').read_text().splitlines():
            r=json.loads(line);rid=r['row_index']
            if rid not in sample.index:continue # February 2019 is not part of this run.
            for column in ['root_fen','active_elo','other_elo','policy_model','win_before','initial_damage']:
                if sample.loc[rid,column]!=original.loc[rid,column]:raise ValueError('Pilot policy input mismatch.')
            existing[rid]=r;cached.append(r)
        with checkpoint.open('w') as f:
            for r in cached:f.write(json.dumps(r)+'\n')
        print(f'Reused {len(cached)} policies from non-overlap pilot rows.',flush=True)
    for kind in ['blitz','rapid']:
        pending=sample[sample.policy_model.eq(kind)&~sample.index.isin(existing)]
        if pending.empty:continue
        network=core.load_policy(kind,Path(protocol['models']))
        for start in range(0,len(pending),32):
            batch=pending.iloc[start:start+32]
            probabilities=core.full_precision_policy(network,batch[['root_fen','active_elo','other_elo']])
            with checkpoint.open('a') as f:
                for row,p in zip(batch.itertuples(index=False),probabilities):
                    r=dict(row_index=int(row.row_index),model=kind,move_probabilities=p)
                    f.write(json.dumps(r)+'\n');existing[row.row_index]=r
            if start%1024==0 or start+32>=len(pending):
                print(f'Policies: {len(existing)}/{len(sample)}',flush=True)
        del network
    if set(existing)!=set(sample.index):raise ValueError('Policy rows differ from allowed four-archive universe.')
    return existing


def make_plan(out,protocol,policies):
    plan_path=out/'selected_reply_plan.csv.gz'
    coverage_path=out/'position_coverage.csv'
    if plan_path.exists() and coverage_path.exists():return pd.read_csv(plan_path),pd.read_csv(coverage_path)
    sample=pd.read_csv(out/'pre_reply_inputs.csv').set_index('row_index')
    plans=[];coverage=[]
    for row_id,row in sample.iterrows():
        board=chess.Board(row.root_fen)
        distribution=policies[int(row_id)]['move_probabilities']
        if set(distribution)!=set(m.uci() for m in board.legal_moves):raise ValueError('Incomplete legal move policy.')
        ordered=sorted(distribution.items(),key=lambda x:(-x[1],x[0]))
        probabilities=np.array([x[1] for x in ordered],float)
        n=top_selection(probabilities,coverage=protocol['coverage'])
        covered=float(probabilities[:n].sum());residual=max(0.,1-covered)
        if residual>1-protocol['coverage']+1e-10:raise ValueError('Prefix missed the coverage target.')
        coverage.append(dict(row_index=row_id,legal_replies=len(ordered),selected_replies=n,
                             covered_probability=covered,unscored_probability=residual))
        for rank,(move,probability) in enumerate(ordered[:n],1):
            child=board.copy(stack=False);child.push_uci(move)
            plans.append(dict(row_index=row_id,rank=rank,reply_uci=move,policy_probability=probability,
                              child_fen=child.fen()))
    plan=pd.DataFrame(plans);coverage=pd.DataFrame(coverage)
    plan.to_csv(plan_path,index=False);coverage.to_csv(coverage_path,index=False)
    print(f'Selected {len(plan)} reply branches instead of {coverage.legal_replies.sum()} total legal branches.',flush=True)
    return plan,coverage


def score(out,plan):
    db=sqlite3.connect(out/'selected_child_evaluations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS evaluations (fen TEXT PRIMARY KEY,record TEXT NOT NULL)')
    records={fen:json.loads(record) for fen,record in db.execute('SELECT fen,record FROM evaluations')}
    required=plan.child_fen.drop_duplicates().tolist()
    pilot=sqlite3.connect(f'file:{PILOT/"candidate_child_evaluations.sqlite"}?mode=ro',uri=True)
    legacy=sqlite3.connect(f'file:{core.LEGACY/"positions.sqlite"}?mode=ro',uri=True)
    for fen in required:
        if fen in records and records[fen]['status']!='error':continue
        row=pilot.execute('SELECT record FROM evaluations WHERE fen=?',(fen,)).fetchone()
        record=json.loads(row[0]) if row else None
        if record is not None:record['reuse_source']='pilot'
        else:
            cached=legacy.execute('SELECT status,white_cp,white_win,is_mate FROM positions WHERE fen=?',(fen,)).fetchone()
            if cached and cached[0] in ['engine','board_terminal']:
                record=dict(fen=fen,status=cached[0],white_cp=cached[1],white_win=cached[2],mate=bool(cached[3]),
                    depth=18,legacy_cache=True,error='',elapsed_ms=None,reuse_source='legacy')
        if record is not None:
            records[fen]=record
            db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?)',(fen,json.dumps(record)))
    db.commit();pilot.close();legacy.close()
    pending=[f for f in required if f not in records or records[f]['status']=='error']
    print(f'Selected children: {len(required)} unique; {len(required)-len(pending)} reused; '
          f'{len(pending)} new depth-18 evaluations.',flush=True)
    tasks=[(pending[i:i+32],str(core.ENGINE)) for i in range(0,len(pending),32)]
    count=0
    if tasks:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for values in pool.map(core.score_child_batch,tasks,chunksize=1):
                with db:
                    for record in values:
                        record['reuse_source']='new'
                        db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?)',(record['fen'],json.dumps(record)))
                        records[record['fen']]=record
                count+=len(values)
                if count%512<len(values) or count==len(pending):
                    print(f'New selected evaluations: {count}/{len(pending)}',flush=True)
    db.close()
    errors=[r for r in records.values() if r['status']=='error']
    if errors:
        core.json_write(out/'engine_errors.json',errors);raise ValueError('Resume to resolve engine errors.')
    if not set(required).issubset(records):raise ValueError('Missing selected evaluations.')
    return records


def aggregate(out,protocol,plan,coverage,records):
    sample=pd.read_csv(out/'pre_reply_inputs.csv').set_index('row_index')
    cover=coverage.set_index('row_index');result=[]
    for row_id,part in plan.groupby('row_index'):
        row=sample.loc[row_id];lower=0.
        for r in part.itertuples(index=False):
            white=records[r.child_fen]['white_win']
            chance=white if row.side=='White' else 100-white
            if core.good_reply(row.win_before,row.initial_damage,chance):lower+=r.policy_probability
        lower=float(np.clip(lower,0.,1.))
        residual=float(cover.loc[row_id,'unscored_probability'])
        upper=min(1.,lower+residual);midpoint=(lower+upper)/2
        if not 0<=lower<=upper<=1+1e-12 or upper-lower>1-protocol['coverage']+1e-10:
            raise ValueError('Invalid probability bounds.')
        result.append(dict(row_index=row_id,source_pgn=row.source_pgn,game_index=row.game_index,ply=row.ply,
            site=row.site,maia_probability_lower=lower,maia_probability_upper=upper,
            **{FEATURE:midpoint},probability_bound_width=upper-lower,
            selected_replies=int(cover.loc[row_id,'selected_replies']),
            legal_replies=int(cover.loc[row_id,'legal_replies'])))
    result=pd.DataFrame(result).sort_values('row_index')
    if len(result)!=protocol['rows'] or result[FEATURE].isna().any():raise ValueError('Incomplete feature rows.')
    exact=pd.read_csv(PILOT/'pre_reply_probabilities.csv').set_index('row_index')
    common=result[result.row_index.isin(exact.index)].set_index('row_index')
    oracle=exact.loc[common.index,'probability_any_preserving_reply']
    if not ((oracle>=common.maia_probability_lower-1e-10)&(oracle<=common.maia_probability_upper+1e-10)).all():
        raise ValueError('Known exact pilot probabilities lie outside production bounds.')
    result.to_csv(out/'maia_bounds_features.csv',index=False)
    reuse=pd.Series([r['reuse_source'] for r in records.values()]).value_counts().to_dict()
    core.json_write(out/'features_ready.json',dict(rows=len(result),source_archives=protocol['source_archives'],
        probability_width_max=float(result.probability_bound_width.max()),
        probability_width_mean=float(result.probability_bound_width.mean()),
        pilot_exact_rows_verified=len(common),pilot_mean_oracle_error=float(np.abs(common[FEATURE]-oracle).mean()),
        selected_reply_branches=len(plan),all_legal_reply_branches=int(coverage.legal_replies.sum()),
        selected_unique_children=len(records),evaluation_reuse_counts=reuse,
        new_stockfish_analyse_calls=sum(r['reuse_source']=='new' and r['status']=='engine' for r in records.values()),
        feature_sha256=core.digest(out/'maia_bounds_features.csv'),actual_reply_parsed=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if not out.exists():prepare(out)
    elif not args.resume:raise FileExistsError('Use a new output directory or --resume.')
    protocol=json.loads((out/'protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        if core.digest(ROOT/path)!=expected:raise ValueError(f'Source changed: {path}')
    if core.digest(core.ENGINE)!=protocol['stockfish_sha256']:raise ValueError('Stockfish changed.')
    if args.prepare_only:return
    if (out/'features_ready.json').exists():print('Features already complete; nothing overwritten.');return
    policies=infer(out,protocol)
    plan,coverage=make_plan(out,protocol,policies)
    records=score(out,plan)
    aggregate(out,protocol,plan,coverage,records)
    for path,expected in protocol['source_sha256'].items():
        if core.digest(ROOT/path)!=expected:raise ValueError(f'Source modified: {path}')
    print('Bounded features complete:',out,flush=True)


if __name__=='__main__':main()
