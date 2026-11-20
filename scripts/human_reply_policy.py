#!/usr/bin/env python3
"""Pre-reply probability of ANY damage-preserving reply, using frozen Maia-2.

Prepare: --out RUN --per-archive 100 --prepare-only
Run/resume: --out RUN --resume
Use --per-archive 0 in a NEW output directory for all five development archives.
No actual reply or future outcome is read before probability generation completes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'development_with_multipv.csv'
LEGACY=ROOT/'notebook_runs/opponent_reply_targets_development_v1'
ROOT_MOVES=ROOT/'notebook_runs/pre_reply_features_development_20261117_v1/reply_agreement_positions.csv'
MODEL_DIR=ROOT/'notebook_runs/human_reply_policy_development_20261117_v1/model_assets'
ENGINE=Path('/private/tmp/stockfish/stockfish-macos-universal')
ARCHIVES={f'lichess_db_standard_rated_{x}.pgn.zst' for x in
          ('2017-04','2017-06','2017-07','2018-02','2019-02')}
FIELDS=['source_pgn','game_index','ply','site','fen_before','uci','side','win_before',
        'initial_damage','player_elo','opponent_elo','event','initial_time_seconds',
        'increment_seconds','opponent_clock_before_move']
WEIGHT_HASHES={
    'rapid':'65aae8465eed5e65df66a24ea7370715579f9e5435098d06fe18bdb1e267e997',
    'blitz':'5090d5d0d49dc29787c08d13febbed5b2da81a18c2ddcd8a90070cd3b43c44b2'}
SEED,KEEP_DAMAGE,WORKERS=1701,.75,8


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_write(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:
        x.item() if isinstance(x,np.generic) else str(x))+'\n')


def chance(cp):
    return 100/(1+math.exp(-.00368208*cp))


def good_reply(win_before,initial_damage,win_after_reply):
    if not initial_damage>0:raise ValueError('Initial damage must be positive.')
    return (win_before-win_after_reply)/initial_damage>=KEEP_DAMAGE


def choose_policy(event,initial,increment):
    event=str(event).casefold()
    if 'blitz' in event:return 'blitz','in_domain_blitz'
    if 'rapid' in event:return 'rapid','in_domain_rapid'
    if 'bullet' in event:return 'blitz','out_of_domain_bullet'
    if 'classical' in event or 'correspondence' in event:return 'rapid','out_of_domain_slow'
    effective=(initial+40*increment) if pd.notna(initial) and pd.notna(increment) else 600
    return ('blitz' if effective<480 else 'rapid'),'out_of_domain_unclassified'


def load_known_positions():
    frame=pd.read_csv(DATA,usecols=FIELDS,low_memory=False)
    if (set(frame.source_pgn.map(lambda x:Path(x).name))!=ARCHIVES or len(frame)!=23167
            or frame.site.nunique()!=6512 or frame.duplicated(['source_pgn','game_index','ply']).any()
            or not (frame.initial_damage.gt(20)&frame.win_before.between(10,90)).all()):
        raise ValueError('Input is not the canonical five development archives.')
    return frame


def prepare(out,per_archive,models):
    if (out/'protocol.json').exists():raise FileExistsError('Protocol already exists.')
    frame=load_known_positions()
    # Neither sampling nor inference receives actual_reply_uci, future moves or targets.
    sample=(frame.groupby('source_pgn',group_keys=False).sample(n=per_archive,random_state=SEED)
            if per_archive else frame.copy()).sort_index()
    sample=sample.assign(row_index=sample.index)
    roots=pd.read_csv(ROOT_MOVES).set_index('fen').depth18_best_reply_uci
    engine_manifest=json.loads((LEGACY/'manifest.json').read_text())
    if digest(ENGINE)!=engine_manifest['engine_sha256'] or engine_manifest['depth']!=18:
        raise ValueError('Stockfish 19 binary/depth does not match existing evaluations.')
    if digest(DATA)!=engine_manifest['data_sha256']:raise ValueError('Development data changed.')
    for kind,expected in WEIGHT_HASHES.items():
        if digest(models/f'{kind}_model.pt')!=expected:raise ValueError('Unexpected Maia-2 weights.')
    prepared=[]; plans=[]
    for row in sample.itertuples(index=False):
        before=chess.Board(row.fen_before)
        original=chess.WHITE if row.side=='White' else chess.BLACK
        if before.turn!=original:raise ValueError('Original perspective disagrees with FEN.')
        move=chess.Move.from_uci(row.uci)
        if move not in before.legal_moves:raise ValueError('Illegal mistake.')
        before.push(move); fen=before.fen()
        if before.turn==original or not before.legal_moves.count():raise ValueError('Invalid reply position.')
        kind,domain=choose_policy(row.event,row.initial_time_seconds,row.increment_seconds)
        active=int(row.opponent_elo) if pd.notna(row.opponent_elo) else 1500
        other=int(row.player_elo) if pd.notna(row.player_elo) else 1500
        record=row._asdict()
        record.update(root_fen=fen,active_elo=active,other_elo=other,policy_model=kind,policy_domain=domain,
            rating_default_used=pd.isna(row.opponent_elo) or pd.isna(row.player_elo),
            stockfish_favorite_uci=roots.loc[fen],
            pretraining_window_overlap=Path(row.source_pgn).name.endswith('2019-02.pgn.zst'),
            legal_replies=before.legal_moves.count())
        prepared.append(record)
        for reply in before.legal_moves:
            child=before.copy(stack=False);child.push(reply)
            plans.append(dict(row_index=row.row_index,reply_uci=reply.uci(),child_fen=child.fen()))
    prepared=pd.DataFrame(prepared);plan=pd.DataFrame(plans)
    import maia2
    package=Path(maia2.__file__).parent
    package_files=[package/name for name in ('model.py','inference.py','utils.py','main.py','configs/maia2-training.yaml')]
    sources=[DATA,LEGACY/'manifest.json',LEGACY/'positions.sqlite',LEGACY/'reply_targets.csv',ROOT_MOVES,
        ROOT/'config.json',ROOT/'my_tuning/selected.json',
        ROOT/'notebook_runs/final_optuna_development_20261117_v1/outer_fold_assignments.csv',
        Path(__file__).resolve(),ROOT/'scripts/evaluate_human_reply_policy.py']
    wal=LEGACY/'positions.sqlite-wal'
    if wal.exists():sources.append(wal)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        kind='Development-only pre-reply human-policy mass over all damage-preserving legal replies',
        source_archives=sorted(ARCHIVES),sample_per_archive=per_archive,sample_mistakes=len(sample),
        sample_games=sample.site.nunique(),seed=SEED,
        sampling='100 mistake rows per archive (or CLI count), seed 1701; sampled from all canonical rows '
                 'without reading actual opponent replies or future outcomes. Month-balanced pilot, '
                 'not a population-weighted prevalence estimate.',
        event='(historical win_before - Stockfish19 winning chance after candidate reply) / '
              'historical initial_damage >= 0.75, from original mistake-maker perspective',
        probability='Sum frozen Maia-2 full-precision legal-move probabilities over ALL legal replies '
                    'satisfying the event. No top-k truncation, no residual unscored mass.',
        policy_inputs=['post_mistake_fen','opponent_elo (active)','mistake_maker_elo (other)'],
        routing='Released Blitz for Blitz/Bullet, Rapid for Rapid/Classical/Correspondence; '
                'otherwise initial+40*increment <480 -> Blitz. All non-Blitz/Rapid domains flagged.',
        clocks='Policy does not condition on live clocks. Opponent pre-reply clock is metadata only.',
        package_versions={x:importlib.metadata.version(x) for x in
            ('maia2','torch','numpy','pandas','scipy','scikit-learn','chess','matplotlib')},
        package_file_sha256={str(p.relative_to(package)):digest(p) for p in package_files},
        model_assets=str(models),model_sha256=WEIGHT_HASHES,
        documented_pretraining_window=['2018-05','2023-11'],
        external_pretraining_caveat='2019-02 may overlap model pretraining; raw/calibrated results for '
            'the four earlier archives are reported separately. This is a documented-date check, '
            'not an independently verified training-game membership audit. External model training '
            'players are not guaranteed disjoint from evaluation players.',
        stockfish=engine_manifest['engine_id'],stockfish_sha256=digest(ENGINE),depth=18,threads=1,hash_mb=64,
        mate_score_cp=1000,terminal_chances=[0,50,100],workers=WORKERS,
        planned_reply_branches=len(plan),unique_child_positions=plan.child_fen.nunique(),
        engine_cache_policy='Read-only reuse of existing exact-FEN Stockfish19 depth18 evaluations; '
            'remaining unique children use the same binary/settings at depth18. New search processes '
            'retain TT within 32-position batches, matching the legacy warm-cache approach. '
            'Only the NEW output cache is written.',
        evaluation='Actual reply/results read only AFTER all pre-reply predictions are saved. '
            'Reuse original game outer assignments. Fixed sigmoid probability calibration fits '
            'only other outer-training games, with player purge in the player protocol. '
            'No Optuna, classification cutoff selection or primary-model changes.',
        evaluation_scopes=['all_five_exploratory','predates_documented_pretraining'],
        calibration='LogisticRegression C=1, logit(raw policy mass), training-only; '
                    'known probability endpoints 0 and 1 preserved exactly.',
        bootstrap_repeats=2000,bootstrap='Paired percentile game-cluster intervals on saved OOF '
            'probability errors, conditional on fixed folds/model. No refitting within bootstrap.',
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
        limitations=['The probability is an estimate from a human-move model, not ground-truth human behavior.',
            'Exact aggregation means every legal move is scored under the defined depth18 procedure; '
            'Stockfish estimates themselves are finite-search approximations.',
            'Damage denominator and win_before come from historical engine evaluations of unrecorded version.',
            'Maia-2 rating bins collapse ratings below 1100 and at/above 2000; no clock conditioning.',
            'Bullet, classical, correspondence, low clocks and early plies can be outside training domain.',
            'FEN reconstruction cannot recover earlier repetition history; PGN outcomes are not used '
            'to fill missing opponent replies. Primary horizon targets remain unchanged.'])
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/'protocol.json',protocol)
    prepared.to_csv(out/'pre_reply_inputs.csv',index=False)
    plan.to_csv(out/'all_legal_reply_plan.csv.gz',index=False)
    print(f'Prepared {len(prepared)} mistakes, {len(plan)} legal replies, '
          f'{plan.child_fen.nunique()} unique child positions.',flush=True)


def load_policy(kind,models):
    from maia2 import model
    if digest(models/f'{kind}_model.pt')!=WEIGHT_HASHES[kind]:raise ValueError('Unexpected weights.')
    return model.from_pretrained(type=kind,device='cpu',save_root=str(models)).eval()


def full_precision_policy(network,inputs):
    """Only root position and two Elo values are passed to this inference function."""
    import torch
    from maia2 import inference
    from maia2.utils import mirror_move
    all_moves,elo_dict,reverse=inference.prepare()
    if list(inputs.columns)!=['root_fen','active_elo','other_elo']:
        raise ValueError('Human-policy inference input allowlist rejected a column.')
    encoded=[inference.preprocessing(r.root_fen,r.active_elo,r.other_elo,elo_dict,all_moves)
             for r in inputs.itertuples(index=False)]
    boards=torch.stack([x[0] for x in encoded])
    active=torch.tensor([x[1] for x in encoded],dtype=torch.long)
    other=torch.tensor([x[2] for x in encoded],dtype=torch.long)
    masks=torch.stack([x[3] for x in encoded]).bool()
    with torch.inference_mode():
        logits,_,_=network(boards,active,other)
        probs=logits.masked_fill(~masks,float('-inf')).softmax(-1).numpy()
    results=[]
    for i,fen in enumerate(inputs.root_fen):
        indices=masks[i].nonzero().flatten().tolist()
        names=[mirror_move(reverse[j]) if chess.Board(fen).turn==chess.BLACK else reverse[j] for j in indices]
        values=np.array([probs[i,j] for j in indices],dtype=np.float64)
        values=values/values.sum() # only floating-point roundoff; never truncate/round move probabilities
        policy=dict(zip(names,values.tolist()))
        if set(policy)!=set(m.uci() for m in chess.Board(fen).legal_moves):raise ValueError('Legal mask/mirroring mismatch.')
        if not np.isclose(sum(policy.values()),1,atol=1e-12):raise ValueError('Policy mass is not one.')
        results.append(policy)
    return results


def infer_policies(out,models):
    path=out/'human_move_probabilities.jsonl'
    completed={}
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);completed[row['row_index']]=row
    sample=pd.read_csv(out/'pre_reply_inputs.csv')
    import torch
    torch.set_num_threads(4);torch.manual_seed(SEED)
    for kind in ('blitz','rapid'):
        pending=sample[sample.policy_model.eq(kind)&~sample.row_index.isin(completed)]
        if pending.empty:continue
        network=load_policy(kind,models)
        for start in range(0,len(pending),32):
            batch=pending.iloc[start:start+32]
            policies=full_precision_policy(network,batch[['root_fen','active_elo','other_elo']])
            with path.open('a') as handle:
                for row,policy in zip(batch.itertuples(index=False),policies):
                    record=dict(row_index=int(row.row_index),model=kind,move_probabilities=policy)
                    handle.write(json.dumps(record)+'\n');completed[row.row_index]=record
            print(f'Human policies: {len(completed)}/{len(sample)}',flush=True)
        del network
    if set(completed)!=set(sample.row_index):raise ValueError('Incomplete human policy coverage.')
    return completed


def score_child_batch(task):
    fens,engine_path=task
    results=[]
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({'Threads':1,'Hash':64})
        for fen in fens:
            start=time.monotonic();board=chess.Board(fen)
            record=dict(fen=fen,status='engine',white_cp=None,white_win=None,mate=False,depth=18,
                        actual_search_depth=None,legacy_cache=False,error='')
            try:
                if board.is_game_over(claim_draw=False):
                    winner=board.outcome(claim_draw=False).winner
                    record.update(status='board_terminal',white_win=50. if winner is None else 100. if winner else 0.)
                else:
                    info=engine.analyse(board,chess.engine.Limit(depth=18),
                                        info=chess.engine.INFO_SCORE|chess.engine.INFO_BASIC)
                    score=info['score'].pov(chess.WHITE);mate=score.mate()
                    cp=(1000 if mate>0 else -1000) if mate is not None else int(score.score())
                    record.update(white_cp=cp,white_win=chance(cp),mate=mate is not None,
                                  actual_search_depth=info.get('depth'))
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}')
            record['elapsed_ms']=1000*(time.monotonic()-start);results.append(record)
    return results


def score_all_children(out):
    plan=pd.read_csv(out/'all_legal_reply_plan.csv.gz')
    required=sorted(plan.child_fen.unique())
    db=sqlite3.connect(out/'candidate_child_evaluations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS evaluations (fen TEXT PRIMARY KEY, record TEXT NOT NULL)')
    existing={fen:json.loads(record) for fen,record in db.execute('SELECT fen,record FROM evaluations')}
    legacy=sqlite3.connect(f'file:{LEGACY/"positions.sqlite"}?mode=ro',uri=True)
    reused=0
    for fen in required:
        if fen in existing and existing[fen]['status']!='error':continue
        cached=legacy.execute('SELECT status,white_cp,white_win,is_mate FROM positions WHERE fen=?',(fen,)).fetchone()
        if cached and cached[0] in ('engine','board_terminal'):
            record=dict(fen=fen,status=cached[0],white_cp=cached[1],white_win=cached[2],mate=bool(cached[3]),
                        depth=18,actual_search_depth=None,legacy_cache=True,error='',elapsed_ms=None)
            existing[fen]=record;db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?)',(fen,json.dumps(record)))
            reused+=1
    db.commit();legacy.close()
    pending=[fen for fen in required if fen not in existing or existing[fen]['status']=='error']
    print(f'Candidate children: {len(required)} unique; {len(required)-len(pending)} cached; {len(pending)} new depth18 searches.',flush=True)
    if pending:
        tasks=[(pending[i:i+32],str(ENGINE)) for i in range(0,len(pending),32)]
        done=0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for records in pool.map(score_child_batch,tasks,chunksize=1):
                with db:
                    for record in records:
                        db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?)',(record['fen'],json.dumps(record)))
                        existing[record['fen']]=record
                done+=len(records)
                if done%256<len(records) or done==len(pending):
                    print(f'New child evaluations: {done}/{len(pending)}',flush=True)
    db.close()
    errors=[r for r in existing.values() if r['status']=='error']
    if errors:
        json_write(out/'engine_errors.json',errors);raise ValueError('Candidate scoring has errors; resume before aggregating.')
    if not set(required).issubset(existing):raise ValueError('Unscored legal replies remain.')
    return existing


def aggregate(out,policies,evaluations):
    sample=pd.read_csv(out/'pre_reply_inputs.csv').set_index('row_index')
    plan=pd.read_csv(out/'all_legal_reply_plan.csv.gz')
    rows=[];details=[]
    for row_id,branches in plan.groupby('row_index'):
        row=sample.loc[row_id];policy=policies[int(row_id)]['move_probabilities']
        legal={m.uci() for m in chess.Board(row.root_fen).legal_moves}
        if set(branches.reply_uci)!=set(policy) or set(policy)!=legal:
            raise ValueError('Candidate set does not cover every legal reply.')
        if min(policy.values())<0 or not np.isclose(sum(policy.values()),1,atol=1e-10):
            raise ValueError('Invalid legal-move probability distribution.')
        good=[];all_chances={}
        for branch in branches.itertuples(index=False):
            value=evaluations[branch.child_fen]['white_win']
            value=value if row.side=='White' else 100-value
            qualifies=good_reply(row.win_before,row.initial_damage,value)
            all_chances[branch.reply_uci]=value
            if qualifies:good.append(branch.reply_uci)
            details.append(dict(row_index=row_id,reply_uci=branch.reply_uci,child_fen=branch.child_fen,
                policy_probability=policy[branch.reply_uci],original_player_chance=value,
                remaining_damage_fraction=(row.win_before-value)/row.initial_damage,
                preserves_075_damage=qualifies,legacy_engine_cache=evaluations[branch.child_fen]['legacy_cache']))
        probability=sum(policy[move] for move in good)
        favorite=row.stockfish_favorite_uci
        favorite_good=favorite in good
        favorite_mass=policy[favorite] if favorite_good else 0.
        if not -1e-12<=probability<=1+1e-12 or probability<favorite_mass-1e-12:raise ValueError('Invalid set probability.')
        record=row.to_dict()
        record.update(row_index=row_id,acceptable_reply_count=len(good),acceptable_reply_ucis=json.dumps(good),
            probability_any_preserving_reply=float(np.clip(probability,0,1)),
            probability_stockfish_exact=policy[favorite],stockfish_favorite_qualifies=favorite_good,
            probability_favorite_acceptable_only=favorite_mass,
            probability_other_acceptable_replies=probability-favorite_mass,
            uniform_legal_acceptable_probability=len(good)/len(policy),
            scored_reply_count=len(branches),unscored_probability_mass=0.)
        rows.append(record)
    predictions=pd.DataFrame(rows).sort_values('row_index')
    pd.DataFrame(details).to_csv(out/'all_legal_reply_scores.csv.gz',index=False)
    predictions.to_csv(out/'pre_reply_probabilities.csv',index=False)
    json_write(out/'predictions_ready.json',dict(rows=len(predictions),all_legal_replies_scored=True,
        actual_replies_read=False,probabilities_sha256=digest(out/'pre_reply_probabilities.csv'),
        created_utc=datetime.now(timezone.utc).isoformat()))
    print('All pre-reply probabilities saved before reading observed replies.',flush=True)


def verify_sources(protocol):
    for source,expected in protocol['source_sha256'].items():
        if digest(ROOT/source)!=expected:raise ValueError(f'Frozen source changed: {source}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--per-archive',type=int,default=100)
    parser.add_argument('--models',type=Path,default=MODEL_DIR)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if not (out/'protocol.json').exists():prepare(out,args.per_archive,args.models.resolve())
    elif not args.resume:raise FileExistsError('Use --resume or a new output directory.')
    protocol=json.loads((out/'protocol.json').read_text());verify_sources(protocol)
    if args.prepare_only:return
    if (out/'completed.json').exists():print('Already complete; nothing overwritten.');return
    models=Path(protocol['model_assets'])
    policies=infer_policies(out,models)
    evaluations=score_all_children(out)
    aggregate(out,policies,evaluations)
    verify_sources(protocol)
    from evaluate_human_reply_policy import evaluate
    evaluate(out)
    verify_sources(protocol)
    json_write(out/'completed.json',dict(status='complete',development_only=True,
        exact_model_mass_all_legal_replies=True,sample_mistakes=protocol['sample_mistakes'],
        source_files_preserved=True,new_optuna_trials=0,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Completed:',out,flush=True)


if __name__=='__main__':
    main()
