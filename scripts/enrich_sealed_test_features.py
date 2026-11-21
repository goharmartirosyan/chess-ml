#!/usr/bin/env python3
"""Authorized March input preparation using frozen development feature routines.

No fitted research model, evaluator, tuning or performance reporting is imported.
Stages checkpoint separately. Labels are constructed only after features exist.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import extract_post_mistake_multipv as multipv
import human_reply_policy as core
import scale_bounded_maia_feature as bounded
import extract_opponent_reply_targets as replies
from prepare_sealed_test_csvs import model_schema

BASE = ROOT/'test_data/march_2025_prepared_v2'
ARCHIVE = 'lichess_db_standard_rated_2025-03.pgn.zst'
IDENTITY = ['source_pgn', 'game_index', 'ply', 'site']
META = ['row_id'] + IDENTITY + ['win_before', 'event', 'time_control', 'multipv_lines_returned']
CONFIG = ROOT/'freeze/approved_models.json'
INPUT_PROTOCOL = ROOT/'freeze/final_input_protocol.json'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_environment():
    expected = json.loads((ROOT/'freeze/environment-policy.json').read_text())
    if platform.python_version()!=expected['python']:
        raise ValueError('Use the recorded policy Python environment.')
    for name, value in expected['packages'].items():
        if metadata.version(name)!=value:
            raise ValueError('Policy environment differs: '+name)
    config = json.loads(CONFIG.read_text())
    if core.digest(core.ENGINE)!=config['engine']['sha256']:
        raise ValueError('Frozen Stockfish binary differs.')
    import maia2
    for name, value in config['maia']['installed_package_file_sha256'].items():
        if core.digest(Path(maia2.__file__).parent/name)!=value:
            raise ValueError('Installed Maia source differs: '+name)
    for kind, value in core.WEIGHT_HASHES.items():
        if core.digest(core.MODEL_DIR/f'{kind}_model.pt')!=value:
            raise ValueError('Frozen Maia checkpoint differs.')
    return config


def build_pre_reply(frame):
    needed = set(core.FIELDS+['row_id', 'event', 'time_control'])
    if set(frame)-needed-set(model_schema()['FULL']):
        raise ValueError('Unexpected column in base pre-reply feature table.')
    if needed-set(frame) or set(frame.source_pgn)!={ARCHIVE} or frame.row_id.duplicated().any():
        raise ValueError('Pre-reply input schema/identities differ.')
    frame = frame.copy()
    frame['row_index'] = np.arange(len(frame))
    records = []
    for row in frame.itertuples(index=False):
        board = chess.Board(row.fen_before)
        if row.side not in ('White', 'Black') or board.turn!=(row.side=='White'):
            raise ValueError('Mistake-maker perspective differs from FEN.')
        move = chess.Move.from_uci(row.uci)
        if move not in board.legal_moves:
            raise ValueError('Illegal recorded mistake.')
        board.push(move)
        if board.legal_moves.count()==0:
            raise ValueError('Post-mistake position has no legal reply; no probability invented.')
        kind, domain = core.choose_policy(row.event, row.initial_time_seconds, row.increment_seconds)
        record = row._asdict()
        record.update(root_fen=board.fen(), policy_model=kind, policy_domain=domain,
            active_elo=int(row.opponent_elo) if pd.notna(row.opponent_elo) else 1500,
            other_elo=int(row.player_elo) if pd.notna(row.player_elo) else 1500,
            active_elo_missing=bool(pd.isna(row.opponent_elo)),
            other_elo_missing=bool(pd.isna(row.player_elo)))
        records.append(record)
    return pd.DataFrame(records)


def prepare(out):
    if out.exists():
        raise FileExistsError('Use a new enrichment directory.')
    c = verify_environment()
    base_manifest = json.loads((BASE/'preparation_manifest.json').read_text())
    for name, value in base_manifest['output_sha256'].items():
        if core.digest(BASE/name)!=value:
            raise ValueError('Base preparation output changed: '+name)
    sources = [ROOT/'scripts/enrich_sealed_test_features.py',
        ROOT/'tests/test_enrich_sealed_test_features.py', ROOT/'extract_post_mistake_multipv.py',
        ROOT/'scripts/scale_bounded_maia_feature.py', ROOT/'scripts/human_reply_policy.py',
        ROOT/'scripts/simulate_human_reply_budget.py', ROOT/'scripts/extract_opponent_reply_targets.py',
        CONFIG, ROOT/'freeze/environment-policy.json']
    cache_paths = [core.LEGACY/'positions.sqlite', bounded.PILOT/'candidate_child_evaluations.sqlite']
    sources += cache_paths + [p.with_name(p.name+'-wal') for p in cache_paths
                             if p.with_name(p.name+'-wal').exists()]
    protocol = dict(schema_version=1, date='2026-11-21',
        authorization='User explicitly authorized the existing MultiPV and Maia routines for test-data preparation; no evaluation.',
        evaluation_authorized=False, source_archives=[ARCHIVE],
        base_manifest_sha256=core.digest(BASE/'preparation_manifest.json'),
        base_output_sha256=base_manifest['output_sha256'],
        source_sha256={str(p.relative_to(ROOT)):core.digest(p) for p in sources},
        environment=json.loads((ROOT/'freeze/environment-policy.json').read_text()),
        engine=c['engine'], maia=c['maia'], models=str(core.MODEL_DIR),
        coverage=c['maia']['coverage'], multipv_workers=2, candidate_workers=bounded.WORKERS,
        rows=base_manifest['counts']['selected_rows'],
        multipv='Unchanged analyze_one/init_worker; 2 persistent one-thread workers, depth18 MultiPV5, Hash64, legacy mate100000.',
        candidate_cache='Exact-FEN read-only development pilot/legacy reuse, then own resumable cache. Fresh one-thread Stockfish per ordered 32-FEN batch, depth18 Hash64 mate1000; TT retained within batch.',
        policy='Frozen routing/rating defaults; CPU4 threads seed1701; original ordered batches32 per checkpoint, atomic batch checkpoints; no pilot row-index reuse.',
        labels='Only after features_ready.json: existing reconstruct() and good_reply() event. Actual positions reuse selected candidate scores or same depth18 batch scorer in separate label cache; no feedback to features.',
        original_extraction_settings='Remain unverified; enrichment does not certify the legacy core extraction.',
        fitted_research_models_accessed=False, metrics_authorized=False, commits_created=False)
    out.mkdir(parents=True)
    # Record the protocol before parsing any test feature values for this stage.
    write_json(out/'protocol.json', protocol)
    frame = build_pre_reply(pd.read_csv(BASE/'pre_reply_inputs.csv'))
    if len(frame)!=protocol['rows']:
        raise ValueError('Prepared feature row count differs.')
    frame.to_csv(out/'pre_reply_inputs.csv', index=False)
    write_json(ROOT/'freeze/test_enrichment_protocol.json', dict(protocol,
        local_output_directory=str(out.relative_to(ROOT)),
        local_protocol_sha256=core.digest(out/'protocol.json')))
    print('Preparation protocol and feature-only reconstruction saved.', flush=True)


def verify(out):
    p = json.loads((out/'protocol.json').read_text())
    if p['evaluation_authorized'] or p['metrics_authorized']:
        raise PermissionError('This entry point permits preparation only.')
    verify_environment()
    for name, value in p['source_sha256'].items():
        if core.digest(ROOT/name)!=value:
            raise ValueError('Frozen preparation source/cache differs: '+name)
    for name, value in p['base_output_sha256'].items():
        if core.digest(BASE/name)!=value:
            raise ValueError('Frozen base output differs: '+name)
    return p


def generate_multipv(out, protocol):
    frame = pd.read_csv(out/'pre_reply_inputs.csv')
    checkpoint = out/'multipv_checkpoint.jsonl'
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            row = json.loads(line)
            if row['_row_id'] in completed: raise ValueError('Duplicated MultiPV checkpoint.')
            completed[row['_row_id']] = row
    tasks = [(int(row.row_index), row.fen_before, row.uci) for row in frame.itertuples()
             if row.row_index not in completed]
    with ProcessPoolExecutor(max_workers=protocol['multipv_workers'], initializer=multipv.init_worker,
            initargs=(str(core.ENGINE),18,64)) as pool:
        with checkpoint.open('a') as handle:
            for result in pool.map(multipv.analyze_one, tasks, chunksize=1):
                handle.write(json.dumps(result)+'\n'); handle.flush()
                completed[result['_row_id']] = result
                if len(completed)%25==0 or len(completed)==len(frame):
                    print(f'MultiPV checkpointed: {len(completed)}/{len(frame)}', flush=True)
    if set(completed)!=set(frame.row_index): raise ValueError('Incomplete MultiPV coverage.')
    rows = pd.DataFrame(completed.values()).rename(columns={'_row_id':'row_index'})
    if rows.multipv_error.fillna('').ne('').any():
        rows.to_csv(out/'multipv_errors.csv', index=False)
        raise ValueError('MultiPV errors remain; no enriched output exported.')
    rows.sort_values('row_index').to_csv(out/'multipv_features.csv', index=False)
    write_json(out/'multipv_ready.json', dict(rows=len(rows), feature_sha256=core.digest(out/'multipv_features.csv')))


def infer_policy_batches(out):
    import torch
    torch.set_num_threads(4); torch.manual_seed(1701)
    sample = pd.read_csv(out/'pre_reply_inputs.csv')
    folder = out/'policy_batches'; folder.mkdir(exist_ok=True)
    completed = []
    for kind in ('blitz', 'rapid'):
        part = sample[sample.policy_model.eq(kind)]
        network = None
        for start in range(0,len(part),32):
            batch = part.iloc[start:start+32]
            path = folder/f'{kind}_{start:06d}.json'
            if path.exists():
                records = json.loads(path.read_text())
                if [r['row_index'] for r in records]!=batch.row_index.tolist():
                    raise ValueError('Policy batch identities differ.')
            else:
                if network is None: network = core.load_policy(kind,core.MODEL_DIR)
                policies = core.full_precision_policy(network,batch[['root_fen','active_elo','other_elo']])
                records = [dict(row_index=int(row.row_index),model=kind,move_probabilities=p)
                    for row,p in zip(batch.itertuples(),policies)]
                write_json(path,records)
            completed.extend(records)
            if len(completed)%256<32 or len(completed)==len(sample):
                print(f'Maia policies checkpointed: {len(completed)}/{len(sample)}', flush=True)
        del network
    return {r['row_index']:r for r in completed}


def aggregate_bounds(sample, plan, coverage, evaluations):
    sample = sample.set_index('row_index'); cover = coverage.set_index('row_index')
    result = []
    for index, part in plan.groupby('row_index'):
        row = sample.loc[index]; lower = 0.
        for branch in part.itertuples():
            record = evaluations[branch.child_fen]
            if record['status'] not in ('engine','board_terminal'):
                raise ValueError('Invalid candidate score status.')
            white = record['white_win']
            value = white if row.side=='White' else 100-white
            if core.good_reply(row.win_before,row.initial_damage,value):
                lower += branch.policy_probability
        lower = float(np.clip(lower,0.,1.))
        residual = float(cover.loc[index,'unscored_probability'])
        upper = min(1.,lower+residual)
        if not 0<=lower<=upper<=1 or upper-lower>.025+1e-10:
            raise ValueError('Invalid fixed-coverage probability bound.')
        result.append(dict(row_index=int(index), row_id=row.row_id,
            maia_probability_lower=lower,maia_probability_upper=upper,
            maia_preserving_reply_probability=(lower+upper)/2))
    if len(result)!=len(sample): raise ValueError('Missing probability rows.')
    return pd.DataFrame(result).sort_values('row_index')


def generate_maia(out, protocol):
    policies = infer_policy_batches(out)
    plan, coverage = bounded.make_plan(out,protocol,policies)
    records = bounded.score(out,plan)
    result = aggregate_bounds(pd.read_csv(out/'pre_reply_inputs.csv'),plan,coverage,records)
    result.to_csv(out/'maia_bounds_features.csv',index=False)
    write_json(out/'maia_ready.json',dict(rows=len(result),coverage=.975,
        actual_reply_parsed=False,feature_sha256=core.digest(out/'maia_bounds_features.csv')))


def export_features(out):
    for kind in ('multipv','maia'):
        ready = json.loads((out/f'{kind}_ready.json').read_text())
        name = 'multipv_features.csv' if kind=='multipv' else 'maia_bounds_features.csv'
        if ready['feature_sha256']!=core.digest(out/name): raise ValueError('Feature artifact differs.')
    sample = pd.read_csv(out/'pre_reply_inputs.csv')
    enriched = sample.merge(pd.read_csv(out/'multipv_features.csv'),on='row_index',validate='one_to_one')
    enriched = enriched.merge(pd.read_csv(out/'maia_bounds_features.csv'),on=['row_index','row_id'],validate='one_to_one')
    schema = model_schema()
    features = enriched[list(dict.fromkeys(META+schema['FULL']+[schema['PROB']]))]
    if len(features)!=len(sample) or features[schema['PROB']].isna().any():
        raise ValueError('Incomplete model feature table.')
    gaps = features.multipv_best_second_gap_pp
    if (gaps.isna() & ~features.multipv_lines_returned.eq(1)).any():
        raise ValueError('Unexpected missing MultiPV gap.')
    features.to_csv(out/'features.csv',index=False)
    write_json(out/'features_ready.json',dict(rows=len(features),features_sha256=core.digest(out/'features.csv'),
        observed_reply_inputs_read=False,fitted_research_models_accessed=False,performance_metrics_computed=False))
    print('Strict final feature table saved; no observed reply was used.',flush=True)


def prepare_labels(out):
    # This is label construction, not an assessment of feature predictions.
    ready = json.loads((out/'features_ready.json').read_text())
    if ready['features_sha256']!=core.digest(out/'features.csv'):
        raise ValueError('Pre-reply feature snapshot differs.')
    cols = ['row_id','fen_before','uci','side','next_moves_san','win_before','initial_damage']
    actual = pd.read_csv(BASE/'selected_test.csv',usecols=cols)
    actual_fens = {}; statuses = []
    for row in actual.to_dict('records'):
        board,move,status,error = replies.reconstruct(row)
        if status=='reconstruction_error':
            raise ValueError('Observed-reply reconstruction failed; no negative label substituted.')
        if move is not None:
            board.push_uci(move); actual_fens[row['row_id']] = board.fen()
        statuses.append(dict(row_id=row['row_id'],status=status))
    pd.DataFrame(statuses).to_csv(out/'reply_label_status.csv',index=False)
    values = score_label_positions(out,list(dict.fromkeys(actual_fens.values())))
    immediate = []
    for row in actual.itertuples():
        if row.row_id not in actual_fens:
            value = np.nan
        else:
            white = values[actual_fens[row.row_id]]['white_win']
            chance = white if row.side=='White' else 100-white
            value = float(core.good_reply(row.win_before,row.initial_damage,chance))
        immediate.append(dict(row_id=row.row_id,actual_preserved=value))
    later = pd.read_csv(BASE/'labels_later_play.csv')
    labels = later.merge(pd.DataFrame(immediate),on='row_id',validate='one_to_one')
    features_ids = pd.read_csv(out/'features.csv',usecols=['row_id']).row_id
    if not np.array_equal(labels.row_id,features_ids): raise ValueError('Feature/label identities differ.')
    labels.to_csv(out/'labels.csv',index=False)
    input_manifest = dict(features=dict(path='features.csv',sha256=core.digest(out/'features.csv')),
        labels=dict(path='labels.csv',sha256=core.digest(out/'labels.csv')),
        input_protocol_sha256=core.digest(INPUT_PROTOCOL),enrichment_protocol_sha256=core.digest(out/'protocol.json'))
    write_json(out/'input_manifest.json',input_manifest)
    # Convenience table retains labels; only features.csv is a model input.
    selected = pd.read_csv(BASE/'selected_test.csv')
    extra = pd.read_csv(out/'features.csv')[['row_id']+multipv.RESULT_COLUMNS[:3]+[bounded.FEATURE]]
    combined = selected.merge(extra,on='row_id',validate='one_to_one').merge(
        labels[['row_id','actual_preserved']],on='row_id',validate='one_to_one')
    combined.to_csv(out/'test_with_frozen_features.csv',index=False)
    completion = dict(status='feature_and_label_tables_prepared',rows=len(labels),
        features_sha256=core.digest(out/'features.csv'),labels_sha256=core.digest(out/'labels.csv'),
        combined_sha256=core.digest(out/'test_with_frozen_features.csv'),
        original_extraction_provenance_verified=False,final_evaluation_enabled=False,
        frozen_research_models_loaded=False,predictions_run=False,performance_metrics_computed=False,
        distributions_inspected=False,commits_created=False)
    write_json(out/'preparation_complete.json',completion)
    print('Separate feature/label tables and combined convenience CSV saved; no model evaluation.',flush=True)


def score_label_positions(out,required):
    # Read-only reuse; actual replies never alter the saved policy prefix/cache.
    source = sqlite3.connect(f'file:{out/"selected_child_evaluations.sqlite"}?mode=ro',uri=True)
    candidates = {fen:json.loads(record) for fen,record in source.execute('SELECT fen,record FROM evaluations')}
    source.close()
    db = sqlite3.connect(out/'actual_reply_evaluations.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS evaluations (fen TEXT PRIMARY KEY,record TEXT NOT NULL)')
    records = {fen:json.loads(record) for fen,record in db.execute('SELECT fen,record FROM evaluations')}
    for fen in required:
        if fen in candidates and fen not in records:
            records[fen] = candidates[fen]
            db.execute('INSERT INTO evaluations VALUES (?,?)',(fen,json.dumps(records[fen])))
    db.commit()
    pending = [fen for fen in required if fen not in records or records[fen]['status']=='error']
    tasks = [(pending[i:i+32],str(core.ENGINE)) for i in range(0,len(pending),32)]
    with ProcessPoolExecutor(max_workers=bounded.WORKERS) as pool:
        done = 0
        for batch in pool.map(core.score_child_batch,tasks,chunksize=1):
            with db:
                for record in batch:
                    records[record['fen']] = record
                    db.execute('INSERT OR REPLACE INTO evaluations VALUES (?,?)',(record['fen'],json.dumps(record)))
            done += len(batch)
            if done%256<32 or done==len(pending):
                print(f'Actual-reply label positions checkpointed: {done}/{len(pending)}',flush=True)
    db.close()
    if any(records[fen]['status'] not in ('engine','board_terminal') for fen in required):
        raise ValueError('Observed-reply engine errors remain.')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','multipv','maia','export','labels'],required=True)
    args = p.parse_args(); out = args.out.resolve()
    if args.stage=='prepare':prepare(out); return
    protocol = verify(out)
    if args.stage=='multipv':generate_multipv(out,protocol)
    elif args.stage=='maia':generate_maia(out,protocol)
    elif args.stage=='export':export_features(out)
    else:prepare_labels(out)


if __name__=='__main__':
    main()
