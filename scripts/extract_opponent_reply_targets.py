#!/usr/bin/env python3
"""Depth-18 development-only opponent-reply targets with resumable position cache.

The actual reply is reconstructed from FEN, mistake UCI, and next_moves_san.
One single-PV Stockfish analysis after the mistake selects the best reply and
scores W0. The same engine then scores the positions after the actual and best
replies. Scores are normalized to the original mistake-maker's perspective.

The SQLite position cache is an append-only checkpoint. Existing source CSVs
are never modified. Missing replies and engine errors remain explicit rows.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.engine
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVES = {
    f'lichess_db_standard_rated_{month}.pgn.zst'
    for month in ('2017-04', '2017-06', '2017-07', '2018-02', '2019-02')
}
K = .00368208
MATE_CP = 1000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def win_chance(cp):
    return 100 / (1 + math.exp(-K * cp))


def read_development(path):
    if path.name != 'development_with_multipv.csv':
        raise ValueError('Only the canonical development_with_multipv.csv is accepted.')
    frame = pd.read_csv(path, low_memory=False)
    needed = {'source_pgn', 'game_index', 'ply', 'site', 'side', 'fen_before', 'uci',
              'next_moves_san', 'win_before', 'win_after', 'initial_damage',
              'eval_after_cp', 'multipv_top5_uci_json',
              'multipv_top5_score_cp_json', 'multipv_top5_win_chance_json'}
    if needed - set(frame):
        raise ValueError(f'Development table lacks {sorted(needed - set(frame))}')
    archives = set(frame['source_pgn'].astype(str).map(lambda s: Path(s).name))
    if archives != EXPECTED_ARCHIVES or len(frame) != 23167:
        raise ValueError('Input is not exactly the canonical five development months.')
    if frame.duplicated(['source_pgn', 'game_index', 'ply']).any():
        raise ValueError('Duplicate mistake identifiers.')
    if frame['site'].nunique() != 6512 or not (
            (frame['initial_damage'] > 20) & frame['win_before'].between(10, 90)).all():
        raise ValueError('Development counts or fixed selection changed.')
    return frame


def reconstruct(row):
    """Return (post-mistake board, actual UCI, status, error)."""
    try:
        board = chess.Board(row['fen_before'])
        original = chess.WHITE if row['side'] == 'White' else chess.BLACK
        if row['side'] not in ('White', 'Black') or board.turn != original:
            raise ValueError('Mistake-maker side does not match FEN turn')
        move = chess.Move.from_uci(row['uci'])
        if move not in board.legal_moves:
            raise ValueError('Stored mistake UCI is illegal')
        board.push(move)
        if board.turn == original:
            raise ValueError('Turn did not change after mistake')
        context = row['next_moves_san']
        if pd.isna(context) or not str(context).strip():
            return board, None, 'missing_actual_reply', ''
        tokens = str(context).split()
        if len(tokens) < 2:
            raise ValueError('Next-move SAN context is malformed')
        reply = board.parse_san(tokens[1])
        if reply not in board.legal_moves:
            raise ValueError('Parsed actual reply is illegal')
        return board, reply.uci(), 'ok', ''
    except Exception as exc:
        return None, None, 'reconstruction_error', f'{type(exc).__name__}: {exc}'[:300]


def audit_sample(frame, output, n=80, seed=1701):
    """Stratified development-only legality and perspective audit, 16 per month."""
    if n < 50 or n > 100 or n % 5:
        raise ValueError('Audit sample must be 50–100 and divisible by five months.')
    recorded = frame[frame['next_moves_san'].notna()]
    sample = recorded.groupby('source_pgn', group_keys=False).sample(n=n // 5, random_state=seed)
    rows = []
    for _, row in sample.sort_values(['source_pgn', 'game_index', 'ply']).iterrows():
        board, actual, status, error = reconstruct(row)
        if status != 'ok':
            raise ValueError(f'Audit reconstruction failed: {error}')
        lines = json.loads(row['multipv_top5_uci_json'])
        cps = json.loads(row['multipv_top5_score_cp_json'])
        chances = json.loads(row['multipv_top5_win_chance_json'])
        if not lines or len(lines) != len(cps) or len(lines) != len(chances):
            raise ValueError('MultiPV arrays disagree.')
        if not all(chess.Move.from_uci(move) in board.legal_moves for move in lines):
            raise ValueError('Stored MultiPV line is illegal after the mistake.')
        if abs(win_chance(row['eval_after_cp']) - row['win_after']) > 1e-8:
            raise ValueError('Core evaluation perspective/conversion mismatch.')
        opponent_best = win_chance(cps[0])
        if abs(opponent_best - chances[0]) > 1e-6:
            raise ValueError('MultiPV winning chance conversion mismatch.')
        rows.append(dict(source_pgn=row['source_pgn'], game_index=row['game_index'],
                         ply=row['ply'], site=row['site'], side=row['side'],
                         mistake_uci=row['uci'], actual_reply_san=str(row['next_moves_san']).split()[1],
                         actual_reply_uci=actual, stored_multipv_best_uci=lines[0],
                         actual_in_stored_top5=actual in lines,
                         stored_best_opponent_cp=cps[0],
                         stored_best_original_win_chance=100-opponent_best,
                         core_win_after=row['win_after'], audit_status='ok'))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite audit: {output}')
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Audit passed: {len(rows)} rows, all five months, '
          f'{sum(r["actual_in_stored_top5"] for r in rows)} actual replies in stored top five.')
    return rows


def score_position(fen, engine, depth):
    """Return white-perspective score plus root PV move, or a recorded error."""
    started = time.monotonic()
    record = dict(fen=fen, status='error', white_cp=None, white_win=None,
                  is_mate=False, best_move_uci=None, error='', elapsed_ms=None)
    try:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            outcome = board.outcome(claim_draw=False)
            if outcome is None:
                raise ValueError('Terminal board has no chess outcome')
            white_win = 50.0 if outcome.winner is None else (100.0 if outcome.winner else 0.0)
            record.update(status='board_terminal', white_win=white_win)
        else:
            info = engine.analyse(board, chess.engine.Limit(depth=depth),
                                  info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
            if info.get('score') is None or not info.get('pv'):
                raise ValueError('Engine omitted score or root PV')
            score = info['score'].pov(chess.WHITE)
            mate = score.mate()
            cp = (MATE_CP if mate > 0 else -MATE_CP) if mate is not None else int(score.score())
            record.update(status='engine', white_cp=cp, white_win=win_chance(cp),
                          is_mate=mate is not None, best_move_uci=info['pv'][0].uci())
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'[:300]
    record['elapsed_ms'] = round(1000*(time.monotonic()-started), 3)
    return record


def score_batch(task):
    fens, engine_path, depth, hash_mb = task
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({'Threads': 1, 'Hash': hash_mb})
        return [score_position(fen, engine, depth) for fen in fens]


def open_cache(path):
    database = sqlite3.connect(path)
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('CREATE TABLE IF NOT EXISTS positions ('
                     'fen TEXT PRIMARY KEY, status TEXT NOT NULL, white_cp REAL, '
                     'white_win REAL, is_mate INTEGER, best_move_uci TEXT, '
                     'error TEXT, elapsed_ms REAL)')
    return database


def analyze_missing(database, fens, engine_path, depth, hash_mb, workers, batch, phase):
    cached = {row[0] for row in database.execute('SELECT fen FROM positions')}
    pending = sorted(set(fens) - cached)
    print(f'{phase}: {len(set(fens))} unique positions, {len(pending)} uncached', flush=True)
    if not pending:
        return
    tasks = [(pending[start:start+batch], str(engine_path), depth, hash_mb)
             for start in range(0, len(pending), batch)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        count = 0
        for results in pool.map(score_batch, tasks, chunksize=1):
            values = [(record['fen'], record['status'], record['white_cp'],
                       record['white_win'], int(record['is_mate']),
                       record['best_move_uci'], record['error'], record['elapsed_ms'])
                      for record in results]
            with database:
                database.executemany('INSERT INTO positions VALUES (?,?,?,?,?,?,?,?)', values)
            count += len(results)
            if count % 500 < len(results) or count == len(pending):
                print(f'{phase}: checkpointed {count}/{len(pending)} new positions', flush=True)


def cache_lookup(database):
    keys = ('fen', 'status', 'white_cp', 'white_win', 'is_mate',
            'best_move_uci', 'error', 'elapsed_ms')
    return {record[0]: dict(zip(keys, record)) for record in
            database.execute('SELECT * FROM positions')}


def original_win(record, original_side):
    if record is None or record['status'] not in ('engine', 'board_terminal'):
        return None
    return record['white_win'] if original_side == 'White' else 100-record['white_win']


def original_cp(record, original_side):
    if record is None or record['status'] != 'engine':
        return None
    return record['white_cp'] if original_side == 'White' else -record['white_cp']


def run_extraction(args):
    source = args.input.resolve()
    engine_path = args.stockfish.resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f'Stockfish executable not found: {engine_path}')
    frame = read_development(source)
    full_rows = len(frame)
    if args.sample_size:
        if args.sample_size % 5 or args.sample_size < 5:
            raise ValueError('Sample size must be positive and divisible by five archives.')
        frame = (frame[frame['next_moves_san'].notna()]
                 .groupby('source_pgn', group_keys=False)
                 .sample(n=args.sample_size // 5, random_state=args.sample_seed)
                 .sort_index())
    data_hash = sha256(source)
    engine_hash = sha256(engine_path)
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine_id = dict(engine.id)
        if not {'Threads', 'Hash'}.issubset(engine.options):
            raise ValueError('Engine lacks required Threads/Hash options.')
    out = args.out.resolve()
    manifest_path = out / 'manifest.json'
    identity = dict(data_sha256=data_hash, engine_sha256=engine_hash,
                    depth=args.depth, threads=1, hash_mb=args.hash_mb,
                    mate_score_cp=MATE_CP, k=K,
                    sample_size=args.sample_size, sample_seed=args.sample_seed)
    if out.exists():
        if not args.resume or not manifest_path.exists():
            raise FileExistsError(f'Output exists; use --resume only for this run: {out}')
        prior = json.loads(manifest_path.read_text())
        if {key: prior.get(key) for key in identity} != identity:
            raise ValueError('Resume metadata disagrees with data or engine settings.')
        if (out / 'reply_targets.csv').exists():
            raise FileExistsError('Completed target CSV exists; choose a new output directory.')
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(identity, kind='development-only immediate opponent reply targets',
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        input_path=str(source), engine_path=str(engine_path),
                        engine_id=engine_id, engine_source=args.engine_source,
                        source_archives=sorted(EXPECTED_ARCHIVES), mistakes=len(frame),
                        full_input_mistakes=full_rows, smoke_only=bool(args.sample_size),
                        games=frame['site'].nunique(), workers=args.workers,
                        checkpoint_batch=args.batch,
                        definitions={'missed_opportunity_damage_pp': 'W_actual_reply - W_after_mistake',
                                     'opponent_reply_regret_pp': 'W_actual_reply - W_best_reply',
                                     'normalized': 'raw percentage-point difference / historical initial_damage'},
                        note='All three W evaluations use one engine/config. Best move is from '
                             'single-PV search after the mistake; best and actual resulting '
                             'positions are each analyzed at the same depth. Terminal board '
                             'outcomes use exact 0/50/100. No PGN termination inference.',
                        versions={name: importlib.metadata.version(name)
                                  for name in ('python-chess', 'pandas')},
                        script_sha256=sha256(Path(__file__)))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    rows, roots = [], []
    for _, row in frame.iterrows():
        board, actual, status, error = reconstruct(row)
        root_fen = board.fen() if board is not None else None
        if root_fen is not None and actual is not None:
            roots.append(root_fen)
        rows.append(dict(source_pgn=row['source_pgn'], game_index=int(row['game_index']),
                         ply=int(row['ply']), site=row['site'], side=row['side'],
                         fen_after_mistake=root_fen, actual_reply_uci=actual,
                         reconstruction_status=status, reconstruction_error=error,
                         historical_initial_damage=float(row['initial_damage']),
                         historical_win_after=float(row['win_after'])))
    database = open_cache(out / 'positions.sqlite')
    try:
        analyze_missing(database, roots, engine_path, args.depth, args.hash_mb,
                        args.workers, args.batch, 'Post-mistake/root')
        cache = cache_lookup(database)
        reply_fens = []
        for row in rows:
            if row['reconstruction_status'] != 'ok':
                continue
            root = cache.get(row['fen_after_mistake'])
            if root is None or root['status'] != 'engine' or not root['best_move_uci']:
                row['root_error'] = root['error'] if root else 'No cached root evaluation'
                continue
            board = chess.Board(row['fen_after_mistake'])
            best = chess.Move.from_uci(root['best_move_uci'])
            actual = chess.Move.from_uci(row['actual_reply_uci'])
            if best not in board.legal_moves or actual not in board.legal_moves:
                row['root_error'] = 'Engine best or actual reply is illegal in root FEN'
                continue
            actual_board = board.copy(stack=False)
            actual_board.push(actual)
            best_board = board.copy(stack=False)
            best_board.push(best)
            row['best_reply_uci'] = best.uci()
            row['fen_after_actual_reply'] = actual_board.fen()
            row['fen_after_best_reply'] = best_board.fen()
            reply_fens.extend((actual_board.fen(), best_board.fen()))
        analyze_missing(database, reply_fens, engine_path, args.depth, args.hash_mb,
                        args.workers, args.batch, 'After actual/best replies')
        cache = cache_lookup(database)
        output = []
        for row in rows:
            root = cache.get(row.get('fen_after_mistake'))
            actual = cache.get(row.get('fen_after_actual_reply'))
            best = cache.get(row.get('fen_after_best_reply'))
            w0 = original_win(root, row['side'])
            wa = original_win(actual, row['side'])
            wb = original_win(best, row['side'])
            missed = wa-w0 if wa is not None and w0 is not None else None
            regret = wa-wb if wa is not None and wb is not None else None
            status = ('ok' if missed is not None and regret is not None else
                      row['reconstruction_status'] if row['reconstruction_status'] != 'ok' else
                      'root_evaluation_error' if w0 is None or not row.get('best_reply_uci') else
                      'actual_reply_evaluation_error' if wa is None else
                      'best_reply_evaluation_error')
            output.append(dict(
                source_pgn=row['source_pgn'], game_index=row['game_index'], ply=row['ply'],
                site=row['site'], side=row['side'], status=status,
                fen_after_mistake=row.get('fen_after_mistake'),
                fen_after_actual_reply=row.get('fen_after_actual_reply'),
                fen_after_best_reply=row.get('fen_after_best_reply'),
                actual_reply_uci=row['actual_reply_uci'],
                best_reply_uci=row.get('best_reply_uci'),
                actual_matches_best=(row['actual_reply_uci'] == row.get('best_reply_uci'))
                                    if row['actual_reply_uci'] and row.get('best_reply_uci') else None,
                win_after_mistake_new=w0, win_after_actual_reply=wa,
                win_after_best_reply=wb,
                eval_after_mistake_new_cp=original_cp(root, row['side']),
                eval_after_actual_reply_cp=original_cp(actual, row['side']),
                eval_after_best_reply_cp=original_cp(best, row['side']),
                missed_opportunity_damage_pp=missed, opponent_reply_regret_pp=regret,
                historical_initial_damage=row['historical_initial_damage'],
                missed_opportunity_over_initial_damage=(missed / row['historical_initial_damage']
                                                        if missed is not None else None),
                reply_regret_over_initial_damage=(regret / row['historical_initial_damage']
                                                  if regret is not None else None),
                historical_win_after=row['historical_win_after'],
                new_minus_historical_win_after_pp=(w0-row['historical_win_after']
                                                   if w0 is not None else None),
                actual_reply_missing=row['actual_reply_uci'] is None,
                root_evaluation_missing=w0 is None,
                actual_evaluation_missing=wa is None,
                best_evaluation_missing=wb is None,
                root_mate_score_used=bool(root and root['is_mate']),
                actual_mate_score_used=bool(actual and actual['is_mate']),
                best_mate_score_used=bool(best and best['is_mate']),
                actual_board_terminal=bool(actual and actual['status'] == 'board_terminal'),
                best_board_terminal=bool(best and best['status'] == 'board_terminal'),
                negative_regret=(regret < 0 if regret is not None else None),
                negative_regret_over_1pp=(regret < -1 if regret is not None else None),
                engine_name=engine_id.get('name'), engine_sha256=engine_hash,
                engine_depth=args.depth, engine_threads=1, engine_hash_mb=args.hash_mb,
                root_error=row.get('root_error') or (root['error'] if root else ''),
                actual_error=actual['error'] if actual else '',
                best_error=best['error'] if best else '',
                reconstruction_error=row['reconstruction_error']))
        result = pd.DataFrame(output)
        if len(result) != len(frame) or result.duplicated(['source_pgn', 'game_index', 'ply']).any():
            raise ValueError('Output identities/counts changed.')
        target_path = out / 'reply_targets.csv'
        temp_path = out / 'reply_targets.csv.tmp'
        result.to_csv(temp_path, index=False)
        os.replace(temp_path, target_path)
        errors = result[result['status'] != 'ok']
        errors.to_csv(out / 'reply_target_errors.csv', index=False)
        counts = result['status'].value_counts().to_dict()
        (out / 'status_counts.json').write_text(json.dumps(counts, indent=2))
        print('Saved:', target_path, 'status counts:', counts, flush=True)
    finally:
        database.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'development_with_multipv.csv')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stockfish', type=Path)
    parser.add_argument('--engine-source', default='')
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--hash-mb', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--audit-size', type=int, default=80)
    parser.add_argument('--audit-seed', type=int, default=1701)
    parser.add_argument('--sample-size', type=int, default=0,
                        help='Stratified five-month smoke extraction; omit for full development')
    parser.add_argument('--sample-seed', type=int, default=1701)
    args = parser.parse_args()
    if args.depth != 18 or args.workers < 1 or args.hash_mb < 1 or args.batch < 1:
        parser.error('Use depth 18 and positive workers, hash, and batch.')
    if args.audit_only:
        frame = read_development(args.input.resolve())
        audit_sample(frame, args.out.resolve() / 'audit_sample.csv', args.audit_size, args.audit_seed)
    elif args.stockfish is None:
        parser.error('--stockfish is required for extraction')
    else:
        run_extraction(args)


if __name__ == '__main__':
    main()
