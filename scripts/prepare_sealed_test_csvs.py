#!/usr/bin/env python3
"""Stage the five explicitly authorized test CSVs; never fit, predict or score.

The CLI accepts only the fixed input list in the recorded preparation protocol.
No chess engine, policy model, evaluator or learned preprocessing is imported.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_chess_modeling import HORIZONS, DERIVED, prepare

PROTOCOL = ROOT / 'test_data/march_2025_prepared_v2/provenance/source_inputs.json'
ARCHIVE = 'lichess_db_standard_rated_2025-03.pgn.zst'
SOURCE_ALIASES = {ARCHIVE, 'lichess_2025_03_partial.pgn.zst'}
IDENTITY = ['source_pgn', 'game_index', 'ply', 'site']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def model_schema():
    # Read constant declarations only; do not import the evaluator or models.
    tree = ast.parse((ROOT / 'scripts/frozen_chess_evaluation.py').read_text())
    names = {'FULL', 'PROB'}
    return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in names}


def identity(row):
    if Path(row['source_pgn']).name not in SOURCE_ALIASES:
        raise ValueError('A supplied file contains a non-March-2025 archive.')
    if not row['site'].strip():
        raise ValueError('Missing stable game identity.')
    values = []
    for key in ('game_index', 'ply'):
        n = float(row[key])
        if not math.isfinite(n) or n <= 0 or n != int(n):
            raise ValueError('Invalid game/move index.')
        values.append(int(n))
    # Preserve the original source string in the combined CSV; final features
    # use the canonical archive basename, as required by the frozen evaluator.
    return (ARCHIVE, *values, row['site'])


def row_id(key):
    return hashlib.sha256(json.dumps(list(key), separators=(',', ':')).encode()).hexdigest()


def protocol_input_paths(protocol):
    """Source filenames belong to the local manifest, not repository code."""
    inputs = protocol['inputs']
    if len(inputs) != 5:
        raise ValueError('The authorized source manifest must contain five extracts.')
    names = [item['path'] for item in inputs]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate source paths in the local manifest.')
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.csv':
            raise ValueError('Source paths must name CSVs within the repository workspace.')
    return [ROOT/name for name in names]


def stage(inputs, out, expected_hashes, protocol_hash):
    """Technical preparation only. Tests supply synthetic files explicitly."""
    inputs, out = [Path(p) for p in inputs], Path(out)
    if out.exists():
        raise FileExistsError('Use a new preparation directory.')
    if len(inputs) != len(expected_hashes):
        raise ValueError('Input hash roster differs.')
    headers = []
    for path, expected in zip(inputs, expected_hashes):
        if digest(path) != expected:
            raise ValueError('A supplied input hash differs from the recorded protocol.')
        with path.open(newline='', encoding='utf-8-sig') as f:
            header = next(csv.reader(f))
        if not header or len(header) != len(set(header)):
            raise ValueError('Missing or duplicated CSV headers.')
        headers.append(header)
    if not headers or any(h != headers[0] for h in headers):
        raise ValueError('Input schemas differ; no merge performed.')
    cols = headers[0]
    required = set(IDENTITY + ['fen_before', 'uci', 'side', 'result', 'win_before', 'win_after'])
    for h in HORIZONS:
        required.update(f'{prefix}_{h}ply' for prefix in ['horizon', 'game_ended_before',
            'win_after', 'eval_after', 'terminal_result_used_for', 'mate_score_used_after',
            'remaining_damage', 'persistence_ratio', 'persistence_category'])
        required.discard(f'horizon_{h}ply')
        required.add(f'horizon_{h}ply_reached')
        required.discard(f'eval_after_{h}ply')
        required.add(f'eval_after_{h}ply_cp')
        required.add(f'persistence_ratio_{h}ply_clipped')
    if required - set(cols):
        raise ValueError('Inputs lack fields required by the frozen target construction.')
    schema = model_schema()
    known_features = [f for f in schema['FULL'] if f in cols]
    features = list(dict.fromkeys(['row_id'] + IDENTITY + ['win_before', 'event',
        'time_control', 'fen_before', 'uci', 'side'] + known_features))
    if set(features) - set(cols) - {'row_id'}:
        raise ValueError('Inputs lack required pre-reply metadata.')
    later_labels = ['row_id', 'win_after'] + [f'{prefix}_{h}ply'
        for h in HORIZONS for prefix in ['win_after', 'change_in_damage']]
    prepared_cols = ['row_id'] + cols + [c for c in DERIVED if c not in cols]
    out.mkdir(parents=True)
    # Membership checks read development identifiers only, never development labels.
    with (ROOT / 'development_with_multipv.csv').open(newline='') as f:
        development_sites = {r['site'] for r in csv.DictReader(f)}
    counts = dict(input_rows=0, unique_rows=0, exact_duplicates_removed=0,
                  selected_rows=0, excluded_rows=0)
    seen, seen_game_ply, seen_primary, games, selected_games = {}, {}, {}, set(), set()
    output_names = ['combined_test.csv', 'selected_test.csv', 'pre_reply_inputs.csv',
                    'labels_later_play.csv', 'exclusions.csv', 'row_lineage.csv']
    handles = []
    try:
        writers = []
        for name, fields in zip(output_names, [cols, prepared_cols, features, later_labels,
                ['row_id', 'reason'], ['input_file', 'input_row', 'row_id', 'status']]):
            handle = (out / name).open('x', newline='', encoding='utf-8')
            handles.append(handle)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writers.append(writer)
        merged, selected, pre_reply, labels, exclusions, lineage = writers
        for path in inputs:
            with path.open(newline='', encoding='utf-8-sig') as f:
                for position, row in enumerate(csv.DictReader(f), start=2):
                    counts['input_rows'] += 1
                    if None in row or any(v is None for v in row.values()):
                        raise ValueError('Malformed input row.')
                    key = identity(row); rid = row_id(key)
                    if key[3] in development_sites:
                        raise ValueError('Test game overlaps development; refusing preparation.')
                    canonical = dict(row, source_pgn=ARCHIVE)
                    # Compare all fields for duplicates; never choose between
                    # conflicting outcomes or predictor values.
                    signature = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                    game_ply = (key[3], key[2])
                    if key in seen:
                        if seen[key] != signature:
                            raise ValueError('Conflicting duplicate rows; no version selected.')
                        counts['exact_duplicates_removed'] += 1
                        lineage.writerow(dict(input_file=path.name, input_row=position,
                                              row_id=rid, status='exact_duplicate'))
                        continue
                    if game_ply in seen_game_ply:
                        raise ValueError('Game/ply occurs with conflicting source identities.')
                    if key[:3] in seen_primary:
                        raise ValueError('Source/game/ply occurs with conflicting game identities.')
                    seen[key] = signature; seen_game_ply[game_ply] = key
                    seen_primary[key[:3]] = key
                    games.add(key[3]); counts['unique_rows'] += 1
                    merged.writerow(row)
                    lineage.writerow(dict(input_file=path.name, input_row=position,
                                          row_id=rid, status='retained_unique'))
                    try:
                        changed, reasons = prepare(canonical.copy(), 20.0, True)
                    except (ValueError, KeyError, OverflowError):
                        changed, reasons = canonical, ['invalid_target_construction_fields']
                    if reasons:
                        counts['excluded_rows'] += 1
                        exclusions.writerow(dict(row_id=rid, reason=';'.join(dict.fromkeys(reasons))))
                        continue
                    changed['row_id'] = rid
                    # Match the frozen target equation exactly; the shared
                    # preparation function also retains its terminal semantics.
                    after = float(changed['win_after'])
                    for h in HORIZONS:
                        changed[f'change_in_damage_{h}ply'] = repr(after-float(changed[f'win_after_{h}ply']))
                    selected.writerow(changed)
                    pre_reply.writerow({c: changed[c] for c in features})
                    labels.writerow({c: changed[c] for c in later_labels})
                    selected_games.add(key[3]); counts['selected_rows'] += 1
        for handle in handles: handle.close()
        for path, expected in zip(inputs, expected_hashes):
            if digest(path) != expected:
                raise ValueError('An original input changed during preparation.')
        counts.update(unique_games=len(games), selected_games=len(selected_games))
        missing = [c for c in schema['FULL'] + [schema['PROB']] if c not in cols]
        manifest = dict(schema_version=1, status='base_tables_prepared_enrichment_required',
            protocol_sha256=protocol_hash, input_sha256={p.name: h for p, h in zip(inputs, expected_hashes)},
            counts=counts, canonical_archive=ARCHIVE, selection='10 <= win_before <= 90; initial_damage > 20',
            target_construction='Existing prepare_chess_modeling.prepare, require_all=True; frozen sign equation',
            output_sha256={name: digest(out/name) for name in output_names},
            missing_model_predictors=missing, missing_quality_metadata=['multipv_lines_returned'],
            missing_immediate_label='actual_preserved: actual reply must be reconstructed and scored separately',
            ready_for_frozen_evaluator=False, predictions_run=False, metrics_computed=False,
            engine_calls=0, policy_inference_calls=0, distributions_inspected=False,
            original_sources_modified=False, fitted_models_accessed=False,
            accepted_source_aliases=sorted(SOURCE_ALIASES),
            lineage_note='The supplied partial-PGN name is mapped only for row identity; this does not certify original archive selection or extraction settings.',
            original_extraction_settings='Unverified; supplied CSVs accepted only for authorized preparation')
        write_json(out/'preparation_manifest.json', manifest)
        return manifest
    except Exception:
        for handle in handles: handle.close()
        write_json(out/'PREPARATION_FAILED.json', dict(status='failed', ready_for_frozen_evaluator=False,
                   reason='Technical validation failed. Partial outputs must not be used.'))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['status'] != 'authorized_preparation_only' or protocol['evaluation_authorized']:
        raise PermissionError('Protocol must authorize preparation only.')
    if protocol['accepted_source_aliases'] != sorted(SOURCE_ALIASES):
        raise ValueError('Archive alias contract differs from the recorded protocol.')
    for name, value in protocol['code_sha256'].items():
        if digest(ROOT/name) != value:
            raise ValueError('Preparation implementation differs from the recorded protocol.')
    paths = protocol_input_paths(protocol)
    result = stage(paths, args.out,
                   [x['sha256'] for x in protocol['inputs']], digest(PROTOCOL))
    print(json.dumps(dict(status=result['status'], counts=result['counts'],
        ready_for_frozen_evaluator=False, predictions_run=False, metrics_computed=False), indent=2))


if __name__ == '__main__':
    main()
