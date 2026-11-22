#!/usr/bin/env python3
"""Read-only development summaries for immediate opponent-reply targets."""

import argparse
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.extract_opponent_reply_targets import (EXPECTED_ARCHIVES, read_development,
                                                    sha256, win_chance)  # noqa: E402

TARGETS = ('missed_opportunity_damage_pp', 'opponent_reply_regret_pp',
           'missed_opportunity_over_initial_damage', 'reply_regret_over_initial_damage')


def load_validated(data_path, target_path):
    frame = read_development(data_path)
    manifest = json.loads((target_path.parent / 'manifest.json').read_text())
    if (manifest['data_sha256'] != sha256(data_path)
            or set(manifest['source_archives']) != EXPECTED_ARCHIVES
            or manifest['depth'] != 18 or manifest['mate_score_cp'] != 1000):
        raise ValueError('Target extraction provenance does not match fixed development protocol.')
    targets = pd.read_csv(target_path, low_memory=False)
    keys = ['source_pgn', 'game_index', 'ply']
    if targets.duplicated(keys).any() or targets['source_pgn'].isna().any():
        raise ValueError('Target identities are missing or duplicated.')
    if not manifest['smoke_only'] and len(targets) != len(frame):
        raise ValueError('Full target file must cover all canonical development mistakes.')
    if manifest['smoke_only'] and len(targets) != manifest['mistakes']:
        raise ValueError('Smoke target count differs from manifest.')
    joined = frame.merge(targets, on=keys, how='inner', validate='one_to_one',
                         suffixes=('', '_target'))
    if len(joined) != len(targets):
        raise ValueError('Some target identities do not match development input.')
    if not (joined['site'] == joined['site_target']).all():
        raise ValueError('Game site differs between source and targets.')
    valid = joined[joined['status'] == 'ok']
    if not (valid['side'] == valid['side_target']).all():
        raise ValueError('Mistake-maker perspective changed during extraction.')
    for row in valid.itertuples(index=False):
        board = chess.Board(row.fen_before)
        original = chess.WHITE if row.side == 'White' else chess.BLACK
        mistake = chess.Move.from_uci(row.uci)
        if board.turn != original or mistake not in board.legal_moves:
            raise ValueError('Invalid source mistake board or perspective.')
        board.push(mistake)
        if board.fen() != row.fen_after_mistake or board.turn == original:
            raise ValueError('Post-mistake FEN or side to move changed.')
        actual = chess.Move.from_uci(row.actual_reply_uci)
        best = chess.Move.from_uci(row.best_reply_uci)
        if actual not in board.legal_moves or best not in board.legal_moves:
            raise ValueError('Reply move is illegal in stored post-mistake FEN.')
        if board.parse_san(str(row.next_moves_san).split()[1]) != actual:
            raise ValueError('Actual reply differs from source SAN continuation.')
        actual_board = board.copy(stack=False)
        best_board = board.copy(stack=False)
        actual_board.push(actual)
        best_board.push(best)
        if (actual_board.fen() != row.fen_after_actual_reply or
                best_board.fen() != row.fen_after_best_reply):
            raise ValueError('Reply-position FEN disagrees with stored move.')
        for label, reply_board in (('actual', actual_board), ('best', best_board)):
            terminal = reply_board.is_game_over(claim_draw=False)
            if bool(getattr(row, f'{label}_board_terminal')) != terminal:
                raise ValueError(f'{label} terminal flag disagrees with reply board.')
            if terminal:
                outcome = reply_board.outcome(claim_draw=False)
                expected = (50.0 if outcome.winner is None else
                            100.0 if outcome.winner == original else 0.0)
                if abs(getattr(row, f'win_after_{label}_reply') - expected) > 1e-9:
                    raise ValueError(f'{label} terminal chance disagrees with board outcome.')
    for cp_column, chance_column in (
            ('eval_after_mistake_new_cp','win_after_mistake_new'),
            ('eval_after_actual_reply_cp','win_after_actual_reply'),
            ('eval_after_best_reply_cp','win_after_best_reply')):
        scored = valid[valid[cp_column].notna()]
        expected = scored[cp_column].map(win_chance).to_numpy(float)
        if not np.allclose(expected, scored[chance_column].to_numpy(float), atol=1e-9):
            raise ValueError(f'Perspective or mate conversion failed for {chance_column}.')
        terminal = valid[valid[cp_column].isna()]
        if not terminal[chance_column].isin([0.0,50.0,100.0]).all():
            raise ValueError(f'Board-terminal chance is invalid for {chance_column}.')
    if not (valid['actual_matches_best'].astype(bool).to_numpy() ==
            (valid['actual_reply_uci'] == valid['best_reply_uci']).to_numpy()).all():
        raise ValueError('Best-reply match flag disagrees with stored moves.')
    if not np.isfinite(valid[list(TARGETS)].to_numpy(float)).all():
        raise ValueError('Validated targets include nonfinite values.')
    if not np.allclose(valid['missed_opportunity_damage_pp'],
                       valid['win_after_actual_reply'] - valid['win_after_mistake_new'], atol=1e-9):
        raise ValueError('Missed-opportunity formula failed.')
    if not np.allclose(valid['opponent_reply_regret_pp'],
                       valid['win_after_actual_reply'] - valid['win_after_best_reply'], atol=1e-9):
        raise ValueError('Reply-regret formula failed.')
    if not np.allclose(valid['missed_opportunity_over_initial_damage'],
                       valid['missed_opportunity_damage_pp'] / valid['initial_damage'], atol=1e-9):
        raise ValueError('Normalized missed-opportunity formula failed.')
    if not np.allclose(valid['reply_regret_over_initial_damage'],
                       valid['opponent_reply_regret_pp'] / valid['initial_damage'], atol=1e-9):
        raise ValueError('Normalized reply-regret formula failed.')
    matching = valid[valid['actual_matches_best'] == True]
    if not np.allclose(matching['opponent_reply_regret_pp'], 0, atol=1e-9):
        raise ValueError('Matching actual/best replies must have zero regret.')
    return joined, manifest


def describe_values(series):
    values = pd.to_numeric(series, errors='coerce').dropna().to_numpy(float)
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                std=float(np.std(values, ddof=1)) if len(values)>1 else np.nan,
                p05=float(np.quantile(values,.05)), p25=float(np.quantile(values,.25)),
                p75=float(np.quantile(values,.75)), p95=float(np.quantile(values,.95)),
                min=float(np.min(values)), max=float(np.max(values)),
                fraction_negative=float(np.mean(values<0)),
                fraction_positive=float(np.mean(values>0)),
                fraction_zero=float(np.mean(np.isclose(values,0,atol=1e-9))))


def summaries(joined):
    valid = joined[joined['status']=='ok'].copy()
    valid['elo_band'] = pd.cut(pd.to_numeric(valid['player_elo'],errors='coerce'),
                               bins=[-np.inf,1600,2000,np.inf], right=False,
                               labels=['<1600','1600–<2000','>=2000']).astype('string').fillna('missing')
    valid['time_control'] = valid['time_control'].astype('string').fillna('missing')
    overall = pd.DataFrame([dict(target=name,**describe_values(valid[name])) for name in TARGETS])
    subgroups=[]
    for field in ('elo_band','time_control','game_phase'):
        for group,part in valid.groupby(field,dropna=False):
            for name in TARGETS[:2]:
                subgroups.append(dict(group_type=field,group=str(group),target=name,
                                      games=part['site'].nunique(),
                                      actual_matches_best_fraction=float(part['actual_matches_best'].mean()),
                                      **describe_values(part[name])))
    status=joined['status'].value_counts(dropna=False).rename_axis('status').reset_index(name='n')
    diagnostic=dict(total=len(joined),valid=len(valid),
                    actual_matches_best=int(valid['actual_matches_best'].sum()),
                    match_fraction=float(valid['actual_matches_best'].mean()),
                    negative_regret=int(valid['negative_regret'].sum()),
                    negative_regret_over_1pp=int(valid['negative_regret_over_1pp'].sum()),
                    actual_board_terminal=int(valid['actual_board_terminal'].sum()),
                    best_board_terminal=int(valid['best_board_terminal'].sum()),
                    root_mate_score_used=int(valid['root_mate_score_used'].sum()),
                    actual_mate_score_used=int(valid['actual_mate_score_used'].sum()),
                    best_mate_score_used=int(valid['best_mate_score_used'].sum()),
                    historical_baseline_median_absolute_gap_pp=float(
                        valid['new_minus_historical_win_after_pp'].abs().median()))
    return overall,pd.DataFrame(subgroups),status,diagnostic


def game_cluster_intervals(joined, repeats, seed):
    valid = joined[joined['status']=='ok']
    games = valid['site'].to_numpy()
    positions = pd.Series(np.arange(len(games))).groupby(games,sort=True).apply(
        lambda values: values.to_numpy()).tolist()
    arrays = {name: valid[name].to_numpy(float) for name in TARGETS}
    match = valid['actual_matches_best'].to_numpy(bool)
    rng = np.random.default_rng(seed)
    draws = {(name,metric): [] for name in TARGETS for metric in ('mean','median')}
    draws[('actual_matches_best','fraction')] = []
    for _ in range(repeats):
        sample = np.concatenate([positions[j] for j in rng.integers(0,len(positions),len(positions))])
        for name,values in arrays.items():
            draws[(name,'mean')].append(np.mean(values[sample]))
            draws[(name,'median')].append(np.median(values[sample]))
        draws[('actual_matches_best','fraction')].append(np.mean(match[sample]))
    rows=[]
    for (name,metric),values in draws.items():
        raw = match if name=='actual_matches_best' else arrays[name]
        point = np.median(raw) if metric=='median' else np.mean(raw)
        low,high = np.quantile(values,[.025,.975])
        rows.append(dict(target=name,metric=metric,estimate=point,
                         lower95=low,upper95=high,games=len(positions),
                         bootstrap_repeats=repeats))
    return pd.DataFrame(rows)


def make_plots(joined,subgroups,out):
    valid=joined[joined['status']=='ok']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,name,title in zip(axes,TARGETS[:2],
                             ('Actual reply minus post-mistake','Actual reply minus best reply')):
        values=valid[name].dropna().to_numpy(float)
        low,high=np.quantile(values,[.01,.99])
        ax.hist(values,bins=70,range=(low,high),color='#4676b4',alpha=.85)
        ax.axvline(0,color='black',linestyle='--')
        ax.set(xlabel='Percentage points (display clipped at 1st–99th percentiles)',
               ylabel='Mistakes',title=title)
    fig.savefig(out/'target_distributions.png',dpi=160)
    plt.close(fig)
    part=subgroups[(subgroups['group_type']=='elo_band') &
                   (subgroups['target'].isin(TARGETS[:2]))]
    fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
    for name,label in [(TARGETS[0],'Missed-opportunity damage'),(TARGETS[1],'Reply regret')]:
        item=part[part['target']==name]
        ax.plot(item['group'],item['median'],marker='o',label=label)
    ax.axhline(0,color='gray',linestyle='--')
    ax.set(xlabel='Original player Elo',ylabel='Median percentage points',
           title='Descriptive development subgroup medians')
    ax.legend()
    fig.savefig(out/'elo_subgroup_medians.png',dpi=160)
    plt.close(fig)


def run(args):
    if args.out.resolve().exists():
        raise FileExistsError(f'Refusing to overwrite: {args.out}')
    joined,source_manifest=load_validated(args.input.resolve(),args.targets.resolve())
    overall,groups,status,diagnostic=summaries(joined)
    intervals=game_cluster_intervals(joined,args.bootstrap_repeats,args.seed)
    valid=joined[joined['status']=='ok']
    board_terminal=(valid['actual_board_terminal'].astype(bool) |
                    valid['best_board_terminal'].astype(bool))
    engine_mate=(valid['root_mate_score_used'].astype(bool) |
                 valid['actual_mate_score_used'].astype(bool) |
                 valid['best_mate_score_used'].astype(bool))
    subsets={
        'all_valid': valid,
        'exclude_board_terminal': valid[~board_terminal],
        'exclude_board_terminal_and_engine_mate': valid[~(board_terminal | engine_mate)],
    }
    sensitivity=pd.DataFrame([
        dict(subset=label,target=name,**describe_values(part[name]))
        for label,part in subsets.items() for name in TARGETS[:2]])
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    metadata=dict(kind='descriptive immediate-reply target analysis; development only',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  source_data_sha256=source_manifest['data_sha256'],
                  targets_sha256=sha256(args.targets.resolve()),
                  engine_id=source_manifest['engine_id'],
                  engine_sha256=source_manifest['engine_sha256'],
                  source_archives=source_manifest['source_archives'],
                  smoke_only=source_manifest['smoke_only'],
                  seed=args.seed,bootstrap_repeats=args.bootstrap_repeats,
                  version=importlib.metadata.version('pandas'),
                  script_sha256=sha256(Path(__file__)),
                  note='Descriptive associations only. Signed values are not clipped. '
                       'Historical initial_damage denominators may differ from the new engine.')
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    overall.to_csv(out/'target_distribution_summary.csv',index=False)
    sensitivity.to_csv(out/'terminal_mate_sensitivity.csv',index=False)
    groups.to_csv(out/'elo_time_phase_subgroups.csv',index=False)
    intervals.to_csv(out/'game_cluster_distribution_intervals.csv',index=False)
    status.to_csv(out/'status_counts.csv',index=False)
    (out/'audit_counts.json').write_text(json.dumps(diagnostic,indent=2))
    make_plots(joined,groups,out)
    print(status.to_string(index=False))
    print(overall.to_string(index=False))
    print('Audit:',diagnostic)
    print('Saved:',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'development_with_multipv.csv')
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap-repeats',type=int,default=300)
    parser.add_argument('--seed',type=int,default=1701)
    args=parser.parse_args()
    if args.bootstrap_repeats<1:
        parser.error('Use at least one bootstrap repetition.')
    run(args)


if __name__=='__main__':
    main()
