"""Synthetic feature-preparation tests; no real test data or model weights."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine
import numpy as np
import pandas as pd
from scripts import enrich_sealed_test_features as e


def feature_row():
    row = {f:0. for f in e.model_schema()['FULL'] if not f.startswith('multipv_')}
    row.update(row_id='synthetic',source_pgn=e.ARCHIVE,game_index=1,ply=1,site='synthetic',
        fen_before=chess.STARTING_FEN,uci='e2e4',side='White',win_before=60.,initial_damage=30.,
        player_elo=1700.,opponent_elo=1800.,event='Blitz',time_control='synthetic',game_phase='opening')
    return row


class EnrichmentTests(unittest.TestCase):
    def test_reconstructs_only_pre_reply_position(self):
        frame=e.build_pre_reply(pd.DataFrame([feature_row()]))
        expected=chess.Board();expected.push_uci('e2e4')
        self.assertEqual(frame.iloc[0].root_fen,expected.fen())
        self.assertEqual(frame.iloc[0].active_elo,1800)
        self.assertEqual(frame.iloc[0].other_elo,1700)

    def test_target_column_rejected(self):
        row=dict(feature_row(),change_in_damage_10ply=99)
        with self.assertRaisesRegex(ValueError,'Unexpected column'):
            e.build_pre_reply(pd.DataFrame([row]))

    def test_side_and_move_checked(self):
        for change in [dict(side='Black'),dict(uci='e2e5')]:
            with self.assertRaises(ValueError):
                e.build_pre_reply(pd.DataFrame([dict(feature_row(),**change)]))

    def test_missing_ratings_and_routing_match_original(self):
        row=dict(feature_row(),player_elo=np.nan,opponent_elo=np.nan,event='UltraBullet')
        actual=e.build_pre_reply(pd.DataFrame([row])).iloc[0]
        self.assertEqual((actual.active_elo,actual.other_elo),(1500,1500))
        self.assertEqual(actual.policy_model,'blitz')
        self.assertTrue(actual.active_elo_missing and actual.other_elo_missing)

    def test_existing_multipv_one_legal_reply_keeps_missing_gap(self):
        board=chess.Board('k7/8/2K5/8/1Q6/8/8/8 w - - 0 1')
        after=board.copy();after.push_uci('b4b5')
        self.assertEqual(after.legal_moves.count(),1)
        move=next(iter(after.legal_moves))
        class FakeEngine:
            def analyse(self, b, limit, multipv, info):
                self_args=(b.fen(),limit.depth,multipv)
                if self_args!=(after.fen(),18,1):raise AssertionError(self_args)
                return [dict(pv=[move],score=chess.engine.PovScore(chess.engine.Cp(50),b.turn))]
        with patch.object(e.multipv,'_worker_engine',FakeEngine()),patch.object(e.multipv,'_worker_depth',18):
            result=e.multipv.analyze_one((0,board.fen(),'b4b5'))
        self.assertEqual(result['multipv_lines_returned'],1)
        self.assertIsNone(result['multipv_best_second_gap_pp'])
        self.assertEqual(result['multipv_good_reply_count_10pp'],1)

    def test_bounds_keep_residual_and_original_side(self):
        plan=pd.DataFrame([dict(row_index=0,child_fen='a',policy_probability=.7),
                           dict(row_index=0,child_fen='b',policy_probability=.28)])
        coverage=pd.DataFrame([dict(row_index=0,unscored_probability=.02)])
        for side,white_a,white_b in [('White',30.,50.),('Black',70.,50.)]:
            sample=pd.DataFrame([dict(row_index=0,row_id='synthetic',side=side,win_before=60.,initial_damage=30.)])
            records={k:dict(status='engine',white_win=v) for k,v in [('a',white_a),('b',white_b)]}
            result=e.aggregate_bounds(sample,plan,coverage,records).iloc[0]
            self.assertAlmostEqual(result.maia_probability_lower,.7)
            self.assertAlmostEqual(result.maia_preserving_reply_probability,.71)

    def test_original_prefix_selection_uses_policy_only(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            frame=e.build_pre_reply(pd.DataFrame([feature_row()]))
            frame.to_csv(out/'pre_reply_inputs.csv',index=False)
            board=chess.Board(frame.iloc[0].root_fen)
            moves=sorted(m.uci() for m in board.legal_moves)
            policy={m:0. for m in moves};policy.update({moves[0]:.7,moves[1]:.28,moves[2]:.02})
            plan,coverage=e.bounded.make_plan(out,dict(coverage=.975),{0:dict(move_probabilities=policy)})
            self.assertEqual(plan.reply_uci.tolist(),moves[:2])
            self.assertAlmostEqual(coverage.iloc[0].unscored_probability,.02)

    def test_labels_cannot_open_before_feature_snapshot(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(e.pd,'read_csv') as read:
            with self.assertRaises(FileNotFoundError):e.prepare_labels(Path(temp))
            read.assert_not_called()

    def test_policy_resume_preserves_original_batch_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            pd.DataFrame([dict(row_index=i,policy_model='blitz',root_fen=chess.STARTING_FEN,
                active_elo=1700,other_elo=1700) for i in range(65)]).to_csv(out/'pre_reply_inputs.csv',index=False)
            sizes=[]
            def infer(network,frame):
                sizes.append(len(frame));return [{'e2e4':1.} for _ in range(len(frame))]
            with patch.object(e.core,'load_policy',return_value=object()),patch.object(e.core,'full_precision_policy',side_effect=infer):
                first=e.infer_policy_batches(out)
                second=e.infer_policy_batches(out)
            self.assertEqual(sizes,[32,32,1])
            self.assertEqual(first,second)
            self.assertEqual(len(first),65)


if __name__=='__main__':unittest.main()
