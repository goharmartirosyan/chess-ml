"""Synthetic preparation-audit checks, without research-model inference."""
import hashlib
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import subprocess

import numpy as np
import pandas as pd
import chess
from scripts import audit_sealed_test_readiness as a


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((a.ROOT/'freeze/approved_models.json').read_text())
        row={k:0. for k in self.config['features']['full']}
        row.update(source_pgn=a.ARCHIVE,game_index=1,ply=1,site='synthetic',
            win_before=60.,initial_damage=30.,event='synthetic',time_control='synthetic',
            game_phase='synthetic',multipv_lines_returned=1,multipv_good_reply_count_10pp=1,
            multipv_best_second_gap_pp=np.nan,maia_preserving_reply_probability=.71)
        row['row_id']=hashlib.sha256(json.dumps([a.ARCHIVE,1,1,'synthetic'],separators=(',',':')).encode()).hexdigest()
        self.x=pd.DataFrame([row])
        label=dict(row_id=row['row_id'],win_after=30.,actual_preserved=1.)
        for h in self.config['horizons']:
            label[f'win_after_{h}ply']=40.;label[f'change_in_damage_{h}ply']=-10.
        self.y=pd.DataFrame([label])
        self.bounds=pd.DataFrame([dict(row_id=row['row_id'],maia_probability_lower=.7,maia_probability_upper=.72)])
        self.status=pd.DataFrame([dict(row_id=row['row_id'],status='ok')])

    def check(self):
        return a.validate_tables(self.x,self.y,self.bounds,self.status,self.config)

    def test_complete_synthetic_tables(self):
        self.assertTrue(all(self.check().values()))

    def test_target_leakage_rejected(self):
        self.x['actual_reply_uci']='e7e5'
        with self.assertRaisesRegex(ValueError,'predictor columns'):self.check()

    def test_future_target_sign_checked(self):
        self.y['change_in_damage_10ply']=10.
        with self.assertRaisesRegex(ValueError,'target equation'):self.check()

    def test_missing_reply_stays_missing(self):
        self.status['status']='missing_actual_reply';self.y['actual_preserved']=np.nan
        self.assertTrue(self.check()['missing_reply_handling'])
        self.y['actual_preserved']=0.
        with self.assertRaisesRegex(ValueError,'Missing reply labels'):self.check()

    def test_unknown_multipv_gap_is_not_accepted(self):
        self.x['multipv_lines_returned']=2
        with self.assertRaisesRegex(ValueError,'gap semantics'):self.check()

    def test_bound_width_and_midpoint_checked(self):
        self.x['maia_preserving_reply_probability']=.7
        with self.assertRaisesRegex(ValueError,'midpoint'):self.check()

    def test_one_reply_gap_verified_against_the_board(self):
        board=chess.Board('k7/8/2K5/8/1Q6/8/8/8 w - - 0 1');board.push_uci('b4b5')
        roots=pd.DataFrame(dict(row_id=self.x.row_id,root_fen=board.fen()))
        self.assertTrue(a.validate_reply_metadata(self.x,roots))

    def test_truncated_multipv_is_not_treated_as_one_legal_reply(self):
        roots=pd.DataFrame(dict(row_id=self.x.row_id,root_fen=chess.STARTING_FEN))
        with self.assertRaisesRegex(ValueError,'line coverage'):
            a.validate_reply_metadata(self.x,roots)

    def test_committed_reference_requires_matching_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'source.py').write_bytes(b'synthetic')
            with patch.object(a,'ROOT',root),patch.object(a.subprocess,'check_output',side_effect=[b'ref\n',b'synthetic']):
                self.assertEqual(a.committed_reference(['source.py']),'ref')
            with patch.object(a,'ROOT',root),patch.object(a.subprocess,'check_output',side_effect=[b'ref\n',b'old']):
                self.assertIsNone(a.committed_reference(['source.py']))

    def test_uncommitted_source_is_reported_without_git_mutation(self):
        with patch.object(a.subprocess,'check_output',side_effect=subprocess.CalledProcessError(128,['git','show'])) as git:
            self.assertIsNone(a.committed_reference(['missing.py']))
        self.assertEqual(git.call_args.args[0],['git','rev-parse','HEAD'])


if __name__=='__main__':unittest.main()
