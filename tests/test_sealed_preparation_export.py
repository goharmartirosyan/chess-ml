"""End-to-end synthetic export, without weights, engine calls or test access."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scripts import enrich_sealed_test_features as e
from tests.test_enrich_sealed_test_features import feature_row


class ExportTests(unittest.TestCase):
    def test_feature_snapshot_then_labels_and_convenience_table(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'base';out=root/'out';base.mkdir();out.mkdir()
            e.write_json(out/'protocol.json',dict(scope='synthetic_export_only'))
            rows=[feature_row(),dict(feature_row(),row_id='synthetic_missing',site='synthetic2',game_index=2)]
            frame=e.build_pre_reply(pd.DataFrame(rows))
            frame.to_csv(out/'pre_reply_inputs.csv',index=False)
            mv=pd.DataFrame([dict(row_index=i,multipv_good_reply_count_10pp=1,
                multipv_best_second_gap_pp=np.nan,multipv_lines_returned=1) for i in range(2)])
            mv.to_csv(out/'multipv_features.csv',index=False)
            maia=pd.DataFrame([dict(row_index=i,row_id=rows[i]['row_id'],
                maia_probability_lower=.7,maia_probability_upper=.72,
                maia_preserving_reply_probability=.71) for i in range(2)])
            maia.to_csv(out/'maia_bounds_features.csv',index=False)
            for kind,name in [('multipv','multipv_features.csv'),('maia','maia_bounds_features.csv')]:
                e.write_json(out/f'{kind}_ready.json',dict(feature_sha256=e.core.digest(out/name)))
            e.export_features(out)
            feature_hash=e.core.digest(out/'features.csv')
            features=pd.read_csv(out/'features.csv')
            self.assertEqual(set(features),set(e.META+e.model_schema()['FULL']+[e.bounded.FEATURE]))
            self.assertFalse(set(features)&{'next_moves_san','win_after','actual_preserved'})
            self.assertTrue(features.multipv_best_second_gap_pp.isna().all())
            selected=pd.DataFrame(rows)
            selected['win_after']=30.
            selected['next_moves_san']=['e4 e5 Nf3','']
            selected.to_csv(base/'selected_test.csv',index=False)
            later=pd.DataFrame(dict(row_id=selected.row_id,win_after=30.))
            for h in (2,4,6,10):
                later[f'win_after_{h}ply']=40.;later[f'change_in_damage_{h}ply']=-10.
            later.to_csv(base/'labels_later_play.csv',index=False)
            def scores(folder,fens):
                self.assertEqual(e.core.digest(folder/'features.csv'),feature_hash)
                return {fen:dict(white_win=30.,status='engine') for fen in fens}
            with patch.object(e,'BASE',base),patch.object(e,'score_label_positions',side_effect=scores):
                e.prepare_labels(out)
            labels=pd.read_csv(out/'labels.csv')
            self.assertEqual(labels.loc[0,'actual_preserved'],1)
            self.assertTrue(pd.isna(labels.loc[1,'actual_preserved']))
            self.assertEqual(e.core.digest(out/'features.csv'),feature_hash)
            manifest=json.loads((out/'input_manifest.json').read_text())
            self.assertEqual(manifest['features']['sha256'],feature_hash)
            self.assertEqual(manifest['labels']['sha256'],e.core.digest(out/'labels.csv'))
            completion=json.loads((out/'preparation_complete.json').read_text())
            self.assertFalse(completion['predictions_run'])
            self.assertFalse(completion['performance_metrics_computed'])
            combined=pd.read_csv(out/'test_with_frozen_features.csv')
            self.assertEqual(len(combined),2)
            self.assertIn(e.bounded.FEATURE,combined)


if __name__=='__main__':unittest.main()
