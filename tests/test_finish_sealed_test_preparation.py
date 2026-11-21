"""Synthetic preparation monitor checks; no real input data."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import finish_sealed_test_preparation as f


class CompletionTests(unittest.TestCase):
    def test_status_reads_only_checkpoint_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            (out/'multipv_checkpoint.jsonl').write_text('{}\n{}\n')
            with sqlite3.connect(out/'selected_child_evaluations.sqlite') as db:
                db.execute('CREATE TABLE evaluations (fen TEXT)')
                db.execute('INSERT INTO evaluations VALUES (?)',('synthetic',))
            value=f.status(out,'feature_generation_running')
            self.assertEqual(value['multipv_rows_checkpointed'],2)
            self.assertEqual(value['candidate_positions_checkpointed'],1)
            self.assertFalse(value['performance_metrics_computed'])
            self.assertFalse(value['final_evaluation_enabled'])

    def test_completed_lineage_keeps_final_evaluation_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);freeze=root/'freeze';freeze.mkdir();out=root/'run';out.mkdir()
            for name in ['features.csv','labels.csv','test_with_frozen_features.csv']:
                (out/name).write_text('synthetic_header\n')
            f.write_json(out/'preparation_complete.json',dict(
                features_sha256=f.digest(out/'features.csv'),labels_sha256=f.digest(out/'labels.csv'),
                combined_sha256=f.digest(out/'test_with_frozen_features.csv'),predictions_run=False))
            f.write_json(out/'protocol.json',dict(preparation_only=True))
            f.write_json(out/'input_manifest.json',dict(input_protocol_sha256='old'))
            f.write_json(freeze/'final_input_protocol.json',dict(status='unresolved'))
            f.write_json(freeze/'code_manifest.json',dict(document_sha256={}))
            with patch.object(f,'ROOT',root):f.refresh_records(out)
            protocol=json.loads((freeze/'final_input_protocol.json').read_text())
            self.assertEqual(protocol['status'],'unresolved')
            self.assertIn('intermediate_lineage',protocol)
            contract=json.loads((out/'input_manifest.json').read_text())
            self.assertEqual(contract['input_protocol_sha256'],f.digest(freeze/'final_input_protocol.json'))
            status=json.loads((freeze/'test_enrichment_status.json').read_text())
            self.assertFalse(status['predictions_run'])


if __name__=='__main__':unittest.main()
