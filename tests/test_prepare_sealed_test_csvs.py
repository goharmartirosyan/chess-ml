"""Synthetic preparation checks. No real test CSV is opened by these tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from scripts import prepare_sealed_test_csvs as p


def example():
    row = {n: '0' for n in p.model_schema()['FULL'] if not n.startswith('multipv_')}
    row.update(source_pgn=p.ARCHIVE, game_index='1', ply='12', site='https://example.test/synthetic',
        side='White', result='1-0', fen_before='synthetic_fen_not_used_for_analysis', uci='e2e4',
        win_before='60', win_after='30', initial_damage='30', event='synthetic', time_control='synthetic',
        game_phase='synthetic')
    for h in p.HORIZONS:
        row.update({f'horizon_{h}ply_reached': 'True', f'game_ended_before_{h}ply': 'False',
            f'win_after_{h}ply': '40', f'eval_after_{h}ply_cp': '0',
            f'terminal_result_used_for_{h}ply': 'False', f'mate_score_used_after_{h}ply': 'False',
            f'remaining_damage_{h}ply': '20', f'persistence_ratio_{h}ply': '0.666',
            f'persistence_ratio_{h}ply_clipped': '0.666', f'persistence_category_{h}ply': 'synthetic'})
    return row


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def source(self, name, rows):
        path = self.root/name
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        return path

    def run_stage(self, paths):
        return p.stage(paths, self.root/'out', [p.digest(x) for x in paths], 'synthetic_protocol_hash')

    def test_merge_exact_duplicates_and_feature_label_separation(self):
        row = example()
        a = self.source('synthetic_a.csv', [row])
        b = self.source('synthetic_b.csv', [row])
        before = [p.digest(x) for x in [a, b]]
        result = self.run_stage([a, b])
        self.assertEqual(result['counts']['exact_duplicates_removed'], 1)
        self.assertEqual(result['counts']['selected_rows'], 1)
        self.assertFalse(result['ready_for_frozen_evaluator'])
        self.assertIn('maia_preserving_reply_probability', result['missing_model_predictors'])
        with (self.root/'out/pre_reply_inputs.csv').open() as f:
            features = next(csv.DictReader(f))
        self.assertFalse(set(features) & {'result', 'win_after', 'change_in_damage_10ply'})
        with (self.root/'out/labels_later_play.csv').open() as f:
            labels = next(csv.DictReader(f))
        self.assertEqual(float(labels['change_in_damage_10ply']), -10)
        self.assertEqual(labels['row_id'], features['row_id'])
        self.assertNotIn('actual_preserved', labels)
        self.assertEqual(before, [p.digest(x) for x in [a, b]])

    def test_conflicting_duplicate_is_not_chosen(self):
        row = example()
        other = dict(row, win_after='29')
        path = self.source('synthetic.csv', [row, other])
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            self.run_stage([path])
        self.assertTrue((self.root/'out/PREPARATION_FAILED.json').exists())
        self.assertFalse((self.root/'out/preparation_manifest.json').exists())

    def test_mismatched_schemas_fail_before_outputs(self):
        a = self.source('synthetic_a.csv', [example()])
        b = self.source('synthetic_b.csv', [dict(example(), extra='unapproved')])
        with self.assertRaisesRegex(ValueError, 'schemas differ'):
            self.run_stage([a, b])
        self.assertFalse((self.root/'out').exists())

    def test_non_test_archive_rejected(self):
        path = self.source('synthetic.csv', [dict(example(), source_pgn='unexpected_archive')])
        with self.assertRaisesRegex(ValueError, 'non-March'):
            self.run_stage([path])

    def test_partial_source_name_keeps_raw_lineage(self):
        row = dict(example(), source_pgn='/synthetic/lichess_2025_03_partial.pgn.zst')
        self.run_stage([self.source('synthetic.csv', [row])])
        with (self.root/'out/combined_test.csv').open() as f:
            raw = next(csv.DictReader(f))
        with (self.root/'out/pre_reply_inputs.csv').open() as f:
            feature = next(csv.DictReader(f))
        self.assertEqual(raw['source_pgn'], row['source_pgn'])
        self.assertEqual(feature['source_pgn'], p.ARCHIVE)

    def test_fixed_strict_selection(self):
        path = self.source('synthetic.csv', [dict(example(), win_before='50')])
        result = self.run_stage([path])
        self.assertEqual(result['counts']['selected_rows'], 0)
        self.assertEqual(result['counts']['excluded_rows'], 1)

    def test_terminal_targets_keep_original_side(self):
        row = dict(example(), side='Black', result='1-0')
        for h in p.HORIZONS:
            row.update({f'horizon_{h}ply_reached': 'False', f'game_ended_before_{h}ply': 'True'})
        result = self.run_stage([self.source('synthetic.csv', [row])])
        with (self.root/'out/labels_later_play.csv').open() as f:
            labels = next(csv.DictReader(f))
        self.assertEqual(float(labels['win_after_10ply']), 0)
        self.assertEqual(float(labels['change_in_damage_10ply']), 30)

    def test_ambiguous_game_identity_rejected(self):
        row = example()
        path = self.source('synthetic.csv', [row, dict(row, site='https://example.test/other')])
        with self.assertRaisesRegex(ValueError, 'conflicting game identities'):
            self.run_stage([path])

    def test_input_hash_must_match(self):
        path = self.source('synthetic.csv', [example()])
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            p.stage([path], self.root/'out', ['incorrect_hash'], 'synthetic')

    def test_source_paths_come_from_the_local_manifest(self):
        manifest=dict(inputs=[dict(path=f'synthetic_part_{i}.csv') for i in range(5)])
        self.assertEqual([x.name for x in p.protocol_input_paths(manifest)],
                         [f'synthetic_part_{i}.csv' for i in range(5)])

    def test_source_manifest_rejects_duplicate_and_external_paths(self):
        for names in [['same.csv']*5,['../outside.csv','b.csv','c.csv','d.csv','e.csv']]:
            with self.assertRaises(ValueError):
                p.protocol_input_paths(dict(inputs=[dict(path=name) for name in names]))


if __name__ == '__main__':
    unittest.main()
