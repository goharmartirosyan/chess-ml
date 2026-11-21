"""Synthetic contract/metric tests; no archived game or final-test file is read."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts import frozen_chess_evaluation as f
from prepare_chess_modeling import terminal_chance, prepare


def synthetic(n=240):
    rng=np.random.default_rng(8)
    x=pd.DataFrame({name:rng.normal(size=n) for name in f.FULL if name!='game_phase'})
    x['initial_damage']=rng.uniform(22,60,n)
    x['win_before']=rng.uniform(70,90,n)
    x['game_phase']=np.where(np.arange(n)%2,'opening','endgame')
    x['player_elo']=rng.integers(1000,2400,n);x['opponent_elo']=rng.integers(1000,2400,n)
    x['source_pgn']='lichess_db_standard_rated_2017-04.pgn.zst'
    x['game_index']=np.arange(n)//6+1;x['ply']=np.arange(n)%6+1
    x['site']=['synthetic-game-'+str(v) for v in x.game_index]
    x['event']='Rated Bullet game';x['time_control']='60+0'
    x['multipv_lines_returned']=5;x['multipv_good_reply_count_10pp']=2
    x.loc[0,'multipv_lines_returned']=1;x.loc[0,'multipv_best_second_gap_pp']=np.nan
    x.loc[::4,'player_clock_before_move']=np.nan
    x[f.PROB]=rng.uniform(0,1,n)
    x['row_id']=f.row_ids(x)
    y=pd.DataFrame({'row_id':x.row_id,'win_after':x.win_before-x.initial_damage})
    for h in [2,4,6,10]:
        future=np.clip(y.win_after+rng.normal(0,20,n),0,100)
        y[f'win_after_{h}ply']=future
        y[f'change_in_damage_{h}ply']=y.win_after-future
    y['actual_preserved']=(x[f.PROB]>.4).astype(float)
    y.loc[::17,'actual_preserved']=np.nan
    return x,y


class FrozenEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=f.load_config();cls.x,cls.y=synthetic()
        cls.models,cls.constants,cls.fitting=f.fit_models(cls.x,cls.y,cls.c)
        cls.pred=f.predict_features(cls.models,cls.constants,cls.x,cls.c)

    def test_approved_roster_and_predictors(self):
        self.assertEqual(set(self.models),{'primary_2','primary_4','primary_6','primary_10',
                         'ten_control','ten_maia','reply_control','reply_maia'})
        self.assertEqual(f.model_specs(self.c)['reply_maia']['features'],
                         ['initial_damage','opponent_elo',f.PROB])
        self.assertEqual(f.model_specs(self.c)['ten_maia']['features'],f.FULL+[f.PROB])
        self.assertEqual(len(self.pred),18*len(self.x))

    def test_future_columns_rejected(self):
        for name in ['actual_preserved','result','win_after_10ply','horizon_10ply_reached',
                     'actual_reply_uci','change_in_damage_2ply']:
            with self.subTest(name=name),self.assertRaises(ValueError):
                f.predict_features(self.models,self.constants,self.x.assign(**{name:1}),self.c)

    def test_training_baselines_and_missing_labels(self):
        for h in self.c['horizons']:
            self.assertEqual(self.constants['primary_mean'][str(h)],self.y[f'change_in_damage_{h}ply'].mean())
        self.assertEqual(self.fitting['reply_maia']['rows'],self.y.actual_preserved.notna().sum())
        self.assertEqual(self.constants['reply_prevalence'],self.y.actual_preserved.mean())

    def test_training_imputer_does_not_refit_on_prediction(self):
        model=self.models['reply_control'];before=model.named_steps['imputer'].statistics_.copy()
        held=self.x.copy();held['opponent_elo']=1000000;held['game_phase']='unseen_phase'
        f.predict_features(self.models,self.constants,held,self.c)
        np.testing.assert_array_equal(before,model.named_steps['imputer'].statistics_)
        valid=self.x.loc[self.y.actual_preserved.notna()]
        np.testing.assert_allclose(before,[valid.initial_damage.median(),valid.opponent_elo.median()])

    def test_auc_uses_continuous_score(self):
        y=np.array([1,0,1,0]);score=np.array([.4,.1,.3,.2])
        result=f.binary_metrics(y,score)
        self.assertEqual(result['roc_auc'],1.)
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(roc_auc_score(y,score>=.5),.5)

    def test_recovery_boundary_is_strict(self):
        result=f.regression_metrics([-25,-25],[-25,-25],[100,100],False)
        self.assertEqual(result['tp'],0);self.assertEqual(result['tn'],2)

    def test_single_class_and_no_positive_prediction(self):
        m=f.binary_metrics([0,0],[.1,.2]);self.assertTrue(np.isnan(m['roc_auc']))
        self.assertTrue(np.isnan(m['precision']));self.assertTrue(np.isnan(m['balanced_accuracy']))

    def test_perspective_and_post_end_targets(self):
        self.assertEqual(terminal_chance('1-0','White'),100)
        self.assertEqual(terminal_chance('1-0','Black'),0)
        self.assertEqual(terminal_chance('1/2-1/2','Black'),50)
        row=dict(win_before='75',win_after='30',side='Black',result='1-0',initial_damage='45')
        for h in [2,4,6,10]:
            row.update({f'horizon_{h}ply_reached':'False',f'game_ended_before_{h}ply':'True'})
        result,reasons=prepare(row,20,True)
        self.assertEqual(reasons,[])
        for h in [2,4,6,10]:self.assertEqual(float(result[f'change_in_damage_{h}ply']),30)

    def test_one_reply_gap_and_identity_checks(self):
        bad=self.x.copy();bad.loc[0,'multipv_best_second_gap_pp']=0
        with self.assertRaises(ValueError):f.check_features(bad)
        with self.assertRaises(ValueError):f.row_ids(pd.concat([self.x,self.x.iloc[:1]]))

    def test_training_rejects_non_development_archive(self):
        x=self.x.copy();x['source_pgn']='unapproved.pgn.zst';x['row_id']=f.row_ids(x)
        y=self.y.copy();y['row_id']=x.row_id
        with self.assertRaises(ValueError):f.fit_models(x,y,self.c)

    def test_label_identity_and_prediction_grid(self):
        with self.assertRaises(ValueError):
            f.score_predictions(self.x,self.y.iloc[:-1],self.pred,self.constants,self.c,repeats=5)
        with self.assertRaises(ValueError):
            f.score_predictions(self.x,self.y,self.pred.iloc[:-1],self.constants,self.c,repeats=5)
        wrong=self.y.copy();wrong.loc[0,'change_in_damage_10ply']*=-1
        with self.assertRaises(ValueError):
            f.score_predictions(self.x,wrong,self.pred,self.constants,self.c,repeats=5)

    def test_game_cluster_intervals_against_expanded_rows(self):
        ci=f.paired_intervals(self.x,self.pred,self.y,self.c,repeats=40)
        row=ci[(ci.task=='primary')&(ci.horizon==2)&(ci.threshold==20)&
               (ci.baseline=='no_change')&(ci.metric=='mae_improvement_pp')].iloc[0]
        p=self.pred[(self.pred.task=='primary')&(self.pred.horizon==2)&
                    (self.pred.model=='primary_2')].set_index('row_id')
        actual=self.y.set_index('row_id').loc[p.index,'change_in_damage_2ply']
        games=self.x.set_index('row_id').loc[p.index,'site']
        errors=np.abs(actual)-np.abs(actual-p.prediction)
        ordered=sorted(games.unique());rng=np.random.default_rng(self.c['seed']);draws=[]
        for _ in range(40):
            sample=rng.integers(0,len(ordered),len(ordered))
            expanded=np.concatenate([errors[games.eq(ordered[i])].to_numpy() for i in sample])
            draws.append(expanded.mean())
        np.testing.assert_allclose([row.lower95,row.upper95],np.quantile(draws,[.025,.975]),atol=1e-12)

    def test_final_access_guard_precedes_any_input_open(self):
        with patch.object(Path,'read_text',return_value=json.dumps({'status':'unresolved'})),\
             patch.object(f,'load_bundle',side_effect=AssertionError('Bundle accessed before guard')):
            with self.assertRaises(PermissionError):
                f.evaluate_prepared('unused','never-open-this-input.json','unused-out',False)
            with self.assertRaises(PermissionError):
                f.evaluate_prepared('unused','never-open-this-input.json','unused-out',True)

    def test_metrics_robustness_and_subgroup_accounting(self):
        metrics,ci=f.score_predictions(self.x,self.y,self.pred,self.constants,self.c,repeats=10)
        self.assertEqual(set(metrics.threshold),{20,25,30})
        self.assertEqual(set(ci.threshold),{20,25,30})
        subset=f.subgroup_metrics(self.x,self.y,self.pred,self.constants,self.c)
        overall=metrics[(metrics.task=='reply')&(metrics.model=='raw_maia')&(metrics.threshold==20)].iloc[0]
        for field in ['opponent_elo_band','time_control','speed_label','game_phase']:
            group=subset[(subset.task=='reply')&(subset.model=='raw_maia')&(subset.variable==field)]
            self.assertEqual(group.n.sum(),overall.n)
            for count in ['tn','fp','fn','tp']:self.assertEqual(group[count].sum(),overall[count])

    def test_prepared_pipeline_persists_predictions_before_labels_and_prevents_reruns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            features=self.x.copy();features['site']='held-'+features.site;features['row_id']=f.row_ids(features)
            labels=self.y.copy();labels['row_id']=features.row_id
            features.to_csv(root/'features.csv',index=False);labels.to_csv(root/'labels.csv',index=False)
            self.x[['row_id']+f.IDENTITY].to_csv(bundle/'training_membership.csv.gz',index=False)
            f.distributions(self.x,'development').to_csv(bundle/'development_distributions.csv',index=False)
            (bundle/'manifest.json').write_text('{}')
            contract=dict(input_protocol_sha256='synthetic-protocol',
                features=dict(path='features.csv',sha256=f.digest(root/'features.csv')),
                labels=dict(path='labels.csv',sha256=f.digest(root/'labels.csv')))
            (root/'input.json').write_text(json.dumps(contract))
            c=dict(self.c,bootstrap_repeats=5)
            manifest=dict(configuration=c,constants=self.constants)
            real_read=pd.read_csv;out=root/'output';label_reads=[]
            def read(path,*args,**kwargs):
                if Path(path).resolve()==(root/'labels.csv').resolve():
                    self.assertTrue((out/'predictions_before_labels.csv.gz').exists())
                    label_reads.append(str(path))
                return real_read(path,*args,**kwargs)
            with patch.object(f,'load_bundle',return_value=(self.models,manifest)),patch.object(f.pd,'read_csv',side_effect=read):
                f._evaluate_prepared_files(bundle,root/'input.json',out,'synthetic-protocol',True)
                with self.assertRaises(FileExistsError):
                    f._evaluate_prepared_files(bundle,root/'input.json',root/'second-output','synthetic-protocol',True)
            self.assertEqual(len(label_reads),1)
            self.assertTrue(json.loads((out/'completion.json').read_text())['development_validation'])
            self.assertFalse((bundle/'FINAL_EVALUATION_STARTED.json').exists())


if __name__=='__main__':unittest.main()
