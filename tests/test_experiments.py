import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.squat.tracking import AttemptTracker, run_method, fit_reference, VARIANTS
from src.evaluation.metrics import event_metrics, match_events, summarize
from src.evaluation.data import load_manifest, load_repetitions
from src.pose.cache import cache_signature, cache_key

CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/experiment.json').read_text())


def fixture(fps=30, depths=(1.,.5)):
    t=np.arange(0,8+1/fps/2,1/fps)
    d=np.zeros_like(t)
    truth=[]
    for i,depth in enumerate(depths):
        start=1+3*i; end=start+2
        mask=(t>=start)&(t<=end)
        d[mask]=depth*(1-np.cos(2*np.pi*(t[mask]-start)/2))/2
        truth.append(dict(start_s=start,bottom_s=start+1,end_s=end,label='acceptable' if depth>=.8 else 'shallow'))
    features=np.column_stack([175-85*d,175-85*d,165-80*d,165-80*d,5+20*d,1.8-.7*d,np.ones_like(d)])
    return dict(times=t,features=features,angles=175-85*d,extraction_ms=np.zeros(len(t)),metadata={'fps':fps}),truth


class ExperimentTests(unittest.TestCase):
    def test_all_variants_emit_attempts_and_depth_labels(self):
        cache,truth=fixture()
        reference=dict(standing=[1.8,175,175],span=[.7,85,85])
        for variant in VARIANTS:
            events,history,runtime=run_method(cache,variant,CONFIG,reference)
            self.assertEqual([e['label'] for e in events],['acceptable','shallow'],variant)
            self.assertEqual(len(history),len(cache['times']))
            self.assertEqual(event_metrics(events,truth)['event_tp'],2)

    def test_temporal_behavior_across_frame_rates(self):
        for fps in (15,24,30,60):
            cache,_=fixture(fps)
            events,_,_=run_method(cache,'temporal_3d',CONFIG,dict(standing=[1.8,175,175],span=[.7,85,85]))
            self.assertEqual([e['label'] for e in events],['acceptable','shallow'],fps)

    def test_missing_gap_cancels_attempt(self):
        cache,_=fixture()
        cache['features'][(cache['times']>1.7)&(cache['times']<2.4)]=np.nan
        events,_,_=run_method(cache,'temporal_3d',CONFIG,dict(standing=[1.8,175,175],span=[.7,85,85]))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['label'],'shallow')

    def test_static_and_incomplete_motion_do_not_count(self):
        tracker=AttemptTracker(CONFIG,True)
        for i,x in enumerate([0.]*30+[.01]*30+[.5]*30):tracker.update(i/30,x)
        self.assertEqual(tracker.events,[])

    def test_reference_uses_only_acceptable_cycles(self):
        cache,truth=fixture()
        ref=fit_reference({'ref':cache},{'ref':truth})
        self.assertEqual(ref['cycles'],1)
        self.assertGreater(ref['span'][0],.6)

    def test_one_prediction_cannot_match_two_truth_events(self):
        a=dict(start_s=0,end_s=2,label='shallow')
        b=dict(start_s=2,end_s=4,label='shallow')
        pred=dict(start_s=0,end_s=4,label='shallow')
        metrics=event_metrics([pred],[a,b])
        self.assertEqual((metrics['event_tp'],metrics['event_fn'],metrics['shallow_fn']),(1,1,1))

    def test_unclear_ground_truth_excluded_from_form_false_positives(self):
        metrics=event_metrics([dict(start_s=0,end_s=2,label='shallow')],[dict(start_s=0,end_s=2,label='unclear')])
        self.assertEqual(metrics['shallow_fp'],0)
        self.assertEqual(metrics['matched_known'],0)

    def test_empty_predictions_retain_false_negatives(self):
        metrics=event_metrics([],[dict(start_s=0,end_s=2,label='shallow')])
        self.assertEqual(metrics['event_fn'],1)
        self.assertEqual(metrics['shallow_fn'],1)

    def test_manifest_rejects_source_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'a.mp4').write_bytes(b'a');(root/'b.mp4').write_bytes(b'b')
            path=root/'clips.csv'
            path.write_text('clip_id,path,split,source_group,attempts,acceptable_depth_reps\na,a.mp4,reference,same,1,1\nb,b.mp4,test,same,1,1\n')
            with self.assertRaisesRegex(ValueError,'crosses splits'):load_manifest(path)

    def test_labels_reject_inconsistent_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reps.csv'
            path.write_text('clip_id,start_s,bottom_s,end_s,depth_label\nx,0,1,2,shallow\n')
            with self.assertRaisesRegex(ValueError,'depth count disagrees'):
                load_repetitions(path,[dict(clip_id='x',attempts=1,acceptable_depth_reps=1)])

    def test_cache_changes_when_video_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'video';model=Path(tmp)/'model'
            video.write_bytes(b'a');model.write_bytes(b'm')
            first=cache_key(cache_signature(video,model,{}))
            video.write_bytes(b'b')
            self.assertNotEqual(first,cache_key(cache_signature(video,model,{})))

    def test_runner_writes_complete_report_from_cached_fixture(self):
        from unittest.mock import patch
        import importlib.util
        spec=importlib.util.spec_from_file_location('runner',Path(__file__).resolve().parents[1]/'scripts/experimental/run_event_experiment.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('reference','evaluation'):(root/(name+'.mp4')).write_bytes(name.encode())
            manifest=root/'clips.csv'
            manifest.write_text('clip_id,path,participant,view,split,exercise,attempts,acceptable_depth_reps,source_group\nreference,reference.mp4,p1,side,reference,squat,2,1,g1\nevaluation,evaluation.mp4,p2,side,dev,squat,2,1,g2\n')
            cache,truth=fixture()
            labels=root/'reps.csv'
            with labels.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['clip_id','start_s','bottom_s','end_s','depth_label']);writer.writeheader()
                for cid in ('reference','evaluation'):
                    for e in truth:writer.writerow(dict(clip_id=cid,start_s=e['start_s'],bottom_s=e['bottom_s'],end_s=e['end_s'],depth_label=e['label']))
            out=root/'out'
            argv=['runner','--manifest',str(manifest),'--repetitions',str(labels),'--out',str(out),'--no-plots']
            with patch('sys.argv',argv),patch.object(runner,'extract_cached',return_value=(cache,root/'fixture.npz')),patch('builtins.print'):
                runner.main()
            report=json.loads((out/'results.json').read_text())
            self.assertTrue(report['participant_disjoint'])
            self.assertEqual(len(report['summary']),4)
            self.assertEqual(report['summary']['temporal_3d']['attempt_mae'],0)
            self.assertTrue((out/'report.md').exists())
            self.assertEqual(len(list(out.glob('*_events.json'))),4)
