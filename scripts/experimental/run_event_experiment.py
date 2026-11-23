"""Cache poses, fit reference-only anchors, compare four fixed methods.

python scripts/experimental/run_event_experiment.py --manifest data/annotations/clips.csv \
    --repetitions data/annotations/repetitions.csv --split dev --out data/results/dev
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.evaluation.data import load_manifest,load_repetitions,sha256
from src.pose.cache import extract_cached
from src.squat.tracking import VARIANTS,fit_reference,run_method
from src.evaluation.metrics import event_metrics,summarize

ROOT=Path(__file__).resolve().parents[2]


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_results(out,summary,examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(summary)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar(names,[summary[n]['attempt_mae'] for n in names])
    ax.set(ylabel='Mean absolute counting error per clip',title='Same clips, four fixed methods')
    ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(out/'counting_comparison.png',dpi=160);plt.close(fig)
    for index,(cid,cache,histories,truth) in enumerate(examples):
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        for variant,history in histories.items():
            axis=axes[0] if variant in VARIANTS[:2] else axes[1]
            axis.plot(cache['times'],[h['signal'] if h['signal'] is not None else np.nan for h in history],label=variant)
        for ax in axes:
            for e in truth:ax.axvspan(e['start_s'],e['end_s'],alpha=.12,color='green' if e['label']=='acceptable' else 'orange')
            ax.legend();ax.grid(alpha=.2)
        axes[0].set_ylabel('2D knee angle (degrees)');axes[1].set_ylabel('Reference-relative depth');axes[1].set_xlabel('Seconds')
        fig.suptitle(cid+' — shaded intervals are supplied annotations');fig.tight_layout();fig.savefig(out/f'trace_{index:03d}.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--repetitions',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/experiment.json')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--cache',type=Path,default=ROOT/'data/cache/poses')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path)
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--allow-participant-overlap',action='store_true',help='Allow a within-person study; prominently record this limitation')
    args=p.parse_args()
    clips=load_manifest(args.manifest);annotations=load_repetitions(args.repetitions,clips)
    config=json.loads(args.config.read_text())
    if config['smooth_tau_s']<=0 or not 0<=config['stand_depth']<config['start_depth']<config['acceptable_depth']:
        raise ValueError('Invalid smoothing/depth configuration')
    if not 0<config['match_iou']<=1 or not 0<=config['min_coverage']<=1 or not 0<config['max_gap_s']:
        raise ValueError('Invalid matching/coverage/gap configuration')
    if not 0<config['min_duration_s']<config['max_duration_s'] or config['bottom_hold_s']<0 or config['direction_epsilon']<0:
        raise ValueError('Invalid temporal configuration')
    reference=[c for c in clips if c['split']=='reference']
    selected=[c for c in clips if c['split']==args.split]
    if not reference or not selected:raise ValueError('Need reference clips and clips in selected split')
    if any(c['exercise'].lower() not in {'squat','squats'} for c in reference):
        raise ValueError('Reference clips must be squats')
    # Unknown person IDs cannot establish a participant-disjoint study.
    train_people={c['participant'].strip() for c in clips if c['split']!=args.split}
    test_people={c['participant'].strip() for c in selected}
    overlap=sorted(train_people & test_people)
    unknown=any(c['participant'].strip().lower() in {'','unknown'} for c in reference+selected)
    if (overlap or unknown) and not args.allow_participant_overlap:
        raise ValueError('Participant split is overlapping/unknown. Fix it or explicitly use --allow-participant-overlap for a within-person pilot.')
    args.out.mkdir(parents=True,exist_ok=True)
    caches={};cache_paths={}
    for c in reference+selected:
        print('Extract/cache:',c['clip_id'],flush=True)
        cache,path=extract_cached(c['path'],args.cache,args.model)
        if annotations[c['clip_id']] and max(e['end_s'] for e in annotations[c['clip_id']]) > cache['times'][-1]+1/cache['metadata']['fps']:
            raise ValueError(f'{c["clip_id"]}: annotation extends beyond video')
        caches[c['clip_id']]=cache;cache_paths[c['clip_id']]=str(path)
    anchors=fit_reference({c['clip_id']:caches[c['clip_id']] for c in reference},annotations)
    anchors['sources']=[dict(clip_id=c['clip_id'],video_sha256=c['video_sha256']) for c in reference]
    (args.out/'reference.json').write_text(json.dumps(anchors,indent=2)+'\n')
    rows=[];examples=[]
    for clip in selected:
        cid=clip['clip_id'];cache=caches[cid];histories={}
        for variant in VARIANTS:
            events,history,runtime=run_method(cache,variant,config,anchors)
            metrics=event_metrics(events,annotations[cid],config['match_iou'])
            depth=sum(e['label']=='acceptable' for e in events)
            truth_depth=clip['acceptable_depth_reps']
            row=dict(clip_id=cid,variant=variant,participant=clip['participant'],view=clip['view'],
                     true_attempts=clip['attempts'],predicted_attempts=len(events),
                     attempt_error=abs(len(events)-clip['attempts']),true_depth=truth_depth,predicted_depth=depth,
                     depth_error=abs(depth-truth_depth) if truth_depth is not None else None,
                     uncertain_predictions=sum(e['label']=='unclear' for e in events),
                     usable_2d_fraction=float(np.isfinite(cache['angles']).mean()),
                     usable_depth_fraction=float(np.isfinite(cache['features'][:,[5,0,1]]).all(axis=1).mean()),
                     algorithm_median_ms=float(np.median(runtime)),algorithm_p95_ms=float(np.percentile(runtime,95)),
                     extraction_median_ms=float(np.median(cache['extraction_ms'])),
                     extraction_p95_ms=float(np.percentile(cache['extraction_ms'],95)),**metrics)
            rows.append(row);histories[variant]=history
            stem=f'clip_{selected.index(clip):03d}_{variant}'
            write_csv(args.out/(stem+'_frames.csv'),history)
            (args.out/(stem+'_events.json')).write_text(json.dumps(events,indent=2)+'\n')
        examples.append((cid,cache,histories,annotations[cid]))
    summary={v:summarize([r for r in rows if r['variant']==v]) for v in VARIANTS}
    groups={field:{value:{v:summarize([r for r in rows if r[field]==value and r['variant']==v]) for v in VARIANTS}
                   for value in sorted({r[field] for r in rows})} for field in ('view','participant')}
    code_hashes={str(path.relative_to(ROOT)):sha256(path) for path in sorted((ROOT/'src').rglob('*.py'))}
    code_hashes['scripts/experimental/run_event_experiment.py']=sha256(__file__)
    report=dict(split=args.split,config=config,summary=summary,by_group=groups,clips=rows,
                reference=anchors,cache_paths=cache_paths,participant_overlap=overlap,
                participant_disjoint=not overlap and not unknown,
                manifest_sha256=sha256(args.manifest),labels_sha256=sha256(args.repetitions),code_sha256=code_hashes,
                timing='Extraction and algorithm timed separately; excludes video decode/rendering. Not end-to-end webcam latency.',
                interpretation='Prospective controlled configurations; improvements must be demonstrated, not assumed.')
    (args.out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write_csv(args.out/'clips.csv',[{**r,'confusion':json.dumps(r['confusion'])} for r in rows])
    failures=sorted(rows,key=lambda r:(r['attempt_error'],r['shallow_fn']+r['shallow_fp']),reverse=True)
    write_csv(args.out/'failure_review.csv',[{k:r[k] for k in ('clip_id','variant','attempt_error','event_fp','event_fn','shallow_fp','shallow_fn','usable_depth_fraction')} for r in failures])
    table=['# Experiment results', '', f'Split: {args.split}. Participant-disjoint: {report["participant_disjoint"]}.',
           '', '| Method | Count MAE | Exact count | Depth MAE | Rep F1 | Shallow F1 |', '|---|---:|---:|---:|---:|---:|']
    def fmt(value):return 'N/A' if value is None else f'{value:.3f}'
    for v,s in summary.items():
        table.append('| '+' | '.join([v,fmt(s['attempt_mae']),fmt(s['exact_count_fraction']),fmt(s['depth_mae']),fmt(s['event_detection']['f1']),fmt(s['shallow_detection']['f1'])])+' |')
    table += ['', report['timing'], '', 'Inspect results.json for per-view/participant metrics, uncertainty coverage and false counts on negatives.',
              'These numbers are evidence for this dataset only. Do not tune settings on test results.']
    (args.out/'report.md').write_text('\n'.join(table)+'\n')
    if not args.no_plots:plot_results(args.out,summary,examples)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
