"""Temporal one-to-one event matching and transparent confusion counts."""
import numpy as np


def interval_iou(a,b):
    overlap=max(0.,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
    union=max(a['end_s'],b['end_s'])-min(a['start_s'],b['start_s'])
    return overlap/union if union else 0.


def match_events(predicted, truth, threshold=.3):
    """Order-preserving DP: maximize match count, then total IoU.

    Squat events are chronological and non-overlapping. Unlike greedy matching,
    this does not sacrifice a second valid match to one slightly higher IoU.
    """
    n,m=len(predicted),len(truth)
    dp=[[(0,0.) for _ in range(m+1)] for _ in range(n+1)]
    action={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            choices=[(dp[i-1][j],'p'),(dp[i][j-1],'t')]
            iou=interval_iou(predicted[i-1],truth[j-1])
            if iou>=threshold:
                prev=dp[i-1][j-1]; choices.append(((prev[0]+1,prev[1]+iou),'match'))
            dp[i][j],action[i,j]=max(choices,key=lambda x:x[0])
    pairs=[];i,j=n,m
    while i and j:
        step=action[i,j]
        if step=='match':pairs.append((i-1,j-1));i-=1;j-=1
        elif step=='p':i-=1
        else:j-=1
    return pairs[::-1]


def event_metrics(predicted,truth,threshold=.3):
    pairs=match_events(predicted,truth,threshold)
    confusion={f'{a}_to_{b}':0 for a in ('acceptable','shallow') for b in ('acceptable','shallow','unclear')}
    for i,j in pairs:
        if truth[j]['label']!='unclear':
            confusion[f'{truth[j]["label"]}_to_{predicted[i]["label"]}']+=1
    # End-to-end shallow detection: missing shallow reps are false negatives.
    # Predictions matched to uncertain GT are excluded from form scoring.
    matched={i:j for i,j in pairs}
    shallow_tp=sum(predicted[i]['label']=='shallow' and truth[j]['label']=='shallow' for i,j in pairs)
    shallow_fp=sum(p['label']=='shallow' and (i not in matched or truth[matched[i]]['label']=='acceptable') for i,p in enumerate(predicted))
    shallow_fn=sum(t['label']=='shallow' for t in truth)-shallow_tp
    return dict(event_tp=len(pairs),event_fp=len(predicted)-len(pairs),event_fn=len(truth)-len(pairs),
                shallow_tp=shallow_tp,shallow_fp=shallow_fp,shallow_fn=shallow_fn,
                matched_known=sum(confusion.values()),
                matched_decided=sum(v for k,v in confusion.items() if not k.endswith('_unclear')),
                confusion=confusion)


def prf(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def summarize(rows):
    if not rows:return {}
    errors=[r['attempt_error'] for r in rows]
    depth=[r['depth_error'] for r in rows if r['depth_error'] is not None]
    sums={key:sum(r[key] for r in rows) for key in ('event_tp','event_fp','event_fn','shallow_tp','shallow_fp','shallow_fn','matched_known','matched_decided')}
    negatives=[r for r in rows if r['true_attempts']==0]
    return dict(clips=len(rows),attempt_mae=float(np.mean(errors)),exact_count_fraction=float(np.mean(np.array(errors)==0)),
                depth_labeled_clips=len(depth),depth_mae=float(np.mean(depth)) if depth else None,
                event_detection=prf(sums['event_tp'],sums['event_fp'],sums['event_fn']),
                shallow_detection=prf(sums['shallow_tp'],sums['shallow_fp'],sums['shallow_fn']),
                form_decision_coverage=sums['matched_decided']/sums['matched_known'] if sums['matched_known'] else None,
                negative_clips=len(negatives),false_reps_on_negatives=sum(r['predicted_attempts'] for r in negatives),
                counts=sums)
