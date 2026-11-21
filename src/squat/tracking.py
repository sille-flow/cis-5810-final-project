"""Four fixed, comparable methods; all emit completed-attempt events.

Iterations are prospective experiments, not claims of improvement.
"""
from dataclasses import dataclass, asdict
import numpy as np
from src.squat.baseline import SquatTracker

VARIANTS = ('baseline', 'confidence_2d', 'reference_3d', 'temporal_3d')


@dataclass
class Event:
    start_s: float
    bottom_s: float
    end_s: float
    label: str
    max_depth: float
    coverage: float = 1.0


def fit_reference(caches, annotations):
    """Fit standing/bottom anchors from manually delimited acceptable reference reps.

    Uses hip height and bilateral knee angles only. Reference source identities
    and hashes are added by the runner. No test labels influence these anchors.
    """
    standing, bottom = [], []
    for cid, cache in caches.items():
        t, x = cache['times'], cache['features'][:,[5,0,1]]
        for e in annotations[cid]:
            if e['label'] != 'acceptable':
                continue
            interval = (t>=e['start_s']) & (t<=e['end_s'])
            if interval.sum() < 6:
                raise ValueError(f'{cid}: reference repetition too short')
            complete = np.isfinite(x).all(axis=1)
            if complete[interval].mean() < .8:
                raise ValueError(f'{cid}: reference repetition has <80% usable geometry')
            # Local neighborhoods around the labeled anchors avoid isolated noise.
            s = interval & ((t<=e['start_s']+.2) | (t>=e['end_s']-.2)) & complete
            b = interval & (np.abs(t-e['bottom_s'])<=.15) & complete
            if not s.any() or not b.any():
                raise ValueError(f'{cid}: no valid reference anchor frames')
            standing.append(np.median(x[s],axis=0)); bottom.append(np.median(x[b],axis=0))
    if not standing:
        raise ValueError('Need annotated acceptable repetitions in reference clips')
    top, low = np.median(standing,axis=0), np.median(bottom,axis=0)
    span = top-low
    if np.any(span < np.array([.1,15,15])):
        raise ValueError('Reference motion range is insufficient; inspect landmarks and labels')
    return dict(standing=top.tolist(), bottom=low.tolist(), span=span.tolist(), cycles=len(standing))


def depth_from_features(features, reference):
    values = np.asarray(features)[[5,0,1]]
    if not np.isfinite(values).all():
        return np.nan
    d = (values-np.array(reference['standing'])) / -np.array(reference['span'])
    # Both legs must flex. Ratios remove scale and fixed angle thresholds.
    return float(.5*d[0] + .5*min(d[1],d[2]))


class AttemptTracker:
    """Causal, timestamp-based counting with explicit descent/ascent validation."""
    def __init__(self, config, temporal=False):
        self.c = config
        self.temporal = temporal
        self.events = []
        self.state = 'unarmed'
        self.last_t = self.last_valid = self.previous = None
        self.pending = None
        self.signal = np.nan
        self.last_standing = None

    def update(self, t, value):
        if self.last_t is not None and t <= self.last_t:
            raise ValueError('Timestamps must strictly increase')
        dt = t-self.last_t if self.last_t is not None else 0
        self.last_t = t
        if not np.isfinite(value):
            if self.pending:
                self.pending['missing'] += dt
            if self.last_valid is not None and t-self.last_valid > self.c['max_gap_s']:
                self._cancel()
            return None
        gap = t-self.last_valid if self.last_valid is not None else 0
        if gap > self.c['max_gap_s']:
            self._cancel()
        self.last_valid = t
        alpha = 1-np.exp(-gap/self.c['smooth_tau_s']) if self.previous is not None else 1
        depth = value if self.previous is None else self.previous + alpha*(value-self.previous)
        delta = depth-self.previous if self.previous is not None else 0
        self.previous = self.signal = depth
        if self.state == 'unarmed':
            if depth <= self.c['stand_depth']:
                self.state = 'standing'; self.last_standing = t
            return None
        if self.state == 'standing':
            if depth <= self.c['stand_depth']:
                self.last_standing = t
            if depth >= self.c['start_depth']:
                self.state = 'descending'
                self.pending = dict(start=self.last_standing if self.last_standing is not None else t,
                                    bottom=t, peak=depth, missing=0., hold=0., ascent=False)
            return None
        p = self.pending
        if depth > p['peak']:
            p['peak'], p['bottom'] = depth,t
        # Only observed, consecutive intervals contribute to the depth hold.
        if depth >= self.c['acceptable_depth'] and gap <= max(dt*1.5,1e-6):
            p['hold'] += dt
        if gap > 0 and delta / gap < -self.c['direction_epsilon']:
            p['ascent'] = True
            self.state = 'ascending'
        duration = t-p['start']
        if duration > self.c['max_duration_s']:
            self._cancel(); return None
        if depth <= self.c['stand_depth']:
            coverage = max(0., 1-p['missing']/max(duration,1e-6))
            accept_event = not self.temporal or (duration >= self.c['min_duration_s'] and p['ascent'])
            deep = p['peak'] >= self.c['acceptable_depth']
            if self.temporal:
                deep = deep and p['hold'] >= self.c['bottom_hold_s']
            label = 'acceptable' if deep else 'shallow'
            if self.temporal and coverage < self.c['min_coverage']:
                label = 'unclear'
            event = Event(p['start'],p['bottom'],t,label,p['peak'],coverage) if accept_event else None
            if event:
                self.events.append(event)
            self.pending = None; self.state = 'standing'; self.last_standing = t
            return event
        return None

    def _cancel(self):
        self.state = 'unarmed'; self.pending = None; self.previous = None
        self.signal = np.nan; self.last_standing = None


def run_method(cache, variant, config, reference):
    if variant not in VARIANTS:
        raise ValueError(variant)
    import time
    times, feats, angles = cache['times'],cache['features'],cache['angles']
    history, runtime, events = [], [], []
    baseline = SquatTracker()
    tracker = AttemptTracker(config,temporal=variant=='temporal_3d')
    smooth_angle = None
    last_valid = last_t = None
    start_s = bottom_s = None
    minimum = 180.
    for t,x,angle in zip(times,feats,angles):
        tick = time.perf_counter()
        if variant in {'baseline','confidence_2d'}:
            if variant == 'confidence_2d':
                if not np.isfinite(angle):
                    if last_valid is not None and t-last_valid>config['max_gap_s']:
                        baseline.state='standing'; baseline.min_angle=180
                        start_s=None; smooth_angle=None
                else:
                    if last_valid is not None and t-last_valid>config['max_gap_s']:
                        baseline.state='standing'; baseline.min_angle=180
                        start_s=None; smooth_angle=None
                    alpha=1-np.exp(-(t-last_valid)/config['smooth_tau_s']) if last_valid is not None else 1
                    smooth_angle=angle if smooth_angle is None else smooth_angle+alpha*(angle-smooth_angle)
                    last_valid=t; angle=smooth_angle
            old = baseline.reps
            previous_state=baseline.state
            baseline.update(float(angle))
            if previous_state=='standing' and baseline.state=='squatting':
                start_s=float(t); minimum=angle; bottom_s=float(t)
            if start_s is not None and np.isfinite(angle) and angle<minimum:
                minimum=angle; bottom_s=float(t)
            if baseline.reps>old:
                events.append(Event(start_s,bottom_s,float(t),'acceptable' if minimum<=95 else 'shallow',float((175-minimum)/80)))
                start_s=None
            signal=float(angle) if np.isfinite(angle) else None
            state=baseline.state
        else:
            raw=depth_from_features(x,reference)
            tracker.update(float(t),raw)
            events=tracker.events
            signal=float(tracker.signal) if np.isfinite(tracker.signal) else None
            state=tracker.state
        runtime.append((time.perf_counter()-tick)*1000)
        history.append(dict(time_s=float(t), signal=signal, state=state, attempts=len(events),
                            acceptable=sum(e.label=='acceptable' for e in events)))
        last_t=t
    return [asdict(e) for e in events], history, np.asarray(runtime)
