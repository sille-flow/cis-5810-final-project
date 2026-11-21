"""Demo the experiment's temporal counter with fitted reference.json anchors."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.squat.tracking import AttemptTracker, depth_from_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'configs/experiment.json')
    p.add_argument('--source',type=Path)
    p.add_argument('--camera',type=int,default=0)
    args=p.parse_args()
    import cv2
    from src.pose.stream import make_landmarker,features_from_result,to_mp_image
    from src.visualization.overlay import draw_skeleton
    tracker=AttemptTracker(json.loads(args.config.read_text()),True)
    reference=json.loads(args.reference.read_text())
    cap=cv2.VideoCapture(str(args.source) if args.source else args.camera)
    if not cap.isOpened():raise ValueError('Cannot open video/camera')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    origin=time.monotonic();index=0;last_ms=-1;feedback='Stand fully upright to begin'
    try:
        with make_landmarker() as detector:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if args.source:
                    raw_ms=cap.get(cv2.CAP_PROP_POS_MSEC)
                    ms=round(raw_ms) if np.isfinite(raw_ms) and raw_ms>last_ms else last_ms+max(1,round(1000/fps))
                else:ms=max(last_ms+1,round((time.monotonic()-origin)*1000))
                last_ms=ms;index+=1
                result=detector.detect_for_video(to_mp_image(frame),ms)
                features,view,landmarks=features_from_result(result)
                depth=depth_from_features(features,reference) if features is not None else np.nan
                event=tracker.update(ms/1000,depth)
                if event:feedback={'acceptable':'Depth reached','shallow':'Shallow attempt','unclear':'Depth uncertain'}[event.label]
                if landmarks is not None:draw_skeleton(frame,landmarks)
                lines=[f'Attempts: {len(tracker.events)}  Depth reached: {sum(e.label=="acceptable" for e in tracker.events)}',
                       f'State: {tracker.state}', feedback if np.isfinite(depth) else 'Pose unclear - show hips, knees and feet']
                for j,line in enumerate(lines):
                    cv2.putText(frame,line,(15,30+30*j),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),4)
                    cv2.putText(frame,line,(15,30+30*j),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                scale=min(960/frame.shape[1],720/frame.shape[0])
                cv2.imshow('Squat experiment',cv2.resize(frame,None,fx=scale,fy=scale))
                if cv2.waitKey(1)&255==ord('q'):break
    finally:
        cap.release();cv2.destroyAllWindows()


if __name__=='__main__':main()
