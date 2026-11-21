"""Extract unlabeled videos for quality review; does not assign labels or splits."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.pose.cache import extract_cached


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('videos',nargs='+',type=Path)
    p.add_argument('--cache',type=Path,default=Path('output/pose_cache'))
    p.add_argument('--model',type=Path)
    p.add_argument('--report',type=Path,default=Path('output/pose_quality.json'))
    args=p.parse_args();rows=[]
    for video in args.videos:
        cache,path=extract_cached(video,args.cache,args.model)
        rows.append(dict(video=str(video),cache=str(path),frames=len(cache['times']),
                         duration_s=float(cache['times'][-1]),
                         usable_2d_fraction=float(np.isfinite(cache['angles']).mean()),
                         usable_depth_fraction=float(np.isfinite(cache['features'][:,[5,0,1]]).all(axis=1).mean()),
                         per_feature_valid_fraction=np.isfinite(cache['features']).mean(axis=0).tolist(),
                         metadata=cache['metadata']))
        print(video,rows[-1]['usable_depth_fraction'],flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
