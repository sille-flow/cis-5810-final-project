"""Extract poses from reference videos and save a phase template.

Mix side-view and facing-camera clips in the same folder. The builder tags
each clip automatically and writes one multi-view template.

Example:
    python scripts/build_template.py --videos data/references/squat/good --out data/templates/squat_good.npz
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synthetic import synthetic_squat_sequence
from src.template import build_multiview_templates

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, default=None, help="Folder of reference clips (defaults to squat references)")
    parser.add_argument("--exercise", default="squat", help="Exercise label used for the default reference folder")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--phases", type=int, default=32)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--per-view",
        action="store_true",
        help="Keep separate front/side templates. Default pools every clip into one "
        "template, because the features are yaw invariant.",
    )
    args = parser.parse_args()
    if args.videos is None and not args.synthetic:
        args.videos = ROOT / "data" / "references" / args.exercise / "good"
    if args.out is None:
        args.out = ROOT / "data" / "templates" / f"{args.exercise}_good.npz"

    by_view: dict[str, list] = defaultdict(list)
    if args.synthetic or args.videos is None:
        print("Building a synthetic bootstrap template. Replace with real videos.")
        by_view["side"] = [
            synthetic_squat_sequence(n_frames=72, n_reps=3, depth=88.0, noise=1.5, view="side"),
            synthetic_squat_sequence(n_frames=90, n_reps=3, depth=92.0, noise=1.5, torso_lean=22.0, view="side"),
        ]
        by_view["front"] = [
            synthetic_squat_sequence(n_frames=72, n_reps=3, depth=88.0, noise=1.5, view="front"),
        ]
    else:
        from src.pose_stream import extract_feature_sequence

        clips = sorted(p for p in args.videos.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if not clips:
            raise SystemExit(f"No video files in {args.videos}")
        for clip in clips:
            print(f"Extracting {clip.name} ...")
            try:
                sequence, view = extract_feature_sequence(clip)
            except RuntimeError as exc:
                print(f"  skipping clip: {exc}")
                continue
            print(f"  detected view: {view} ({len(sequence)} frames)")
            by_view[view].append(sequence)

        if not any(by_view.values()):
            raise SystemExit(
                "No usable pose features were found in any clip. Make sure the full body is visible, "
                "well lit, and large enough in frame; for floor exercises, include the shoulders, hips, "
                "knees, and ankles where possible."
            )

    if not args.per_view:
        pooled = [seq for seqs in by_view.values() for seq in seqs]
        counts = {v: len(s) for v, s in by_view.items()}
        by_view = defaultdict(list, {"oblique": pooled})
        print(f"Pooling {len(pooled)} clips into one template (clips per detected view: {counts})")

    templates = build_multiview_templates(dict(by_view), n_phases=args.phases)
    templates.save(args.out)
    print(f"Wrote {args.out} with views: {', '.join(templates.views)}")


if __name__ == "__main__":
    main()
