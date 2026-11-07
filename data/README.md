# Reference workout videos

The live system does not score form with fixed squat angles. It follows a
**phase template** built from one or more good-form clips of the same exercise.

## What to record yourselves

Best option for this class project: film good-form squats from **both** a
slight side angle and facing the camera.

- Camera roughly at hip height, full body in frame
- 3–8 continuous reps per clip
- Put **side and front** clips in the same folder `data/references/squat/good/`
- The builder labels each clip automatically (`front`, `side`, or `oblique`)
- Rebuild the template:

```text
python scripts/build_template.py --videos data/references/squat/good --out data/templates/squat_good.npz
```

Later you can add other folders (`squat/shallow`, `pushup/good`) and build one
template per class. The matcher only needs the template `.npz` at runtime.

Build the three reference templates from the existing exercise folders:

```text
python scripts/build_template.py --exercise squat
python scripts/build_template.py --exercise pushups
python scripts/build_template.py --exercise situps
python scripts/run_live.py --templates data/templates
```

Automatic exercise selection compares each live feature sequence with all
loaded reference templates. A template must exist for every exercise to be
recognized. Pushups and situps currently use generic cycle phase counting;
the squat retains its depth-based counter.

## Public datasets worth using as extra references

These are starting points, not required downloads. After you obtain clips,
run the same `build_template.py` extractor. Many datasets already include
skeleton tracks; if so, convert those joints into our feature layout instead
of re-running MediaPipe.

| Dataset | What it gives you | How it helps here |
| --- | --- | --- |
| [InfiniteForm / InfiniteRep](https://github.com/SeungBack/infiniteform) | Large synthetic fitness videos with pose labels | Lots of squat variation without filming everyone in the class |
| [MM-Fit](https://github.com/mmfit/mm-fit) | Multi-camera gym exercises with IMU + vision | Real gym footage; useful if you want more than webcam side views |
| [UESTC RGB-D](https://github.com/HeliXonProtein/UESTC-RGB-D) | 32 aerobic exercises, RGB-D | Extra motion diversity for testing phase tracking |
| [Fit3D](https://fit3d.imar.ro/) | Lab-captured fitness with 3D mocap | High-quality joint trajectories if you need a clean template |
| Kaggle / YouTube squat compilations | Uncontrolled form videos | Good for *testing* the matcher, weaker as a “correct form” prior |

Do not treat a random YouTube squat as ground-truth coaching. For form
feedback, prefer clips you have labeled as good, or datasets with explicit
quality ratings.

## Why not put raw videos in git

Clips are large and often not yours to redistribute. Keep videos local, commit
only the extracted `data/templates/*.npz` files if they do not contain
identifiable imagery.
