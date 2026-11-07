"""Build a phase-indexed pose template from reference feature sequences.

Sequences are grouped by camera view (front / side / oblique). Live matching
then uses the template for the current view so frontal squats are not scored
against a side-view knee-angle trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, feature_min_std, interpolate_nans


@dataclass
class PhaseTemplate:
    mean: np.ndarray
    std: np.ndarray
    pca_mean: np.ndarray
    pca_axis: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    view: str = "side"

    @property
    def n_phases(self) -> int:
        return int(self.mean.shape[0])

    def project(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            return (x - self.pca_mean) @ self.pca_axis
        return (x - self.pca_mean) @ self.pca_axis

    def pose_at_index(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        i = int(index) % self.n_phases
        return self.mean[i], self.std[i]


@dataclass
class MultiViewTemplates:
    views: dict[str, PhaseTemplate] = field(default_factory=dict)

    def get(self, view: str) -> PhaseTemplate:
        if view in self.views:
            return self.views[view]
        if view == "front" and "oblique" in self.views:
            return self.views["oblique"]
        if view == "side" and "oblique" in self.views:
            return self.views["oblique"]
        if "oblique" in self.views:
            return self.views["oblique"]
        if "side" in self.views:
            return self.views["side"]
        return next(iter(self.views.values()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "view_names": np.array(list(self.views.keys())),
        }
        for name, template in self.views.items():
            payload[f"{name}_mean"] = template.mean
            payload[f"{name}_std"] = template.std
            payload[f"{name}_pca_mean"] = template.pca_mean
            payload[f"{name}_pca_axis"] = template.pca_axis
            payload[f"{name}_feature_names"] = np.array(template.feature_names)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MultiViewTemplates":
        data = np.load(path, allow_pickle=False)
        if "view_names" in data.files:
            views = {}
            for name in data["view_names"].tolist():
                name = str(name)
                names = tuple(str(n) for n in data[f"{name}_feature_names"].tolist())
                views[name] = PhaseTemplate(
                    mean=data[f"{name}_mean"],
                    std=data[f"{name}_std"],
                    pca_mean=data[f"{name}_pca_mean"],
                    pca_axis=data[f"{name}_pca_axis"],
                    feature_names=names,
                    view=name,
                )
            return cls(views=views)
        names = tuple(str(n) for n in data["feature_names"].tolist())
        legacy = PhaseTemplate(
            mean=data["mean"],
            std=data["std"],
            pca_mean=data["pca_mean"],
            pca_axis=data["pca_axis"],
            feature_names=names,
            view="side",
        )
        return cls(views={"side": legacy})


def _pca_axis(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = frames.mean(axis=0)
    scale = np.maximum(frames.std(axis=0), 1e-6)
    centered = (frames - mean) / scale
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0] / scale
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    projection = (frames - mean) @ axis
    if projection[0] < 0:
        axis = -axis
    return mean, axis


def _smooth_1d(signal: np.ndarray, window: int = 7) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _segment_cycles(progress: np.ndarray) -> list[tuple[int, int]]:
    if len(progress) < 12:
        return [(0, len(progress))]

    smooth = _smooth_1d(progress)
    high = np.quantile(smooth, 0.65)
    peaks: list[int] = []
    for i in range(2, len(smooth) - 2):
        if smooth[i] >= high and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            if not peaks or i - peaks[-1] >= 8:
                peaks.append(i)
    if len(peaks) < 2:
        return [(0, len(progress))]
    return [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1) if peaks[i + 1] - peaks[i] >= 8]


def _resample_cycle(cycle: np.ndarray, n_phases: int) -> np.ndarray:
    t_src = np.linspace(0.0, 1.0, cycle.shape[0])
    t_dst = np.linspace(0.0, 1.0, n_phases, endpoint=False)
    out = np.zeros((n_phases, cycle.shape[1]), dtype=np.float64)
    for j in range(cycle.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, cycle[:, j])
    return out


def build_phase_template(
    sequences: list[np.ndarray],
    n_phases: int = 32,
    min_std: np.ndarray | float | None = None,
    view: str = "side",
) -> PhaseTemplate:
    if not sequences:
        raise ValueError("Need at least one reference sequence")

    cleaned = [interpolate_nans(np.asarray(seq, dtype=np.float64)) for seq in sequences]
    stacked = np.concatenate(cleaned, axis=0)
    pca_mean, pca_axis = _pca_axis(stacked)

    resampled: list[np.ndarray] = []
    for seq in cleaned:
        progress = (seq - pca_mean) @ pca_axis
        for start, end in _segment_cycles(progress):
            cycle = seq[start:end]
            if len(cycle) < 6:
                continue
            resampled.append(_resample_cycle(cycle, n_phases))

    if not resampled:
        resampled = [_resample_cycle(seq, n_phases) for seq in cleaned if len(seq) >= 6]
    if not resampled:
        raise ValueError("Reference sequences are too short to build a template")

    cube = np.stack(resampled, axis=0)
    mean = cube.mean(axis=0)
    std = cube.std(axis=0)
    floor = feature_min_std() if min_std is None else np.asarray(min_std, dtype=np.float64)
    std = np.maximum(std, floor)
    return PhaseTemplate(
        mean=mean,
        std=std,
        pca_mean=pca_mean,
        pca_axis=pca_axis,
        view=view,
    )


def build_multiview_templates(
    sequences_by_view: dict[str, list[np.ndarray]],
    n_phases: int = 32,
) -> MultiViewTemplates:
    views = {}
    for view, sequences in sequences_by_view.items():
        if not sequences:
            continue
        views[view] = build_phase_template(sequences, n_phases=n_phases, view=view)
    if not views:
        raise ValueError("Need at least one reference sequence")
    return MultiViewTemplates(views=views)
