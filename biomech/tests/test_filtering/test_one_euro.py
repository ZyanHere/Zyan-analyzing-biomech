"""Smoothing: does it remove jitter without eating the signal or adding lag?"""

from __future__ import annotations

import numpy as np
import pytest

from biomech.config import FilterConfig
from biomech.filtering.one_euro import LandmarkFilter, OneEuroArrayFilter
from biomech.types import LandmarkSet

from ..synthetic import build, landmark_set

DT = 1.0 / 30.0


def _run(values: list[float], cfg: FilterConfig | None = None) -> list[float]:
    """Feed a scalar series through the filter at a steady 30 FPS."""
    f = OneEuroArrayFilter(cfg or FilterConfig())
    return [float(f.apply(np.array([v]), i * DT)[0]) for i, v in enumerate(values)]


class TestSmoothing:
    def test_first_sample_passes_through_unchanged(self) -> None:
        assert _run([42.0])[0] == 42.0

    def test_noise_around_a_constant_is_reduced(self) -> None:
        rng = np.random.default_rng(0)
        noisy = [100.0 + float(rng.normal(0, 3.0)) for _ in range(200)]
        smoothed = _run(noisy)[50:]
        assert np.std(smoothed) < np.std(noisy[50:]) * 0.6

    def test_a_steady_ramp_is_tracked_not_flattened(self) -> None:
        """Adaptive cutoff: a moving signal should not be smoothed away."""
        ramp = [float(i) for i in range(100)]
        out = _run(ramp)
        assert out[-1] == pytest.approx(ramp[-1], rel=0.05)

    def test_lag_on_a_fast_move_stays_small(self) -> None:
        """The failure this filter exists to avoid: EMA alpha=0.15 bought low
        jitter at 240 ms of lag - seven frames (FINDINGS.md F15)."""
        step = [0.0] * 20 + [100.0] * 20
        out = _run(step)
        frames_to_90pct = next(i for i, v in enumerate(out[20:]) if v >= 90.0)
        assert frames_to_90pct * DT * 1000 < 120.0

    def test_range_is_mostly_retained(self) -> None:
        """One Euro keeps 88-89% of movement range. Peaks must therefore be
        read from the raw signal, not this one."""
        swing = [50.0 * np.sin(i * 0.2) for i in range(200)]
        out = _run(swing)[20:]
        assert (max(out) - min(out)) > 0.8 * (max(swing[20:]) - min(swing[20:]))


class TestTimeHandling:
    def test_dt_comes_from_the_supplied_timestamp(self) -> None:
        """Not from a wall clock read inside the filter: that would fold
        queueing delay into the velocity estimate."""
        f = OneEuroArrayFilter(FilterConfig())
        f.apply(np.array([0.0]), 100.0)
        moved = f.apply(np.array([10.0]), 100.0 + DT)
        assert 0.0 < float(moved[0]) < 10.0

    def test_a_long_gap_resets_instead_of_interpolating(self) -> None:
        """Beyond the clamp the previous sample is no longer evidence about
        this one, so restarting is honest and inventing motion is not."""
        cfg = FilterConfig()
        f = OneEuroArrayFilter(cfg)
        f.apply(np.array([0.0]), 0.0)
        after_gap = f.apply(np.array([100.0]), cfg.dt_max_s * 3)
        assert float(after_gap[0]) == 100.0, "should have reset and passed through"

    def test_time_going_backwards_resets(self) -> None:
        f = OneEuroArrayFilter(FilterConfig())
        f.apply(np.array([0.0]), 10.0)
        assert float(f.apply(np.array([55.0]), 9.0)[0]) == 55.0

    def test_dropped_frames_are_handled_by_the_dt_term(self) -> None:
        """Being dt-aware is why dropped frames degrade gracefully here, where
        a fixed-alpha EMA silently changes meaning with the frame rate."""
        f = OneEuroArrayFilter(FilterConfig())
        f.apply(np.array([0.0]), 0.0)
        f.apply(np.array([10.0]), DT)
        big_step = f.apply(np.array([20.0]), DT + 4 * DT)
        assert 10.0 < float(big_step[0]) <= 20.0

    def test_reset_clears_history(self) -> None:
        f = OneEuroArrayFilter(FilterConfig())
        f.apply(np.array([0.0]), 0.0)
        f.reset()
        assert float(f.apply(np.array([99.0]), DT)[0]) == 99.0


class TestLandmarkFilter:
    def test_raw_landmarks_are_not_mutated(self) -> None:
        """The validity layer judges raw data; if smoothing modified it in
        place, it would be judging the smoother instead."""
        marks = landmark_set(build(side="L", body_yaw=90.0))
        before = marks.image_xy.copy()
        LandmarkFilter(FilterConfig()).apply(marks, 0.0)
        assert np.array_equal(marks.image_xy, before)

    def test_visibility_passes_through_untouched(self) -> None:
        """Confidence is the model's opinion, not a measurement to smooth."""
        marks = landmark_set(build(side="L"), visibility=0.73)
        out = LandmarkFilter(FilterConfig()).apply(marks, 0.0)
        assert np.array_equal(out.visibility, marks.visibility)

    def test_disabling_returns_the_input_unchanged(self) -> None:
        from dataclasses import replace

        marks = landmark_set(build(side="L"))
        cfg = replace(FilterConfig(), filter_landmarks=False)
        assert LandmarkFilter(cfg).apply(marks, 0.0) is marks

    def test_jitter_in_landmarks_becomes_less_jitter_in_angles(self) -> None:
        """The end-to-end claim: this is applied to positions specifically so
        the angles derived from them are steadier."""
        from biomech.biomechanics.angles import elbow_flexion
        from biomech.types import Side

        rng = np.random.default_rng(3)
        base = build(side="L", body_yaw=90.0)
        marks = landmark_set(base)
        filt = LandmarkFilter(FilterConfig())

        raw_angles, smooth_angles = [], []
        for i in range(120):
            noisy_xy = marks.image_xy + rng.normal(0, 1.5, marks.image_xy.shape)
            noisy = LandmarkSet(noisy_xy, marks.world_xyz, marks.visibility)
            raw_angles.append(elbow_flexion(noisy.image_xy, Side.LEFT))
            smooth = filt.apply(noisy, i * DT)
            smooth_angles.append(elbow_flexion(smooth.image_xy, Side.LEFT))

        assert np.nanstd(smooth_angles[30:]) < np.nanstd(raw_angles[30:]) * 0.7
