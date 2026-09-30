"""Phase 0 acceptance tests: configuration and CLI."""

from __future__ import annotations

import pytest

from biomech.cli import build_parser, config_from_args, validate_args
from biomech.config import Config, OrientationConfig
from biomech.errors import ConfigError


def test_default_config_is_valid() -> None:
    Config().validate()


def test_config_serialises_for_recording() -> None:
    d = Config().to_dict()
    assert d["validity"]["min_visibility"] == 0.55
    assert d["model"]["running_mode"] == "video"


def test_unknown_model_variant_is_rejected() -> None:
    from dataclasses import replace

    cfg = Config()
    cfg = replace(cfg, model=replace(cfg.model, variant="enormous"))
    with pytest.raises(ConfigError, match="lite, full, heavy"):
        cfg.validate()


def test_overlapping_orientation_bands_are_rejected() -> None:
    """The gap between the bands is the region where neither plane is valid.

    If frontal_max rises above sagittal_min there is no such gap, and a
    measurement could be considered valid in two mutually exclusive planes.
    """
    from dataclasses import replace

    cfg = replace(
        Config(),
        orientation=OrientationConfig(sagittal_min_yaw_deg=40.0, frontal_max_yaw_deg=50.0),
    )
    with pytest.raises(ConfigError, match="bands overlap"):
        cfg.validate()


def test_visibility_outside_zero_to_one_is_rejected() -> None:
    from dataclasses import replace

    cfg = Config()
    cfg = replace(cfg, validity=replace(cfg.validity, min_visibility=1.5))
    with pytest.raises(ConfigError, match="between 0 and 1"):
        cfg.validate()


def test_replay_source_requires_a_path() -> None:
    args = build_parser().parse_args(["--source", "video"])
    with pytest.raises(ConfigError, match="needs --path"):
        validate_args(args)


def test_missing_replay_file_fails_before_the_model_loads() -> None:
    args = build_parser().parse_args(["--source", "landmarks", "--path", "nope.jsonl"])
    with pytest.raises(ConfigError, match="No such file"):
        validate_args(args)


def test_video_recording_rejected_for_landmark_replay() -> None:
    """A landmark recording contains no images, so there is nothing to write."""
    args = build_parser().parse_args(
        ["--source", "landmarks", "--path", __file__, "--record-video", "out.mp4"]
    )
    with pytest.raises(ConfigError, match="no images"):
        validate_args(args)


def test_cli_flags_reach_the_config() -> None:
    args = build_parser().parse_args(["--device", "2", "--model", "lite"])
    cfg = config_from_args(args)
    assert cfg.capture.device_index == 2
    assert cfg.model.variant == "lite"
