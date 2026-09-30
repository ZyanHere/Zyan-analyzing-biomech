"""Entry point: python -m biomech

Owns: process lifecycle, logging setup, and turning a deliberate error into a
readable message instead of a traceback.
Does NOT own: any pipeline logic. It parses, configures, delegates, reports.
"""

from __future__ import annotations

import logging
import sys

from .cli import build_parser, config_from_args, validate_args
from .errors import BiomechError

log = logging.getLogger("biomech")


def setup_logging(verbose: bool) -> None:
    """Configure logging once, for the whole process."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the application. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    try:
        validate_args(args)
        cfg = config_from_args(args)

        # Logged at startup so any session can be reconstructed from its log
        # alone - which matters because every threshold here came from an
        # experiment, and a result is only meaningful with its settings.
        log.info("configuration:")
        for section, values in cfg.to_dict().items():
            log.info("  %s: %s", section, values)

        if args.probe:
            from .capture.probe import probe_camera

            probe_camera(cfg.capture)
            return 0

        from .pipeline.runner import run  # imported late: it loads MediaPipe

        return run(cfg, args)

    except BiomechError as exc:
        # Deliberate errors carry actionable messages. A traceback would only
        # bury the sentence the user actually needs.
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
