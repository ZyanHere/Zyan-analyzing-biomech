"""Exception hierarchy.

Owns: the error types this package raises, and the contract that their messages
are actionable.
Does NOT own: how errors are presented (see __main__.py).

Every message states what failed, what was expected, and what to do about it.
A user who sees one of these should not need to read the source to act on it.
"""

from __future__ import annotations


class BiomechError(Exception):
    """Base for every error this package raises deliberately.

    The entry point catches this and prints the message without a traceback.
    Anything not derived from it keeps its traceback, because an unexpected
    error should look unexpected.
    """


class ConfigError(BiomechError):
    """Configuration is invalid or self-contradictory."""


class CaptureError(BiomechError):
    """A frame source could not be opened, or failed irrecoverably."""


class ModelError(BiomechError):
    """The pose model could not be found or loaded."""


class SourceExhaustedError(BiomechError):
    """A finite source (video or landmark file) reached its end.

    Terminal, but not a fault: a file ending is expected. Raised rather than
    returning None so that "stop, there will never be another frame" cannot be
    confused with "wait, the camera has not produced one yet" - the two need
    opposite responses from the pipeline.

    Named with the Error suffix for convention; it signals the end of a source,
    not a malfunction.
    """
