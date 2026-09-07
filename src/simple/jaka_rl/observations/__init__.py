"""Jaka MF observation classes (registered by name for yaml ``_target_``).

Only the v1 MF observation (620-dim) is supported; the v2 (835-dim) policy was
dropped in favor of ``jaka_mf_v1_dr``.
"""

from .base import Observation, ObsGroup
from .jaka_mf import jaka_frame_stack_mf

__all__ = [
    "Observation",
    "ObsGroup",
    "jaka_frame_stack_mf",
]
