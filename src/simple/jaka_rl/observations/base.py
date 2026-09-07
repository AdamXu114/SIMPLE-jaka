"""Observation base + registry for the Jaka MF policy.

Ported from sim2real-jaka ``rl_policy/observations/base.py``. Subclasses are
auto-registered by class name via ``__init_subclass__`` so the yaml
``_target_`` key can resolve them by name.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from simple.jaka_rl.base_policy import BasePolicy


class _RegistryMixin:
    def __init_subclass__(cls) -> None:
        if not hasattr(cls, "registry"):
            cls.registry = {}

        cls_name = cls.__name__
        cls._file = inspect.getfile(cls)
        cls._line = inspect.getsourcelines(cls)[1]
        if cls_name.startswith("_"):
            return
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")


class Observation(_RegistryMixin):
    def __init__(self, env: "BasePolicy", **kwargs):
        self.env = env
        self.state_processor = env.state_processor

    def reset(self):
        pass

    def update(self, data: Dict[str, Any]) -> None:
        pass

    def compute(self) -> np.ndarray:
        raise NotImplementedError


class ObsGroup:
    def __init__(
        self,
        name: str,
        funcs: Dict[str, Observation],
    ):
        self.name = name
        self.funcs = funcs

    def compute(self) -> np.ndarray:
        tensors = [func.compute() for func in self.funcs.values()]
        return np.concatenate(tensors, axis=-1)


__all__ = ["Observation", "ObsGroup"]
