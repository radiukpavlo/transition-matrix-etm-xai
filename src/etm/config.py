"""Minimal YAML->dataclass loader.

We intentionally keep configuration simple and transparent.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T")


def dataclass_from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    try:
        # Resolve forward references and string annotations
        types = typing.get_type_hints(cls)
    except Exception:
        # Fallback if resolution fails (e.g. strict dependencies missing)
        types = {f.name: f.type for f in dataclasses.fields(cls)}

    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]

        # Get resolved type
        field_type = types.get(f.name, f.type)

        # Handle recursive dataclasses
        if dataclasses.is_dataclass(field_type) and isinstance(val, dict):
            kwargs[f.name] = dataclass_from_dict(field_type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)  # type: ignore[arg-type]
