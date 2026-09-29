"""Typed configuration: nested dataclasses loaded from YAML with inheritance and overrides.

A config file is plain YAML mirroring a dataclass tree. Two extras:

  - `base: <path>` (or a list of paths) at the top of a file inherits from other
    YAML files, resolved relative to the including file. Mappings merge recursively;
    anything else (scalars, lists) is replaced by the child.
  - Command-line overrides `a.b.c=value`; the value is parsed as YAML, so `3e-4`,
    `true`, `[2, 3]` and `null` get their natural types.

Unknown keys are errors (a typo must not silently fall back to a default), and
values are type-checked against the dataclass annotations.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_tree(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, recursively resolving its `base:` includes."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    bases = data.pop("base", None)
    if bases is None:
        return data
    merged: dict[str, Any] = {}
    for base in [bases] if isinstance(bases, str) else bases:
        merged = _merge(merged, load_yaml_tree(path.parent / base))
    return _merge(merged, data)


def apply_overrides(tree: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    tree = dict(tree)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"override {override!r} must look like key.path=value")
        dotted, raw = override.split("=", 1)
        keys = dotted.strip().split(".")
        node = tree
        for key in keys[:-1]:
            child = node.get(key)
            node[key] = dict(child) if isinstance(child, dict) else {}
            node = node[key]
        node[keys[-1]] = yaml.safe_load(raw)
    return tree


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is Any:
        return value
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, where)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
        raise TypeError(f"{where}: {value!r} matches none of {annotation} ({'; '.join(errors)})")
    if dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping for {annotation.__name__}, got {value!r}")
        return from_dict(annotation, value, where)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        item_type = args[0] if args else Any
        items = [_coerce(item, item_type, f"{where}[{i}]") for i, item in enumerate(value)]
        return items if origin is list else tuple(items)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        value_type = args[1] if len(args) == 2 else Any
        return {key: _coerce(item, value_type, f"{where}.{key}") for key, item in value.items()}
    if origin is typing.Literal:
        if value not in args:
            raise ValueError(f"{where}: {value!r} is not one of {args}")
        return value
    if annotation is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if annotation is float and isinstance(value, str):
        return float(value)  # YAML reads "3e-4" (no dot) as a string
    if annotation in (int, str, bool) and not isinstance(value, annotation):
        raise TypeError(f"{where}: expected {annotation.__name__}, got {value!r}")
    if annotation is int and isinstance(value, bool):
        raise TypeError(f"{where}: expected int, got {value!r}")
    return value


def from_dict(cls: type[T], data: dict[str, Any], where: str = "") -> T:
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"{where or cls.__name__}: unknown key(s) {sorted(unknown)}; known: {sorted(fields)}")
    kwargs = {
        name: _coerce(value, hints[name], f"{where}.{name}" if where else name) for name, value in data.items()
    }
    return cls(**kwargs)


def to_dict(config: Any) -> Any:
    if dataclasses.is_dataclass(config):
        return {f.name: to_dict(getattr(config, f.name)) for f in dataclasses.fields(config)}
    if isinstance(config, (list, tuple)):
        return [to_dict(item) for item in config]
    if isinstance(config, dict):
        return {key: to_dict(value) for key, value in config.items()}
    return config


def load_config(cls: type[T], path: str | Path | None, overrides: list[str] | None = None) -> T:
    tree = load_yaml_tree(path) if path else {}
    return from_dict(cls, apply_overrides(tree, overrides or []))


def dump_yaml(config: Any) -> str:
    return yaml.safe_dump(to_dict(config), sort_keys=False, allow_unicode=True)
