"""Load private-server adapter factories without importing private code here."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_plugin(specification: str, **kwargs) -> Any:
    """Load ``package.module:factory`` and instantiate it with keyword args."""

    if ":" not in specification:
        raise ValueError("plugin specification must be 'package.module:factory'")
    module_name, attribute_name = specification.split(":", maxsplit=1)
    module = import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"plugin factory {specification!r} is not callable")
    return factory(**kwargs)
