# -*- coding: utf-8 -*-
"""
Provides a compatibility layer for Python typing features,
ensuring support across different Python versions.

It imports various typing constructs from the built-in `typing` module.
For the `TypeGuard` feature, which is available in Python 3.10 and later,
it attempts to import from `typing_extensions` if not found in the built-in module.
"""

import sys
import subprocess
from typing import (
    List,
    Tuple,
    Sequence,
    Dict,
    Iterable,
    Callable,
    Union,
    Any,
    Generic,
    Optional,
    Type,
    Mapping,
    Text,
    TypeVar,
    Iterator,
    SupportsInt,
    Set,
    ContextManager,
    Deque,
    FrozenSet,
    NamedTuple,
    NewType,
    TypedDict,
    Generator, 
    Pattern, 
    Literal
)

# Check if Python version is 3.10 or higher
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    try:
        # Try to import TypeGuard from typing_extensions
        from typing_extensions import TypeGuard
    except ImportError:
        # Try to install typing_extensions if it's not available
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "typing_extensions"])
            from typing_extensions import TypeGuard
        except (ImportError, subprocess.CalledProcessError):
            # Provide a fallback for TypeGuard if installation and import both fail
            from typing import Type #noqa

            # Mimic TypeGuard behavior
            class TypeGuard(Type[bool]):
                pass

__all__ = [
    "List",
    "Tuple",
    "Sequence",
    "Dict",
    "Iterable",
    "Callable",
    "Union",
    "Any",
    "Generic",
    "Optional",
    "Type",
    "Mapping",
    "Text",
    "TypeVar",
    "Iterator",
    "SupportsInt",
    "Set",
    "ContextManager",
    "Deque",
    "FrozenSet",
    "NamedTuple",
    "NewType",
    "TypedDict",
    "Generator",
    "TypeGuard",
    "Pattern", 
    "Literal"
]
