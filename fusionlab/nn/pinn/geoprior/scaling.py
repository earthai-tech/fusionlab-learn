# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

r"""GeoPrior scaling config helpers (Keras-serializable)."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import os

import numpy as np

from ...._fusionlog import fusionlog
from ... import KERAS_DEPS, dependency_message
from .utils import (
    canonicalize_scaling_kwargs,
    enforce_scaling_alias_consistency,
    load_scaling_kwargs,
    validate_scaling_kwargs,
)

K = KERAS_DEPS
register_keras_serializable = K.register_keras_serializable

DEP_MSG = dependency_message("nn.pinn.geoprior.scaling")
logger = fusionlog().get_fusionlab_logger(__name__)


def _jsonify(x):
    r"""
    Convert nested objects into JSON-serializable Python types.
    
    This helper walks common container types and converts values
    into plain Python objects suitable for storage in a Keras
    configuration dictionary.
    
    It is intended for defensive serialization, where values may
    include NumPy scalars, tuples, sets, or mapping-like objects.
    
    Parameters
    ----------
    x : object
        Input object to convert. This may be a mapping, list,
        tuple, set, NumPy scalar, or any other Python object.
    
    Returns
    -------
    out : object
        A JSON-serializable representation of ``x`` when possible.
        Containers are converted recursively. Objects that do not
        require conversion are returned unchanged.
    
    Notes
    -----
    - Mapping keys are cast to ``str`` to avoid non-JSON keys.
    - Sets are converted to sorted lists to ensure stability.
    - NumPy scalar types are converted using ``.item()``.
    
    Examples
    --------
    >>> _jsonify({"a": 1})
    {'a': 1}
    
    >>> import numpy as np
    >>> _jsonify({"v": np.float32(2.0)})
    {'v': 2.0}
    
    See Also
    --------
    GeoPriorScalingConfig.get_config :
        Uses this function to serialize configuration safely.
    """
    # Dict-like: ensure keys are strings.
    if isinstance(x, Mapping):
        return {str(k): _jsonify(v) for k, v in x.items()}

    # List/tuple: keep ordering.
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]

    # Set: stable ordering for deterministic configs.
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)

    # NumPy scalar: convert to Python scalar.
    if hasattr(x, "item") and isinstance(
        x,
        (np.generic,),
    ):
        return x.item()

    # Fall back: return as-is.
    return x


@register_keras_serializable(
    "fusionlab.nn.pinn.geoprior",
    name="GeoPriorScalingConfig",
)
@dataclass
class GeoPriorScalingConfig:
    r"""
    Scaling configuration utilities for GeoPrior PINN.

    This module defines :class:`~GeoPriorScalingConfig`, a small
    Keras-serializable container used to store and reconstruct
    the physics scaling and slicing controls used by
    GeoPriorSubsNet.

    The scaling configuration is critical because it governs how
    coordinates, time units, groundwater variables, and physics
    residuals are interpreted and non-dimensionalized. If this
    configuration is not faithfully serialized via Keras
    ``get_config()``, a reloaded model may be reconstructed with
    a different effective physics behavior.

    The main entry point is :meth:`GeoPriorScalingConfig.from_any`,
    which accepts a ``dict``-like mapping, a file path ``str``,
    or an existing :class:`~GeoPriorScalingConfig` instance. The
    resolved configuration is produced by :meth:`resolve`, which
    runs the same canonicalization and validation pipeline used
    during training.

    Notes
    -----
    - The resolved scaling dictionary should be JSON-safe and
      stable under Keras serialization.
    - Use :func:`_jsonify` to defensively convert nested values
      (NumPy scalars, tuples, sets) into plain Python types.

    See Also
    --------
    load_scaling_kwargs :
        Load scaling configuration from mapping or file.
    canonicalize_scaling_kwargs :
        Normalize keys and fill defaults consistently.
    enforce_scaling_alias_consistency :
        Ensure alias keys agree and do not conflict.
    validate_scaling_kwargs :
        Validate schema and value ranges.

    References
    ----------
    .. [1] Chollet, F. et al. "Keras: Deep Learning for Humans".
           Keras serialization and configuration patterns.
    .. [2] Python Software Foundation. "dataclasses - Data
           Classes" (Python standard library documentation).
    """
    # Raw payload (may be incomplete or aliased).
    payload: dict = field(default_factory=dict)

    # Optional provenance (e.g., file path).
    source: str | None = None

    # Schema version tag (for future migrations).
    schema_version: str = "1"

    @classmethod
    def from_any(cls, obj, *, copy=True):
        r"""
        Serializable container for GeoPrior scaling configuration.
        
        This dataclass stores a "payload" dictionary that holds all
        scaling and physics-control parameters required to reproduce
        the model behavior after saving and reloading with Keras.
        
        The container supports flexible construction from:
        - ``None`` (empty config),
        - a mapping (dict-like),
        - a file path ``str`` (loaded via ``load_scaling_kwargs``),
        - an existing :class:`~GeoPriorScalingConfig` instance.
        
        The canonical and validated configuration is produced by
        :meth:`resolve`, which applies the GeoPrior scaling pipeline:
        loading, canonicalization, alias consistency checks, and
        validation.
        
        Parameters
        ----------
        payload : dict, optional
            Raw scaling configuration payload. This may be incomplete
            or contain aliases prior to canonicalization.
        source : str or None, optional
            Optional provenance string, typically a file path used to
            load the payload. This is stored for traceability only.
        schema_version : str, optional
            Version label for the payload schema. This can be used
            to implement migrations when the scaling format evolves.
        
        Attributes
        ----------
        payload : dict
            The raw payload stored in this object.
        source : str or None
            The provenance hint, if provided.
        schema_version : str
            Schema version label.
        
        Notes
        -----
        - The resolved scaling dictionary returned by :meth:`resolve`
          is the one you should pass to the model internals.
        - ``get_config`` returns JSON-safe objects only. This avoids
          subtle reconstruction drift caused by non-serializable
          values.
        
        Examples
        --------
        Construct from a mapping:
        
        >>> cfg = GeoPriorScalingConfig.from_any(
        ...     {"coords_normalized": True}
        ... )
        >>> sk = cfg.resolve()
        >>> isinstance(sk, dict)
        True
        
        Construct from a file path:
        
        >>> cfg = GeoPriorScalingConfig.from_any(
        ...     "path/to/scaling_kwargs.json"
        ... )
        >>> sk = cfg.resolve()
        
        Use in a model constructor (pattern):
        
        >>> cfg = GeoPriorScalingConfig.from_any(scaling_kwargs)
        >>> scaling_kwargs_resolved = cfg.resolve()
        
        See Also
        --------
        GeoPriorScalingConfig.from_any :
            Build config from dict, path, or config instance.
        GeoPriorScalingConfig.resolve :
            Produce canonical and validated scaling dictionary.
        load_scaling_kwargs, canonicalize_scaling_kwargs :
            Scaling pipeline functions.
        
        References
        ----------
        .. [1] Chollet, F. et al. "Keras: Deep Learning for Humans".
               Keras object serialization via get_config/from_config.
        """
        
        r"""
        Create a scaling config from common input types.
        
        This factory method normalizes user input into a
        :class:`~GeoPriorScalingConfig` instance.
        
        Accepted inputs
        ---------------
        - ``None``: create an empty config.
        - :class:`~GeoPriorScalingConfig`: returned as-is.
        - ``str``: treated as a file path and loaded via
          :func:`load_scaling_kwargs`.
        - ``Mapping``: converted to a dict payload by default.
        
        Parameters
        ----------
        obj : object
            Scaling configuration input to normalize.
        copy : bool, optional
            If ``True``, copy mapping payloads into a new ``dict``.
            This helps avoid accidental mutation of user state.
        
        Returns
        -------
        cfg : GeoPriorScalingConfig
            A normalized config container.
        
        Raises
        ------
        TypeError
            If ``obj`` is not ``None``, ``str``, ``Mapping``, or a
            :class:`~GeoPriorScalingConfig` instance.
        
        Notes
        -----
        - When ``obj`` is a file path, the path is stored in the
          ``source`` attribute for traceability.
        - Canonicalization and validation happen in :meth:`resolve`,
          not in this constructor.
        
        Examples
        --------
        >>> GeoPriorScalingConfig.from_any(None)
        GeoPriorScalingConfig(payload={}, source=None, ...)
        
        >>> GeoPriorScalingConfig.from_any({"a": 1}).payload["a"]
        1
        """
        # ``None`` -> empty payload.
        if obj is None:
            logger.debug(
                "GeoPriorScalingConfig.from_any: obj=None",
            )
            return cls(payload={})

        # Already a config object.
        if isinstance(obj, cls):
            logger.debug(
                "GeoPriorScalingConfig.from_any: "
                "received GeoPriorScalingConfig",
            )
            return obj

        # Path-like: load via existing loader.
        if isinstance(obj, str):
            logger.info(
                "GeoPriorScalingConfig.from_any: "
                "loading scaling kwargs from path=%r",
                obj,
            )
            payload = load_scaling_kwargs(
                obj,
                copy=copy,
            )
            logger.debug(
                "GeoPriorScalingConfig.from_any: "
                "loaded keys=%d source=%r",
                len(payload),
                obj,
            )
            return cls(
                payload=payload,
                source=obj,
            )

        # Mapping-like: accept dict-like payload.
        if isinstance(obj, Mapping):
            logger.debug(
                "GeoPriorScalingConfig.from_any: "
                "received Mapping keys=%d copy=%s",
                len(obj),
                bool(copy),
            )
            payload = dict(obj) if copy else obj
            return cls(payload=payload)

        # Unsupported type.
        msg = (
            "Unsupported scaling_kwargs type: "
            f"{type(obj)!r}"
        )
        logger.error(
            "GeoPriorScalingConfig.from_any: %s",
            msg,
        )
        raise TypeError(msg)

    def resolve(self):
        r"""
        Resolve the payload into a canonical, validated scaling dict.
        
        This method runs the GeoPrior scaling pipeline and returns a
        dictionary suitable for direct use inside model computations.
        
        The pipeline is:
        1) Load payload (mapping or file-style behavior),
        2) Canonicalize keys and fill defaults,
        3) Enforce alias consistency,
        4) Validate values and required fields.
        
        Returns
        -------
        scaling_kwargs : dict
            Canonical and validated scaling configuration.
        
        Raises
        ------
        ValueError
            If validation fails due to missing keys or invalid values.
        KeyError
            If canonicalization expects keys that are absent.
        TypeError
            If the payload contains unsupported types.
        
        Notes
        -----
        - The returned dict is intended to be stable under Keras
          serialization and safe to store in model state.
        - This method always loads with ``copy=True`` to avoid
          mutating the stored payload.
        
        Examples
        --------
        >>> cfg = GeoPriorScalingConfig.from_any(
        ...     {"coords_normalized": True}
        ... )
        >>> sk = cfg.resolve()
        >>> sk["coords_normalized"]
        True
        
        See Also
        --------
        canonicalize_scaling_kwargs :
            Normalizes scaling keys and defaults.
        validate_scaling_kwargs :
            Enforces schema and constraints.
        enforce_scaling_alias_consistency :
            Prevents conflicting aliases.
        """
        logger.debug(
            "GeoPriorScalingConfig.resolve: start "
            "(source=%r, schema_version=%r)",
            self.source,
            self.schema_version,
        )

        # Load payload defensively (copy).
        sk = load_scaling_kwargs(
            self.payload,
            copy=True,
        )
        logger.debug(
            "GeoPriorScalingConfig.resolve: loaded "
            "payload keys=%d",
            len(sk),
        )

        # Normalize keys and fill defaults.
        sk = canonicalize_scaling_kwargs(sk)
        logger.debug(
            "GeoPriorScalingConfig.resolve: "
            "canonicalized keys=%d",
            len(sk),
        )

        # Enforce alias agreement (no conflicts).
        enforce_scaling_alias_consistency(sk)
        logger.debug(
            "GeoPriorScalingConfig.resolve: "
            "alias consistency OK",
        )

        # Validate schema and value ranges.
        validate_scaling_kwargs(sk)
        logger.info(
            "GeoPriorScalingConfig.resolve: OK "
            "(keys=%d, source=%r)",
            len(sk),
            self.source,
        )

        return sk

    def get_config(self):
        r"""
        Return a JSON-safe Keras configuration dictionary.

        Keras uses this method to serialize the object. The returned
        dictionary must contain only JSON-serializable values.

        This implementation uses :func:`_jsonify` to defensively
        convert nested structures such as NumPy scalars, tuples, and
        sets into plain Python types.

        Returns
        -------
        config : dict
            JSON-safe configuration dictionary with the following
            keys:
            - ``payload``: JSON-safe payload mapping,
            - ``source``: provenance hint (may be ``None``),
            - ``schema_version``: schema version label.

        Notes
        -----
        - ``source`` is stored for traceability and does not affect
          :meth:`resolve`.
        - When saved as part of a model config, this makes scaling
          reconstruction deterministic.

        See Also
        --------
        GeoPriorScalingConfig.from_config :
            Recreate a config instance from this dictionary.
        """
        cfg = {
            "payload": _jsonify(self.payload),
            "source": self.source,
            "schema_version": self.schema_version,
        }
        logger.debug(
            "GeoPriorScalingConfig.get_config: "
            "payload keys=%d source=%r",
            len(cfg.get("payload", {})),
            self.source,
        )
        return cfg

    @classmethod
    def from_config(cls, config):
        r"""
        Recreate an instance from a Keras configuration dictionary.

        This class method is used by Keras deserialization to rebuild
        the object from the dictionary returned by :meth:`get_config`.

        Parameters
        ----------
        config : dict
            Configuration dictionary produced by :meth:`get_config`.

        Returns
        -------
        cfg : GeoPriorScalingConfig
            Reconstructed config instance.

        Notes
        -----
        - This method does not call :meth:`resolve`. Resolution is
          deferred to the consumer so that reconstruction remains
          explicit and testable.

        See Also
        --------
        GeoPriorScalingConfig.get_config :
            Produces the configuration dictionary.
        """
        logger.debug(
            "GeoPriorScalingConfig.from_config: "
            "keys=%s",
            sorted(list(config.keys())),
        )
        return cls(**config)

def _deep_update(base: dict, over: dict) -> dict:
    """
    Deep-merge nested dicts.

    Override wins. Update is in-place; returns `base`.
    """
    if not over:
        return base
    for k, v in over.items():
        is_d = isinstance(v, dict)
        is_b = isinstance(base.get(k), dict)
        if is_d and is_b:
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_json_path(
    path: str,
    base_dir: Optional[str],
) -> str:
    """
    Resolve a JSON path.

    Expands env vars and `~`. Relative paths are
    resolved against `base_dir` (or CWD).
    """
    p = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(p):
        return p
    b = base_dir or os.getcwd()
    return os.path.abspath(os.path.join(b, p))


def override_scaling_kwargs(
    sk: Mapping[str, Any],
    cfg: Optional[Mapping[str, Any]],
    *,
    finalize: Optional[Callable[[dict], dict]] = None,
    dyn_names: Optional[Sequence[str]] = None,
    gwl_dyn_index: Optional[int] = None,
    base_dir: Optional[str] = None,
    path_key: str = "SCALING_KWARGS_JSON_PATH",
    strict: bool = True,
    add_path: bool = True,
    log_fn: Optional[Callable[[str], Any]] = None,
) -> dict:
    r"""
    Override ``scaling_kwargs`` from a JSON file or dict.
    
    This helper applies an optional, precedence-based override to
    an existing ``scaling_kwargs`` mapping. The override source is
    read from ``cfg[path_key]``. If the key is missing or empty, the
    input ``sk`` is returned (optionally finalized).
    
    The override can be provided as:
    
    - a file path to a JSON object (mapping), or
    - a Python dict-like mapping embedded in ``cfg``.
    
    Overrides are applied via a deep-merge strategy:
    
    - for nested dict values, keys are merged recursively,
    - for non-dict values, the override replaces the base value.
    
    Optionally, the merged result is passed through ``finalize`` to
    recompute derived or canonical fields (for example, coordinate
    ranges, unit flags, or other normalization metadata).
    
    Parameters
    ----------
    sk : Mapping[str, Any]
        Base scaling configuration (``scaling_kwargs``). This is
        typically computed by Stage-2 or loaded from Stage-1 output.
        The input is copied to a plain ``dict`` before modification.
    
    cfg : Mapping[str, Any] or None
        Configuration mapping that may contain the override source
        under ``path_key``. If ``None``, no override is applied.
    
    finalize : callable or None, optional
        Function applied to the scaling dict to enforce canonical
        structure or to compute derived fields. If provided, it is
        applied before and after the override merge:
    
        - pre-merge: normalize the base dict,
        - post-merge: ensure the merged dict is consistent.
    
        The callable must accept a dict and return a dict.
    
    dyn_names : Sequence[str] or None, optional
        Expected dynamic feature names for safety validation. If
        provided and the override contains ``dynamic_feature_names``,
        the two sequences are compared. A mismatch raises an error
        when ``strict=True``.
    
    gwl_dyn_index : int or None, optional
        Expected dynamic index for the groundwater-level feature.
        If provided and the override contains ``gwl_dyn_index``, the
        values are compared. A mismatch raises an error when
        ``strict=True``.
    
    base_dir : str or None, optional
        Base directory used to resolve relative JSON paths. If
        ``None``, the current working directory is used.
    
    path_key : str, default="SCALING_KWARGS_JSON_PATH"
        Name of the key in ``cfg`` that specifies the override. The
        value may be a dict-like mapping or a path to a JSON file.
    
    strict : bool, default=True
        Controls behavior on safety-check mismatches. When ``True``,
        mismatches raise a ``ValueError``. When ``False``, mismatches
        can be logged via ``log_fn`` and the override still proceeds.
    
    add_path : bool, default=True
        If ``True``, store the resolved override source in the output
        dict under ``scaling_kwargs_override_path``. When the override
        is provided as a mapping (not a file), the value is set to
        ``"<dict>"``.
    
    log_fn : callable or None, optional
        Optional logger function. If provided, it is called with
        informative messages such as successful override application
        and (when ``strict=False``) mismatch warnings. Common choices
        are ``print`` or ``logger.info``.
    
    Returns
    -------
    out : dict
        Final scaling dict after optional override and optional
        finalization. The returned dict is independent from the input
        mapping object ``sk`` (a copy is always created).
    
    Raises
    ------
    FileNotFoundError
        If ``cfg[path_key]`` is a path and the file does not exist.
    
    ValueError
        If a path is provided but the file does not contain valid
        JSON, or if a safety check fails while ``strict=True``.
    
    TypeError
        If the loaded override is not a JSON object (dict-like).
    
    Notes
    -----
    Path resolution
        When ``cfg[path_key]`` is a string path, it is resolved as:
    
        1. Expand environment variables and ``~``.
        2. If relative, join with ``base_dir`` (or CWD).
    
    Safety checks
        The checks are intentionally conservative. They prevent using
        an override file produced for a different dataset or feature
        layout. Recommended checks are:
    
        - ``dynamic_feature_names`` equality when known.
        - ``gwl_dyn_index`` equality when known.
    
        You can extend validation by checking additional keys such as
        ``coord_epsg_used``, ``coords_normalized``, or unit flags.
    
    Finalization
        In GeoPrior pipelines, ``finalize`` is typically a helper that
        enforces defaults and recomputes derived entries. Applying it
        both before and after the override helps reduce edge cases
        where the override only supplies partial information.
    
    Examples
    --------
    Stage-2: override computed scaling with a file
        In Stage-2, call this right after the auto-computed scaling
        is available, so the override takes precedence:
    
        >>> sk = subsmodel_params["scaling_kwargs"]
        >>> sk = override_scaling_kwargs(
        ...     sk,
        ...     cfg,
        ...     finalize=finalize_scaling_kwargs,
        ...     dyn_names=DYN_NAMES,
        ...     gwl_dyn_index=GWL_DYN_INDEX,
        ...     base_dir=os.path.dirname(__file__),
        ...     strict=True,
        ...     log_fn=print,
        ... )
        >>> subsmodel_params["scaling_kwargs"] = sk
    
    Stage-3: override Stage-1 scaling prior to enforcing bounds
        In Stage-3, apply the override before injecting Stage-3 bounds:
    
        >>> sk_model = dict(cfg.get("scaling_kwargs", {}) or {})
        >>> sk_model = override_scaling_kwargs(
        ...     sk_model,
        ...     cfg,
        ...     dyn_names=sk_model.get("dynamic_feature_names"),
        ...     gwl_dyn_index=sk_model.get("gwl_dyn_index"),
        ...     base_dir=os.path.dirname(__file__),
        ... )
        >>> sk_model["bounds"] = {
        ...     **(sk_model.get("bounds", {}) or {}),
        ...     **bounds_for_scaling,
        ... }
    
    Inline dict override (no JSON file)
        If the override is embedded in config, it is used directly:
    
        >>> cfg = {
        ...     "SCALING_KWARGS_JSON_PATH": {
        ...         "coords_normalized": True,
        ...         "coord_ranges": {"t": 7.0, "x": 1000.0, "y": 900.0},
        ...     }
        ... }
        >>> out = override_scaling_kwargs({}, cfg)
    
    See Also
    --------
    finalize_scaling_kwargs :
        Canonicalize and complete ``scaling_kwargs`` entries.
    compute_scaling_kwargs :
        Build a base scaling dict from data and pipeline settings.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics
       environment. Computing in Science and Engineering, 9(3),
       90-95.
    """

    base = dict(sk) if sk is not None else {}
    if finalize is not None:
        base = finalize(base)

    cfg = cfg or {}
    raw = cfg.get(path_key, None)
    if raw in (None, "", False):
        return base

    if isinstance(raw, Mapping):
        over = dict(raw)
        over_path = "<dict>"
    else:
        over_path = _resolve_json_path(str(raw), base_dir)
        if not os.path.isfile(over_path):
            raise FileNotFoundError(
                f"{path_key} not found: {over_path}"
            )
        try:
            with open(over_path, "r", encoding="utf-8") as f:
                over = json.load(f)
        except Exception as e:
            raise ValueError(
                f"Invalid JSON: {over_path}"
            ) from e

    if not isinstance(over, dict):
        raise TypeError("Override must be a JSON object.")

    if dyn_names and "dynamic_feature_names" in over:
        names = list(over["dynamic_feature_names"])
        if names != list(dyn_names):
            msg = "Override mismatch: dynamic_feature_names."
            if strict:
                raise ValueError(msg)
            if log_fn:
                log_fn(msg)

    if gwl_dyn_index is not None and "gwl_dyn_index" in over:
        ov = int(over["gwl_dyn_index"])
        if ov != int(gwl_dyn_index):
            msg = "Override mismatch: gwl_dyn_index."
            if strict:
                raise ValueError(msg)
            if log_fn:
                log_fn(msg)

    out = _deep_update(base, over)

    if finalize is not None:
        out = finalize(out)

    if add_path:
        out["scaling_kwargs_override_path"] = over_path

    if log_fn:
        log_fn(f"[INFO] scaling_kwargs override: {over_path}")

    return out








