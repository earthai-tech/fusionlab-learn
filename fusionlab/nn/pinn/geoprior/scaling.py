# -*- coding: utf-8 -*-
# Author: LKouadio <etanoyau@gmail.com>
# License: BSD-3-Clause

r"""GeoPrior scaling config helpers (Keras-serializable)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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











