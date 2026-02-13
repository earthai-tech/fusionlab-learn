## -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author : LKouadio <etanoyau@gmail.com>

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple


import numpy as np 

__all__ = [
    "canonicalize_BHQO",
    "canonicalize_BHQO_quantiles_np" 
]


def canonicalize_BHQO(
    y_pred: Any,
    *,
    y_true: Any | None = None,
    q_values: Sequence[float] = (0.1, 0.5, 0.9),
    n_q: int | None = None,
    layout: str | None = None,
    enforce_monotone: bool = True,
    return_layout: bool = False,
    verbose: int = 0,
    log_fn: Callable[[str], None] = print,
) -> Any:
    """
    Canonicalize quantile outputs to (B, H, Q, O).

    Supported layouts (rank-4):
      - BHQO: (B, H, Q, O) -> unchanged
      - BQHO: (B, Q, H, O) -> transpose(0, 2, 1, 3)
      - BHOQ: (B, H, O, Q) -> transpose(0, 1, 3, 2)

    If ambiguous (e.g., H == Q), and y_true is given,
    pick the transform with smallest MAE for q50.

    If y_true is not given, fallback is:
      1) use `layout` if provided
      2) else prefer BHQO if plausible
      3) else pick by min crossing score

    Parameters
    ----------
    y_pred:
        Quantile tensor, NumPy array or TF tensor.
    y_true:
        Target tensor (B, H, O) or (B, H, 1).
        Used only to resolve ambiguity robustly.
    q_values:
        Quantiles in order, e.g. (0.1, 0.5, 0.9).
    n_q:
        Number of quantiles. Defaults to len(q_values).
    layout:
        Force interpretation: "BHQO", "BQHO", "BHOQ".
    enforce_monotone:
        Sort along Q axis after canonicalization.
    return_layout:
        If True, return (arr, chosen_layout).
    verbose, log_fn:
        Logging controls.

    Returns
    -------
    arr or (arr, layout)
        Canonical (B, H, Q, O) and optionally the layout.
    """
    tf = _maybe_tf()
    if tf is not None and tf.is_tensor(y_pred):
        out = _canonicalize_tf(
            y_pred,
            y_true=y_true,
            q_values=q_values,
            n_q=n_q,
            layout=layout,
            enforce_monotone=enforce_monotone,
            verbose=verbose,
            log_fn=log_fn,
        )
        if return_layout:
            return out
        return out[0]

    out = _canonicalize_np(
        y_pred,
        y_true=y_true,
        q_values=q_values,
        n_q=n_q,
        layout=layout,
        enforce_monotone=enforce_monotone,
        verbose=verbose,
        log_fn=log_fn,
    )
    if return_layout:
        return out
    return out[0]


# ---------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------

def _canonicalize_np(
    y_pred: Any,
    *,
    y_true: Any | None,
    q_values: Sequence[float],
    n_q: int | None,
    layout: str | None,
    enforce_monotone: bool,
    verbose: int,
    log_fn: Callable[[str], None],
) -> tuple[np.ndarray, str]:
    y = np.asarray(y_pred)

    y = _ensure_rank4_np(
        y,
        verbose=verbose,
        log_fn=log_fn,
    )
    if y.ndim != 4:
        return y, "UNCHANGED"

    n_q = int(n_q or len(q_values))
    if n_q <= 0:
        raise ValueError("n_q must be > 0.")

    if layout is not None:
        arr = _apply_layout_np(y, layout)
        arr = _post_np(arr, enforce_monotone)
        return arr, layout

    opts = _build_options_np(y, n_q)
    if not opts:
        if verbose:
            log_fn(
                "canonicalize_BHQO: no options; "
                "return unchanged."
            )
        return y, "UNCHANGED"

    if len(opts) == 1:
        name, arr = opts[0]
        arr = _post_np(arr, enforce_monotone)
        return arr, name

    yt = None
    if y_true is not None:
        yt = np.asarray(y_true)
        yt = _ensure_ytrue_np(yt)

    med = _median_q_index(q_values)

    best = _pick_best_np(
        opts,
        y_true=yt,
        med=med,
        verbose=verbose,
        log_fn=log_fn,
    )
    name, arr = best
    arr = _post_np(arr, enforce_monotone)
    return arr, name


def _build_options_np(
    y: np.ndarray,
    n_q: int,
) -> list[tuple[str, np.ndarray]]:
    opts: list[tuple[str, np.ndarray]] = []

    if y.shape[2] == n_q:
        opts.append(("BHQO", y))

    if y.shape[1] == n_q:
        opts.append(
            (
                "BQHO",
                np.transpose(y, (0, 2, 1, 3)),
            )
        )

    if y.shape[3] == n_q:
        opts.append(
            (
                "BHOQ",
                np.transpose(y, (0, 1, 3, 2)),
            )
        )

    return opts


def _pick_best_np(
    opts: list[tuple[str, np.ndarray]],
    *,
    y_true: np.ndarray | None,
    med: int,
    verbose: int,
    log_fn: Callable[[str], None],
) -> tuple[str, np.ndarray]:
    best_name = opts[0][0]
    best_arr = opts[0][1]
    best_score = float("inf")

    for name, arr in opts:
        mae = None
        if y_true is not None:
            mae = _mae_q50_np(arr, y_true, med)

        cross = _cross_score_np(arr)

        if y_true is not None:
            score = float(mae)
        else:
            score = float(cross)

        if verbose >= 2:
            log_fn(
                "canonicalize_BHQO: "
                f"cand={name} "
                f"mae={_fmt(mae)} "
                f"cross={cross:.6f}"
            )

        if score < best_score:
            best_score = score
            best_name = name
            best_arr = arr

    if verbose:
        log_fn(
            "canonicalize_BHQO: "
            f"chose={best_name} "
            f"score={best_score:.6f}"
        )

    return best_name, best_arr


def _cross_score_np(arr: np.ndarray) -> float:
    q10 = arr[:, :, 0, :]
    q50 = arr[:, :, 1, :]
    q90 = arr[:, :, 2, :]

    if q10.shape[-1] == 1:
        q10 = q10[..., 0]
        q50 = q50[..., 0]
        q90 = q90[..., 0]

    c1 = float(np.mean(q10 > q50))
    c2 = float(np.mean(q50 > q90))
    c3 = float(np.mean(q10 > q90))
    return c1 + c2 + c3


def _mae_q50_np(
    arr: np.ndarray,
    y_true: np.ndarray,
    med: int,
) -> float:
    q50 = arr[:, :, med, :]
    yt = y_true

    if q50.shape != yt.shape:
        return float("inf")

    return float(np.mean(np.abs(q50 - yt)))


def _ensure_rank4_np(
    y: np.ndarray,
    *,
    verbose: int,
    log_fn: Callable[[str], None],
) -> np.ndarray:
    if y.ndim == 3:
        return y[..., None]
    if y.ndim != 4:
        if verbose:
            log_fn(
                "canonicalize_BHQO: "
                f"rank={y.ndim}, skipping."
            )
    return y


def _ensure_ytrue_np(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2:
        return y[..., None]
    return y


def _apply_layout_np(
    y: np.ndarray,
    layout: str,
) -> np.ndarray:
    lay = layout.upper().strip()
    if lay == "BHQO":
        return y
    if lay == "BQHO":
        return np.transpose(y, (0, 2, 1, 3))
    if lay == "BHOQ":
        return np.transpose(y, (0, 1, 3, 2))
    raise ValueError(
        "layout must be one of: "
        "BHQO, BQHO, BHOQ."
    )


def _post_np(
    arr: np.ndarray,
    enforce_monotone: bool,
) -> np.ndarray:
    if enforce_monotone:
        return np.sort(arr, axis=2)
    return arr


# ---------------------------------------------------------------------
# TensorFlow backend (lazy import; eager-first)
# ---------------------------------------------------------------------


def _canonicalize_tf(
    y_pred: Any,
    *,
    y_true: Any | None,
    q_values: Sequence[float],
    n_q: int | None,
    layout: str | None,
    enforce_monotone: bool,
    verbose: int,
    log_fn: Callable[[str], None],
) -> tuple[Any, str]:
    tf = _maybe_tf()
    if tf is None:
        arr = np.asarray(y_pred)
        return arr, "UNCHANGED"

    y = y_pred
    if y.shape.rank == 3:
        y = tf.expand_dims(y, axis=-1)

    if y.shape.rank != 4:
        if verbose:
            log_fn(
                "canonicalize_BHQO: "
                f"rank={y.shape.rank}, skipping."
            )
        return y, "UNCHANGED"

    n_q = int(n_q or len(q_values))
    if n_q <= 0:
        raise ValueError("n_q must be > 0.")

    if layout is not None:
        arr = _apply_layout_tf(y, layout)
        arr = _post_tf(arr, enforce_monotone)
        return arr, layout

    opts = _build_options_tf(y, n_q)
    if not opts:
        if verbose:
            log_fn(
                "canonicalize_BHQO: "
                "no options; unchanged."
            )
        return y, "UNCHANGED"

    if len(opts) == 1:
        name, arr = opts[0]
        arr = _post_tf(arr, enforce_monotone)
        return arr, name

    if not tf.executing_eagerly():
        name, arr = _prefer_bhqo(opts)
        arr = _post_tf(arr, enforce_monotone)
        return arr, name

    yt = None
    if y_true is not None:
        yt = y_true
        if not tf.is_tensor(yt):
            yt = tf.convert_to_tensor(yt)
        if yt.shape.rank == 2:
            yt = tf.expand_dims(yt, axis=-1)

    med = _median_q_index(q_values)

    best_name, best_arr = _pick_best_tf(
        opts,
        y_true=yt,
        med=med,
        verbose=verbose,
        log_fn=log_fn,
    )
    best_arr = _post_tf(best_arr, enforce_monotone)
    return best_arr, best_name


def _build_options_tf(
    y: Any,
    n_q: int,
) -> list[tuple[str, Any]]:
    opts: list[tuple[str, Any]] = []
    shp = y.shape

    if shp[2] == n_q:
        opts.append(("BHQO", y))

    if shp[1] == n_q:
        opts.append(
            (
                "BQHO",
                _maybe_tf().transpose(y, [0, 2, 1, 3]),
            )
        )

    if shp[3] == n_q:
        opts.append(
            (
                "BHOQ",
                _maybe_tf().transpose(y, [0, 1, 3, 2]),
            )
        )

    return opts


def _pick_best_tf(
    opts: list[tuple[str, Any]],
    *,
    y_true: Any | None,
    med: int,
    verbose: int,
    log_fn: Callable[[str], None],
) -> tuple[str, Any]:
    # tf = _maybe_tf()

    best_name = opts[0][0]
    best_arr = opts[0][1]
    best_score = float("inf")

    for name, arr in opts:
        mae = None
        if y_true is not None:
            mae = _mae_q50_tf(arr, y_true, med)
            mae = float(mae.numpy())

        cross = float(_cross_score_tf(arr).numpy())

        score = float(mae) if y_true is not None else cross

        if verbose >= 2:
            log_fn(
                "canonicalize_BHQO: "
                f"cand={name} "
                f"mae={_fmt(mae)} "
                f"cross={cross:.6f}"
            )

        if score < best_score:
            best_score = score
            best_name = name
            best_arr = arr

    if verbose:
        log_fn(
            "canonicalize_BHQO: "
            f"chose={best_name} "
            f"score={best_score:.6f}"
        )

    return best_name, best_arr


def _cross_score_tf(arr: Any) -> Any:
    tf = _maybe_tf()

    q10 = arr[:, :, 0, :]
    q50 = arr[:, :, 1, :]
    q90 = arr[:, :, 2, :]

    if q10.shape.rank is not None and q10.shape[-1] == 1:
        q10 = q10[..., 0]
        q50 = q50[..., 0]
        q90 = q90[..., 0]

    c1 = tf.reduce_mean(tf.cast(q10 > q50, tf.float32))
    c2 = tf.reduce_mean(tf.cast(q50 > q90, tf.float32))
    c3 = tf.reduce_mean(tf.cast(q10 > q90, tf.float32))
    return c1 + c2 + c3


def _mae_q50_tf(
    arr: Any,
    y_true: Any,
    med: int,
) -> Any:
    tf = _maybe_tf()
    q50 = arr[:, :, med, :]

    if q50.shape.rank != y_true.shape.rank:
        return tf.constant(np.inf, dtype=tf.float32)

    if (
        q50.shape.rank is not None
        and y_true.shape.rank is not None
        and q50.shape.rank == 3
    ):
        if q50.shape[1] != y_true.shape[1]:
            return tf.constant(np.inf, dtype=tf.float32)

    return tf.reduce_mean(tf.abs(q50 - y_true))


def _apply_layout_tf(
    y: Any,
    layout: str,
) -> Any:
    tf = _maybe_tf()
    lay = layout.upper().strip()
    if lay == "BHQO":
        return y
    if lay == "BQHO":
        return tf.transpose(y, [0, 2, 1, 3])
    if lay == "BHOQ":
        return tf.transpose(y, [0, 1, 3, 2])
    raise ValueError(
        "layout must be one of: "
        "BHQO, BQHO, BHOQ."
    )


def _post_tf(
    arr: Any,
    enforce_monotone: bool,
) -> Any:
    tf = _maybe_tf()
    if enforce_monotone:
        return tf.sort(arr, axis=2)
    return arr


def _prefer_bhqo(
    opts: list[tuple[str, Any]],
) -> tuple[str, Any]:
    for name, arr in opts:
        if name == "BHQO":
            return name, arr
    return opts[0]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _maybe_tf() -> Any | None:
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return None
    return tf


def _median_q_index(
    q_values: Sequence[float],
) -> int:
    q = np.asarray(q_values, dtype=float)
    return int(np.argmin(np.abs(q - 0.5)))


def _fmt(v: Any) -> str:
    if v is None:
        return "None"
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v)

def canonicalize_BHQO_quantiles_np(
    y: Any,
    n_q: int = 3,
    *,
    verbose: int = 0,
    log_fn: Callable[[str], None] = print,
) -> Any:
    """
    Return y in canonical (B,H,Q,O).

    Accepts common layouts:
      - (B,H,Q,O) -> unchanged
      - (B,Q,H,O) -> transpose(0,2,1,3)
      - (B,H,O,Q) -> transpose(0,1,3,2)

    If ambiguous (multiple axes match n_q), choose the transform
    with minimal quantile crossing score.
    """
    y_np = np.asarray(y)

    # Not quantile tensor (we only canonicalize rank-4).
    if y_np.ndim != 4:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"rank={y_np.ndim}, skipping."
            )
        return y

    n_q = int(n_q)
    if n_q <= 0:
        raise ValueError("n_q must be a positive integer.")

    # candidates: interpret which axis is Q among {1,2,3}
    cand = [ax for ax in (1, 2, 3) if y_np.shape[ax] == n_q]

    # No axis matches n_q => not quantile mode, return unchanged.
    if not cand:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"no axis matches n_q={n_q}, "
                "return unchanged."
            )
        return y_np

    options: list[tuple[str, np.ndarray]] = []

    # already (B,H,Q,O): q_axis=2 and O axis=3
    if y_np.shape[2] == n_q:
        # Keep as-is. This is the canonical layout.
        options.append(("BHQO", y_np))

    # (B,Q,H,O) -> (B,H,Q,O)
    if y_np.shape[1] == n_q:
        # Swap axes 1 and 2.
        options.append(
            (
                "BQHO->BHQO",
                _safe_transpose(y_np, (0, 2, 1, 3)),
            )
        )

    # (B,H,O,Q) -> (B,H,Q,O)
    if y_np.shape[3] == n_q:
        # Swap axes 2 and 3.
        options.append(
            (
                "BHOQ->BHQO",
                _safe_transpose(y_np, (0, 1, 3, 2)),
            )
        )

    # If nothing matched the supported transforms, keep unchanged.
    if not options:
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                "no supported transform matched; "
                "return unchanged."
            )
        return y_np

    # If only one option, pick it directly.
    if len(options) == 1:
        name, arr = options[0]
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"chose={name} (only option)."
            )
        return arr

    # Multiple candidates (e.g., H==Q==3):
    # pick best by minimal quantile crossing score.
    best_name: Optional[str] = None
    best_arr: Optional[np.ndarray] = None
    best_score = float("inf")

    for name, arr in options:
        # score in canonical BHQO along axis=2
        sc = _mean_crossing_score(arr, q_axis=2)
        if verbose >= 2:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                f"candidate={name} score={sc:.6f}"
            )

        if sc < best_score:
            best_score = sc
            best_name = name
            best_arr = arr

    if best_arr is None:
        # Defensive fallback; should not happen.
        if verbose:
            log_fn(
                "canonicalize_BHQO_quantiles_np: "
                "no best candidate found; "
                "return unchanged."
            )
        return y_np

    if verbose:
        log_fn(
            "canonicalize_BHQO_quantiles_np: "
            f"chose={best_name} score={best_score:.6f}"
        )

    return best_arr


def _mean_crossing_score(
    arr: np.ndarray,
    q_axis: int,
    q_idx: Sequence[int] = (0, 1, 2),
) -> float:
    # Compute quantile crossing score:
    #   mean(q10>q50) + mean(q50>q90) + mean(q10>q90)
    q10 = np.take(arr, int(q_idx[0]), axis=q_axis)
    q50 = np.take(arr, int(q_idx[1]), axis=q_axis)
    q90 = np.take(arr, int(q_idx[2]), axis=q_axis)

    # squeeze last dim if O=1
    if q10.ndim >= 1 and q10.shape[-1] == 1:
        q10 = q10[..., 0]
        q50 = q50[..., 0]
        q90 = q90[..., 0]

    c1 = float(np.mean(q10 > q50))
    c2 = float(np.mean(q50 > q90))
    c3 = float(np.mean(q10 > q90))
    return c1 + c2 + c3


def _safe_transpose(
    y: np.ndarray,
    axes: Tuple[int, int, int, int],
) -> np.ndarray:
    return np.transpose(y, axes)
