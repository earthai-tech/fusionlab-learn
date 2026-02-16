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

def canonicalize_to_BHQO_using_contract(
    s_pred,
    *,
    q_values=(0.1, 0.5, 0.9),
    enforce_monotone=True,
    verbose=0,
    log_fn=print,
):
    """
    Canonicalize quantile tensor to (B,H,Q,O).

    Accepts common layouts:
      - (B,H,Q,O) : unchanged
      - (B,Q,H,O) : transpose -> (B,H,Q,O)
      - (B,H,O,Q) : transpose -> (B,H,Q,O)
      - rank-3 (B,H,Q) / (B,Q,H) / (B,H,O):
          expanded with O=1, then canonicalized

    If enforce_monotone=True, quantiles are sorted
    along axis=2 (Q axis).

    Notes
    -----
    - TF backend returns a tf.Tensor.
    - NumPy backend returns a np.ndarray.
    """

    def _log(level: int, msg: str) -> None:
        if verbose >= level:
            log_fn(msg)

    tf = _maybe_tf()
    n_q = len(q_values) if q_values else 0

    # ==================================================
    # NumPy backend
    # ==================================================
    if tf is None:
        s = np.asarray(s_pred)
        _log(1, f"canon_BHQO[np]: in={s.shape}")

        if n_q <= 0:
            _log(1, "canon_BHQO[np]: n_q<=0, keep.")
            return s

        # Rank-3 -> add O axis (O=1).
        if s.ndim == 3:
            s = np.expand_dims(s, axis=-1)
            _log(
                2,
                "canon_BHQO[np]: "
                f"expanded rank-3 -> {s.shape}",
            )

        if s.ndim != 4:
            _log(
                1,
                "canon_BHQO[np]: "
                f"rank={s.ndim}, unchanged.",
            )
            return s

        b, d1, d2, d3 = s.shape
        _log(
            2,
            "canon_BHQO[np]: "
            f"d1={d1} d2={d2} d3={d3} n_q={n_q}",
        )

        # Accept BHQO
        if d2 == n_q:
            out = s
            _log(1, "canon_BHQO[np]: BHQO keep.")

        # Accept BQHO -> transpose to BHQO
        elif d1 == n_q:
            out = np.transpose(s, (0, 2, 1, 3))
            _log(1, "canon_BHQO[np]: BQHO swap.")

        # Accept BHOQ -> transpose to BHQO
        elif d3 == n_q:
            out = np.transpose(s, (0, 1, 3, 2))
            _log(1, "canon_BHQO[np]: BHOQ move.")

        else:
            raise ValueError(
                "Cannot locate quantile axis. "
                f"shape={s.shape}, n_q={n_q}"
            )

        if enforce_monotone:
            out = np.sort(out, axis=2)
            _log(2, "canon_BHQO[np]: sorted axis=2.")

        _log(1, f"canon_BHQO[np]: out={out.shape}")
        return out


    # TensorFlow backend (safe conversion + verbose)

    if n_q <= 0:
        _log(1, "canon_BHQO[tf]: n_q<=0, keep.")
        return s_pred

    if not tf.is_tensor(s_pred):
        s_pred = tf.convert_to_tensor(s_pred)

    rank = int(s_pred.shape.rank or 0)
    _log(
        1,
        "canon_BHQO[tf]: "
        f"in rank={rank} "
        f"shape={tuple(s_pred.shape.as_list())}",
    )

    # Rank-3 -> add O axis (O=1).
    if s_pred.shape.rank == 3:
        s_pred = tf.expand_dims(s_pred, axis=-1)
        _log(
            2,
            "canon_BHQO[tf]: "
            f"expanded rank-3 -> "
            f"{tuple(s_pred.shape.as_list())}",
        )

    if s_pred.shape.rank != 4:
        return s_pred

    # Accept BHQO
    if int(s_pred.shape[2]) == n_q:
        out = s_pred
        _log(1, "canon_BHQO[tf]: BHQO keep.")

    # Accept BQHO -> transpose to BHQO
    elif int(s_pred.shape[1]) == n_q:
        out = tf.transpose(s_pred, [0, 2, 1, 3])
        _log(1, "canon_BHQO[tf]: BQHO swap.")

    # Accept BHOQ -> transpose to BHQO
    elif int(s_pred.shape[3]) == n_q:
        out = tf.transpose(s_pred, [0, 1, 3, 2])
        _log(1, "canon_BHQO[tf]: BHOQ move.")

    else:
        raise ValueError(
            "Cannot locate quantile axis. "
            f"shape={s_pred.shape}, n_q={n_q}"
        )

    if enforce_monotone:
        out = tf.sort(out, axis=2)
        _log(2, "canon_BHQO[tf]: sorted axis=2.")

    return out

def ensure_subs_bhq(
    s_pred_b,
    *,
    y_true_b,
    q_values,
    enforce_monotone=True,
    verbose=0,
    log_fn=print,
):
    """
    Ensure subsidence quantile predictions are in (B,H,Q,O).

    Expected quantile mode shape is rank-4, typically one of:
      - (B,H,Q,O)  : canonical
      - (B,Q,H,O)  : swap axes 1<->2
    Ambiguous case (often H == Q == n_q):
      - choose orientation by q50 MAE vs y_true_b
      - if MAE cannot be computed (shape mismatch), fallback to
        a quantile-crossing score (smaller is better)

    Notes
    -----
    - If enforce_monotone=True, we sort along Q axis (axis=2).
    - NumPy backend returns np.ndarray.
    """

    def _log(level: int, msg: str) -> None:
        if verbose >= level:
            log_fn(msg)

    def _mae_safe_np(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return float("inf")
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) == 0:
            return float("inf")
        return float(np.mean(np.abs(a[mask] - b[mask])))

    def _cross_score_np(arr: np.ndarray) -> float:
        # arr is expected (B,H,Q,O)
        if arr.ndim != 4 or arr.shape[2] < 2:
            return float("inf")
        # Adjacent crossings + end-to-end crossing.
        a = arr[:, :, :-1, :]
        b = arr[:, :, 1:, :]
        c_adj = float(np.mean(a > b))
        c_end = float(
            np.mean(arr[:, :, 0, :] > arr[:, :, -1, :])
        )
        return c_adj + c_end

    def _ensure_ytrue_bho_np(
        y_true,
        *,
        n_q: int,
        med: int,
    ) -> np.ndarray | None:
        if y_true is None:
            return None
        try:
            yt = np.asarray(y_true)
        except Exception:
            return None

        # (B,H) -> (B,H,1)
        if yt.ndim == 2:
            return yt[..., None]

        # (B,H,O) OK
        if yt.ndim == 3:
            return yt

        # If y_true was tiled to quantile shape, drop to med.
        if yt.ndim == 4:
            if yt.shape[2] == n_q:
                return yt[:, :, med, :]
            if yt.shape[1] == n_q:
                yt2 = np.transpose(yt, (0, 2, 1, 3))
                return yt2[:, :, med, :]
            if yt.shape[3] == n_q:
                yt2 = np.transpose(yt, (0, 1, 3, 2))
                return yt2[:, :, med, :]

            # Last-resort squeeze trailing singleton.
            if yt.shape[-1] == 1:
                return yt[..., 0]

        return None

    tf = _maybe_tf()

    # NumPy backend
    if tf is None:
        s = np.asarray(s_pred_b)
        _log(1, f"ensure_subs_bhq[np]: in={s.shape}")

        # Expect rank-4 in quantile mode.
        if s.ndim != 4:
            _log(
                1,
                "ensure_subs_bhq[np]: "
                f"rank={s.ndim}, unchanged.",
            )
            return s

        n_q = len(q_values) if q_values else 0
        if n_q <= 0:
            _log(
                1,
                "ensure_subs_bhq[np]: "
                "n_q<=0, unchanged.",
            )
            return s

        d1 = int(s.shape[1])
        d2 = int(s.shape[2])
        _log(
            2,
            "ensure_subs_bhq[np]: "
            f"d1={d1} d2={d2} n_q={n_q}",
        )

        if (d2 == n_q) and (d1 != n_q):
            out = s
            _log(1, "ensure_subs_bhq[np]: chose keep.")
        elif (d1 == n_q) and (d2 != n_q):
            out = np.transpose(s, (0, 2, 1, 3))
            _log(1, "ensure_subs_bhq[np]: chose swap.")
        else:
            # Ambiguous: decide by q50 MAE vs y_true.
            q = np.asarray(q_values, dtype=float)
            med = int(np.argmin(np.abs(q - 0.5)))

            _log(
                2,
                "ensure_subs_bhq[np]: "
                f"ambiguous, med={med} q={_fmt(q[med])}",
            )

            keep = s
            swap = np.transpose(s, (0, 2, 1, 3))

            yt = _ensure_ytrue_bho_np(
                y_true_b,
                n_q=n_q,
                med=med,
            )

            if yt is None:
                mae_keep = float("inf")
                mae_swap = float("inf")
            else:
                mae_keep = _mae_safe_np(
                    keep[:, :, med, :],
                    yt,
                )
                mae_swap = _mae_safe_np(
                    swap[:, :, med, :],
                    yt,
                )

            _log(
                2,
                "ensure_subs_bhq[np]: "
                f"mae_keep={_fmt(mae_keep)} "
                f"mae_swap={_fmt(mae_swap)}",
            )

            if np.isfinite(mae_keep) or np.isfinite(mae_swap):
                if mae_swap < mae_keep:
                    out = swap
                    _log(1, "ensure_subs_bhq[np]: swap (mae).")
                else:
                    out = keep
                    _log(1, "ensure_subs_bhq[np]: keep (mae).")
            else:
                # Fallback: use quantile crossing score.
                cs_keep = _cross_score_np(keep)
                cs_swap = _cross_score_np(swap)
                _log(
                    2,
                    "ensure_subs_bhq[np]: "
                    f"cross_keep={cs_keep:.6f} "
                    f"cross_swap={cs_swap:.6f}",
                )
                if cs_swap < cs_keep:
                    out = swap
                    _log(
                        1,
                        "ensure_subs_bhq[np]: swap (cross).",
                    )
                else:
                    out = keep
                    _log(
                        1,
                        "ensure_subs_bhq[np]: keep (cross).",
                    )

        if enforce_monotone:
            out = np.sort(out, axis=2)
            _log(
                2,
                "ensure_subs_bhq[np]: "
                "sorted along axis=2.",
            )

        _log(1, f"ensure_subs_bhq[np]: out={out.shape}")
        return out

    # TensorFlow backend (original logic + safe conversion)
    if not tf.is_tensor(s_pred_b):
        s_pred_b = tf.convert_to_tensor(s_pred_b)

    _log(
        1,
        "ensure_subs_bhq[tf]: "
        f"in rank={int(s_pred_b.shape.rank or 0)} "
        f"shape={tuple(s_pred_b.shape.as_list())}",
    )

    # Expect rank-4 in quantile mode.
    if s_pred_b.shape.rank != 4:
        return s_pred_b

    n_q = len(q_values) if q_values else 0
    if n_q <= 0:
        return s_pred_b

    # Easy cases when dims differ.
    d1 = int(s_pred_b.shape[1])
    d2 = int(s_pred_b.shape[2])

    if (d2 == n_q) and (d1 != n_q):
        out = s_pred_b
        _log(1, "ensure_subs_bhq[tf]: chose keep.")
    elif (d1 == n_q) and (d2 != n_q):
        out = tf.transpose(s_pred_b, [0, 2, 1, 3])
        _log(1, "ensure_subs_bhq[tf]: chose swap.")
    else:
        # Ambiguous (often H==Q): choose by median MAE vs y_true.
        q = np.asarray(q_values, dtype=float)
        med = int(np.argmin(np.abs(q - 0.5)))

        keep = s_pred_b
        swap = tf.transpose(s_pred_b, [0, 2, 1, 3])

        if not tf.is_tensor(y_true_b):
            y_true_b = tf.convert_to_tensor(y_true_b)

        mae_keep = tf.reduce_mean(
            tf.abs(keep[:, :, med, :] - y_true_b)
        )
        mae_swap = tf.reduce_mean(
            tf.abs(swap[:, :, med, :] - y_true_b)
        )

        if verbose >= 2:
            _log(
                2,
                "ensure_subs_bhq[tf]: "
                f"mae_keep={_fmt(float(mae_keep.numpy()))} "
                f"mae_swap={_fmt(float(mae_swap.numpy()))}",
            )

        out = tf.cond(
            mae_swap < mae_keep,
            lambda: swap,
            lambda: keep,
        )

    if enforce_monotone:
        out = tf.sort(out, axis=2)

    return out

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
