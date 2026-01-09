
from __future__ import annotations
from typing import Dict, List, Optional, Union, Callable
from typing import Tuple, Any

import os
import warnings
import inspect
import numpy as np
import matplotlib.pyplot as plt

from ... import KERAS_DEPS 

History = KERAS_DEPS.History 


def _as_history_dict(history: Any) -> Dict[str, List[float]]:
    # Accept History or dict-like.
    if isinstance(history, History):
        return dict(history.history or {})
    if isinstance(history, dict):
        return dict(history)
    h = getattr(history, "history", None)
    if isinstance(h, dict):
        return dict(h)
    raise TypeError(
        "history must be keras History or dict-like."
    )

def _get_valid_kwargs(fn, kwargs: Dict) -> Dict:
    # Filter kwargs to avoid matplotlib signature errors.
    try:
        sig = inspect.signature(fn)
        valid = set(sig.parameters.keys())
    except Exception:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in valid}


def _has_pos_only(arrs: List[np.ndarray]) -> bool:
    # True if all finite and strictly positive.
    if not arrs:
        return False
    for a in arrs:
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        if np.any(a <= 0.0):
            return False
    return True


def _pick_scale(
    scale: str,
    arrs: List[np.ndarray],
) -> Tuple[str, Dict]:
    # For requested "log", fall back to symlog if needed.
    if scale != "log":
        return scale, {}
    if _has_pos_only(arrs):
        return "log", {}
    # symlog keeps zeros/negatives visible.
    # linthresh from smallest positive, else 1e-12.
    pos = []
    for a in arrs:
        a = a[np.isfinite(a)]
        pos.append(a[a > 0.0])
    pos = np.concatenate(pos) if pos else np.asarray([])
    lt = float(np.min(pos)) if pos.size else 1e-12
    lt = max(lt, 1e-12)
    return "symlog", {"linthresh": lt}


def plot_history_in(
    history: Union[History, Dict],
    metrics: Optional[Dict[str, List[str]]] = None,
    layout: str = "subplots",
    title: str = "Model Training History",
    figsize: Optional[Tuple[float, float]] = None,
    style: str = "default",
    savefig: Optional[str] = None,
    max_cols: Union[int, str] = "auto",
    show_grid: bool = True,
    grid_props: Optional[Dict] = None,
    yscale_settings: Optional[Dict[str, str]] = None,
    log_fn: Optional[Callable[..., None]] = None,
    **plot_kwargs,
) -> None:
    """
    Plot Keras history (train + val) robustly.
    """
    log = log_fn if log_fn is not None else print
    hist = _as_history_dict(history)

    if not hist:
        warnings.warn("Empty history: nothing to plot.")
        return

    # Style (never crash)
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("default")

    # Auto-group if not provided
    if metrics is None:
        metrics = {}
        for k in hist.keys():
            if k.startswith("val_"):
                continue
            g = "Losses" if "loss" in k.lower() else k
            g = g.replace("_", " ").title()
            metrics.setdefault(g, []).append(k)

    if not metrics:
        warnings.warn("No metrics to plot.")
        return

    if yscale_settings is None:
        yscale_settings = {}

    grid_kws = grid_props or {"linestyle": ":", "alpha": 0.7}

    # Layout
    n_plots = len(metrics)
    if layout == "single":
        n_rows, n_cols = 1, 1
        if figsize is None:
            figsize = (10.0, 6.0)
    else:
        cols = 2 if max_cols == "auto" else int(max_cols)
        n_cols = max(1, min(cols, n_plots))
        n_rows = (n_plots + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (float(n_cols) * 6.0, float(n_rows) * 5.0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    axflat = axes.flatten()
    fig.suptitle(title, fontsize=14, weight="bold")

    def _plot_one_axis(ax, keys: List[str], subttl: str) -> None:
        # Collect arrays for scale decision.
        arrs = []
        for k in keys:
            if k in hist and len(hist[k]):
                arrs.append(np.asarray(hist[k], dtype=float))

        # Choose scale for this subplot.
        req = yscale_settings.get(subttl, "linear")
        scale, skws = _pick_scale(req, arrs)
        try:
            ax.set_yscale(scale, **skws)
        except Exception:
            ax.set_yscale("linear")

        # Plot requested keys.
        seen = set()
        for k in keys:
            if k not in hist or not len(hist[k]):
                continue
            if k in seen:
                continue
            seen.add(k)

            y = np.asarray(hist[k], dtype=float)
            x = np.arange(1, len(y) + 1)

            # If user passed val_* explicitly, treat it as val.
            is_val = k.startswith("val_")
            base = k[4:] if is_val else k

            pk = _get_valid_kwargs(ax.plot, plot_kwargs)
            lab = ("Val " if is_val else "Train ")
            lab = lab + base.replace("_", " ").title()

            ax.plot(x, y, label=lab, **pk)

            # Auto-add val only for non-val keys.
            if not is_val:
                vk = "val_" + k
                if vk in hist and len(hist[vk]):
                    vy = np.asarray(hist[vk], dtype=float)
                    ax.plot(
                        x,
                        vy,
                        linestyle="--",
                        label=("Val " + base.replace("_", " ").title()),
                        **pk,
                    )

        ax.set_title(subttl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        if show_grid:
            ax.grid(**grid_kws)

    if layout == "single":
        ax = axflat[0]
        keys = []
        for grp_keys in metrics.values():
            keys.extend(grp_keys)
        _plot_one_axis(ax, keys, "All Metrics")
        for i in range(1, len(axflat)):
            axflat[i].set_visible(False)
    else:
        i = 0
        for subttl, keys in metrics.items():
            if i >= len(axflat):
                break
            _plot_one_axis(axflat[i], keys, subttl)
            i += 1
        for j in range(i, len(axflat)):
            axflat[j].set_visible(False)

    plt.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])

    if savefig:
        root, ext = os.path.splitext(savefig)
        if not ext:
            savefig = root + ".png"
        os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
        try:
            plt.savefig(savefig, dpi=300)
            log(f"[OK] Saved figure -> {savefig}")
        except Exception as e:
            warnings.warn(f"Save failed: {e}")
        plt.close(fig)
    else:
        plt.show()

def gather_coords_flat(
    dataset,
    *,
    coord_key="coords",
    log_fn=None,
    max_batches=None,
):
    """
    Collect flat (t, x, y) arrays from a tf.data dataset.
    """


    log = log_fn if log_fn is not None else (lambda *_: None)

    ts = []
    xs = []
    ys = []

    n_seen = 0
    for batch in dataset:
        # dataset can yield inputs or (inputs, targets)
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch

        # inputs can be dict, sequence, or coords tensor directly
        if isinstance(inputs, dict):
            coords = inputs.get(coord_key, None)
            if coords is None:
                raise KeyError(
                    f"Missing '{coord_key}' in inputs dict."
                )
        elif isinstance(inputs, (tuple, list)):
            coords = inputs[0]
        else:
            coords = inputs

        # tf.Tensor -> numpy
        if hasattr(coords, "numpy"):
            coords = coords.numpy()

        coords = np.asarray(coords)

        if coords.shape[-1] != 3:
            raise ValueError(
                "coords[..., -1] must be 3 (t, x, y)."
            )

        # coords can be (B, 3) or (B, T, 3)
        t = np.asarray(coords[..., 0]).ravel()
        x = np.asarray(coords[..., 1]).ravel()
        y = np.asarray(coords[..., 2]).ravel()

        ts.append(t)
        xs.append(x)
        ys.append(y)

        n_seen += 1
        if max_batches is not None and n_seen >= max_batches:
            break

    if not xs:
        raise ValueError("Dataset yielded no coords.")

    out = {
        "t": np.concatenate(ts, axis=0),
        "x": np.concatenate(xs, axis=0),
        "y": np.concatenate(ys, axis=0),
    }

    log(
        "gather_coords_flat:",
        f"n={out['x'].shape[0]}",
    )
    return out


def plot_physics_values_in(
    payload,
    *,
    keys=None,
    dataset=None,
    coords=None,
    mode="map",
    title="Physics diagnostics",
    n_cols=2,
    figsize=None,
    savefig=None,
    show=True,
    clip_q=(0.01, 0.99),
    transform=None,
    bins=80,
    s=8,
    log_fn=None,
    **scatter_kwargs,
):
    """
    Plot physics arrays (residuals/fields) from a payload dict.
    """


    log = log_fn if log_fn is not None else print

    def _finite(a):
        a = np.asarray(a, dtype=float)
        m = np.isfinite(a)
        return a[m]

    def _safe_vlim(v):
        v = _finite(v)
        if v.size == 0:
            return None, None
        lo, hi = clip_q
        lo = float(lo)
        hi = float(hi)
        if not (0.0 <= lo < hi <= 1.0):
            return None, None
        try:
            vmin = float(np.quantile(v, lo))
            vmax = float(np.quantile(v, hi))
        except Exception:
            return None, None
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None, None
        if vmin == vmax:
            return None, None
        return vmin, vmax

    def _apply_transform(v):
        if transform is None:
            return v

        if callable(transform):
            return transform(v)

        t = str(transform).lower().strip()

        if t == "abs":
            return np.abs(v)

        if t == "log10":
            v = np.asarray(v, dtype=float)
            v = np.where(v > 0.0, v, np.nan)
            return np.log10(v)

        if t in ("signed_log10", "slog10"):
            v = np.asarray(v, dtype=float)
            return np.sign(v) * np.log10(1.0 + np.abs(v))

        return v

    def _pick_keys(d):
        if keys is not None:
            return list(keys)

        # reasonable defaults for GeoPrior payloads
        pref = [
            "cons_res_vals",
            "R_cons",
            "epsilon_cons",
            "epsilon_gw",
            "epsilon_prior",
            "log10_tau",
            "log10_tau_prior",
            "K",
            "Ss",
            "Hd",
            "H",
        ]
        out = [k for k in pref if k in d]
        if out:
            return out

        # fallback: any numeric arrays
        out = []
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                if v.dtype.kind in ("f", "i", "u"):
                    out.append(k)
        return out

    def _get_xy_for_values(v):
        # user provided coords dict
        if coords is not None:
            x = np.asarray(coords["x"]).ravel()
            y = np.asarray(coords["y"]).ravel()
            return x, y

        # derive coords from dataset (preferred)
        if dataset is not None:
            c = gather_coords_flat(
                dataset,
                log_fn=log_fn,
            )
            x = np.asarray(c["x"]).ravel()
            y = np.asarray(c["y"]).ravel()
            return x, y

        return None, None

    def _align_xy(x, y, v):
        # try to align lengths without guessing too much
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        v = np.asarray(v).ravel()

        nx = x.shape[0]
        nv = v.shape[0]

        if nx == nv:
            return x, y, v

        # common case: payload is (B) but coords are (B*T)
        if nx % nv == 0:
            step = nx // nv
            x2 = x[::step]
            y2 = y[::step]
            if x2.shape[0] == nv:
                return x2, y2, v

        # opposite: coords (B) but values (B*T)
        if nv % nx == 0:
            step = nv // nx
            v2 = v[::step]
            if v2.shape[0] == nx:
                return x, y, v2

        # last resort: truncate to min length
        n = min(nx, nv)
        return x[:n], y[:n], v[:n]

    d = payload
    if not isinstance(d, dict):
        raise TypeError("payload must be a dict.")

    klist = _pick_keys(d)
    if not klist:
        warnings.warn("No plot-able keys found in payload.")
        return

    mode = str(mode).lower().strip()
    if mode not in ("map", "hist", "both"):
        mode = "map"

    n_plots = len(klist)
    if mode == "both":
        n_plots = len(klist) * 2

    if n_cols < 1:
        n_cols = 1

    n_cols = min(int(n_cols), n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (n_cols * 6, n_rows * 5)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False,
    )
    axes = axes.ravel()

    fig.suptitle(title, fontsize=16, weight="bold")

    plot_i = 0

    for k in klist:
        v = d.get(k, None)
        if v is None:
            continue

        v = np.asarray(v).ravel()
        v = _apply_transform(v)

        if mode in ("map", "both"):
            ax = axes[plot_i]
            plot_i += 1

            x, y = _get_xy_for_values(v)
            if x is None or y is None:
                ax.set_visible(False)
            else:
                x, y, vv = _align_xy(x, y, v)
                m = np.isfinite(vv)
                x = x[m]
                y = y[m]
                vv = vv[m]

                vmin, vmax = _safe_vlim(vv)

                sc = ax.scatter(
                    x,
                    y,
                    c=vv,
                    s=s,
                    vmin=vmin,
                    vmax=vmax,
                    **scatter_kwargs,
                )

                ax.set_title(k)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_aspect("equal", adjustable="box")

                try:
                    fig.colorbar(sc, ax=ax, shrink=0.85)
                except Exception:
                    pass

        if mode in ("hist", "both"):
            ax = axes[plot_i]
            plot_i += 1

            vv = _finite(v)
            if vv.size == 0:
                ax.set_visible(False)
            else:
                ax.hist(vv, bins=int(bins))
                ax.set_title(f"{k} (hist)")
                ax.set_xlabel(k)
                ax.set_ylabel("count")

    # hide unused axes
    for j in range(plot_i, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if savefig:
        try:
            root, ext = os.path.splitext(savefig)
            if not ext:
                savefig = root + ".png"

            out_dir = os.path.dirname(savefig)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)

            plt.savefig(
                savefig,
                dpi=300,
                bbox_inches="tight",
            )
            log(f"Saved: {savefig}")
        except Exception as e:
            warnings.warn(f"Save failed: {e}")
        finally:
            plt.close(fig)
    else:
        if show:
            plt.show()
        else:
            plt.close(fig)
            
def _keys_starting(hist: Dict[str, List[float]], p: str) -> List[str]:
    ks = [k for k in hist.keys() if not k.startswith("val_")]
    ks = [k for k in ks if k.startswith(p)]
    return sorted(ks)


def _existing(hist: Dict[str, List[float]], keys: List[str]) -> List[str]:
    out = []
    for k in keys:
        if k in hist and len(hist[k]):
            out.append(k)
    return out


def plot_epsilons_in(
    history: Union[History, Dict],
    *,
    title: str = "Epsilons",
    savefig: Optional[str] = None,
    style: str = "default",
    log_fn: Optional[Callable[..., None]] = None,
) -> None:
    # Plot only epsilon_* (incl. *_raw) with safe symlog.
    hist = _as_history_dict(history)
    eps_keys = _keys_starting(hist, "epsilon_")

    if not eps_keys:
        (log_fn or print)("[plot] No epsilon_* keys.")
        return

    groups = {"Epsilons": eps_keys}

    ysc = {"Epsilons": "log"}
    plot_history_in(
        hist,
        metrics=groups,
        layout="single",
        title=title,
        style=style,
        savefig=savefig,
        yscale_settings=ysc,
        log_fn=log_fn,
    )


def plot_physics_losses_in(
    history: Union[History, Dict],
    *,
    title: str = "Physics Loss Terms",
    savefig: Optional[str] = None,
    style: str = "default",
    log_fn: Optional[Callable[..., None]] = None,
) -> None:
    # Auto-plot key physics loss terms with log/symlog.
    hist = _as_history_dict(history)

    keys = [
        "physics_loss",
        "physics_loss_scaled",
        "consolidation_loss",
        "gw_flow_loss",
        "prior_loss",
        "smooth_loss",
        "mv_prior_loss",
        "bounds_loss",
        # optional diagnostics if enabled:
        "q_reg_loss",
        "q_rms",
        "q_gate",
        "subs_resid_gate",
    ]
    keys = _existing(hist, keys)
    if not keys:
        (log_fn or print)("[plot] No physics loss keys.")
        return

    groups = {"Physics": keys}

    # Request log; plot_history_in will fall back to symlog.
    ysc = {"Physics": "log"}
    plot_history_in(
        hist,
        metrics=groups,
        layout="single",
        title=title,
        style=style,
        savefig=savefig,
        yscale_settings=ysc,
        log_fn=log_fn,
    )


def autoplot_geoprior_history(
    history: Union[History, Dict],
    *,
    outdir: str,
    prefix: str = "geoprior",
    style: str = "default",
    log_fn: Optional[Callable[..., None]] = None,
) -> None:
    # Minimal, robust: epsilons + physics loss terms.
    os.makedirs(outdir, exist_ok=True)

    plot_epsilons_in(
        history,
        title=f"{prefix} | epsilons",
        savefig=os.path.join(outdir, f"{prefix}_epsilons.png"),
        style=style,
        log_fn=log_fn,
    )

    plot_physics_losses_in(
        history,
        title=f"{prefix} | physics terms",
        savefig=os.path.join(outdir, f"{prefix}_physics_terms.png"),
        style=style,
        log_fn=log_fn,
    )
