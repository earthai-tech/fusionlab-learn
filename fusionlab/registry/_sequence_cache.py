
from __future__ import annotations
import json, hashlib, joblib, logging
from pathlib import Path
from typing import Any, Optional, Literal 

from platformdirs import user_cache_dir
_log = logging.getLogger("fusionlab.seqcache")



_APP_NAME   = "fusionlab-learn"
_APP_AUTHOR = "earthai-tech"


def resolve_sequence_cache(
    run_dir: str | Path | None = None,
    *,
    mode: Literal["get", "ensure"] = "get",
) -> Optional[Path]:
    """
    Unified helper to **get** or **make** the `<run_dir>/sequence_cache` folder.

    Parameters
    ----------
    run_dir : str | Path | None, default **None**
        The experiment’s run directory.  If *None*, the global *runs* root is
        used.
    mode : {"get", "ensure"}, default **"get"**
        - ``"get"``   → return the path **only if it already exists**,
          otherwise *None*.
        - ``"ensure"``→ create the directory (and parents) if needed and return
          the `Path`.

    Returns
    -------
    pathlib.Path | None
        The absolute path of the sequence-cache folder, or *None* when
        ``mode="get"`` and it doesn’t exist.

    Examples
    --------
    >>> cache = resolve_sequence_cache("~/runs/exp-42", mode="ensure")
    >>> resolve_sequence_cache("~/runs/exp-42")   # "get" is default
    PosixPath('/home/you/runs/exp-42/sequence_cache')
    """
    run_dir = Path(run_dir or _runs_root()).expanduser().resolve()
    seq_dir = run_dir / "sequence_cache"

    if mode == "ensure":
        seq_dir.mkdir(parents=True, exist_ok=True)
        return seq_dir

    if mode == "get":
        return seq_dir if seq_dir.exists() else None

    raise ValueError(f"mode must be 'get' or 'ensure', not {mode!r}")



def user_runs_root() -> Path:
    """Return the OS-appropriate *runs* cache root."""
    root = Path(user_cache_dir(appname=_APP_NAME, appauthor=_APP_AUTHOR))
    return (root / "runs").expanduser().resolve()


def ensure_sequence_cache_dir(run_dir: str | Path) -> Path:
    """
    Guarantee the `<run_dir>/sequence_cache` folder exists
    and return it as a `Path`.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    cache_dir = run_dir / "sequence_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_sequence_cache_dir(run_dir: str | Path) -> Optional[Path]:
    """
    Return `<run_dir>/sequence_cache` **only if it already exists**,
    else return `None` (do *not* create anything).
    """
    cache_dir = Path(run_dir).expanduser().resolve() / "sequence_cache"
    return cache_dir if cache_dir.exists() else None

def _runs_root() -> Path:
    """
    OS-independent *runs* cache root:

    * Windows →  C:\\Users\\<...>\\AppData\\Local\\earthai-tech\\fusionlab-learn\\Cache\\runs
    * macOS   →  ~/Library/Caches/earthai-tech/fusionlab-learn/runs
    * Linux   →  ~/.cache/earthai-tech/fusionlab-learn/runs
    """
    return (Path(user_cache_dir(_APP_NAME, _APP_AUTHOR)) / "runs").resolve()




def _sha1_of(obj: Any) -> str:
    """Stable SHA-1 of any JSON-serialisable object."""
    js = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(js.encode("utf-8")).hexdigest()


def _cache_dir_for_run(run_dir: str | Path) -> Path:
    """<run_dir>/sequence_cache  (created lazily)."""
    p = Path(run_dir).expanduser().resolve() / "sequence_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(run_dir: str | Path, signature: dict) -> Path:
    cache_key = _sha1_of(signature)
    return _cache_dir_for_run(run_dir) / f"{cache_key}.joblib"


def save_seq_cache(
    *,
    signature: dict,
    payload:   dict,
    run_dir:   str | Path,
    overwrite: bool = False,
) -> Path:
    """
    Persist *payload* under a hash of *signature*.

    Returns the full path of the written cache file.
    """
    fp = _cache_path(run_dir, signature)
    if fp.exists() and not overwrite:
        _log.debug("sequence cache already exists: %s", fp.name)
        return fp

    joblib.dump({"signature": signature, "payload": payload}, fp, compress="lz4")
    _log.info("sequence cache saved: %s", fp.name)
    return fp


def load_seq_cache(
    *,
    signature: dict,
    run_dir:   str | Path,
    drop_bad:  bool = True,
) -> Optional[dict]:
    """
    Try loading a cached payload.  Returns **dict** or **None**.

    If the stored signature mismatches, removes the stale file when
    *drop_bad* is True.
    """
    fp = _cache_path(run_dir, signature)
    if not fp.exists():
        return None

    try:
        blob = joblib.load(fp)
        if blob.get("signature") != signature:
            raise ValueError("signature mismatch")
        return blob.get("payload")
    except Exception as exc:           # corrupted or mismatching
        _log.warning("invalid sequence cache %s (%s)", fp.name, exc)
        if drop_bad:
            try:
                fp.unlink()
            except OSError:
                pass
        return None
