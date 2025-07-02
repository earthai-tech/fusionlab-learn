# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides a robust system for managing and tracking training run artifacts
through a centralized manifest registry.
"""
from __future__ import annotations

from pathlib import Path
import os
import json
import tempfile
import shutil
import atexit
import datetime
from typing import (
    Callable,
    Optional,
    Any,
    Dict,
    Union,
    Final,
    Literal,
    Tuple,  
    final,
)

try:
    from platformdirs import user_cache_dir
except ImportError as exc:
    raise ImportError(
        "The ManifestRegistry requires the `platformdirs` library. "
        "Please install it by running: pip install platformdirs"
    ) from exc

__all__ = ["ManifestRegistry", "_update_manifest"]



@final
class ManifestRegistry:
    """Manages training runs in a centralized cache directory.

    **Warning**  
    This is a core, singleton class. Subclassing or modifying its
    attributes at runtime is prohibited and will raise errors.

    """
    __slots__ = (
        "_initialized",
        "root_dir",
        "persistent_root",
        "session_root",
        "log",
        "_debug_mode",
        "_run_registry_path",
        "_manifest_filename",
        
    )

    _instance: Final[Optional["ManifestRegistry"]] = None
    _ENV_VAR: Final[str] = "FUSIONLAB_RUN_DIR"
    _RUNS_SUBDIR: Final[str] = "runs"
    _TRAIN_FILE_NAME : Final[str] = "run_manifest.json"
    _TUNER_FILE_NAME : Final[str] = "tuner_run_manifest.json"


    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(f"Subclassing {cls.__name__} is not allowed")

    def __init__(
        self,
        session_only: bool = False,
        manifest_kind: Literal["training","tuning"] = "training",
        log_callback: Optional[Callable[[str], None]] = None,
        debug_mode: Optional[str] = None,
    ) -> None:
        super().__setattr__("_initialized", False) 
        
        if getattr(self, "_initialized", False):
            return

        super().__setattr__("log", log_callback or print)
        super().__setattr__("_debug_mode", debug_mode or "silence")
        
        # pick correct file‐name once, use everywhere else
        _manifest_filename = (
            self._TRAIN_FILE_NAME if manifest_kind == "training"
            else self._TUNER_FILE_NAME
        )
        super().__setattr__("_manifest_filename", _manifest_filename)
        
        default_root = Path(
            user_cache_dir("fusionlab-learn", "earthai-tech")
        ) / self._RUNS_SUBDIR
        env_root = os.getenv(self._ENV_VAR)
        super().__setattr__(
            "persistent_root",
            Path(env_root) if env_root else default_root
        )

        super().__setattr__("session_root", None)

        if session_only:
            sess = tempfile.mkdtemp(prefix="fusionlab_run_")
            super().__setattr__("session_root", Path(sess))
            super().__setattr__("root_dir", Path(sess))
            atexit.register(self.purge_session)
            if self._debug_mode == "debug":
                self.log(f"[Registry] Session-only mode: {self.root_dir}")
        else:
            super().__setattr__("root_dir", self.persistent_root)
            if self._debug_mode == "debug":
                self.log(f"[Registry] Persistent mode: {self.root_dir}")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_root.mkdir(parents=True, exist_ok=True)

        super().__setattr__("_run_registry_path", None)
        super().__setattr__("_initialized", True)

    def __setattr__(self, name: str, value: Any):
        # if getattr(self, "_initialized", False):
        #     raise AttributeError(
        #         f"{self.__class__.__name__} is immutable; cannot assign {name!r}"
        #     )
        # super().__setattr__(name, value)
        # allow the very first write to `_initialized`
        if name == "_initialized":
            super().__setattr__(name, value)
            return
    
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"{self.__class__.__name__} is immutable; cannot assign {name!r}"
            )
        super().__setattr__(name, value)

    def new_run_dir(self, *, city: str = "unset", model: str = "unset") -> Path:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}_{city}_{model}_run"
        run_path = self.root_dir / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        super().__setattr__("_run_registry_path", run_path)
        return run_path

    def import_manifest(self, source_manifest_path: Union[str, Path]) -> Path:
        source = Path(source_manifest_path).resolve()
        if (
            source.is_relative_to(self.persistent_root)
            or (self.session_root and source.is_relative_to(self.session_root))
        ):
            if self._debug_mode == "debug":
                self.log(f"Already in registry: {source}")
            return source
        if not source.is_file() or source.suffix.lower() != ".json":
            raise FileNotFoundError("Must provide a valid .json file")
        try:
            content = json.loads(source.read_text("utf-8"))
            city = content.get("configuration", {}).get("city_name", "imported")
            model = content.get("configuration", {}).get("model_name", "run")
        except Exception:
            city, model = "imported", "run"
        new_dir = self.new_run_dir(city=city, model=model)
        dest = new_dir / self._manifest_filename
        shutil.copyfile(source, dest)
        super().__setattr__("_run_registry_path", dest)
        if self._debug_mode == "debug":
            self.log(f"Imported manifest into: {new_dir}")
        return dest

    def latest_manifest(self) -> Optional[Path]:
        all_manifests = []
        if self._debug_mode == "debug":
            self.log("Searching for latest manifest...")
        all_manifests.extend(
            self.persistent_root.glob(f"*/{self._manifest_filename}")
        )
        if self.session_root:
            if self._debug_mode == "debug":
                self.log(f"Checking session cache: {self.session_root}")
            all_manifests.extend(
                self.session_root.glob(f"*/{self._manifest_filename}")
            )
        if not all_manifests:
            if self._debug_mode == "debug":
                self.log("No manifests found")
            return None
        latest = max(all_manifests, key=lambda p: p.stat().st_mtime)
        if self._debug_mode == "debug":
            self.log(f"Found latest manifest: {latest}")
        return latest

    def update(
        self,
        run_dir: Path,
        section: str,
        item: Union[Any, Dict[str, Any]],
        *,
        as_list: bool = False,
    ) -> None:
        _update_manifest(
            run_dir=run_dir,
            section=section,
            item=item,
            as_list=as_list,
            name=self._manifest_filename,
        )

    def purge_session(self) -> None:
        if self.session_root and self.session_root.exists():
            if self._debug_mode == "debug":
                self.log(f"Purging session: {self.session_root}")
            shutil.rmtree(self.session_root, ignore_errors=True)

    @property
    def registry_path(self) -> Path:
        return self._run_registry_path
    
    @classmethod
    def manifest_filename(cls) -> str:
        #e.g. run_dir / ManifestRegistry.manifest_filename()
        return ( 
            cls._TRAIN_FILE_NAME if cls._kind == "training" 
            else cls._TUNER_FILE_NAME
            )


ManifestRegistry.__doc__ = """
Manages training runs in a centralized cache directory.

**Warning**  
This is a core, singleton class. Subclassing or modifying its
attributes at runtime is prohibited and will raise errors.

Internal Use Only:
- Final singleton class for managing training-run directories and manifests.
- Prevents subclassing and post-init attribute changes.

Methods
-------
__new__(cls, *args, **kwargs)

Internal Use Only:
- Implements singleton pattern at object creation.
- Returns existing instance or allocates new one.

__init_subclass__(cls, *args, **kwargs)

Internal Use Only:
- Raises TypeError to prohibit subclassing at runtime.

__init__(self, session_only: bool, log_callback: Optional[Callable],
         debug_mode: Optional[str])

Internal Use Only:
- Guards single initialization via _initialized flag.
- Configures:
    • persistent_root (from env var or user cache)
    • session_root (temp dir if session_only=True)
    • root_dir pointer to active directory
    • log callback and _debug_mode flag
- Ensures persistent_root and root_dir exist on disk.

__setattr__(self, name: str, value: Any)

Internal Use Only:
- Disallows attribute assignment after initialization.
- Raises AttributeError if setting any attribute post-init.

new_run_dir(self, *, city: str, model: str) → Path

Internal Use Only:
- Creates timestamped run directory under root_dir.
- Stores new path in _run_registry_path.

import_manifest(self, source_manifest_path: Union[str, Path]) → Path

Internal Use Only:
- If manifest already under managed roots, returns it.
- Otherwise:
    • Validates .json file
    • Reads optional metadata (city/model)
    • Creates new run directory
    • Copies manifest into it
    • Updates _run_registry_path

latest_manifest(self) → Optional[Path]

Internal Use Only:
- Scans persistent_root and session_root for run_manifest.json files.
- Returns most recently modified file or None if none found.
- Emits debug logs when _debug_mode == "debug".

update(self, run_dir: Path, section: str, item: Union[Any, Dict], *, as_list: bool)

Internal Use Only:
- Proxy to standalone _update_manifest().
- Passes run_dir, section, item, as_list, and manifest filename.

purge_session(self) → None

Internal Use Only:
- Deletes temporary session_root directory if it exists.
- Registered to run at exit in session-only mode.

registry_path(self) → Path

Internal Use Only:
- Property for last-created or imported run/manifest path (_run_registry_path).

Class-Level Constants (Final)
-----------------------------
_ENV_VAR          = "FUSIONLAB_RUN_DIR"
_RUNS_SUBDIR      = "runs"
_MANIFEST_FILENAME= "run_manifest.json"
- Immutable configuration values throughout the class.
"""

class _ManifestRegistry:
    """Manages training runs in a centralized cache directory.

    This class provides a clean interface for creating unique,
    timestamped directories for each training run. It handles the
    creation and updating of a `run_manifest.json` file within each
    directory, which acts as the single source of truth for all
    configurations and artifact paths for that run.

    This system makes inference workflows robust, as they can reliably
    find all necessary components by simply referencing the latest
    manifest file.

    Attributes
    ----------
    root_dir : pathlib.Path
        The absolute path to the central directory where all training
        run subdirectories are stored. This defaults to a location
        within the user's cache (e.g., `~/.cache/fusionlab-learn/runs`)
        but can be overridden with the `FUSIONLAB_RUN_DIR`
        environment variable.

    Examples
    --------
    >>> from fusionlab.utils.manifest_registry import ManifestRegistry
    >>> registry = ManifestRegistry()
    >>> # Create a new directory for a training run
    >>> run_dir = registry.new_run_dir(city="zhongshan", model="PINN")
    >>> print(run_dir.name)
    2025-06-24_15-30-00_zhongshan_PINN_run

    >>> # Find the most recent manifest file
    >>> latest = registry.latest_manifest()
    >>> if latest:
    ...     print(f"Latest run found: {latest}")

    >>> # Clean up all stored runs
    >>> # registry.purge_session()
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    

    _ENV_VAR = "FUSIONLAB_RUN_DIR"
    _RUNS_SUBDIR = "runs"
    _MANIFEST_FILENAME = "run_manifest.json"

    def __init__(
        self, session_only: bool = False, 
        log_callback: Optional[callable] = None,
        debug_mode : Optional[str] =None, 
        ) -> None:
        """Initializes the registry and ensures the root directory exists."""
        # The singleton pattern ensures __init__ is only run once.
        if self._initialized:
            return 
        
        self.log = log_callback or print 
        self._debug_mode =debug_mode or 'silence'
        
        # Always know the persistent cache path
        default_persistent_root = Path(
            user_cache_dir("fusionlab-learn", "earthai-tech")
        ) / self._RUNS_SUBDIR
        self.persistent_root = Path(os.getenv(
            self._ENV_VAR, default_persistent_root))
        
        self.session_root = None
        
        if session_only:
            # Session mode: create a temp dir and set it as active
            self.session_root = Path(
                tempfile.mkdtemp(prefix="fusionlab_run_"))
            self.root_dir = self.session_root
            atexit.register(self.purge_session)
            
            if self._debug_mode =='debug': 
                self.log(f"[Registry] Initialized in session-only mode."
                          f" Active dir: {self.root_dir}")
        else:
            # Persistent mode: set the cache as active
            self.root_dir = self.persistent_root
            if self._debug_mode =="debug": 
                self.log(f"[Registry] Initialized in persistent mode."
                         f" Active dir: {self.root_dir}") 
                
        # Ensure the root directory exists.
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_root.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        self._run_registry_path =None 
        
        
    def new_run_dir(
        self, *, city: str = "unset", 
        model: str = "unset"
        ) -> Path:
        """Creates a fresh, timestamped run directory.

        The directory name is generated using the current timestamp and
        the provided city and model names to ensure uniqueness and
        discoverability.

        Parameters
        ----------
        city : str, default='unset'
            The name of the city or dataset for this run.
        model : str, default='unset'
            The name of the model being trained.

        Returns
        -------
        pathlib.Path
            The absolute path to the newly created run directory.
        """
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{ts}_{city}_{model}_run"
        run_path = self.root_dir / run_name
        run_path.mkdir(parents=True, exist_ok=True)
        self._run_registry_path = run_path
        return run_path
    
    def import_manifest(self, source_manifest_path: str | Path) -> Path:
        """
        Imports an external run manifest into the registry.
    
        This method copies a user-provided `run_manifest.json` file
        into a new, timestamped directory within the central registry.
        This is the recommended way to register an existing training
        run for subsequent inference.
    
        Parameters
        ----------
        source_manifest_path : str or pathlib.Path
            The path to the external `run_manifest.json` file to import.
    
        Returns
        -------
        pathlib.Path
            The path to the newly created manifest file *inside* the registry.
    
        Raises
        ------
        FileNotFoundError
            If the `source_manifest_path` does not exist or is not a JSON file.
        """
        source_path = Path(source_manifest_path).resolve() 
        
        # Check if the file is already inside any known registry
        if (
            source_path.is_relative_to(self.persistent_root) or
            (self.session_root and source_path.is_relative_to(
                self.session_root))
        ):
            if self._debug_mode=='debug': 
                self.log(
                    f"Manifest '{source_path.name}' "
                    "is already in the registry."
                )
            return source_path # Already managed, do nothing.
        
        if (
            not source_path.is_file() or
            source_path.suffix.lower() != ".json"
        ):
            raise FileNotFoundError(
                "Provided manifest path must be a valid .json file."
            )
            
        # Create a new run directory to house the imported manifest
        # We can try to get city/model name from the manifest if it exists
        try:
            content = json.loads(source_path.read_text("utf-8"))
            city = content.get(
                "configuration", {}).get("city_name", "imported")
            model = content.get(
                "configuration", {}).get("model_name", "run")
        except Exception:
            city, model = "imported", "run"
            
        
        new_run_dir = self.new_run_dir(city=city, model=model)
        
        # Copy the user's manifest into the new directory
        destination_path = new_run_dir / self._MANIFEST_FILENAME
        shutil.copyfile(source_path, destination_path)
        self._run_registry_path = destination_path
        if self._debug_mode=='debug': 
            self.log(f"Imported manifest to new run directory: {new_run_dir}")
            
        return destination_path
    
    def latest_manifest(self) -> Optional[Path]:
        """Finds and returns the path to the most recent manifest file.

        This method scans all subdirectories within the registry's root
        for `run_manifest.json` files and returns the one with the
        latest modification time. This is the primary mechanism for the
        inference pipeline to automatically find the latest trained model.

        Returns
        -------
        pathlib.Path or None
            The path to the most recent manifest file, or None if no
            runs are found in the registry.
        """

        all_manifests = []
        if self._debug_mode =='debug':
            self.log("Searching for the latest training run...")

        # 1. Search in the persistent cache directory
        # Instantiate the registry in default (persistent) mode to get the path

        if self._debug_mode =='debug':
            self.log(f"  -> Checking persistent cache: {self.persistent_root}")
        all_manifests.extend(
            self.persistent_root.glob("*/{self._MANIFEST_FILENAME}"))

        # Search session cache if it exists
        if self.session_root:
            if self._debug_mode =='debug':
                self.log(f"  -> Checking session cache: {self.session_root}")
            all_manifests.extend(
                self.session_root.glob(f"*/{self._MANIFEST_FILENAME}"))
            
        # 3. Find the most recent manifest among all found files
        if not all_manifests:
            if self._debug_mode =='debug':
                self.log("  -> No manifest files found in any known location.")
            return None

        latest_manifest = max(all_manifests, key=lambda p: p.stat().st_mtime)
        if self._debug_mode =='debug':
            self.log(f"  -> Found latest manifest: {latest_manifest}")
        
        return latest_manifest 

    def update(
        self,
        run_dir: Path,
        section: str,
        item: Union[Any, Dict[str, Any]],
        *,
        as_list: bool = False
    ) -> None:
        """Updates the manifest file for a specific run directory.

        This is a convenience method that proxies to the internal
        `_update_manifest` function.

        Parameters
        ----------
        run_dir : pathlib.Path
            The specific run directory containing the manifest to update.
        section : str
            The top-level key in the JSON file (e.g., 'configuration',
            'training', 'artifacts').
        item : dict or any
            The data to write. If a dictionary, it is merged into the
            section. Otherwise, it is stored under a special `_` key.
        as_list : bool, default=False
            If True and `item` is not a dictionary, appends the item
            to a list within the section.
        """
        _update_manifest(
            run_dir,
            section,
            item,
            as_list=as_list,
            name=self._MANIFEST_FILENAME
        )

    def purge_session(self) -> None:
        """Deletes the entire run registry directory.

        Warning: This is a destructive operation and will permanently
        remove all saved training runs and artifacts. It is primarily
        intended for cleanup during testing.
        """
        if self.session_root and self.session_root.exists():
            if self._debug_mode =='debug': 
                self.log(
                    f"[Registry] Cleaning up session"
                    f" directory: {self.session_root}")
            shutil.rmtree(self.session_root, ignore_errors=True)
            
    @property
    def registry_path(self) -> Path:
        return self._run_registry_path
  

def _update_manifest(
    run_dir: Union[str, Path],
    section: str,
    item: Union[Any, Dict[str, Any]],
    *,
    as_list: bool = False,
    manifest_kind:  Literal["training","tuning"] = None,
    name: str = "run_manifest.json",
    check_manifest : bool=True, 
) -> None:
    """
    Safely reads, updates, and writes a JSON manifest file.
    Read, update and atomically write a *training* or *tuning* manifest.

    This function is robust to the `run_dir` argument being either a
    directory path or a direct path to the manifest file itself.
    """
    # ── 1.  Decide the target file-name 
    kind_to_name = {
        "training": "run_manifest.json",
        "tuning":   "tuner_run_manifest.json",
    }
    if manifest_kind is not None:
        if manifest_kind not in kind_to_name:
            raise ValueError(
                f"manifest_kind must be 'training' or 'tuning' "
                f"(got {manifest_kind!r})"
            )
        name = kind_to_name[manifest_kind]

    if check_manifest and name not in kind_to_name.values():
        raise ValueError(
            f"Illegal manifest file name {name!r}. "
            "Allowed: 'run_manifest.json', 'tuner_run_manifest.json'"
        )
        
    # --- Robust Path Handling ---
    path_obj = Path(run_dir)
    manifest_path: Path

    if path_obj.is_dir():
        # If the provided path is a directory, append the default filename.
        manifest_path = path_obj / name
    elif str(path_obj).endswith('.json'):
        # If the provided path already points to a .json file, use it directly.
        manifest_path = path_obj
    else:
        # If it's a file but not a .json file, this is an invalid state.
        raise ValueError(f"Invalid path for manifest. Expected a directory or a"
                         f" .json file, but got: {path_obj}")

    # Ensure the PARENT directory of the manifest file exists.
    # This works correctly for both cases above.
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Read Existing Data (if any) ---
    data: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Handle case where manifest is corrupted
            pass 

    # --- Update Data ---
    sec = data.setdefault(section, {})

    if isinstance(item, dict):
        # Deep-update the section with the new dictionary items
        sec.update(item)
    else:
        if as_list:
            log_list = sec.setdefault("_", [])
            if item not in log_list:
                log_list.append(item)
        else:
            sec["_"] = item

    # --- Atomic Write ---
    # Write to a temporary file, then 
    # replace the original to prevent corruption.
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)

def update_manifest(
    run_dir: Union[str, Path],
    section: str,
    item: Union[Any, Dict[str, Any]],
    *,
    as_list: bool = False,
    name: str = "run_manifest.json",
) -> None:
    """
    Update a JSON manifest file safely.

    Read, modify, and write a manifest JSON file located in a run
    directory or at a direct .json path.

    Parameters
    ----------
    run_dir : Union[str, Path]
        Path to a directory containing the manifest, or to the manifest
        file itself. If a directory is passed, `name` will be appended.
    section : str
        Top-level key under which `item` will be stored.
    item : Union[Any, Dict[str, Any]]
        If dict, its keys merge into the section. Otherwise, stored
        under the '_' subkey (as a single value or list element).
    as_list : bool, optional
        If True and `item` is not a dict, append to a list under '_'.
        Defaults to False.
    name : str, optional
        Filename of the manifest when `run_dir` is a directory.
        Defaults to "run_manifest.json".

    Returns
    -------
    None
        Writes changes directly to disk by delegating to
        `_update_manifest`.

    See Also
    --------
    _update_manifest
        Implementation that handles path resolution, JSON load/update,
        atomic write, and error handling.

    Examples
    --------
    >>> from fusionlab.registry import update_manifest
    >>> # Overwrite a single entry under 'metrics'
    >>> update_manifest("runs/exp1", "metrics", 0.95)
    >>> # Append an error code under 'errors'
    >>> update_manifest(
    ...     "runs/exp1",
    ...     "errors",
    ...     "timeout",
    ...     as_list=True
    ... )
    """
    return _update_manifest(
        run_dir=run_dir,
        section=section,
        item=item,
        as_list=as_list,
        name=name
    )

def locate_manifest(
    log: Callable[[str], None] = print,
    _debug_mode: Optional[str] = "silence",
) -> Optional[Path]:
    """
    Locate the most recent `run_manifest.json` across known locations.

    This function searches both the persistent cache directory and any
    temporary session directories for all `run_manifest.json` files,
    then returns the one with the latest modification time.

    Parameters
    ----------
    log : Callable[[str], None], optional
        Logging function for status messages during search. Defaults to
        the built-in `print`.
    _debug_mode: str, optional  
        Controls verbosity of logging: ``'debug'`` enables messages;
        ``'silence'`` (default) mutes them.

    Returns
    -------
    Optional[Path]
        Absolute path to the most recently modified manifest file, or
        ``None`` if no manifest exists in any known location.

    See Also
    --------
    _locate_manifest
        Internal implementation handling the actual search logic.
    fusionlab.utils.manifest_registry.ManifestRegistry
        Class responsible for creating and managing run directories.

    Examples
    --------
    >>> # Simple call, no logging
    >>> path = locate_manifest()
    >>> if path is not None:
    ...     print(f"Latest manifest at: {path}")
    >>> # Enable debug messages
    >>> locate_manifest(log=lambda msg: print(f"[DEBUG] {msg}"),
    ...                _KIND='debug')
    """
    return _locate_manifest(log=log, _debug_mode=_debug_mode)


def _locate_manifest(
    log: Callable[[str], None] = print, 
    _debug_mode: Optional[str] ="silence", 
    *, 
    manifest_kind : Literal["training", "tuning"] | None = None,
    name          : Optional[str] = None,
    locate_both   : bool = False,
) -> Optional[Path] | Tuple[Optional[Path], Optional[Path]]:
    """Finds the most recent `run_manifest.json` across all possible locations.

    This function provides the definitive way to find the latest
    completed training run. It robustly searches both the persistent
    user cache directory and any temporary session directories created
    by the GUI.

    The search order is as follows:
    1.  Determines the path to the persistent cache
        (e.g., `~/.cache/fusionlab-learn/runs`).
    2.  Determines the system's temporary directory
        (e.g., `/tmp` or `C:\\Users\\...\\AppData\\Local\\Temp`).
    3.  Scans both locations for all `run_manifest.json` files.
    4.  Compares all found manifests and returns the single one with the
        most recent modification time.

    Parameters
    ----------
    log : callable, default=print
        A logging function to output status messages during the search.
    _debug_mode: str, optional 
       Whether for debugging (``'debug'``) or mute (``'silence'``). 
       Default is ``'silence'``.  
    manifest_kind :
        `"training"` → look for ``run_manifest.json``  
        `"tuning"`   → look for ``tuner_run_manifest.json``  
        `None`       → honour *name* (back-compat, default='run_manifest.json'
    name :
        Custom file-name when *manifest_kind* is *None*.
    locate_both :
        When *True* return a **pair** *(latest_training, latest_tuning)*.
        Useful for GUIs that want to ask which one to use.
        
    Returns
    -------
    Path | None  *or*  (Path | None, Path | None)
        The absolute path to the most recently modified manifest file found,
        or None if no runs exist in any location.

    Notes
    -----
    • Searches the persistent cache *and* all temp “session-only” runs.  
    • Chooses the file with the newest *mtime*.
    
    See Also
    --------
    fusionlab.utils.manifest_registry.ManifestRegistry : The class that manages
        the creation of these run directories.

    """
    kind_to_name = {
        "training": "run_manifest.json",
        "tuning":   "tuner_run_manifest.json",
    }
    if manifest_kind is not None:
        name = kind_to_name[manifest_kind]
    name = name or "run_manifest.json"

    def _find_latest(target_name: str) -> Optional[Path]:
        if _debug_mode =='debug':
            log("Searching for the latest training run...")
        all_manifests = []
        
        # 1. Search in the persistent cache directory
        # Instantiate the registry in default (persistent) mode to get the path
        persistent_registry = ManifestRegistry(log_callback=lambda *a: None)
        persistent_path = persistent_registry.root_dir
        if _debug_mode =='debug':
            log(f"  -> Checking persistent cache: {persistent_path}")
        all_manifests.extend(persistent_path.glob(f"*/{target_name}"))
    
        # 2. Search in any active temporary session directories
        temp_dir = Path(tempfile.gettempdir())
        if _debug_mode =='debug':
            log(f"  -> Checking for session directories in: {temp_dir}")
        # The prefix 'fusionlab_run_' matches what ManifestRegistry creates
        temp_run_dirs = temp_dir.glob("fusionlab_run_*")
        for run_dir in temp_run_dirs:
            if run_dir.is_dir():
                all_manifests.extend(run_dir.glob(f"**/{target_name}"))
    
        # 3. Find the most recent manifest among all found files
        if not all_manifests:
            if _debug_mode =='debug':
                log("  -> No manifest files found in any known location.")
            return None
    
        latest_manifest = max(all_manifests, key=lambda p: p.stat().st_mtime)
        if _debug_mode =='debug':
            log(f"  -> Found latest manifest: {latest_manifest}")
    
        return latest_manifest
    
    if locate_both:
        return (_find_latest(kind_to_name["training"]),
                _find_latest(kind_to_name["tuning"]))

    return _find_latest(name)

