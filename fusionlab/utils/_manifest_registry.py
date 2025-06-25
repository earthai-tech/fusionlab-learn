# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
Provides a robust system for managing and tracking training run artifacts
through a centralized manifest registry.
"""
from __future__ import annotations

import os
import json
import shutil
import tempfile
import atexit
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable 

try:
    from platformdirs import user_cache_dir
except ImportError as exc:
    raise ImportError(
        "The ManifestRegistry requires the `platformdirs` library. "
        "Please install it by running: pip install platformdirs"
    ) from exc

__all__ = ["ManifestRegistry", "_update_manifest"]

 
class ManifestRegistry:
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
    >>> # registry.purge_all()
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
        ) -> None:
        """Initializes the registry and ensures the root directory exists."""
        # The singleton pattern ensures __init__ is only run once.
        if self._initialized:
            return 
        
        self.log = log_callback or print 
        if session_only:
            # Create a unique temporary directory for this session.
            # The directory name will have a 'fusionlab_run_' prefix.
            self.root_dir = Path(tempfile.mkdtemp(prefix="fusionlab_run_"))
            # Register the cleanup function to be called on program exit.
            atexit.register(self.purge_all)
            self.log("[Registry] Initialized in session-only mode.")
        else:
            # Use the persistent cache directory.
            default_root = Path(
                user_cache_dir("fusionlab-learn", "earthai-tech")
            ) / self._RUNS_SUBDIR
            self.root_dir = Path(os.getenv(self._ENV_VAR, default_root))

        # Ensure the root directory exists.
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True
        self._run_registry_path =None 
        
    def new_run_dir(self, *, city: str = "unset", model: str = "unset") -> Path:
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
        source_path = Path(source_manifest_path)
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
            city = content.get("configuration", {}).get("city_name", "imported")
            model = content.get("configuration", {}).get("model_name", "run")
        except Exception:
            city, model = "imported", "run"
            
        new_run_dir = self.new_run_dir(city=city, model=model)
        
        # Copy the user's manifest into the new directory
        destination_path = new_run_dir / self._MANIFEST_FILENAME
        shutil.copyfile(source_path, destination_path)
        self._run_registry_path = destination_path
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
        manifests = list(self.root_dir.glob(f"*/{self._MANIFEST_FILENAME}"))
        if not manifests:
            return None
        return max(manifests, key=lambda p: p.stat().st_mtime)

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

    def purge_all(self) -> None:
        """Deletes the entire run registry directory.

        Warning: This is a destructive operation and will permanently
        remove all saved training runs and artifacts. It is primarily
        intended for cleanup during testing.
        """
        if self.root_dir and self.root_dir.exists():
            self.log(
                f"[Registry] Cleaning up session directory: {self.root_dir}")
            shutil.rmtree(self.root_dir, ignore_errors=True)
            
    
    @property
    def registry_path(self) -> Path:
        return self._run_registry_path
    
def _update_manifest(
    run_dir: Union[str, Path],
    section: str,
    item: Union[Any, Dict[str, Any]],
    *,
    as_list: bool = False,
    name: str = "run_manifest.json",
) -> None:
    """
    Safely reads, updates, and writes a JSON manifest file.

    This function is robust to the `run_dir` argument being either a
    directory path or a direct path to the manifest file itself.
    """
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
    # Write to a temporary file, then replace the original to prevent corruption.
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    

def _locate_manifest(
    start_path: Optional[Path] = None, # Kept for signature compatibility but ignored
    max_up: int = 3,                   # Kept for signature compatibility but ignored
    log: Callable[[str], None] = print
) -> Optional[Path]:
    """Finds the most recent `run_manifest.json` file from the central registry.

    This function provides the definitive way to find the latest completed
    training run by searching the cache directory managed by the
    ManifestRegistry. It replaces any previous heuristic search logic.

    Parameters
    ----------
    start_path : Path, optional
        This parameter is ignored and kept only for backward compatibility
        with the GUI's call signature. The function will always search
        the central registry.
    max_up : int, optional
        This parameter is ignored.
    log : callable, default=print
        A logging function to output status messages.

    Returns
    -------
    Optional[Path]
        The path to the most recently modified manifest file, or None if no
        runs are found in the registry.
    """
    log("Searching for the latest training run in the manifest registry...")
    try:
        # Instantiate the registry to get the central run directory
        registry = ManifestRegistry()
        # The registry itself has the logic to find the latest manifest
        latest_manifest = registry.latest_manifest()

        if latest_manifest:
            log(f"  Found latest manifest: {latest_manifest}")
        else:
            log("  No manifest files found in the registry.")
            
        return latest_manifest

    except Exception as e:
        log(f"[Error] Could not access the manifest registry: {e}")
        return None
    
def locate_manifest(
    log: Callable[[str], None] = print
) -> Optional[Path]:
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

    Returns
    -------
    Optional[Path]
        The absolute path to the most recently modified manifest file found,
        or None if no runs exist in any location.

    See Also
    --------
    fusionlab.utils.manifest_registry.ManifestRegistry : The class that manages
        the creation of these run directories.
    """
    import tempfile
    from fusionlab.utils._manifest_registry import ManifestRegistry

    log("Searching for the latest training run...")
    all_manifests = []

    # 1. Search in the persistent cache directory
    # Instantiate the registry in default (persistent) mode to get the path
    persistent_registry = ManifestRegistry(log_callback=lambda *a: None)
    persistent_path = persistent_registry.root_dir
    log(f"  -> Checking persistent cache: {persistent_path}")
    all_manifests.extend(persistent_path.glob("*/run_manifest.json"))

    # 2. Search in any active temporary session directories
    temp_dir = Path(tempfile.gettempdir())
    log(f"  -> Checking for session directories in: {temp_dir}")
    # The prefix 'fusionlab_run_' matches what ManifestRegistry creates
    temp_run_dirs = temp_dir.glob("fusionlab_run_*")
    for run_dir in temp_run_dirs:
        if run_dir.is_dir():
            all_manifests.extend(run_dir.glob("**/run_manifest.json"))

    # 3. Find the most recent manifest among all found files
    if not all_manifests:
        log("  -> No manifest files found in any known location.")
        return None

    latest_manifest = max(all_manifests, key=lambda p: p.stat().st_mtime)
    log(f"  -> Found latest manifest: {latest_manifest}")

    return latest_manifest

