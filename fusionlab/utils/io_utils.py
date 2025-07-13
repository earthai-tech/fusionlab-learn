# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

# Source: Adapted from earthai-tech/gofast/utils/io_utils (https://github.com/earthai-tech/gofast)
# This module is included in the FusionLab package, with modifications
# to fit FusionLab’s I/O conventions.

"""
Input/Output utilities for managing file paths, directories, and
loading serialized data within FusionLab. Provides error-checked
deserialization, directory management, and archive handling
(e.g., .tgz, .zip), streamlining file operations and data recovery.

Adapted for FusionLab from the original fusionlab.utils.io_utils.
"""

import os
import re
import shutil
import pickle
import joblib
import datetime
import warnings
import copy
import csv
import json
import yaml
import h5py
import tarfile
from pathlib import Path
from pprint import pformat
from typing import Optional, Union, Any, Tuple, List, Dict, Text

from six.moves import urllib
from zipfile import ZipFile
from tqdm import tqdm

import numpy as np
import pandas as pd

from .._fusionlog import fusionlog
from ..api.property import BaseClass
from ..compat.sklearn import validate_params
from ..core.array_manager import to_numeric_dtypes
from ..core.checks import (
    exist_features, check_files, is_in_if, str2columns
)
from ..core.io import EnsureFileExists
from ..core.utils import is_iterable, smart_format
from ..decorators import RunReturn, smartFitRun
from .validator import check_is_runned, is_frame
from ._dependency import import_optional_dependency

logger = fusionlog.get_fusionlab_logger(__name__)

__all__ = [
    'FileManager',
    'cpath',
    'deserialize_data',
    'extract_tar_with_progress',
    'fetch_tgz_from_url',
    'fetch_tgz_locally',
    'dummy_csv_translator',
    'fetch_json_data_from_url',
    'get_config_fname_from_varname',
    'get_valid_key',
    'key_checker',
    'key_search',
    'load_serialized_data',
    'load_csv',
    'move_cfile',
    'parse_csv',
    'parse_json',
    'parse_md',
    'parse_yaml',
    'print_cmsg',
    'rename_files',
    'sanitize_unicode_string',
    'save_job',
    'save_path',
    'serialize_data',
    'serialize_data_in',
    'spath',
    'store_or_write_hdf5',
    'to_hdf5',
    'zip_extractor',
    'fetch_joblib_data',
    'to_txt',
]

@smartFitRun 
class FileManager(BaseClass):
    r"""
    A class for managing and organizing files within a directory
    structure. This class provides methods to filter, organize,
    and rename files in bulk based on file extensions and name
    patterns. All operations are executed via the ``run`` method to
    ensure proper initialization and state management.

    Mathematically, if :math:`\mathcal{F}` represents the set of files
    in the root directory and :math:`\phi(f)` is a filtering function
    that selects files based on file type and name pattern, then the
    FileManager produces a subset

    .. math::
       \mathcal{F}' = \{ f \in \mathcal{F} \mid \phi(f) \}

    and performs operations such as moving or copying to reorganize
    these files into a target directory.

    Parameters
    ----------
    root_dir      : str
        The root directory containing the files to be managed.
        This directory must exist and contain the files subject to
        filtering.
    target_dir    : str
        The directory where the organized files will be placed.
        If necessary, this directory can be created when
        ``create_dirs`` is True.
    file_types    : list of str, optional
        A list of file extensions (e.g., ``['.csv', '.json']``) used to
        filter the files. If ``None``, no file type filtering is applied.
    name_patterns : list of str, optional
        A list of substrings (e.g., ``['2023', 'report']``) to filter
        file names. If ``None``, all file names are included.
    move          : bool, optional
        If True, files are moved from the source to the target directory;
        otherwise, they are copied. Default is False.
    overwrite     : bool, optional
        If True, existing files in the target directory will be overwritten.
        If False, existing files are skipped. Default is False.
    create_dirs   : bool, optional
        If True, missing directories in the target path are created.
        Default is False.

    Attributes
    ----------
    root_dir_     : str
        The validated root directory from which files are managed.
    target_dir_   : str
        The directory where the processed files are stored.

    Methods
    -------
    run(pattern, replacement)
        Executes the file organization process. It filters files using
        the criteria provided at initialization and, if a `pattern` and
        corresponding `replacement` are given, performs bulk renaming.
    get_processed_files()
        Returns a list of file paths that have been processed and
        organized into the target directory.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import FileManager
    >>> manager = FileManager(
    ...     root_dir='data/raw',
    ...     target_dir='data/processed',
    ...     file_types=['.csv', '.json'],
    ...     name_patterns=['2023', 'report'],
    ...     move=True,
    ...     overwrite=True,
    ...     create_dirs=True
    ... )
    >>> manager.run(pattern='old', replacement='new')
    >>> processed = manager.get_processed_files()
    >>> print(processed)

    Notes
    -----
    The public method ``run`` orchestrates the file management
    operations by first calling the internal method
    :meth:`_organize_files` to filter and move or copy files from the
    source directory to the target directory. If renaming is needed,
    :meth:`_rename_files` is invoked with the specified `pattern` and
    `replacement`. The method :meth:`get_processed_files` compiles a
    list of all files that have been organized, based on a walk of the
    target directory.

    See Also
    --------
    shutil.move : To move files between directories.
    shutil.copy2: To copy files while preserving file metadata.

    References
    ----------
    .. [1] Python Software Foundation. "os.walk — Directory tree
           generator". Python Documentation.
    .. [2] Python Software Foundation. "shutil — High-level file 
           operations". Python Documentation.
    """

    @validate_params({
        'root_dir'     : [str],
        'target_dir'   : [str],
        'file_types'   : [list, None],
        'name_patterns': [list, None],
        'move'         : [bool],
        'overwrite'    : [bool],
        'create_dirs'  : [bool]
    })
    def __init__(
        self,
        root_dir: str,
        target_dir: str,
        file_types: Optional[List[str]] = None,
        name_patterns: Optional[List[str]] = None,
        move: bool = False,
        overwrite: bool = False,
        create_dirs: bool = False
    ):
        # Assign parameters to instance attributes.
        self.root_dir      = root_dir
        self.target_dir    = target_dir
        self.file_types    = file_types
        self.name_patterns = name_patterns
        self.move          = move
        self.overwrite     = overwrite
        self.create_dirs   = create_dirs

        # Validate that the root directory exists.
        if not os.path.isdir(self.root_dir):
            raise ValueError(
                f"Root directory '{self.root_dir}' does not exist."
            )

        # If requested, create the target directory.
        if self.create_dirs and not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir, exist_ok=True)

        logger.debug(
            f"Initialized FileManager with root_dir: {self.root_dir}, "
            f"target_dir: {self.target_dir}"
        )

    @RunReturn
    def run(
        self,
        pattern: Optional[str] = None,
        replacement: Optional[str] = None
    ):
        r"""
        Executes file organization operations.

        This method filters files based on the specified file types
        and name patterns, then organizes them by moving or copying
        into the target directory. Additionally, if a `pattern` is
        provided, file names containing that pattern are renamed by
        replacing the pattern with the specified `replacement`.

        Parameters
        ----------
        pattern     : str, optional
            The substring to search for in file names. If provided,
            file names containing this pattern will be renamed.
        replacement : str, optional
            The string to replace `pattern` with in file names.
            Required if `pattern` is specified.

        Returns
        -------
        self : FileManager
            The instance itself after executing operations.

        Examples
        --------
        >>> manager = FileManager(...)
        >>> manager.run(pattern='old', replacement='new')
        """
        # Organize (move/copy) filtered files from root to target.
        self._organize_files()

        # If a renaming pattern is provided, execute renaming.
        if pattern is not None:
            if replacement is None:
                raise ValueError(
                    "Replacement string must be provided if pattern is specified."
                )
            self._rename_files(pattern, replacement)

        # Mark the manager as having run its operations.
        self._is_runned = True

    def get_processed_files(self) -> List[str]:
        r"""
        Retrieves a list of processed files in the target directory.

        Returns
        -------
        files : list of str
            A list containing the full paths of the files that have been
            organized into the target directory.

        Examples
        --------
        >>> manager = FileManager(...)
        >>> manager.run()
        >>> files = manager.get_processed_files()
        >>> print(files)
        """
        # Ensure the run method has been executed.
        check_is_runned(self, attributes=['_is_runned'])

        processed_files = []
        # Walk the target directory and accumulate file paths.
        for dirpath, _, filenames in os.walk(self.target_dir):
            for filename in filenames:
                processed_files.append(
                    os.path.join(dirpath, filename)
                )
        return processed_files

    def _organize_files(self):
        r"""Private method to filter and process files from root_dir."""
        try:
            # Retrieve the list of files that match filtering criteria.
            files = self._get_filtered_files()
            # Process each file individually.
            for file_path in files:
                self._handle_file(file_path)
        except Exception as e:
            logger.error(f"Failed to organize files: {str(e)}")
            raise RuntimeError(
                f"Organizing files failed: {str(e)}"
            ) from e

    def _rename_files(self, pattern: str, replacement: str):
        r"""Private method to rename files in target_dir by pattern."""
        try:
            # Traverse target_dir and rename files matching the pattern.
            for dirpath, _, filenames in os.walk(self.target_dir):
                for filename in filenames:
                    if pattern in filename:
                        old_path = os.path.join(dirpath, filename)
                        new_filename = filename.replace(pattern, replacement)
                        new_path = os.path.join(dirpath, new_filename)
                        # Skip renaming if file exists and overwrite is False.
                        if os.path.exists(new_path) and not self.overwrite:
                            logger.info(
                                f"File {new_path} already exists; skipping."
                            )
                            continue
                        os.rename(old_path, new_path)
                        logger.debug(
                            f"Renamed {old_path} to {new_path}"
                        )
        except Exception as e:
            logger.error(f"Failed to rename files: {str(e)}")
            raise RuntimeError(
                f"Renaming files failed: {str(e)}"
            ) from e

    def _get_filtered_files(self):
        r"""Retrieve files from root_dir filtered by file_types
        and name_patterns."""
        matched_files = []
        # Walk through root_dir recursively.
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                # Filter by file extension if file_types specified.
                if self.file_types and not any(
                    filename.endswith(ext) for ext in self.file_types
                ):
                    continue
                # Filter by name patterns if provided.
                if self.name_patterns and not any(
                    pat in filename for pat in self.name_patterns
                ):
                    continue
                matched_files.append(
                    os.path.join(dirpath, filename)
                )
        return matched_files

    def _handle_file(self, file_path):
        r"""Handle moving or copying a single file from root_dir
        to target_dir."""
        # Compute the relative path from the root directory.
        relative_path = os.path.relpath(file_path, self.root_dir)
        target_path   = os.path.join(self.target_dir, relative_path)

        # Ensure that the target directory exists.
        target_dir = os.path.dirname(target_path)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        # If the file exists in the target and overwrite is disabled, skip it.
        if os.path.exists(target_path) and not self.overwrite:
            logger.info(
                f"File {target_path} already exists; skipping."
            )
            return

        # Depending on the 'move' flag, either move or copy the file.
        if self.move:
            shutil.move(file_path, target_path)
            logger.debug(
                f"Moved {file_path} to {target_path}"
            )
        else:
            shutil.copy2(file_path, target_path)
            logger.debug(
                f"Copied {file_path} to {target_path}"
            )

def _update_manifest(
    run_dir: str,
    section: str,
    item: Union [Any, Dict[str, Any]],
    *,
    as_list: bool = False,
    name: str = "run_manifest.json",
) -> None:
    """
    Update <run_dir>/<name>  (=  *run_manifest.json* by default).

    Parameters
    ----------
    run_dir
        Folder that holds the manifest.
    section
        Top-level key inside the JSON file.
    item
        • If *dict*: merged **deep-update** style into `section`
          (same behaviour as before).

        • Otherwise: treated as a *value* to be stored under **`_`
          (underscore)** inside the `section` –  
          If *as_list* is ``True`` the value is **appended** to a list
          (created on first call).
    as_list
        Only meaningful when *item* is **not** a dict.
    name
        File name (default ``run_manifest.json``).
    """
    p = Path(run_dir) / name
    if p.exists():
        data: Dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = {}

    # make sure the section exists
    sec = data.setdefault(section, {})

    # ------------------------------------------------------------------
    if isinstance(item, dict):
        # old behaviour – deep update
        sec.update(item)

    else:  # scalar / list / str …
        if as_list:
            # keep a list under the magic "_" key
            lst = sec.setdefault("_", [])
            if item not in lst:           # avoid duplicates
                lst.append(item)
        else:
            # overwrite scalar value
            sec["_"] = item

    # write back atomically (simple way)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)              # atomic on POSIX & NTFS

def zip_extractor(
    zip_file, 
    samples: Union[int, str] = '*', 
    ftype: Optional[str] = None,  
    savepath: Optional[str] = None,
    pwd: Optional[Union[str, bytes]] = None
) -> list:
    r"""
    Extracts files from a ZIP archive based on various filtering
    criteria and saves them to a specified directory.

    The extraction process can be controlled by the ``samples``
    parameter to limit the number of files extracted, or by the
    ``ftype`` parameter to filter by a specific file extension.
    The resulting file names are returned as a list.

    .. math::
       \text{Extracted Files} = \{ f \in \mathcal{A} \mid \phi(f) \}

    where :math:`\mathcal{A}` is the set of all files in the archive,
    and :math:`\phi(f)` is a predicate that checks if a file matches
    the desired extension and is within the specified sample count.

    Parameters
    ----------
    zip_file  : str
        Full path to the ZIP archive file.
    samples   : int or str, optional
        Number of files to extract. If set to ``'*'``, all files are
        extracted. Default is ``'*'``.
    ftype     : str, optional
        File extension filter (e.g., ``'.csv'``). Only files with this
        extension are extracted. If no matching files are found,
        a ValueError is raised.
    savepath  : str, optional
        Directory where the extracted files will be stored. If not
        provided, files are extracted to the current working directory.
    pwd       : str or bytes, optional
        Password for encrypted ZIP files. If provided as a string,
        it will be used as is (or can be encoded to bytes as needed).

    Returns
    -------
    list of str
        A list of extracted file names (with paths).

    Examples
    --------
    >>> from fusionlab.utils.io_utils import zip_extractor
    >>> extracted_files = zip_extractor(
    ...     'data/archive.zip',
    ...     samples='*',
    ...     ftype='.csv',
    ...     savepath='data/extracted',
    ...     pwd='secret'
    ... )
    >>> print(extracted_files)
    ['folder1/file1.csv', 'folder2/file2.csv', ...]

    Notes
    -----
    The function first validates the input ZIP file using
    ``check_files`` (assumed to be defined in the package). It then
    determines the sample count and filters files by extension if
    ``ftype`` is provided. Extraction is done via the standard
    ``ZipFile.extract`` or ``ZipFile.extractall`` methods.

    See Also
    --------
    zipfile.ZipFile.extract : Extract a single file from a ZIP archive.
    zipfile.ZipFile.extractall : Extract all files from a ZIP archive.
    """

    def raise_msg_when(objn: list, ft: str) -> list:
        """
        Filters the list of file names to only include those that end
        with the given file extension. Raises a ValueError if no such
        file is found.

        Parameters
        ----------
        objn : list of str
            List of file names from the ZIP archive.
        ft   : str
            The file extension filter (e.g., ``'.csv'``).

        Returns
        -------
        list of str
            Filtered list of file names that match the file extension.
        """
        # Filter file names based on the extension.
        filtered = [o for o in objn if o.endswith(ft)]
        if len(filtered) == 0:
            # Determine available extensions.
            available_ext = [s.split('.')[-1] for s in objn if '.' in s]
            available_str = (
                f"{available_ext}" if available_ext else "None"
            )
            raise ValueError(
                f"No files in the archive match {ft!r}. "
                f"Available file types: {available_str}"
            )
        return filtered

    # Validate the ZIP file using check_files (assumed to be defined).
    zip_file = check_files(zip_file, formats='.zip', return_valid=True)

    # Convert samples parameter to an integer if not '*'.
    if isinstance(samples, str):
        if samples != '*':
            try:
                samples = int(samples)
            except Exception:
                raise ValueError(
                    "Parameter `samples` must be an integer or '*'"
                )
    elif not isinstance(samples, int):
        raise TypeError(
            "Parameter `samples` must be int or str,"
            f" not {type(samples).__name__!r}"
        )

    # Open the ZIP file for extraction.
    with ZipFile(zip_file, 'r') as zip_obj:
        # Get list of all file names in the archive.
        objnames = zip_obj.namelist()

        # Determine the sample count.
        if samples == '*':
            samples = len(objnames)

        # If file type filter is specified, filter file names.
        if ftype is not None:
            objnames = raise_msg_when(objn=objnames, ft=ftype)

        # Extract all files if samples exceed available files and
        # no file type filter is provided.
        if samples >= len(objnames) and ftype is None:
            zip_obj.extractall(path=savepath, pwd=pwd)
        else:
            # Extract only the first `samples` files.
            for zf in objnames[:samples]:
                zip_obj.extract(zf, path=savepath, pwd=pwd)

    return objnames

def to_hdf5(
    data,
    fn: str,
    objname: Optional[str] = None,
    close: bool = True,
    **hdf5_kws
) -> Any:
    r"""
    Store a data object in Hierarchical Data Format 5 (HDF5).

    This function serializes the input ``data`` into an HDF5 file. It
    supports both pandas DataFrames and NumPy arrays. If ``data`` is a
    DataFrame, it uses ``pd.HDFStore`` (which requires the ``pytables``
    package) to store the data. If ``data`` is a NumPy array, it uses
    ``h5py.File`` to create a dataset.

    The file path is constructed by concatenating the specified
    ``savepath`` (or the current working directory if ``savepath`` is
    not provided) with the provided filename (``fn``). The function
    automatically appends the appropriate file extension: ``.h5`` for
    DataFrames and ``.hdf5`` for arrays.

    .. math::
       \text{filepath} = \text{savepath} \oplus \text{filename} \oplus
       \text{extension}

    where :math:`\oplus` denotes string concatenation.

    Parameters
    ----------
    data       : Any
        The data object to be stored. Must be either a NumPy array or a
        pandas DataFrame.
    fn         : str
        The file path (without extension) where the HDF5 file will be
        saved.
    objname    : str, optional
        The name under which to store the data within the HDF5 file.
        Defaults to ``'data'`` if not provided.
    close      : bool, default=True
        If ``True``, the file is closed after writing. If ``False``,
        the file remains open for additional modifications.
    **hdf5_kws : dict, optional
        Additional keyword arguments to pass to the HDFStore
        constructor (for DataFrames) or to customize dataset creation
        (for arrays). Examples include:
          - ``mode``: File mode (e.g., ``'a'``, ``'w'``)
          - ``complevel``: Compression level (0-9)
          - ``complib``: Compression library (e.g., ``'zlib'``)
          - ``fletcher32``: Enable Fletcher32 checksum (bool)
          
        In more details:
            
        *  mode : {'a', 'w', 'r', 'r+'}, default 'a'
    
             ``'r'``
                 Read-only; no data can be modified.
             ``'w'``
                 Write; a new file is created (an existing file with the same
                 name would be deleted).
             ``'a'``
                 Append; an existing file is opened for reading and writing,
                 and if the file does not exist it is created.
             ``'r+'``
                 It is similar to ``'a'``, but the file must already exist.
         * complevel : int, 0-9, default None
             Specifies a compression level for data.
             A value of 0 or None disables compression.
         * complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
             Specifies the compression library to be used.
             As of v0.20.2 these additional compressors for Blosc are supported
             (default if no compressor specified: 'blosc:blosclz'):
             {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
              'blosc:zlib', 'blosc:zstd'}.
             Specifying a compression library which is not available issues
             a ValueError.
         * fletcher32 : bool, default False
             If applying compression use the fletcher32 checksum.
    Returns
    -------
    store : object
        An IO interface for the stored data. For DataFrames, this is a
        ``pd.HDFStore`` object; for arrays, an ``h5py.File`` object.

    Examples
    --------
    >>> import os
    >>> import pandas as pd
    >>> from fusionlab.utils.io_utils import to_hdf5
    >>> data = pd.DataFrame({
    ...     'a': [1, 2, 3],
    ...     'b': [4, 5, 6]
    ... })
    >>> save_path = os.path.join('output', 'datafile')
    >>> store = to_hdf5(data, fn=save_path, objname='mydata', verbose=1)
    >>> # Access stored data:
    >>> retrieved = store['mydata']
    >>> print(retrieved.head())

    Notes
    -----
    - Ensure the dependency ``pytables`` is installed when serializing a
      DataFrame. If not, an error will be raised.
    - When serializing NumPy arrays, the dataset is created with the name
      ``"dataset_01"``.
    - If ``close`` is set to ``False``, the caller is responsible for
      closing the store.

    See Also
    --------
    joblib.dump : For serializing objects using Joblib.
    pickle.dump : For serializing objects using Pickle.
    h5py.File   : For working with HDF5 files in Python.

    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011).
           "The NumPy Array: A Structure for Efficient Numerical
           Computation". Computing in Science & Engineering, 13(2), 22-30.
    """

    # Validate that data is either a NumPy array or a pandas DataFrame.
    if not (isinstance(data, np.ndarray) or isinstance(data, pd.DataFrame)):
        raise TypeError(
            f"Expect a numpy array or pandas DataFrame, not "
            f"{type(data).__name__!r}"
        )

    # Remove any existing HDF5 file extension from the provided filename.
    fn = str(fn).replace('.h5', "").replace('.hdf5', "")

    store = None
    if isinstance(data, pd.DataFrame):
        # Ensure the dependency 'pytables' is installed.
        import_optional_dependency('tables')
        # Create an HDFStore with a .h5 extension.
        # remove extension if exist.
        fn = str(fn).replace ('.h5', "").replace(".hdf5", "")
        store = pd.HDFStore(fn + '.h5', **hdf5_kws)
        # Use the provided objname or default to 'data'.
        objname = objname or 'data'
        store[str(objname)] = data
    else:
        # Convert data to a NumPy array (if not already).
        data = np.asarray(data)

        # Create an HDF5 file with a .hdf5 extension.
        store = h5py.File(f"{fn}.hdf5", "w")
        # Create a dataset named "dataset_01" with the array data.
        store.create_dataset(
            "dataset_01",
            data.shape,
            dtype=data.dtype,
            data=data
        )
    # Optionally close the store.
    if close:
        store.close()

    return store

def store_or_write_hdf5 (
    df,  
    key:str= None, 
    mode:str='a',  
    kind: str=None, 
    path_or_buf:str= None, 
    encoding:str="utf8", 
    csv_sep: str=",",
    index: bool=..., 
    columns:Union [str, List[Any]]=None, 
    sanitize_columns:bool=False,  
    func: Optional[callable]= None, 
    args: tuple=(), 
    applyto: Union [str, List[Any]]=None, 
    **func_kwds, 
    )->Union [None, pd.DataFrame]: 
    """ Store data to hdf5 or write data to csv file. 
    
    Note that by default, the data is not store nor write and 
    return data if frame or transform the Path-Like object to data frame. 

    Parameters 
    -----------
    d: Dataframe, shape (m_samples, n_features)
        data to store or write or sanitize.
    key:str
       Identifier for the group in the store.
       
    mode: {'a', 'w', 'r+'}, default 'a'
       Mode to open file:
    
       - 'w': write, a new file is created (an existing file with the 
                                          same name would be deleted).
       - 'a': append, an existing file is opened for reading and writing, 
         and if the file does not exist it is created.
       - 'r+': similar to 'a', but the file must already exist.
       
    kind: str, {'store', 'write', None} , default=None 
       Type of task to perform: 
           
       - 'store': Store data to hdf5
       - 'write': export data to csv file.
       - None: construct a dataframe if array is passed or sanitize it. 

    path_or_buf: str or pandas.HDFStore, or str, path object, file-like \
        object, or None, default=None 
       File path or HDFStore object. String, path object
       (implementing os.PathLike[str]), or file-like object implementing 
       a write() function. If ``write=True`` and  None, the result is returned 
       as a string. If a non-binary file object is passed, it should be 
       opened with newline=" ", disabling universal newlines. If a binary 
       file object is passed, mode might need to contain a 'b'.
      
    encoding: str, default='utf8'
       A string representing the encoding to use in the output file, 
       Encoding is not supported if path_or_buf is a non-binary file object. 

    csv_sep: str, default=',', 
       String of length 1. Field delimiter for the output file.
       
    index: bool, index =False, 
       Write data to csv with index or not. 
       
    columns: list of str, optional 
        Usefull to create a dataframe when array is passed. Be aware to fit 
        the number of array columns (shape[1])
        
    sanitize_columns: bool, default=False, 
       remove undesirable character in the data columns using the default
       argument of `regex` parameters and fill pattern to underscore '_'. 
       The default regex implementation is:: 
           
           >>> import re 
           >>> re.compile (r'[_#&.)(*@!,;\s-]\s*', flags=re.IGNORECASE)
           
    func: callable, Optional 
       A custom sanitizing function and apply to each columns of the dataframe.
       If provide, the expected columns must be listed to `applyto` parameter.
       
    args: tuple, optional 
       Positional arguments of the sanitizing columns 
       
    applyto: str or list of str, Optional 
       The list of columns to apply the function ``func``. To apply the 
       function to all columns, use the ``*`` instead. 
       
    func_kwds: dict, 
       Keywords arguments of the sanitizing function ``func``. 
       
    Return 
    -------
    None or d: None of dataframe. 
      returns None if `kind` is set to ``write`` or ``store`` otherwise 
      return the dataframe. 
  
    Examples
    --------
    >>> from fusionlab.utils.io_utils import store_or_write_hdf5
    >>> from fusionlab.datasets import load_bagoue 
    >>> data = load_bagoue().frame 
    >>> data.geol[:5]
    0    VOLCANO-SEDIM. SCHISTS
    1                  GRANITES
    2                  GRANITES
    3                  GRANITES
    4          GEOSYN. GRANITES
    Name: geol, dtype: object
    >>> data = store_or_write_hdf5 ( data, sanitize_columns = True)
    >>> data[['type', 'geol', 'shape']] # put all to lowercase
      type                    geol shape
    0   cp  volcano-sedim. schists     w
    1   ec                granites     v
    2   ec                granites     v
    >>> # compute using func 
    >>> def test_func ( a, times  , to_percent=False ): 
            return ( a * times / 100)   if to_percent else ( a *times )
    >>> data.sfi[:5]
    0    0.388909
    1    1.340127
    2    0.446594
    3    0.763676
    4    0.068501
    Name: sfi, dtype: float64
    >>> d = store_or_write_hdf5 ( data,  func = test_func, args =(7,), applyto='sfi')
    >>> d.sfi[:5] 
    0    2.722360
    1    9.380889
    2    3.126156
    3    5.345733
    4    0.479507
    Name: sfi, dtype: float64
    >>> store_or_write_hdf5 ( data,  func = test_func, args =(7,),
                          applyto='sfi', to_percent=True).sfi[:5]
    0    0.027224
    1    0.093809
    2    0.031262
    3    0.053457
    4    0.004795
    Name: sfi, dtype: float64
    >>> # write data to hdf5 and outputs to current directory 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data.h5', 
                          kind ='store')
    >>> # export data to csv 
    >>> store_or_write_hdf5 ( d, key='test0', path_or_buf= 'test_data', 
                          kind ='export')
    """
    
    kind= key_search (str(kind), default_keys=(
        "none", "store", "write", "export", "tocsv"), 
        raise_exception=True , deep=True)[0]
    
    kind = "export" if kind in ('write', 'tocsv') else kind 
    
    is_frame(df, df_only =True, raise_exception=True, objname="Data") 
    
    d = to_numeric_dtypes(
        df, columns=columns,sanitize_columns=sanitize_columns, 
        fill_pattern='_')
   
    # get categorical variables 
    if ( sanitize_columns 
        or func is not None
        ): 
        d, _, cf = to_numeric_dtypes(d, return_feature_types= True )
        #( strip then pass to lower case all non-numerical data) 
        # for minimum sanitization  
        for cat in cf : 
            d[cat]= d[cat].str.lower()
            d[cat]= d[cat].str.strip()
            
    if func is not None: 
        if not callable(func): 
            raise TypeError(
                f"Expect a callable for `func`. Got {type(func).__name__!r}")

        if applyto is None:
            raise ValueError("Need to specify the data column to apply"
                             f"{func.__name__!r} to.")
        
        applyto = is_iterable( 
            applyto, exclude_string=True, transform =True 
            ) if applyto !="*" else d.columns 
        # check whether the applyto columns are in data columns 
        exist_features(d, applyto)
        
        # map each colum 
        for col in applyto: 
            d [col]=d[col].apply( func, args=args, **func_kwds )

    # store in h5 file. 
    if kind=='store':
        if path_or_buf is None: 
            print("Destination file is missing. Use 'data.h5' instead outputs"
                  f" in the current directory {os.getcwd()}")
            path_or_buf= 'data.h5'
 
        d.to_hdf ( path_or_buf , key =key, mode =mode )
    # export to csv file
    if kind=="export": 
        d.to_csv(path_or_buf, encoding = encoding  , sep=csv_sep , 
                 index =False if index is ... else index   )
        
    return d if kind not in ("store", "export") else None 

def key_checker (
    keys: str ,   
    valid_keys:List[str], 
    regex:re = None, 
    pattern:str = None , 
    deep_search:bool =False
    ): 
    """check whether a give key exists in valid_keys and return a list if 
    many keys are found.
    
    Parameters 
    -----------
    keys: str, list of str 
       Key value to find in the valid_keys 
       
    valid_keys: list 
       List of valid keys by default. 
       
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    deep_search: bool, default=False 
       If deep-search, the key finder is no sensistive to lower/upper case 
       or whether a numeric data is included. 
 
       
    Returns 
    --------
    keys: str, list , 
      List of keys that exists in the `valid_keys`. 
      
    Examples
    --------
    
    >>> from fusionlab.utils.io_utils import key_checker
    >>> key_checker('h502', valid_keys= ['h502', 'h253','h2601'])  
    Out[68]: 'h502'
    >>> key_checker('h502+h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[69]: ['h502', 'h2601']
    >>> key_checker('h502 h2601', valid_keys= ['h502', 'h253','h2601'])
    Out[70]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2601'], valid_keys= ['h502', 'h253','h2601'])
    Out[73]: ['h502', 'h2601']
    >>> key_checker(['h502',  'h2602'], valid_keys= ['h502', 'h253','h2601'])
    UserWarning: key 'h2602' is missing in ['h502', 'h2602']
    Out[82]: 'h502'
    >>> key_checker(['502',  'H2601'], valid_keys= ['h502', 'h253','h2601'], 
                    deep_search=True )
    Out[57]: ['h502', 'h2601']
    
    """
    _keys = copy.deepcopy(keys)
    valid_keys = is_iterable(valid_keys , exclude_string =True, transform =True )
    if isinstance ( keys, str): 
        pattern = pattern or '[_#&@!_+,;\s-]\s*'
        keys = str2columns (keys, regex = regex , pattern=pattern )
    # If iterbale object , save obj 
    # to improve error 
    kkeys = copy.deepcopy(keys)
    if deep_search: 
        keys = key_search(
            keys, 
            default_keys= valid_keys,
            deep=True, 
            raise_exception= True,
            regex =regex, 
            pattern=pattern 
            )
        return keys[0] if len(keys)==1 else keys 
    # for consistency 
    keys = [ k for k in keys if ''.join(
        [ str(i) for i in valid_keys] ).find(k)>=0 ]
    # assertion error if key does not exist. 
    if len(keys)==0: 
        verb1, verb2 = ('', 'es') if len(kkeys)==1 else ('s', '') 
        msg = (f"key{verb1} {_keys!r} do{verb2} not exist."
               f" Expect {' ,'.join(valid_keys)}") # dont use smartformat here 
        raise KeyError ( msg )
        
    if len(keys) != len(kkeys):
        # dont use is_in_if 
        miss_keys = is_in_if ( kkeys, keys , return_diff= True , error ='ignore')
        miss_keys, verb = (miss_keys[0], 'is') if len( miss_keys) ==1 else ( 
            miss_keys, 'are')
        warnings.warn(f"key{'' if verb=='is' else 's'} {miss_keys!r} {verb}"
                      f" missing in {_keys}")
    keys = keys[0] if len(keys)==1 else keys 
    
    return keys
 

def key_search (
    keys: str,  
    default_keys: Union [Text , List[str]], 
    parse_keys: bool=True, 
    regex :re=None, 
    pattern :str=None, 
    deep: bool =...,
    raise_exception:bool=..., 
    ): 
    """Find key in a list of default keys and select the best match. 
    
    Parameters 
    -----------
    keys: str or list 
       The string or a list of key. When multiple keys is passed as a string, 
       use the space for key separating. 
       
    default_keys: str or list 
       The likehood key to find. Can be a litteral text. When a litteral text 
       is passed, it is better to provide the regex in order to skip some 
       character to parse the text properly. 
       
    parse_keys: bool, default=True 
       Parse litteral string using default `pattern` and `regex`. 
       
       .. versionadded:: 0.2.7 
        
    regex: `re` object,  
        Regular expresion object. Regex is important to specify the kind
        of data to parse. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&*@!_,;\s-]\s*', flags=re.IGNORECASE)
            
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns. Pattern is 
        important especially when some character are considers as a part of 
        word but they are not a separator. For example a data columns with 
        a name `'DH_Azimuth'`, if a pattern is not explicitely provided, 
        the default pattern will parse as two separated word which is far 
        from the expected results. 
        
    deep: bool, default=False 
       Not sensistive to uppercase. 
       
    raise_exception: bool, default=False 
       raise error when key is not find. 
       
    Return 
    -------
    list: list of valid keys or None if not find ( default) 

    Examples
    ---------
    >>> from fusionlab.utils.io_utils import key_search 
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'])
    Out[44]: ['h502']
    >>> key_search('h502-hh2601', default_keys= ['h502', 'h253','HH2601'], 
                   deep=True)
    Out[46]: ['h502', 'HH2601']
    >>> key_search('253', default_keys= ("I m here to find key among h502,
                                             h253 and HH2601"))
    Out[53]: ['h253'] 
    >>> key_search ('east', default_keys= ['DH_East', 'DH_North']  , deep =True,)
    Out[37]: ['East']
    key_search ('east', default_keys= ['DH_East', 'DH_North'], 
                deep =True,parse_keys= False)
    Out[39]: ['DH_East']
    """
    def _ellipsis2false ( param): 
        if param ==...:
            return False 
        return True 
    deep, raise_exception, parse_keys = [_ellipsis2false(
        param) for param in [deep, raise_exception, parse_keys] ] 
    # make a copy of original keys 
    
    kinit = copy.deepcopy(keys)
    if parse_keys: 
        if is_iterable(keys , exclude_string= True ): 
            keys = ' '.join ( [str(k) for k in keys ]) 
             # for consisteny checker 
        pattern = pattern or '[#&@!_+,;\s-]\s*'
        keys = str2columns ( keys , regex = regex , pattern = pattern ) 
            
        if is_iterable ( default_keys , exclude_string=True ): 
            default_keys = ' '. join ( [ str(k) for k in default_keys ])
            # make a copy
        default_keys =  str2columns(
            default_keys, regex =regex , pattern = pattern )
    else : 
        keys = is_iterable(
        keys, exclude_string = True, transform =True )
        default_keys = is_iterable ( 
            default_keys, exclude_string=True, transform =True )
        
    dk_init = copy.deepcopy(default_keys )
    # if deep convert all keys to lower 
    if deep: 
        keys= [str(it).lower() for it in keys  ]
        default_keys = [str(it).lower() for it in default_keys  ]

    valid_keys =[] 
    for key in keys : 
        for ii, dkey in enumerate (default_keys) : 
            vk = re.findall(rf'\w*{key}\w*', dkey)
            # rather than rf'\b\w*{key}\w*\b'
            # if deep take the real values in defaults keys.
            if len(vk) !=0: 
                if deep: valid_keys.append( dk_init[ii] )
                else:valid_keys.extend( vk)
                break     
    if ( raise_exception 
        and len(valid_keys)==0
        ): 
        kverb ='s' if len(kinit)> 1 else ''
        raise KeyError (f"key{kverb} {kinit!r} not found."
                       f" Expect {','.join(dk_init)}")
    return None if len(valid_keys)==0 else valid_keys 


def serialize_data_in(
    data,
    filename: str = None,
    force: bool = True,
    savepath: str = None,
    verbose: int = 0
) -> str:
    r"""
    Serializes a Python object to a binary file using either joblib 
    or pickle.

    This function attempts to serialize the input `data` using the 
    ``joblib.dump`` method. If this attempt fails, it falls back 
    to using ``pickle.dump``. The final file path is constructed 
    by concatenating the directory specified by ``savepath`` (or the 
    current working directory if ``savepath`` is None) with the 
    given ``filename``. Mathematically, the file path is given by:

    .. math::
       \text{filepath} = \text{savepath} \oplus \text{filename}

    where :math:`\oplus` denotes string concatenation.

    Parameters
    ----------
    data: Any
        The Python object to serialize. It must be compatible with 
        either ``joblib`` or ``pickle`` serialization.
    filename     : str, optional
        The target filename for the serialized data. If ``None``, a 
        filename is generated using the current timestamp formatted as 
        ``"%Y%m%d%H%M%S"`` (e.g., ``"serialized_20230315123045.pkl"``).
    force        : bool, default=True
        Determines whether to overwrite an existing file with the same 
        filename. If ``False``, a timestamp is appended to the filename 
        to ensure uniqueness.
    savepath     : str, optional
        The directory in which to save the serialized file. If not 
        specified, the file is saved to the current working directory 
        (``os.getcwd()``).
    verbose      : int, default=0
        Controls the verbosity of output messages. Higher values 
        produce more detailed logging during the serialization process.

    Returns
    -------
    str
        The complete file path to which the data has been serialized.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import serialize_data_in
    >>> data = {"a": 1, "b": 2}
    >>> filepath = serialize_data_in(data, filename='data.pkl', 
    ...                              force=True, verbose=1)
    >>> print(filepath)
    /path/to/current/directory/data.pkl

    Notes
    -----
    The function first tries to serialize the input `data` using 
    ``joblib.dump``. In case of any exception during this attempt, it 
    falls back to using ``pickle.dump``. This dual approach improves 
    robustness in diverse runtime environments where one serialization 
    method might be unsupported or encounter issues with the given data 
    type.

    See Also
    --------
    joblib.dump : Serialize objects to disk using Joblib.
    pickle.dump : Serialize objects to disk using Pickle.
    os.getcwd    : Retrieve the current working directory.

    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011).
           "The NumPy Array: A Structure for Efficient Numerical
           Computation". Computing in Science & Engineering, 13(2), 22-30.
    """
    # Determine the filename: if not provided, generate one using a timestamp.
    if filename is None:
        filename = (
            f"serialized_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
        )

    # Determine the save directory (default is current working directory)
    directory = savepath if savepath is not None else os.getcwd()
    filepath  = os.path.join(directory, filename)

    # If the file exists and force is False, modify the filename to ensure uniqueness.
    if os.path.exists(filepath) and not force:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if filename.endswith('.pkl'):
            filename = filename.replace('.pkl', f"_{timestamp}.pkl")
        else:
            filename = f"{filename}_{timestamp}.pkl"
        filepath = os.path.join(directory, filename)

    # Attempt to serialize the data using joblib.dump.
    try:
        joblib.dump(data, filepath)
        if verbose > 0:
            print(f"[INFO] Data serialized to {filepath} using joblib.")
    except Exception as e:
        if verbose > 0:
            print(
                f"[WARN] joblib.dump failed with error: {e}. "
                "Falling back to pickle.dump..."
            )
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            if verbose > 0:
                print(f"[INFO] Data serialized to {filepath} using pickle.")
        except Exception as e2:
            raise RuntimeError(
                f"Serialization failed using both joblib and pickle: {e2}"
            ) from e2

    return filepath

    if filename is None:
        filename = f"serialized_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    
    filepath = os.path.join(savepath or os.getcwd(), filename)
    
    if os.path.exists(filepath) and not force:
        filename = filename.replace(
            '.pkl', f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")
        filepath = os.path.join(savepath or os.getcwd(), filename)
        
    try:
        joblib.dump(data, filepath)
        if verbose > 0:
            print(f"Data serialized to {filepath}")
    except Exception as e: # noqa 
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        if verbose > 0:
            print(f"Data serialized using pickle to {filepath}")
    
    return filepath


def save_path(nameOfPath: str) -> str:
    """
    Creates a directory if it does not exist.

    Parameters
    ----------
    nameOfPath : str
        Name or path of the directory to create.

    Returns
    -------
    str
        The path of the created directory. If it exists, returns the existing path.

    Examples
    --------
    >>> save_path("test_directory")
    'path/to/test_directory'
    """
    path = os.path.join(os.getcwd(), nameOfPath)
    os.makedirs(path, exist_ok=True)
    return path

def sanitize_unicode_string(str_: str) -> str:
    """
    Removes spaces and replaces accented characters in a string.

    Parameters
    ----------
    str_ : str
        The string to sanitize.

    Returns
    -------
    str
        The sanitized string with removed spaces and replaced accents.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import sanitize_unicode_string 
    >>> sentence ='Nos clients sont extrêmement satisfaits '
        'de la qualité du service fourni. En outre Nos clients '
            'rachètent frequemment nos "services".'
    >>> sanitize_unicode_string  (sentence)
    ... 'nosclientssontextrmementsatisfaitsdelaqualitduservice'
        'fournienoutrenosclientsrachtentfrequemmentnosservices'
    >>> sanitize_unicode_string("Élève à l'école")
    'elevealecole'
    """
    accents_replacements = {'éèê': 'e', 'àâ': 'a'}
    str_ = re.sub(r'\s+', '', str_.lower())
    
    for chars, repl in accents_replacements.items():
        str_ = re.sub(f"[{chars}]", repl, str_)
    
    return str_

def parse_md(pf: str, delimiter: str = ':'):
    """
    Parse a markdown-style file with key-value pairs separated by 
    a delimiter.

    Parameters
    ----------
    pf : str
        Path to the markdown file containing key-value pairs.
    delimiter : str, default=':'
        Delimiter used to separate key-value pairs.

    Yields
    ------
    Tuple[str, str]
        A tuple containing the key and processed value.

    Raises
    ------
    IOError
        If the provided path does not lead to a valid file.

    Notes
    -----
    - This function yields key-value pairs by reading the file line-by-line.
    - It applies `sanitize_unicode_string` to keys to ensure data consistency.

    Examples
    --------
    >>> list(parse_md_data('parser_file.md', delimiter=':'))
    [('key1', 'Value1'), ('key2', 'Value2')]
    """
    if not os.path.isfile(pf):
        raise IOError("Unable to detect the parser file. Need a Path-like object.")

    with open(pf, 'r', encoding='utf8') as f:
        pdata = f.readlines()

    for row in pdata:
        if row in ('\n', ' '):
            continue
        fr, en = row.strip().split(delimiter)
        fr = sanitize_unicode_string(fr)  # Clean up the key
        en = en.strip()
        
        # Capitalize the first letter of the value
        en = en[0].upper() + en[1:]
        
        yield fr, en

def dummy_csv_translator(
        csv_fn: str, pf: str, delimiter: str = ':',
        destfile: str = 'pme.en.csv'):
    """
    Translate a CSV file using a dictionary created from a markdown-style parser file.

    Parameters
    ----------
    csv_fn : str
        Path to the source CSV file.
    pf : str
        Path to the markdown-style file used to create the translation dictionary.
    delimiter : str, default=':'
        Delimiter used in the parser file to separate key-value pairs.
    destfile : str, default='pme.en.csv'
        Name of the destination file for the translated CSV.

    Returns
    -------
    DataFrame
        Translated CSV data as a DataFrame.
    list
        List of untranslated terms found in the source CSV.

    Notes
    -----
    - This function uses `parse_md_data` to read the parser file and apply
      translations to the CSV content.
    - Missing translations are collected and returned for review.

    Examples
    --------
    >>> df, missing = dummy_csv_translator(
        "data.csv", "parser_file.md", delimiter=":", destfile="output.csv")
    >>> print(df.head())
    >>> print(missing)

    """
    parser_data = dict(parse_md(pf, delimiter))
    
    # Read CSV data
    with open(csv_fn, 'r', encoding='utf8') as csv_f:
        csv_reader = csv.reader(csv_f)
        csv_data = [row for row in csv_reader]

    # Locate 'Industry_type' column and split data blocks
    industry_index = csv_data[0].index('Industry_type')
    csv_1b = [row[:industry_index + 1] for row in csv_data]
    csv_2b = [row[industry_index + 1:] for row in csv_data]

    # Clean data in `csv_1b` and collect missing translations
    csv_1b_cleaned = copy.deepcopy(csv_1b)
    untranslated_terms = set()
    for row in csv_1b_cleaned[3:]:
        for i, value in enumerate(row):
            value = sanitize_unicode_string(value.strip())
            if value not in parser_data:
                untranslated_terms.add(value)
            else:
                row[i] = parser_data.get(value, value)

    # Combine cleaned blocks and convert to DataFrame
    combined_data = [r1 + r2 for r1, r2 in zip(csv_1b_cleaned, csv_2b)]
    df = pd.DataFrame(np.array(combined_data[1:]), columns=combined_data[0])

    # Apply parser dictionary and save to destination file
    df.replace(parser_data, inplace=True)
    df.to_csv(destfile, index=False)
    
    return df, list(untranslated_terms)

def rename_files(
    src_files: Union[str, List[str]], 
    dst_files: Union[str, List[str]], 
    basename: Optional[str] = None, 
    extension: Optional[str] = None, 
    how: str = 'py', 
    prefix: bool = True, 
    keep_copy: bool = True, 
    trailer: str = '_', 
    sortby: Union[re.Pattern, callable] = None, 
    **kws
) -> None:
    """Rename files in directory.

    Parameters 
    -----------
    src_files: str, Path-like object 
       Source files to rename 
      
    dst_files: str of PathLike object 
       Destination files renamed. 
       
    extension: str, optional 
       If a path is given in `src_files`, specifying the `extension` will just 
       collect only files with this typical extensions. 
       
    basename: str, optional 
       If `dst_files` is passed as Path-object, name should be needed 
       for a change, otherwise, the number is incremented using the Python 
       index counting defined by the parameter ``how=py` 
        
    how: str, default='py' 
       The way to increment files when `dst_files` is given as a Path object. 
       For instance, for a  ``name=E_survey`` and ``prefix==True``, the first 
       file should be ``E_survey_00`` if ``how='py'`` otherwise it should be 
       ``E_survey_01``.
     
    prefix: bool, default=True
      Prefix is used to position the name before the number incrementation. 
      If ``False`` and `name` is given, the number is positionning before the 
      name. If ``True`` and not `prefix` for a ``name=E_survey``, it should be 
      ``00_E_survey`` and ``01_E_survey``. 

    keep_copy: bool, default=True 
       Keep a copy of the source files. 
       
    trailer: str, default='_', 
       Item used to separate the basename for counter. 
       
    sortby: Regex or Callable, 
       Key to sort the collection of the items when `src_files` is passed as 
       a path-like object.  This is usefull to keep order as the origin files 
       especially  when files includes a specific character.  Furthermore 
       [int| float |'num'|'digit'] sorted the files according to the
       number included in the filename if exists. 

    kws: dict 
       keyword arguments passed to `os.rename`. 

    """ 
    dest_dir =None ; trailer = str(trailer)
    extension = str(extension).lower()
    
    if os.path.isfile (src_files ): 
        src_files = [src_files ] 
        
    elif os.path.isdir (src_files): 
        src_path = src_files
        ldir = os.listdir(src_path) 

        src_files = ldir if extension =='none' else [
             f for f in ldir  if  f.endswith (extension) ]
    
        if sortby: 
            if sortby in ( int, float, 'num', 'number', 'digit'): 
                src_files = sorted(ldir, key=lambda s:int( re.search(
                    '\d+', s).group()) if re.search('\d+', s) else 0 )
            else: 
                src_files = sorted(ldir, key=sortby)

        src_files = [  os.path.join(src_path, f )   for f in src_files  ] 
        # get only the files 
        src_files = [ f for f in src_files if os.path.isfile (f ) ]

    else : raise FileNotFoundError(f"{src_files!r} not found.") 
    
    # Create the directory if it doesn't exist
    if ( dst_files is not None 
        and not os.path.exists (dst_files)
        ): 
        os.makedirs(dst_files)
        
    if os.path.isdir(dst_files): 
        dest_dir = dst_files 
    
    if isinstance (dst_files, str): 
        dst_files = [dst_files]
        #XXX revise 
   # dst_files = is_iterable(dst_files , exclude_string= True, transform =True ) 
    # get_extension of the source_files 
    _, ex = os.path.splitext (src_files[0]) 
    
    if dest_dir: 
        if basename is None: 
            warnings.warn(
                "Missing basename for renaming file. Should use `None` instead.")
            basename =''; trailer =''
            
        basename= str(basename)
        if prefix: 
            dst_files =[ f"{str(basename)}{trailer}" + (
                f"{i:03}" if how=='py' else f"{i+1:03}") + f"{ex}"
                        for i in range (len(src_files))]
        elif not prefix: 
            dst_files =[ (f"{i:03}" if how=='py' else f"{i+1:03}"
                        ) +f"{trailer}{str(basename)}" +f"{ex}"
                        for i in range (len(src_files))]
        
        dst_files = [os.path.join(dest_dir , f) for f in dst_files ] 
    
    for f, nf in zip (src_files , dst_files): 
        try: 
           if keep_copy : shutil.copy (f, nf , **kws )
           else : os.rename (f, nf , **kws )
        except FileExistsError: 
            os.remove(nf)
            if keep_copy : shutil.copy (f, nf , **kws )
            else : os.rename (f, nf , **kws )
            
@EnsureFileExists
def fetch_joblib_data(
    job_file: str, 
    *keys: str, 
    error_mode: str = 'raise', 
    verbose: int = 0
) -> Union[Dict[str, Any], Tuple[Any, ...]]:
    """Dynamically load data from a joblib-saved dictionary with
    flexible key access.

    Parameters
    ----------
    job_file : str
        Path to the joblib file containing a dictionary
    *keys : str
        Variable-length list of dictionary keys to retrieve
    error_mode : {'raise', 'warn', 'ignore'}, default='raise'
        Handling of missing keys:
        - 'raise': Immediately raise KeyError
        - 'warn': Issue warning and skip missing keys
        - 'ignore': Silently skip missing keys
    verbose : int, default=0
        Verbosity level:
        - 0: No output
        - 1: Basic loading information
        - 2: Detailed debugging output

    Returns
    -------
    Union[Dict, Tuple]
        - Full dictionary if no keys specified
        - Tuple of values for requested keys (maintaining order)

    Raises
    ------
    FileNotFoundError
        If specified job_file doesn't exist
    TypeError
        If loaded data isn't a dictionary
    KeyError
        If requested key not found and error_mode='raise'

    Examples
    --------
    >>> from fusionlab.utils.io_utils import fetch_joblib_data
    >>> data = fetch_joblib_data('data.joblib', 'X_train', 'y_train')
    >>> X, y = fetch_joblib_data('data.joblib', 'X_val', 'y_val', verbose=1)
    >>> full_dict = fetch_joblib_data('data.joblib')

    Notes
    -----
    - Maintains original insertion order for Python 3.7+ dictionaries
    - Missing keys in 'warn'/'ignore' modes result in shorter return tuple
    - Joblib files must contain dictionary objects
    """
    try:
        if verbose >= 1:
            print(f"Loading data from {job_file}")
        data = joblib.load(job_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Joblib file {job_file} not found") from None
    except Exception as e:
        raise ValueError(f"Error loading {job_file}: {str(e)}") from e

    if not isinstance(data, dict):
        raise TypeError(f"Loaded data from {job_file} is not a dictionary")

    if not keys:
        if verbose >= 1:
            print("No keys requested - returning full dictionary")
        return data

    results = []
    available_keys = list(data.keys ())
    for key in keys:
        if key in data:
            results.append(data[key])
            if verbose >= 2:
                print(f"Successfully retrieved key: {key}")
        else:
            msg = ( f"Key '{key}' not found in {job_file}."
                   f" Available keys are: {smart_format(available_keys)}"
                  )
            if error_mode == 'raise':
                raise KeyError(msg)
            elif error_mode == 'warn':
                warnings.warn(msg, UserWarning)
                if verbose >= 1:
                    print(f"Warning: {msg}")
            # No action needed for 'ignore' mode

    if verbose >= 1:
        print(f"Retrieved {len(results)}/{len(keys)} requested items")

    return tuple(results) if len(results) > 1 else results[0] if results else ()

def cpath(savepath: str = None, dpath: str = '_default_path_') -> str:
    """
    Ensures a directory exists for saving files, creating it if necessary.

    Parameters
    ----------
    savepath : str, optional
        The target directory to validate or create. If None, `dpath` is used 
        as the directory.

    dpath : str, default='_default_path_'
        Default directory created in the current working directory if 
        `savepath` is None.

    Returns
    -------
    str
        The absolute path to the validated or created directory.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import cpath
    >>> default_path = cpath()
    >>> print(f"Files will be saved to: {default_path}")

    >>> custom_path = cpath('/path/to/save')
    >>> print(f"Files will be saved to: {custom_path}")

    Notes
    -----
    `cpath` validates the directory path and, if necessary, creates the
    directory tree. If a problem occurs during creation, an error message 
    is printed.

    See Also
    --------
    pathlib.Path.mkdir : Utility for directory creation.
    """
    if savepath is None:
        # Use default directory path if none provided
        savepath = Path.cwd() / dpath
    else:
        savepath = Path(savepath)

    try:
        # Create the directory and parents if they do not exist
        savepath.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {savepath}: {e}")

    return str(savepath.resolve())


def spath(name_of_path: str) -> str:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    name_of_path : str
        Path-like object to create if it doesn't exist.

    Returns
    -------
    str
        The absolute path to the created or existing directory.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import spath
    >>> path = spath('data/saved_models')
    >>> print(f"Directory available at: {path}")

    Notes
    -----
    `spath` is useful for quickly ensuring that a specific directory is 
    available for storing files. It provides feedback if the directory
    already exists.
    """
    savepath = os.path.join(os.getcwd(), name_of_path)
    try:
        if not os.path.isdir(savepath):
            os.mkdir(name_of_path)
    except:
        warnings.warn("The path already exists.")
    return savepath

def load_serialized_data(
        filename: str, verbose: int = 0
        ):
    """
    Load data from a serialized file (e.g., pickle or joblib format).

    Parameters
    ----------
    filename : str
        Name of the file to load data from.

    verbose : int, default=0
        Verbosity level. Controls the amount of output information:
        - 0: No output
        - >2: Detailed loading process messages.

    Returns
    -------
    Any
        Data loaded from the file, or None if deserialization fails.

    Raises
    ------
    TypeError
        If `filename` is not a string.

    FileExistsError
        If the specified file does not exist.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import load_serialized_data
    >>> data = load_serialized_data('data/my_data.pkl', verbose=3)

    Notes
    -----
    This function attempts to load serialized data using joblib and 
    fallbacks to pickle if needed. Verbose output provides feedback on 
    the loading process and success or failure of each step.

    See Also
    --------
    joblib.load : High-performance loading utility.
    pickle.load : General-purpose Python serialization library.
    """
    filename = check_files(filename, return_valid = True )

    _filename = os.path.basename(filename)
    data = None

    try:
        # Attempt to load with joblib
        data = joblib.load(filename)
        if verbose > 2:
            print(f"Data from {_filename!r} successfully reloaded using joblib.")
    except:
        if verbose > 2:
            print(f"Fallback: {_filename!r} not loaded with joblib; trying pickle.")
        with open(filename, 'rb') as tod:
            data = pickle.load(tod)
        if verbose > 2:
            print(f"Data from {_filename!r} reloaded using pickle.")

    if verbose > 0:
        if data is None:
            print("Unable to deserialize data. Please check your file.")
        else:
            print(f"Data from {_filename} has been successfully reloaded.")

    return data

def save_job(
    job, 
    savefile ,* ,  
    protocol =None,  
    append_versions=True, 
    append_date=True, 
    fix_imports= True, 
    buffer_callback = None,   
    **job_kws
    ): 
    """ Quick save your job using 'joblib' or persistent Python pickle module.
    
    Parameters 
    -----------
    job: Any 
        Anything to save, preferabaly a models in dict 
        
    savefile: str, or path-like object 
         name of file to store the model.
         The *file* argument must have a write() method that accepts a
         single bytes argument. It can thus be a file object opened for
         binary writing, an io.BytesIO instance, or any other custom
         object that meets this interface.
         
    append_versions: bool, default =True 
        Append the version of Joblib module or Python Pickle module following 
        by the scikit-learn, numpy and also pandas versions. This is useful 
        to have idea about previous versions for loading file when system or 
        modules have been upgraded. This could avoid bottleneck when data 
        have been stored for long times and user has forgotten the date and 
        versions at the time the file was saved. 
        
    append_date: bool, default=True, 
       Append the date  of the day to the filename. 
       
    protocol: int, optional 
        The optional *protocol* argument tells the pickler to use the
        given protocol; supported protocols are 0, 1, 2, 3, 4 and 5.
        The default protocol is 4. It was introduced in Python 3.4, and
        is incompatible with previous versions.
    
        Specifying a negative protocol version selects the highest
        protocol version supported.  The higher the protocol used, the
        more recent the version of Python needed to read the pickle
        produced.
        
    fix_imports: bool, default=True, 
        If *fix_imports* is True and *protocol* is less than 3, pickle
        will try to map the new Python 3 names to the old module names
        used in Python 2, so that the pickle data stream is readable
        with Python 2.
        
    buffer_call_back: int, optional 
        If *buffer_callback* is None (the default), buffer views are
        serialized into *file* as part of the pickle stream.
    
        If *buffer_callback* is not None, then it can be called any number
        of times with a buffer view.  If the callback returns a false value
        (such as None), the given buffer is out-of-band; otherwise the
        buffer is serialized in-band, i.e. inside the pickle stream.
    
        It is an error if *buffer_callback* is not None and *protocol*
        is None or smaller than 5.
        
    job_kws: dict, 
        Additional keywords arguments passed to :func:`joblib.dump`. 
        
    Returns
    -------
    str
        The final filename where the job was saved.

    Notes
    -----
    This function appends system-specific metadata like versions and date to
    the filename, which can aid in tracking compatibility over time.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import save_job
    >>> model = {"key": "value"}  # Replace with actual model object
    >>> savefile = save_job(model, "my_model", append_date=True, append_versions=True)
    >>> print(savefile)
    'my_model.20240101.sklearn_v1.0.numpy_v1.21.joblib'

    """
    def remove_extension(filename: str, extension: str) -> str:
        return filename.replace(extension, '')

    import sklearn

    # check_files(savefile)
    # Generate versioning metadata
    versions = 'sklearn_v{0}.numpy_v{1}.pandas_v{2}'.format(
        sklearn.__version__, np.__version__, pd.__version__)
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    # Handle file extensions
    savefile = str(savefile)
    extension = '.joblib' if '.joblib' in savefile else '.pkl'
    savefile = remove_extension(savefile, extension)

    # Append date and versions if requested
    if append_date:
        savefile += f".{date_str}"
    if append_versions:
        savefile += f".{versions}"

    try:
        joblib.dump(job, f"{savefile}.joblib", **job_kws)
    except Exception:
        with open(f"{savefile}.pkl", 'wb') as wfile:
            pickle.dump(job, wfile, protocol=protocol, 
                        fix_imports=fix_imports, 
                        buffer_callback=buffer_callback)

    return savefile

def _cparser_manager(
    cfile: str,
    savepath: Optional[str] = None, 
    todo: str = 'load', 
    dpath: Optional[str] = None,
    verbose: int = 0, 
    **pkws
) -> None:
    """
    Manages configuration file saving and output messages based on action type.

    Parameters
    ----------
    cfile : str
        Name of the configuration file.
    savepath : str, optional
        Directory path to save the configuration file.
    todo : str, default='load'
        Action to perform with the config file. Options are 'load' or 'dump'.
    dpath : str, optional
        Default path to use if savepath is not specified.
    verbose : int, default=0
        Controls verbosity level of output messages.

    Notes
    -----
    This function uses `move_cfile` to ensure the configuration file is stored 
    in the correct location, and calls `print_cmsg` to provide user feedback.

    """
    check_files(cfile)
    if savepath == 'default':
        savepath = None
    yml_fn, _ = move_cfile(cfile, savepath, dpath=dpath)

    if verbose > 0:
        print(print_cmsg(yml_fn, todo, **pkws))

def move_cfile(
    cfile: str, 
    savepath: Optional[str] = None, 
    **ckws
) -> Tuple[str, str]:
    """
    Moves a file to the specified path. If moving fails, copies and 
    deletes the original.

    Parameters
    ----------
    cfile : str
        Name of the file to move.
    savepath : str, optional
        Target directory. If not specified, uses default path via `cpath`.

    Returns
    -------
    Tuple[str, str]
        The new file path and a confirmation message.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import move_cfile
    >>> new_path, msg = move_cfile('myfile.txt', 'new_directory')
    >>> print(new_path, msg)

    """
    check_files(cfile)
    savepath = cpath(savepath or '_default_path_', **ckws)
    destination_file_path = os.path.join(savepath, os.path.basename(cfile))

    try:
        shutil.move(cfile, destination_file_path)
    except shutil.Error:
        shutil.copy2(cfile, destination_file_path)
        os.remove(cfile)

    msg = (f"--> '{os.path.basename(destination_file_path)}'successfully"
           f" saved to '{os.path.realpath(destination_file_path)}'."
           )
    return destination_file_path, msg

def print_cmsg(
        cfile: str, todo: str = 'load', config: str = 'YAML') -> str:
    """
    Generates output message for configuration file operations.

    Parameters
    ----------
    cfile : str
        Name of the configuration file.
    todo : str, default='load'
        Operation performed ('load' or 'dump').
    config : str, default='YAML'
        Type of configuration file (e.g., 'YAML', 'CSV', 'JSON').

    Returns
    -------
    str
        Confirmation message for the configuration operation.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import print_cmsg
    >>> msg = print_cmsg('config.yml', 'dump')
    >>> print(msg)
    --> YAML 'config.yml' data was successfully saved.

    """
    if todo == 'load':
        msg = f"--> Data successfully loaded from '{os.path.realpath(cfile)}'."
    elif todo == 'dump':
        msg =( 
            f"--> {config.upper()} '{os.path.basename(cfile)}'"
            " data was successfully saved."
            )
    return msg

def parse_csv(
    csv_fn: str = None,
    data: Optional[Union[List[Dict], List[List[str]]]] = None,
    todo: str = 'reader', 
    fieldnames: Optional[List[str]] = None,
    savepath: Optional[str] = None,
    header: bool = False,
    verbose: int = 0,
    **csvkws
) -> Union[List[Dict], List[List[str]], None]:
    """
    Parses a CSV file or serializes data to a CSV file.

    This function allows loading (reading) from or dumping (writing) to a CSV 
    file. It supports standard CSV and dictionary-based CSV formats.

    Parameters
    ----------
    csv_fn : str, optional
        The CSV filename for reading or writing. For writing operations, if 
        `data` is provided and `todo` is set to 'write' or 'dictwriter', this 
        specifies the output CSV filename.
    data : list, optional
        Data to write in the form of a list of lists or dictionaries.
    todo : str, default='reader'
        Specifies the operation type:
        - 'reader' or 'dictreader': Reads data from a CSV file.
        - 'writer' or 'dictwriter': Writes data to a CSV file.
    fieldnames : list of str, optional
        List of keys for dictionary-based writing to specify the field order.
    savepath : str, optional
        Directory to save the CSV file when writing. Defaults to '_savecsv_' 
        if not provided and the path does not exist.
    header : bool, default=False
        If True, includes headers when writing with DictWriter.
    verbose : int, default=0
        Controls the verbosity level for output messages.
    csvkws : dict, optional
        Additional arguments passed to `csv.writer` or `csv.DictWriter`.

    Returns
    -------
    Union[List[Dict], List[List[str]], None]
        Parsed data from the CSV file, as a list of lists or a list of 
        dictionaries, based on the operation. Returns `None` when writing.

    Notes
    -----
    For writing data, the method uses either `csv.writer` for regular CSV or 
    `csv.DictWriter` for dictionary-based CSV depending on the value of `todo`.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import parse_csv
    >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    >>> parse_csv(csv_fn='output.csv', data=data, todo='dictwriter', fieldnames=['name', 'age'])
    >>> loaded_data = parse_csv(csv_fn='output.csv', todo='dictreader', fieldnames=['name', 'age'])
    >>> print(loaded_data)
    [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]

    """
    csv_fn = check_files(csv_fn, formats ='.csv', return_valid=True ) 
    
    todo, domsg = _return_ctask(todo)

    if 'write' in todo:
        csv_fn = get_config_fname_from_varname(
            data, config_fname=csv_fn, config='.csv')

    try:
        if todo == 'reader':
            with open(csv_fn, 'r') as csv_f:
                csv_reader = csv.reader(csv_f)
                data = [row for row in csv_reader]
        elif todo == 'writer':
            with open(f"{csv_fn}.csv", 'w', newline='', encoding='utf8') as new_csvf:
                csv_writer = csv.writer(new_csvf, **csvkws)
                csv_writer.writerows(data) if len(data) > 1 else csv_writer.writerow(data)
        elif todo == 'dictreader':
            with open(csv_fn, 'r', encoding='utf8') as csv_f:
                csv_reader = csv.DictReader(csv_f, fieldnames=fieldnames)
                data = list(csv_reader)
        elif todo == 'dictwriter':
            with open(f"{csv_fn}.csv", 'w') as new_csvf:
                csv_writer = csv.DictWriter(new_csvf, fieldnames=fieldnames, **csvkws)
                if header:
                    csv_writer.writeheader()
                if isinstance(data, dict):
                    csv_writer.writerow(data)
                else:
                    csv_writer.writerows(data)
    except csv.Error as e:
        raise csv.Error(f"Unable {domsg} CSV {csv_fn!r}. {e}")
    except Exception as e:
        msg = "Unrecognizable file" if 'read' in todo else "Unable to write"
        raise TypeError(f"{msg} {csv_fn!r}. Check your"
                        f" {'file' if 'read' in todo else 'data'}. {e}")

    _cparser_manager(f"{csv_fn}.csv", savepath, todo=todo, dpath='_savecsv_',
                    verbose=verbose, config='CSV')
    return data

def _return_ctask(todo: Optional[str] = None) -> Tuple[str, str]:
    """
    Determine the action to perform based on the `todo` input.

    Parameters
    ----------
    todo : str, optional
        Specifies the action. Options:
        - 'load': Load data from a config file (YAML|CSV|JSON).
        - 'dump': Serialize data to a config file (YAML|CSV|JSON).

    Returns
    -------
    Tuple[str, str]
        `todo`: Corrected action string.
        `domsg`: Message for user based on action.

    Notes
    -----
    This function normalizes user input for `todo` to avoid misinterpretations.

    """
    def p_csv(v, cond='dict', base='reader'):
        return f"{cond}{base}" if cond in v else base

    ltags = ('load', 'recover', True, 'fetch')
    dtags = ('serialized', 'dump', 'save', 'write', 'serialize')
    if todo is None:
        raise ValueError(
            "NoneType action cannot be performed. Specify 'load' or 'dump'.")

    todo = str(todo).lower()
    ltags += ('loads',) if todo == 'loads' else ()
    dtags += ('dumps',) if todo == 'dumps' else ()

    if todo in ltags:
        todo, domsg = 'load', 'to parse'
    elif todo in dtags:
        todo, domsg = 'dump', 'to serialize'
    elif 'read' in todo:
        todo, domsg = p_csv(todo), 'to read'
    elif 'write' in todo:
        todo, domsg = p_csv(todo, base='writer'), 'to write'
    else:
        raise ValueError(
            f"Invalid action '{todo}'. Use 'load' or 'dump' (YAML|CSV|JSON).")

    return todo, domsg

def parse_yaml(
    yml_fn: str = None,
    data=None,
    todo: str = 'load',
    savepath: Optional[str] = None,
    verbose: int = 0,
    **ymlkws
):
    """
    Parse and handle YAML configuration files for loading or saving data.

    Parameters
    ----------
    yml_fn : str, optional
        The YAML filename. If `data` is provided and `todo` is set to 'dump',
        `yml_fn` will be used as the output filename. If `todo` is set to 
        'load', `yml_fn` is the input filename to read from.

    data : Any, optional
        Data in a Python object format that will be serialized and saved as a 
        YAML file if `todo` is 'dump'.

    todo : {'load', 'dump'}, default='load'
        Action to perform with the YAML file:
        - 'load': Load data from the YAML file specified by `yml_fn`.
        - 'dump': Serialize `data` into a YAML format and save to `yml_fn`.

    savepath : str, optional
        Path where the YAML file will be saved if `todo` is 'dump'. If not 
        provided, a default path will be used. The function will ensure that 
        the path exists.

    verbose : int, default=0
        Controls verbosity of output messages.

    **ymlkws : dict
        Additional keyword arguments passed to `yaml.dump` when saving data.

    Returns
    -------
    Any
        The data loaded from the YAML file if `todo` is 'load', or `data` 
        after saving if `todo` is 'dump'.

    Raises
    ------
    yaml.YAMLError
        If there is an issue with reading or writing the YAML file.

    Notes
    -----
    This function uses `safe_load` and `safe_dump` methods from PyYAML for 
    secure handling of YAML files.

    See Also
    --------
    `get_config_fname_from_varname` : Utility for generating YAML configuration 
    filenames based on variable names.
    """
    yml_fn = check_files(yml_fn, formats =['.yml', '.yam'], return_valid=True ) 
    
    # Determine task for loading or dumping YAML
    todo = todo.lower()
    if todo.startswith('dump'):
        yml_fn = get_config_fname_from_varname(data, yml_fn)
        try:
            with open(f"{yml_fn}.yml", "w") as fw:
                yaml.safe_dump(data, fw, **ymlkws)
        except yaml.YAMLError:
            raise yaml.YAMLError(
                f"Unable to save data to {yml_fn}. Check file permissions.")
    elif todo.startswith('load'):
        try:
            with open(yml_fn, "r") as fy:
                data = yaml.safe_load(fy)
        except yaml.YAMLError:
            raise yaml.YAMLError(
                f"Unable to load data from {yml_fn}. Check the YAML format.")
    else:
        raise ValueError(f"Invalid value for 'todo': {todo}. Use 'load' or 'dump'.")

    # Manage paths and configurations
    _cparser_manager(f"{yml_fn}.yml", savepath, todo=todo, dpath='_saveyaml_',
                    verbose=verbose, config='YAML')

    return data


def get_config_fname_from_varname(
        data, config_fname: Optional[str] = None, config: str = '.yml') -> str:
    """
    Generate a filename based on a variable name for YAML configuration.

    Parameters
    ----------
    data : Any
        The data object from which the variable name will be derived to 
        create a YAML configuration filename.

    config_fname : str, optional
        Custom configuration filename. If `None`, the name of `data` will 
        be used as the filename.

    config : str, default='.yml'
        The file extension/type for the configuration file. Can be '.yml', 
        '.json', or '.csv'.

    Returns
    -------
    str
        A suitable filename for saving the configuration data.

    Raises
    ------
    ValueError
        If `config_fname` cannot be derived or an invalid file type is provided.

    Notes
    -----
    This function supports dynamic filename generation based on variable names,
    which aids in maintaining a clear configuration structure for serialized 
    data. Files are saved with appropriate extensions based on the `config` type.
    """
    # Clean up file extension and validate config type
    config = config.lstrip('.')
    if config_fname is None:
        try:
            config_fname = f"{data}".split('=')[0].strip()
        except Exception as e:
            raise ValueError(f"Unable to determine configuration filename: {str(e)}")
    else:
        config_fname = config_fname.replace(f".{config}", "").replace(".yaml", "")
    
    # Append correct file extension
    return f"{config_fname}.{config}"


def parse_json(
    json_fn: str = None,
    data=None,
    todo: str = 'load',
    savepath: Optional[str] = None,
    verbose: int = 0,
    **jsonkws
):
    """
    Parse and manage JSON configuration files, either loading data from
    or saving data to a JSON file.

    Parameters
    ----------
    json_fn : str, optional
        JSON filename or URL. If `data` is provided and `todo` is 'dump', 
        `json_fn` will be used as the output filename. If `todo` is 'load', 
        `json_fn` is the input filename or URL.

    data : Any, optional
        Data in Python object format to serialize and save if `todo` is 'dump'.

    todo : {'load', 'loads', 'dump', 'dumps'}, default='load'
        Action to perform with JSON:
        - 'load': Load data from a JSON file.
        - 'loads': Parse a JSON string.
        - 'dump': Serialize `data` to a JSON file.
        - 'dumps': Serialize `data` to a JSON string.

    savepath : str, optional
        Path where the JSON file will be saved if `todo` is 'dump'. If 
        `savepath` does not exist, it will save to the default path '_savejson_'.

    verbose : int, default=0
        Controls verbosity of output messages.

    **jsonkws : dict
        Additional keyword arguments passed to `json.dump` or `json.dumps` 
        when saving data.

    Returns
    -------
    Any
        The data loaded from the JSON file or URL if `todo` is 'load', or `data`
        after saving if `todo` is 'dump'.

    Raises
    ------
    json.JSONDecodeError
        If there is an issue with reading or writing the JSON file.

    TypeError
        If the JSON file or data cannot be processed.

    Notes
    -----
    This function uses `json.load`, `json.loads`, `json.dump`, and `json.dumps` 
    for efficient handling of JSON files and strings.

    See Also
    --------
    `fetch_json_data_from_url` : Fetches JSON data from a given URL.
    `get_config_fname_from_varname` : Utility for generating JSON configuration 
    filenames based on variable names.
    """
    json_fn = check_files(json_fn, formats ='.json', return_valid=True ) 
    # Set task for loading or dumping JSON
    if json_fn and "http" in json_fn:
        todo, json_fn, data = fetch_json_data_from_url(json_fn, todo)

    if 'dump' in todo:
        json_fn = get_config_fname_from_varname(data, json_fn, config='.json')

    JSON = {
        "load": json.load,
        "loads": json.loads,
        "dump": json.dump,
        "dumps": json.dumps
    }

    try:
        if todo == 'load':
            with open(json_fn, "r") as fj:
                data = JSON[todo](fj)
        elif todo == 'loads':
            data = JSON[todo](json_fn)
        elif todo == 'dump':
            with open(f"{json_fn}.json", "w") as fw:
                JSON[todo](data, fw, **jsonkws)
        elif todo == 'dumps':
            data = JSON[todo](data, **jsonkws)
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Unable to {todo} JSON file {json_fn}. Please verify your file.", f'{json_fn!r}', 1)
    except Exception:
        raise TypeError(
            f"Error with {json_fn!r}. Verify your {'file' if 'load' in todo else 'data'}.")

    _cparser_manager(
        f"{json_fn}.json", savepath, todo=todo,
        dpath='_savejson_', verbose=verbose, config='JSON'
    )

    return data

def fetch_json_data_from_url(url: str, todo: str = 'load'):
    """
    Retrieve and parse JSON data from a URL.

    Parameters
    ----------
    url : str
        Universal Resource Locator (URL) from which JSON data is fetched.

    todo : {'load', 'dump'}, default='load'
        Action to perform with JSON:
        - 'load': Load JSON data from the URL.
        - 'dump': Parse and prepare data from the URL for saving in a JSON file.

    Returns
    -------
    tuple
        A tuple of `todo` action, filename (or data source), and parsed data.

    Raises
    ------
    urllib.error.URLError
        If there is an issue accessing the URL.

    Notes
    -----
    The function uses `json.loads` to parse data directly from a URL response,
    supporting convenient access to web-hosted JSON content.
    """
    with urllib.request.urlopen(url) as jresponse:
        source = jresponse.read()
    data = json.loads(source)

    if 'load' in todo:
        todo, json_fn = 'loads', source
    elif 'dump' in todo:
        todo, json_fn = 'dumps', '_urlsourcejsonf.json'

    return todo, json_fn, data

def deserialize_data(filename: str, verbose: int = 0) -> Any:
    """
    Deserialize and load data from a serialized file using `joblib` or `pickle`.

    The function attempts to load the serialized data from the provided file
    `filename` using `joblib` first. If `joblib` fails, it tries to load the
    data using `pickle`. An error is raised if both methods fail.

    Parameters
    ----------
    filename : str
        The name or path of the file containing the serialized data.
        This file is expected to be in a compatible format with either
        `joblib` or `pickle`.
    
    verbose : int, optional
        Verbosity level. Messages indicating loading progress will be displayed
        if `verbose` is greater than 0.

    Returns
    -------
    Any
        The data loaded from the serialized file, or `None` if loading fails.

    Raises
    ------
    TypeError
        If `filename` is not a string, as file paths must be provided as strings.
    
    FileNotFoundError
        If the specified `filename` does not exist or cannot be located.
    
    IOError
        If both `joblib` and `pickle` fail to deserialize the data from the file.
    
    ValueError
        If the file was successfully read but yielded no data (i.e., `None`).

    Examples
    --------
    >>> from fusionlab.utils.io_utils import deserialize_data
    >>> data = deserialize_data('path/to/serialized_data.pkl', verbose=1)
    Data loaded successfully from 'path/to/serialized_data.pkl' using joblib.

    Notes
    -----
    The function first attempts deserialization with `joblib` to leverage 
    efficient file handling for large datasets. If `joblib` encounters an error, 
    it falls back to `pickle`, which provides broader compatibility with Python 
    objects but may be less optimized for large datasets.
    
    See Also
    --------
    joblib.load : Joblib's load function for fast I/O operations on large data.
    pickle.load : Pickle's load function for serializing and deserializing 
                  Python objects.
    
    References
    ----------
    .. [1] Joblib Documentation - https://joblib.readthedocs.io
    .. [2] Python Pickle Module - https://docs.python.org/3/library/pickle.html
    """
    filename = check_files ( filename, return_valid =True )
    
    # Attempt to load data using joblib
    try:
        data = joblib.load(filename)
        if verbose:
            print(f"Data loaded successfully from {filename!r} using joblib.")
    except Exception as joblib_error:
        # Fallback to pickle if joblib fails
        try:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            if verbose:
                print(f"Data loaded successfully from {filename!r} using pickle.")
        except Exception as pickle_error:
            raise IOError(
                f"Failed to load data from {filename!r}. "
                f"Joblib error: {joblib_error}, Pickle error: {pickle_error}"
            )

    # Verify that the data is not None after successful deserialization
    if data is None:
        raise ValueError(
            f"Data in {filename!r} could not be deserialized. "
            "The file may be corrupted or contain no data."
        )

    return data

def serialize_data(
    data: Any,
    filename: Optional[str] = None,
    savepath: Optional[str] = None,
    to: Optional[str] = None,
    force: bool = True,
    compress: Optional[Union[int, str]] = None,
    pickle_protocol: int = pickle.HIGHEST_PROTOCOL,
    verbose: int = 0
) -> str:
    r"""
    Serialize and save a Python object to a binary file using either
    ``joblib`` or ``pickle``. This function is designed to be robust
    and versatile, handling multiple cases including file naming,
    overwriting behavior, and compression options.

    The final file path is computed as:

    .. math::
       \text{filepath} = \text{savepath} \oplus \text{filename}

    where :math:`\oplus` denotes string concatenation.

    Parameters
    ----------
    data          : Any
        The Python object to serialize. The object must be compatible
        with either ``joblib.dump`` or ``pickle.dump``.
    filename      : str, optional
        The target filename for the serialized data. If ``None``, a
        filename is generated using the current timestamp, e.g.,
        ``"__mydumpedfile_20230315_123045.pkl"``.
    savepath      : str, optional
        The directory in which to save the file. If not specified, the
        current working directory (``os.getcwd()``) is used. The directory
        is created if it does not exist.
    to           : str, optional
        The serialization method to use. Acceptable values are
        ``'joblib'`` and ``'pickle'``. If ``None``, the default is
        ``'joblib'``.
    force         : bool, default=True
        If ``True``, any existing file with the same name is overwritten.
        If ``False``, a timestamp is appended to the filename to ensure
        uniqueness.
    compress      : int or str, optional
        Compression level or method for ``joblib.dump``. If ``None``,
        no compression is applied.
    pickle_protocol : int, default=pickle.HIGHEST_PROTOCOL
        The pickle protocol to use when serializing with ``pickle.dump``.
    verbose       : int, default=0
        Controls the verbosity of output messages. Higher values produce
        more detailed logging during the serialization process.

    Returns
    -------
    str
        The full path to the saved serialized file.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import serialize_data
    >>> import numpy as np
    >>> data = {"a": np.arange(10), "b": np.random.rand(10)}
    >>> filepath = serialize_data(
    ...     data, filename="mydata.pkl", savepath="output", 
    ...     to="pickle", force=False, verbose=1
    ... )
    >>> print(filepath)
    /current/working/directory/output/mydata_<timestamp>.pkl

    Notes
    -----
    The function first constructs the file path from ``savepath`` and
    ``filename``. If a file already exists and ``force`` is False, a
    timestamp is appended to ensure uniqueness. Then, depending on the
    value of ``to``, the function attempts to serialize the data using
    either ``joblib.dump`` (with optional compression via the ``compress``
    parameter) or ``pickle.dump`` (using the specified ``pickle_protocol``).
    If an error occurs during serialization, an ``IOError`` is raised.

    See Also
    --------
    joblib.dump : Serialize objects to disk using Joblib.
    pickle.dump : Serialize objects to disk using Pickle.
    os.getcwd    : Retrieve the current working directory.

    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011).
           "The NumPy Array: A Structure for Efficient Numerical
           Computation". Computing in Science & Engineering, 13(2), 22-30.
    """

    # Set default serialization method to joblib if not provided.
    if to is None:
        to = "joblib"
    else:
        if not isinstance(to, str):
            raise TypeError(
                f"Serialization method `to` must be a string, not "
                f"{type(to)}."
            )
        to = to.lower()
        if to not in ("joblib", "pickle"):
            raise ValueError(
                "Unknown serialization method `to`. Must be "
                "'joblib' or 'pickle'."
            )

    # Generate filename using timestamp if not provided.
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"__mydumpedfile_{timestamp}.pkl"

    # Ensure filename ends with .pkl
    if not filename.endswith(".pkl"):
        filename += ".pkl"

    # Determine full save directory.
    directory = savepath if savepath is not None else os.getcwd()
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            if verbose:
                print(f"[INFO] Created directory: {directory}")
        except Exception as e:
            raise IOError(
                f"Failed to create directory '{directory}': {e}"
            ) from e

    full_path = os.path.join(directory, filename)

    # If file exists and force is False, append a timestamp.
    if os.path.exists(full_path) and not force:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if filename.endswith(".pkl"):
            filename = filename.replace(
                ".pkl", f"_{timestamp}.pkl"
            )
        else:
            filename = f"{filename}_{timestamp}.pkl"
        full_path = os.path.join(directory, filename)
        if verbose:
            print(
                f"[INFO] File exists. New filename generated: {filename}"
            )

    # Attempt serialization using the specified method.
    try:
        if to == "joblib":
            joblib.dump(data, full_path, compress=compress)
            if verbose:
                print(
                    f"[INFO] Data serialized using joblib and saved to "
                    f"{full_path!r}."
                )
        elif to == "pickle":
            with open(full_path, "wb") as file:
                pickle.dump(data, file, protocol=pickle_protocol)
            if verbose:
                print(
                    f"[INFO] Data serialized using pickle and saved to "
                    f"{full_path!r}."
                )
    except Exception as e:
        raise IOError(
            f"An error occurred during data serialization: {e}"
        ) from e

    return full_path


def fetch_tgz_from_url(
    data_url: str,
    tgz_filename: str,
    data_path: Optional[Union[str, Path]] = None,
    file_to_retrieve: Optional[str] = None,
    **kwargs
) -> Optional[Path]:
    """
    Downloads a .tgz file from a specified URL, saves it to a directory,
    and optionally extracts a specific file from the archive.

    This function retrieves a .tgz file from the provided `data_url` and saves 
    it to the specified `data_path` directory. If `file_to_retrieve` is specified, 
    the function will extract only that file from the archive; otherwise, the 
    entire archive will be extracted.

    Parameters
    ----------
    data_url : str
        The URL to download the .tgz file from.
    tgz_filename : str
        The name to assign to the downloaded .tgz file.
    data_path : Union[str, Path], optional
        Directory where the downloaded file will be saved. Defaults to a 'tgz_data' 
        directory in the current working directory if not specified.
    file_to_retrieve : str, optional
        Specific filename to extract from the .tgz archive. If not provided,
        the entire archive is extracted.
    **kwargs : dict
        Additional keyword arguments to pass to the extraction method.

    Returns
    -------
    Optional[Path]
        Path to the extracted file if a specific file was requested; otherwise, 
        returns None.

    Raises
    ------
    FileNotFoundError
        If the specified `file_to_retrieve` is not found in the archive.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import fetch_tgz_from_url
    >>> data_url = 'https://example.com/data.tar.gz'
    >>> extracted_file = fetch_tgz_from_url(
    ...     data_url, 'data.tar.gz', data_path='data_dir', file_to_retrieve='file.csv')
    >>> print(extracted_file)

    Notes
    -----
    Uses the `tqdm` progress bar for tracking download progress.
    """
    import urllib.request
    
    data_path = Path(data_path or os.path.join(os.getcwd(), 'tgz_data'))
    data_path.mkdir(parents=True, exist_ok=True)
    tgz_path = data_path / tgz_filename

    # Download with progress bar
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=tgz_filename, ncols=100) as t:
        urllib.request.urlretrieve(data_url, tgz_path, reporthook=_download_progress_hook(t))

    try:
        with tarfile.open(tgz_path, "r:gz") as tar:
            if file_to_retrieve:
                tar.extract(file_to_retrieve, path=data_path, **kwargs)
                return data_path / file_to_retrieve
            tar.extractall(path=data_path)
    except (tarfile.TarError, KeyError) as e:
        print(f"Error extracting {file_to_retrieve or 'archive'}: {e}")
        return None

    return None


def fetch_tgz_locally(
    tgz_file: str,
    filename: str,
    savefile: str = 'tgz',
    rename_outfile: Optional[str] = None
) -> str:
    """
    Extracts a specific file from a local .tgz archive and optionally renames it.

    This function fetches a specific file `filename` from a local tar archive 
    located at `tgz_file`, and saves it to `savefile`. If `rename_outfile` is 
    specified, the file is renamed after extraction.

    Parameters
    ----------
    tgz_file : str
        Full path to the tar file.
    filename : str
        Name of the target file to extract from the archive.
    savefile : str, optional
        Destination directory for the extracted file, defaulting to 'tgz'.
    rename_outfile : str, optional
        New name for the fetched file. If not provided, retains the original name.

    Returns
    -------
    str
        Full path to the fetched and possibly renamed file.

    Raises
    ------
    FileNotFoundError
        If the `tgz_file` or the specified `filename` is not found.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import fetch_tgz_locally
    >>> fetched_file = fetch_tgz_locally(
    ...     'path/to/archive.tgz', 'file.csv', savefile='extracted', rename_outfile='renamed.csv')
    >>> print(fetched_file)
    """
    tgz_path = Path(tgz_file)
    save_path = Path(savefile)
    save_path.mkdir(parents=True, exist_ok=True)

    if not tgz_path.is_file():
        raise FileNotFoundError(f"Source {tgz_file!r} is not a valid file.")

    with tarfile.open(tgz_path) as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(filename)), None)
        if member:
            tar.extract(member, path=save_path)
            extracted_file_path = save_path / member.name
            final_file_path = save_path / (rename_outfile if rename_outfile else filename)
            if extracted_file_path != final_file_path:
                extracted_file_path.rename(final_file_path)
                if extracted_file_path.parent != save_path:
                    shutil.rmtree(extracted_file_path.parent, ignore_errors=True)
        else:
            raise FileNotFoundError(f"File {filename} not found in {tgz_file}.")

    print(f"--> '{final_file_path}' was successfully extracted from '{tgz_path.name}' "
          f"and saved to '{save_path}'.")
    return str(final_file_path)


def extract_tar_with_progress(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    path: Path
):
    """
    Extracts a single file from a tar archive with a progress bar.

    Parameters
    ----------
    tar : tarfile.TarFile
        Opened tar file object.
    member : tarfile.TarInfo
        Tar member (file) to be extracted.
    path : Path
        Directory path where the file will be extracted.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import extract_tar_with_progress
    >>> with tarfile.open('data.tar.gz', 'r:gz') as tar:
    ...     member = tar.getmember('file.csv')
    ...     extract_tar_with_progress(tar, member, Path('output_dir'))

    Notes
    -----
    Uses `tqdm` for progress tracking of the file extraction process.
    """
    with tqdm(total=member.size, desc=f"Extracting {member.name}",
              unit='B', unit_scale=True) as progress_bar:
        with tar.extractfile(member) as member_file:
            with open(path / member.name, 'wb') as out_file:
                shutil.copyfileobj(member_file, out_file, length=1024 * 1024,
                                   callback=lambda x: progress_bar.update(1024 * 1024))


def _download_progress_hook(t):
    """Progress hook for urlretrieve to update tqdm progress bar."""
    last_block = [0]

    def inner(block_count=1, block_size=1, total_size=None):
        if total_size is not None:
            t.total = total_size
        t.update((block_count - last_block[0]) * block_size)
        last_block[0] = block_count

    return inner

def load_csv(
        data_path: str, delimiter: Optional[str] = ',', **kwargs
        ) ->pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    This function reads a comma-separated values (CSV) file into a `pandas`
    DataFrame, with the ability to specify a custom delimiter. It provides
    support for additional options passed to `pandas.read_csv` for more
    granular control over the data loading process.

    Parameters
    ----------
    data_path : str
        The file path to the CSV file that is to be loaded. The file path must
        lead to a `.csv` file. If the file does not exist at the specified path,
        a `FileNotFoundError` is raised.
    
    delimiter : str, optional
        The character used to separate values in the CSV file. The default is
        `,` for standard CSVs. If a different delimiter is used in the file 
        (e.g., `;`), it can be specified here.

    **kwargs : dict
        Additional keyword arguments that will be passed directly to 
        `pandas.read_csv`. For instance, users can specify `header`, `index_col`,
        `dtype`, and other options supported by `read_csv` for more customized 
        data handling.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the loaded data, with the specified
        options applied.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist at the provided `data_path`.
    
    ValueError
        If the file specified by `data_path` is not a CSV file (i.e., does not 
        have a `.csv` extension), a `ValueError` is raised to ensure correct 
        file type.

    Notes
    -----
    This function simplifies the process of loading CSV data into a DataFrame,
    with a straightforward parameter for delimiter customization and full access 
    to `pandas.read_csv` options. It is ideal for basic CSV loading tasks, as well
    as more complex ones requiring specific column handling, type casting, and 
    missing value handling, which can be passed via `**kwargs`.

    Examples
    --------
    Suppose you have a CSV file `example.csv` with the following content:
    
    ```
    name,age,city
    Alice,30,New York
    Bob,25,Los Angeles
    ```

    To load this file into a DataFrame:

    >>> from fusionlab.utils.io_utils import load_csv
    >>> df = load_csv('example.csv')
    >>> print(df)
         name  age         city
    0   Alice   30     New York
    1     Bob   25  Los Angeles

    If the file uses a semicolon (`;`) as the delimiter:

    >>> df = load_csv('example.csv', delimiter=';')

    Additionally, you can pass custom `read_csv` parameters through `**kwargs`,
    such as specifying a column as the index:

    >>> df = load_csv('example.csv', index_col='name')
    >>> print(df)
           age         city
    name                   
    Alice    30     New York
    Bob      25  Los Angeles

    See Also
    --------
    pandas.read_csv : Full documentation for loading CSV files into a DataFrame 
                      with detailed parameter options.

    References
    ----------
    .. [1] Wes McKinney, "Python for Data Analysis," 2nd Edition, O'Reilly Media, 2017.
    """

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"The file '{data_path}' does not exist.")
    
    if not data_path.lower().endswith('.csv'):
        raise ValueError(
            "The specified file is not a CSV file. Please provide a valid CSV file.")

    # Load the CSV data into a DataFrame with the specified delimiter and additional kwargs
    return pd.read_csv(data_path, delimiter=delimiter, **kwargs)

def get_valid_key(input_key, default_key, substitute_key_dict=None,
                  regex_pattern = "[#&*@!,;\s]\s*", deep_search=True):
    """
    Validates an input key and substitutes it with a valid key if necessary,
    based on a mapping of valid keys to their possible substitutes. If the input
    key is not provided or is invalid, a default key is used.

    Parameters
    ----------
    input_key : str
        The key to validate and possibly substitute.
    default_key : str
        The default key to use if input_key is None, empty, or not found in 
        the substitute mapping.
    substitute_key_dict : dict, optional
        A mapping of valid keys to lists of their possible substitutes. This
        allows for flexible key substitution and validation.
    regex_pattern: str, default = '[#&*@!,;\s-]\s*'
        The base pattern to split the text into a columns
    deep_search: bool, default=False 
       If deep-search, the key finder is no sensistive to lower/upper case 
       or whether a numeric data is included. 
    Returns
    -------
    str
        A valid key, which is either the original input_key if valid, a substituted
        key if the original was found in the substitute mappings, or the default_key.

    Notes
    -----
    This function also leverages an external validation through `key_checker` for
    a deep search validation, ensuring the returned key is within the set of valid keys.
    
    Example
    -------
    >>> from fusionlab.utils.io_utils import get_valid_key
    >>> substitute_key_dict = {'valid_key1': ['vk1', 'key1'], 'valid_key2': ['vk2', 'key2']}
    >>> get_valid_key('vk1', 'default_key', substitute_key_dict)
    'valid_key1'
    >>> get_valid_key('unknown_key', 'default_key', substitute_key_dict)
    'KeyError...'
  
    """
    # Ensure substitute_mapping is a dictionary if not provided
    substitute_key_dict = substitute_key_dict or {}

    # Fallback to default_key if input_key is None or empty
    input_key = input_key or default_key

    # Attempt to find a valid substitute for the input_key
    for valid_key, substitutes in substitute_key_dict.items():
        # Case-insensitive comparison for substitutes
        normalized_substitutes = [str(sub).lower() for sub in substitutes]
        
        if str(input_key).lower() in normalized_substitutes:
            input_key = valid_key
            break
    
    regex = re.compile (fr'{regex_pattern}', flags=re.IGNORECASE)
    # use valid keys  only if substitute_key_dict not provided. 
    valid_keys = substitute_key_dict.keys() if substitute_key_dict else is_iterable(
            default_key, exclude_string=True, transform=True)
    valid_keys = set (list(valid_keys) + [default_key])
    # Further validate the (possibly substituted) input_key
    input_key = key_checker(input_key, valid_keys=valid_keys,
                            deep_search=deep_search,regex = regex  )
    
    return input_key


def to_txt(
    d,
    filename=None,
    format='txt',
    indent=2,
    width=80,
    depth=None,
    compat=False,
    include_header=True,
    mode='w',
    encoding='utf-8',
    overwrite=True,
    header=None,
    footer=None,
    serializer=None,
    savepath = None, 
    verbose=1,
    logger =None, 
    **kwargs
):
    """
    Export data objects to a text or JSON file with
    optional custom formatting.

    The function, `<to_txt>`, handles writing `<d>`
    (a string, dict, list, or general object) to a
    file named `<filename>`. When no filename is
    given, it automatically generates one based on
    the current date/time. If `<format>` is `"json"`
    and `<d>` is valid for JSON serialization,
    it attempts a JSON export. Otherwise, it falls
    back to text mode, leveraging Python's built-in
    `pformat` and an optional `<serializer>` for
    advanced transformations.

    .. math::
       \\text{FileName}_{timestamp} \\rightarrow \\text{output}

    where :math:`\\text{FileName}_{timestamp}` is an
    auto-generated name like `output_20230101_123456.txt`
    if `<filename>` is not provided.

    Parameters
    ----------
    d : object
        Data to write. Can be any Python object
        supported by `pformat`, or a dict if
        `<format>` is `'json'`.
    filename : str, optional
        Full path (or name) of the output file. If
        None, a time-stamped name is produced,
        prefixed with `'output_'`.
    format : str, default='txt'
        File format, either ``"txt"`` or ``"json"``.
        If it fails to serialize as JSON, the
        process reverts to text.
    indent : int, default=2
        Indentation level for pretty-printing text
        or JSON.
    width : int, default=80
        Wrap width for formatted text lines.
    depth : int, optional
        Maximum depth to which nested structures
        are expanded. If None, no limit is applied.
    compat : bool, default=False
        If True, instructs `pformat` to produce
        more compact text. Not used when exporting
        JSON.
    include_header : bool, default=True
        Whether to include a decorative header
        (with timestamp) at the top of the file in
        text mode.
    mode : str, default='w'
        File writing mode. Typically `'w'` for
        overwrite, `'a'` for append.
    encoding : str, default='utf-8'
        Text encoding used when opening the file.
    overwrite : bool, default=True
        If False, raises an error if the file
        already exists.
    header : str, optional
        Custom header text (if `<include_header>`
        is True). Overwrites the default header
        if given.
    footer : str, optional
        Custom footer text appended at the end of
        the file, if `<include_header>` is True.
    serializer : callable, optional
        A function that transforms `<d>` before
        printing. If it fails, `<d>` remains
        unchanged.
    verbose : int, default=1
        Verbosity level for logging. Higher
        values yield more console messages
        (e.g., file stats at `<verbose>`>=3).
    **kwargs
        Additional parameters passed to the JSON
        serializer (`json.dump`) or `pformat`.

    Returns
    -------
    str
        The final filename used to store the
        output (potentially auto-generated).

    Notes
    -----
    If `<format>` is `"json"`, the function tries
    `json.dump` with a few standard parameters.
    If an exception occurs, it reverts to text
    export. The `<serializer>` argument allows
    custom transformations, such as flattening
    nested dicts or converting objects to JSON-
    serializable representations [1]_.

    Examples
    --------
    >>> from fusionlab.utils.io_utils import to_txt
    >>> my_data = {"name":"Alice","age":30}
    >>> # Basic text export
    >>> txt_file = to_txt(my_data, verbose=2)
    >>> # Enforce JSON format
    >>> json_file = to_txt(my_data, format='json', indent=4)

    See Also
    --------
    pformat : Pretty-print complex Python data
        structures.

    References
    ----------
    .. [1] van Rossum, Guido, *Python's standard
       library "json" module*, Python Docs.
    """
    log = logger or print 
    
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = 'json' if format == 'json' else 'txt'
        filename = f"output_{timestamp}.{ext}"
    
    # Use the savepath if provided
    if savepath:
        # Ensure directory exists
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        # Join the savepath and filename
        filename = os.path.join(savepath, filename)

    # Check file existence if overwrite disabled
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(
            f"File '{filename}' exists. Set overwrite=True to override.")

    original_format = format.lower()
    success = False
    file_created = False

    # Attempt JSON export if requested
    if original_format == 'json':
        try:
            json_kwargs = {'indent': indent, 'ensure_ascii': False}
            json_kwargs.update(kwargs)
            with open(filename, 'w', encoding=encoding) as f:
                json.dump(d, f, **json_kwargs)
            success = True
            file_created = True
            if verbose >= 1:
                log(f"JSON export successful: {filename}")
        except Exception as e:
            if verbose >= 1:
                log(f"JSON export failed ({e}), falling back to TXT")
            filename = os.path.splitext(filename)[0] + '.txt'
            format = 'txt'

    # TXT export fallback
    if not success:
        # Apply serializer if provided
        processed_data = d
        if serializer:
            try:
                processed_data = serializer(d)
            except Exception as e:
                if verbose >= 2:
                    log(f"Serializer error: {e}, using original data")

        # Generate formatted content
        if isinstance(processed_data, str):
            content = processed_data
        else:
            content = pformat(
                processed_data,
                indent=indent,
                width=width,
                depth=depth,
                compact=compat
            )

        # Add header/footer
        header_section = ""
        if include_header:
            header_section = header if header else (
                "\n" + "=" * 60 + "\n"
                f"DATA EXPORT | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                "=" * 60 + "\n\n"
            )
        
        footer_section = footer if footer else (
            "\n\n" + "=" * 60 + "\n" if include_header else ""
        )

        content = f"{header_section}{content}{footer_section}"

        # Write to file
        with open(filename, mode, encoding=encoding) as f:
            f.write(content)
        file_created = True

    # Verbose reporting
    if verbose >= 1 and file_created:
        log(f"Created '{filename}' ({format.upper()})")
        if verbose >= 3:
            try:
                size = os.path.getsize(filename)
                log(f"Dimensions: {size} bytes | {len(content.splitlines())} lines")
                log(f"Encoding: {encoding} | Mode: {mode}")
            except Exception as e:
                if verbose >= 4:
                    log(f"Metadata error: {e}")

    return filename




























