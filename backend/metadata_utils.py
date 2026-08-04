# metadata_utils.py
"""Shared utilities for normalised metadata access.

Every module that needs the sampling frequency, machining parameters, or
file-header dict should call these helpers instead of re-implementing the
lookup chain.  This eliminates the ``file_header`` / ``File_Header`` casing
mismatch and the inconsistent default values that were scattered across
app.py, fft_streamer.py, inference_streamer.py, and computation.py.
"""

from typing import Any, Dict, Optional


def get_file_header(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the file-header sub-dict regardless of key casing.

    The MATLAB-style upload stores it as ``metadata["file_header"]`` (lowercase)
    while some older code expected ``metadata["File_Header"]`` (title-case).
    This helper checks both.
    """
    if not isinstance(metadata, dict):
        return None
    fh = metadata.get("file_header") or metadata.get("File_Header")
    if isinstance(fh, dict):
        return fh
    return None


def get_sample_frequency(metadata: Dict[str, Any], default: float = 1000.0) -> float:
    """Extract the sampling frequency from session metadata.

    Lookup order:
      1. ``metadata.file_header.SampleFrequency`` (or ``File_Header``)
      2. ``metadata.sample_frequency``
      3. ``metadata.SampleFrequency``
      4. *default* (caller-supplied; 1000.0 unless overridden)

    All streaming/FFT/inference code should use this single function.
    """
    fh = get_file_header(metadata)
    fs_val = None
    if fh is not None:
        fs_val = fh.get("SampleFrequency")
    if fs_val is None and isinstance(metadata, dict):
        fs_val = metadata.get("sample_frequency") or metadata.get("SampleFrequency")
    return float(fs_val or default)


def get_machining_param(metadata: Dict[str, Any], key: str, default=None):
    """Look up a machining parameter (fg, fp, n, z, …) from session metadata.

    ``preprocess_payload`` nests these under ``metadata["machining"]``, but
    older callers expected them directly on ``metadata``.  This helper checks
    both locations.
    """
    if not isinstance(metadata, dict):
        return default
    # Preferred: nested under "machining"
    machining = metadata.get("machining")
    if isinstance(machining, dict):
        val = machining.get(key)
        if val is not None:
            return val
    # Fallback: flat metadata (legacy / alternative upload formats)
    return metadata.get(key, default)
