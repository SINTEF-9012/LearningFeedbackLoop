import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# ---------------------------
# Internal helpers
# ---------------------------

def _hann(n: int) -> np.ndarray:
    return np.hanning(n)

def _rect(n: int) -> np.ndarray:
    return np.ones(n, dtype=float)

def _get_window(window_type: str, n: int) -> np.ndarray:
    """Return a window function of the specified type and length."""
    if window_type in ("hann", "hanning"):
        return _hann(n)
    elif window_type in ("rect", "boxcar", "rectangular"):
        return _rect(n)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

def _coherent_gain(win: np.ndarray) -> float:
    """Coherent gain = sum(win)/N (≈0.5 for Hann)."""
    N = len(win)
    return float(np.sum(win)) / float(N)

def _select_window_by_time(signal: np.ndarray, fs: float, t_min: float, t_max: float) -> Tuple[np.ndarray, int, int]:
    """
    Convert time window [t_min, t_max] to indices and slice signal (clipped to bounds).
    Returns (segment, i0, i1).
    """
    n = len(signal)
    i0 = int(np.floor(t_min * fs))
    i1 = int(np.ceil(t_max * fs))
    i0 = max(0, min(n, i0))
    i1 = max(0, min(n, i1))
    if i1 <= i0:
        raise ValueError("Invalid window: no samples selected after clipping.")
    return signal[i0:i1], i0, i1

def _lookup_channel_unit_from_metadata(metadata: Dict[str, Any], channel_name: str) -> str:
    """
    Attempt to find per-channel 'Unit' by matching SignalName within metadata['Channel_*'] entries.
    Fallback to metadata-level 'Unit' or '' if not found.
    """
    for k, v in (metadata or {}).items():
        if isinstance(v, dict) and k.startswith("Channel_"):
            sig_name = v.get("SignalName")
            if sig_name == channel_name:
                return v.get("Unit", "")
    return metadata.get("Unit", "")

def _pick_channels(data_dict: Dict[str, Any], requested_channels: Optional[List[str]]) -> List[str]:
    """
    Determine which channels to process from sessions[session_id]["data"].
    If requested_channels is provided, intersect with available; otherwise process all.
    """
    available = list(data_dict.keys())
    if not available:
        return []
    if not requested_channels:
        return available
    # preserve request order, include only those that exist
    selected = [ch for ch in requested_channels if ch in data_dict]
    return selected

# ---------------------------
# Amplitude estimators
# ---------------------------

def goertzel_amplitude(x: np.ndarray, fs: float, f0: float,
                       window: np.ndarray,
                       detrend: bool = True,
                       return_peak: bool = False) -> float:
    """
    Narrowband amplitude at f0 using Goertzel.
    Returns RMS by default; peak if return_peak=True.
    """
    N = len(x)
    if detrend:
        x = x - np.mean(x)
    xw = x * window

    # Goertzel parameters
    kf = (N * f0) / fs
    omega = 2.0 * np.pi * kf / N
    coeff = 2.0 * np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0
    for xn in xw:
        s = xn + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    # Complex DFT-equivalent at frequency f0
    real_part = s_prev - s_prev2 * np.cos(omega)
    imag_part = s_prev2 * np.sin(omega)
    Xk = real_part - 1j * imag_part

    cg = _coherent_gain(window)               # correct for window attenuation
    A_peak = (2.0 * np.abs(Xk)) / (N * cg)    # single-tone peak amplitude
    return A_peak if return_peak else (A_peak / np.sqrt(2.0))

def fft_interp_amplitude(x: np.ndarray, fs: float, f0: float,
                         window: np.ndarray,
                         detrend: bool = True,
                         return_peak: bool = False) -> float:
    """
    Alternative: FFT + quadratic interpolation around nearest bin (log-magnitude).
    Returns RMS by default; peak if return_peak=True.
    """
    N = len(x)
    if detrend:
        x = x - np.mean(x)
    xw = x * window

    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    if len(freqs) < 3:
        # too short for interpolation; degrade gracefully to nearest bin
        k = int(np.argmin(np.abs(freqs - f0)))
        mag_interp = float(np.abs(X[k]))
    else:
        k = int(np.argmin(np.abs(freqs - f0)))
        k = int(np.clip(k, 1, len(freqs) - 2))
        mags = np.abs(X[[k-1, k, k+1]])

        # Quadratic interpolation in dB for stability
        a = 20.0 * np.log10(max(mags[0], 1e-24))
        b = 20.0 * np.log10(max(mags[1], 1e-24))
        c = 20.0 * np.log10(max(mags[2], 1e-24))
        denom = (a - 2*b + c)
        delta = 0.0 if np.isclose(denom, 0.0) else 0.5 * (a - c) / denom
        mag_interp_db = b - 0.25 * (a - c) * delta
        mag_interp = 10.0 ** (mag_interp_db / 20.0)

    cg = _coherent_gain(window)
    A_peak = (2.0 * mag_interp) / (N * cg)
    return A_peak if return_peak else (A_peak / np.sqrt(2.0))

# ---------------------------
# Core API computation (multi-channel, session-backed)
# ---------------------------

def compute_fg_fp_for_window_session_multi(
    sess: Dict[str, Any],
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute amplitudes at fg and fp for a given time window using server-stored
    signal & metadata in `sessions[session_id]`, across multiple channels.

    Server state (expected):
      sessions[session_id]["data"]      : Dict[str, List[float]]  # channel_name -> samples
      sessions[session_id]["metadata"]  : Dict[str, Any]          # includes File_Header, maybe fg/fp/n/z, Channel_* entries
      sessions[session_id]["raw_file"]  : Any (unused here)

    Request (examples of supported keys):
      {
        "window":  { "t_min": float, "t_max": float },          # seconds (relative to signal start)
        "channels": ["AcelX", "AcelY"],                         # optional; default = all available channels
        "options": {
          "method": "goertzel" | "fft",                         # default "goertzel"
          "return_peak": bool,                                  # default False => RMS
          "detrend": bool,                                      # default True
          "window_type": "hann" | "rect"                        # default "hann"
        },
        "variables": { "z": 8, ... }                            # optional; used to derive fp if metadata lacks it
      }

    Returns:
      {
        "session_id": str,
        "fs": float,
        "frequencies": {"fg": float, "fp": float},
        "window": {
          "t_min_requested": float, "t_max_requested": float
        },
        "options": {...},
        "variables": {...},                                     # passthrough from request
        "channels": {
          "<channel_name>": {
            "unit": str,
            "effective_window": {"i0": int, "i1": int, "t_min_effective": float, "t_max_effective": float, "N": int},
            "amplitudes": {"fg": float, "fp": float, "amplitude_type": "rms" | "peak"}
          },
          ...
        },
        "missing_channels": [ ... ],                            # requested but not present
        "errors": { "<channel_name>": "error message", ... }    # per-channel errors (if any)
      }
    """
    #if session_id not in sessions:
    #    raise KeyError(f"Unknown session_id: {session_id}")

    data_dict = sess.get("data") or {}
    metadata = sess.get("metadata") or {}

    if not data_dict:
        raise ValueError("Session has no signal data.")
    if not metadata:
        raise ValueError("Session has no metadata.")

    # --- Sampling rate ---
    file_header = metadata.get("File_Header") or {}
    if "SampleFrequency" not in file_header:
        raise ValueError("SampleFrequency missing in metadata.File_Header.")
    fs = float(file_header["SampleFrequency"])

    # --- Frequencies (fg, fp) ---
    # Preferred: explicit metadata["fg"] / ["fp"]
    fg = metadata.get("fg", None)
    if fg is None:
        n_rpm = metadata.get("n", None)
        if n_rpm is None:
            raise ValueError("Missing 'fg' and cannot derive it (no 'n' in metadata).")
        fg = float(n_rpm) / 60.0
    else:
        fg = float(fg)

    fp = metadata.get("fp", None)
    if fp is None:
        # try z from metadata first, else from request.variables
        z = metadata.get("z", None)
        if z is None:
            z = (request.get("variables") or {}).get("z", None)
        if z is None:
            raise ValueError("Missing 'fp' and cannot derive it (no 'z' available).")
        fp = float(z) * float(fg)
    else:
        fp = float(fp)

    # --- Window parameters ---
    win_req = request.get("window") or {}
    t_min = float(win_req.get("t_min", 0.0))
    t_max = float(win_req.get("t_max", 0.0))
    if t_max <= t_min:
        # If not provided, default to full duration of the shortest channel
        # (kept explicit to avoid confusion; you can change defaulting logic)
        # Compute min length in seconds across channels:
        min_len_samples = min(len(np.asarray(sig, dtype=float)) for sig in data_dict.values())
        t_min = 0.0
        t_max = min_len_samples / fs
    if t_max <= t_min:
        raise ValueError("window.t_max must be greater than window.t_min (after defaults).")

    # --- Options ---
    opts = request.get("options", {}) or {}
    method = str(opts.get("method", "goertzel")).lower()
    return_peak = bool(opts.get("return_peak", False))
    detrend = bool(opts.get("detrend", True))
    window_type = str(opts.get("window_type", "hann")).lower()

    if method not in ("goertzel", "fft"):
        raise ValueError("options.method must be 'goertzel' or 'fft'.")
    if window_type not in ("hann", "hanning", "rect", "boxcar", "rectangular"):
        raise ValueError(f"Unsupported window_type='{window_type}'.")

    # --- Channels selection ---
    requested_channels = request.get("channels", None)
    channels = _pick_channels(data_dict, requested_channels)
    if not channels:
        raise ValueError("No valid channels to process (empty session data or none matched request).")
    missing_channels = []
    if requested_channels:
        missing_channels = [ch for ch in requested_channels if ch not in data_dict]

    # --- Prepare response containers ---
    out_channels: Dict[str, Any] = {}
    per_channel_errors: Dict[str, str] = {}

    # --- Process each channel ---
    for ch in channels:
        try:
            sig = np.asarray(data_dict[ch], dtype=float)

            # Slice window for this channel
            seg, i0, i1 = _select_window_by_time(sig, fs, t_min, t_max)
            N = len(seg)
            if N < 8:
                raise ValueError(f"Selected window too short on channel '{ch}': {N} samples.")

            # Build window function
            if window_type in ("hann", "hanning"):
                win = _hann(N)
            else:  # rectangular
                win = _rect(N)

            # Stability check: need ≥ 2 cycles of the lowest target frequency
            fmin = max(1e-6, min(fg, fp))
            min_samples = max(8, int(np.ceil(2.0 * fs / fmin)))
            if N < min_samples:
                raise ValueError(
                    f"Window too short for stable amplitude at {fmin:.3f} Hz "
                    f"(need ≥ {min_samples} samples, got {N})."
                )

            # Choose estimator
            if method == "goertzel":
                estimator = lambda x, f: goertzel_amplitude(x, fs, f, window=win, detrend=detrend, return_peak=return_peak)
            else:
                estimator = lambda x, f: fft_interp_amplitude(x, fs, f, window=win, detrend=detrend, return_peak=return_peak)

            # Compute amplitudes
            amp_fg = estimator(seg, fg)
            amp_fp = estimator(seg, fp)

            # Unit lookup (best-effort)
            unit = _lookup_channel_unit_from_metadata(metadata, ch)

            # Assemble per-channel output
            t0_eff = i0 / fs
            t1_eff = i1 / fs
            out_channels[ch] = {
                "unit": unit,
                "effective_window": {
                    "i0": i0,
                    "i1": i1,
                    "N": N,
                    "t_min_effective": t0_eff,
                    "t_max_effective": t1_eff
                },
                "amplitudes": {
                    "fg": float(amp_fg),
                    "fp": float(amp_fp),
                    "amplitude_type": "peak" if return_peak else "rms"
                }
            }

        except Exception as e:
            per_channel_errors[ch] = str(e)

    # If all channels failed, surface a clear error
    if not out_channels and per_channel_errors:
        raise RuntimeError(f"All requested channels failed: {per_channel_errors}")

    # --- Prepare global response ---
    # Extract session_id from session if available
    session_id = sess.get("session_id", "unknown")
    
    response = {
        "session_id": session_id,
        "fs": fs,
        "frequencies": {
            "fg": float(fg),
            "fp": float(fp)
        },
        "window": {
            "t_min_requested": float(t_min),
            "t_max_requested": float(t_max)
        },
        "options": {
            "method": method,
            "return_peak": return_peak,
            "detrend": detrend,
            "window_type": "hann" if window_type in ("hann", "hanning") else "rect"
        },
        "variables": request.get("variables", {}),
        "channels": out_channels,
        "missing_channels": missing_channels,
        "errors": per_channel_errors
    }
    return response


# ---------------------------
# Optimized amplitude estimators (used by compute_fg_fp_for_window_session_multi_ref)
# ---------------------------

def _goertzel_amplitude(x: np.ndarray, fs: float, f0: float,
                        window: np.ndarray, detrend: bool, cg: float,
                        return_peak: bool) -> float:
    N = len(x)
    if detrend:
        x = x - np.mean(x)
    xw = x * window

    # Goertzel recurrence
    kf = (N * f0) / fs
    omega = 2.0 * np.pi * kf / N
    coeff = 2.0 * np.cos(omega)

    s_prev = 0.0
    s_prev2 = 0.0
    # tight loop over Python is okay for a single window; N is moderate (e.g., fs*Δt)
    for xn in xw:
        s = xn + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    real_part = s_prev - s_prev2 * np.cos(omega)
    imag_part = s_prev2 * np.sin(omega)
    Xk = real_part - 1j * imag_part

    A_peak = (2.0 * np.abs(Xk)) / (N * cg)
    return A_peak if return_peak else (A_peak / np.sqrt(2.0))

def _fft_interp_amplitude(x: np.ndarray, fs: float, f0: float,
                          window: np.ndarray, detrend: bool, cg: float,
                          return_peak: bool) -> float:
    N = len(x)
    if detrend:
        x = x - np.mean(x)
    xw = x * window
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    if len(freqs) < 3:
        k = int(np.argmin(np.abs(freqs - f0)))
        mag_interp = float(np.abs(X[k]))
    else:
        k = int(np.argmin(np.abs(freqs - f0)))
        k = int(np.clip(k, 1, len(freqs) - 2))
        mags = np.abs(X[[k-1, k, k+1]])
        # Quadratic interpolation in dB
        a = 20.0 * np.log10(max(mags[0], 1e-24))
        b = 20.0 * np.log10(max(mags[1], 1e-24))
        c = 20.0 * np.log10(max(mags[2], 1e-24))
        denom = (a - 2*b + c)
        delta = 0.0 if np.isclose(denom, 0.0) else 0.5 * (a - c) / denom
        mag_interp_db = b - 0.25 * (a - c) * delta
        mag_interp = 10.0 ** (mag_interp_db / 20.0)

    A_peak = (2.0 * mag_interp) / (N * cg)
    return A_peak if return_peak else (A_peak / np.sqrt(2.0))

# ---------------------------
# Core API computation (multi-channel, session reference)
# ---------------------------

def compute_fg_fp_for_window_session_multi_ref(
    session: Dict[str, Any],
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Optimized: takes a single session reference (not the whole sessions map).
    Expects:
      session["data"]     : Dict[str, np.ndarray or List[float]]
      session["metadata"] : Dict[str, Any] with File_Header.SampleFrequency and maybe fg/fp/n/z
    """
    data_dict = session.get("data") or {}
    metadata = session.get("metadata") or {}

    if not data_dict:
        raise ValueError("Session has no signal data.")
    if not metadata:
        raise ValueError("Session has no metadata.")

    file_header = metadata.get("File_Header") or {}
    fs = float(file_header["SampleFrequency"])

    # fg
    fg = metadata.get("fg", None)
    if fg is None:
        n_rpm = metadata.get("n", None)
        if n_rpm is None:
            raise ValueError("Missing 'fg' and cannot derive it (no 'n' in metadata).")
        fg = float(n_rpm) / 60.0
    else:
        fg = float(fg)

    # fp
    fp = metadata.get("fp", None)
    if fp is None:
        z = metadata.get("z", None)
        if z is None:
            z = (request.get("variables") or {}).get("z", None)
        if z is None:
            raise ValueError("Missing 'fp' and cannot derive it (no 'z' available).")
        fp = float(z) * float(fg)
    else:
        fp = float(fp)

    win_req = request.get("window") or {}
    t_min = float(win_req.get("t_min", 0.0))
    t_max = float(win_req.get("t_max", 0.0))
    if t_max <= t_min:
        # default to shortest channel duration
        min_len_samples = min(len(np.asarray(sig)) for sig in data_dict.values())
        t_min = 0.0
        t_max = min_len_samples / fs
    if t_max <= t_min:
        raise ValueError("window.t_max must be greater than window.t_min (after defaults).")

    opts = request.get("options", {}) or {}
    method = str(opts.get("method", "goertzel")).lower()
    return_peak = bool(opts.get("return_peak", False))
    detrend = bool(opts.get("detrend", True))
    window_type = str(opts.get("window_type", "hann")).lower()
    if method not in ("goertzel", "fft"):
        raise ValueError("options.method must be 'goertzel' or 'fft'.")
    if window_type not in ("hann", "hanning", "rect", "boxcar", "rectangular"):
        raise ValueError(f"Unsupported window_type='{window_type}'.")

    requested_channels = request.get("channels", None)
    channels = _pick_channels(data_dict, requested_channels)
    if not channels:
        raise ValueError("No valid channels to process.")
    missing_channels = [ch for ch in (requested_channels or []) if ch not in data_dict]

    # Precompute window length and window vector once (same for all channels)
    # because t_min/t_max are the same for all channels
    # Determine N from the FIRST channel
    first_ch = channels[0]
    first_sig = data_dict[first_ch]
    first_sig = np.asarray(first_sig)  # no copy if already ndarray
    seg_tmp, _, _ = _select_window_by_time(first_sig, fs, t_min, t_max)
    N = len(seg_tmp)
    if N < 8:
        raise ValueError(f"Selected window too short: {N} samples.")
    win = _get_window("hann" if window_type in ("hann", "hanning") else "rect", N)
    cg = _coherent_gain(win)

    # Stability check once
    fmin = max(1e-6, min(fg, fp))
    min_samples = max(8, int(np.ceil(2.0 * fs / fmin)))
    if N < min_samples:
        raise ValueError(
            f"Window too short for stable amplitude at {fmin:.3f} Hz "
            f"(need ≥ {min_samples} samples, got {N})."
        )

    # Pick estimator once
    if method == "goertzel":
        estimator = lambda x, f: _goertzel_amplitude(x, fs, f, window=win, detrend=detrend, cg=cg, return_peak=return_peak)
    else:
        estimator = lambda x, f: _fft_interp_amplitude(x, fs, f, window=win, detrend=detrend, cg=cg, return_peak=return_peak)

    out_channels: Dict[str, Any] = {}
    per_channel_errors: Dict[str, str] = {}

    for ch in channels:
        try:
            sig = np.asarray(data_dict[ch])  # zero-copy if already ndarray
            seg, i0, i1 = _select_window_by_time(sig, fs, t_min, t_max)
            # seg length should equal N; if not (channel shorter), an error will be thrown above or here
            if len(seg) != N:
                # If a different length arises (e.g., channel shorter), rebuild window just for this channel
                N_local = len(seg)
                if N_local < 8:
                    raise ValueError(f"Selected window too short on channel '{ch}': {N_local} samples.")
                win_local = _get_window("hann" if window_type in ("hann", "hanning") else "rect", N_local)
                cg_local = _coherent_gain(win_local)
                amp_fg = (_goertzel_amplitude if method == "goertzel" else _fft_interp_amplitude)(
                    seg, fs, fg, window=win_local, detrend=detrend, cg=cg_local, return_peak=return_peak
                )
                amp_fp = (_goertzel_amplitude if method == "goertzel" else _fft_interp_amplitude)(
                    seg, fs, fp, window=win_local, detrend=detrend, cg=cg_local, return_peak=return_peak
                )
            else:
                amp_fg = estimator(seg, fg)
                amp_fp = estimator(seg, fp)

            unit = _lookup_channel_unit_from_metadata(metadata, ch)
            out_channels[ch] = {
                "unit": unit,
                "effective_window": {
                    "i0": i0,
                    "i1": i1,
                    "N": int(len(seg)),
                    "t_min_effective": i0 / fs,
                    "t_max_effective": i1 / fs
                },
                "amplitudes": {
                    "fg": float(amp_fg),
                    "fp": float(amp_fp),
                    "amplitude_type": "peak" if return_peak else "rms"
                }
            }
        except Exception as e:
            per_channel_errors[ch] = str(e)

    if not out_channels and per_channel_errors:
        raise RuntimeError(f"All requested channels failed: {per_channel_errors}")

    return {
        "fs": float(fs),
        "frequencies": {"fg": float(fg), "fp": float(fp)},
        "window": {"t_min_requested": float(t_min), "t_max_requested": float(t_max)},
        "options": {
            "method": method,
            "return_peak": return_peak,
            "detrend": detrend,
            "window_type": "hann" if window_type in ("hann", "hanning") else "rect"
        },
        "variables": request.get("variables", {}),
        "channels": out_channels,
        "missing_channels": missing_channels,
        "errors": per_channel_errors
    }


# ---------------------------
# FFT Processing (compute_rfft_multichannel)
# ---------------------------

def _freq_mask(freqs: np.ndarray, max_freq_hz: Optional[float]) -> np.ndarray:
    if max_freq_hz is None:
        return np.ones_like(freqs, dtype=bool)
    return freqs <= float(max_freq_hz)

def compute_rfft_multichannel(
    segs_by_channel: Dict[str, np.ndarray],
    fs: float,
    window_type: str = "hann",
    detrend: bool = True,
    output: str = "amplitude",   # "amplitude" | "power" | "psd"
    db: bool = False,
    bin_stride: int = 1,
    max_freq_hz: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute single-sided RFFT for multiple channels over the SAME time segment.
    Returns:
      freqs: (K,) frequency bins [Hz]
      spectra: { channel: (K,) array, scaled per 'output', optionally in dB }
    Scaling:
      - "amplitude": single-tone amplitude per bin (like line amplitude, not RMS),
        with window coherent gain & single-sided correction (×2 for non-DC/Nyquist).
      - "power": magnitude^2 scaled single-sided (useful if you want power per bin).
      - "psd": power spectral density (per Hz), approximate (Welch-like without averaging).
    """
    # assume all segments same length
    ch0 = next(iter(segs_by_channel))
    x0 = segs_by_channel[ch0]
    N = x0.size
    win = _get_window(window_type, N)
    cg = _coherent_gain(win)

    # Frequencies
    freqs = np.fft.rfftfreq(N, d=1.0/fs)

    # Build mask for band-limit and decimation
    mask = _freq_mask(freqs, max_freq_hz)
    if bin_stride > 1:
        idxs = np.nonzero(mask)[0][::bin_stride]
        mask = np.zeros_like(mask, dtype=bool)
        mask[idxs] = True

    # Precompute single-sided factor: 2 for bins (1..K-2), except DC and Nyquist
    K = freqs.size
    ss_factor = np.ones(K, dtype=float)
    if K > 1:
        ss_factor[1:K-1] = 2.0

    spectra: Dict[str, np.ndarray] = {}
    for ch, x in segs_by_channel.items():
        xx = np.asarray(x, dtype=float)
        if detrend:
            xx = xx - xx.mean()
        X = np.fft.rfft(xx * win)
        mag = np.abs(X)

        if output == "amplitude":
            # Single-tone peak amplitude per bin:
            # A_peak ≈ (ss_factor * |X|) / (N * cg)
            A_peak = (ss_factor * mag) / (N * max(cg, 1e-12))
            y = A_peak
            if db:
                # dB (peak amplitude): 20*log10
                y = 20.0 * np.log10(np.maximum(y, 1e-24))
        elif output == "power":
            # Power per bin (single-sided): ss_factor * (|X|^2) / (N^2 * cg^2)
            P = (ss_factor * (mag**2)) / ((N * max(cg, 1e-12))**2)
            y = P
            if db:
                y = 10.0 * np.log10(np.maximum(y, 1e-24))
        elif output == "psd":
            # PSD estimate (V^2/Hz if input is V): single-sided, normalized by (fs * sum(win^2))
            # Here we approximate: SSD = ss_factor * |X|^2 / (fs * sum(win^2))
            denom = fs * float(np.sum(win**2))
            PSD = (ss_factor * (mag**2)) / max(denom, 1e-12)
            y = PSD
            if db:
                y = 10.0 * np.log10(np.maximum(y, 1e-24))
        else:
            raise ValueError("output must be 'amplitude', 'power', or 'psd'")

        spectra[ch] = y[mask]

    return freqs[mask], spectra