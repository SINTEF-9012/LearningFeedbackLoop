import numpy as np


def compute_harmonics(
    accel: np.ndarray,
    fg: float,
    harm_mults: list[int],
    fft_win: int = 4096,
    fft_step: int = 1024,
) -> np.ndarray:
    """Compute harmonic magnitudes for each accel channel via sliding FFT.

    Args:
        accel: (N, C) accelerometer signals
        fg: spindle frequency in Hz (RPM / 60)
        harm_mults: list of harmonic multipliers (e.g. [1,2,3,4,6,8,10])
        fft_win: FFT window size in samples
        fft_step: step between consecutive windows

    Returns:
        (T, C*len(harm_mults)) array of harmonic magnitudes.
        Columns ordered: [h1_ch0, h2_ch0, ..., hN_ch0, h1_ch1, ..., hN_chC]
    """
    n_samples, n_channels = accel.shape
    n_harm = len(harm_mults)
    n_steps = max(0, (n_samples - fft_win) // fft_step + 1)
    result = np.zeros((n_steps, n_harm * n_channels), dtype=np.float32)

    for t in range(n_steps):
        start = t * fft_step
        for ch in range(n_channels):
            seg = accel[start : start + fft_win, ch]
            spectrum = np.abs(np.fft.rfft(seg))
            for hi, mult in enumerate(harm_mults):
                bin_idx = int(round(fg * mult))
                col = ch * n_harm + hi
                if 0 < bin_idx < len(spectrum):
                    result[t, col] = spectrum[bin_idx]
    return result
