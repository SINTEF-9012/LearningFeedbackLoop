"""
Unit tests for the computation module.

Tests cover:
- Window functions (_hann, _rect)
- Time window selection (_select_window_by_time)
- Goertzel amplitude estimation
- FFT amplitude estimation
- Multi-channel FFT computation
"""

import pytest
import numpy as np

from backend.computation import (
    _hann,
    _rect,
    _coherent_gain,
    _select_window_by_time,
    goertzel_amplitude,
    fft_interp_amplitude,
    compute_rfft_multichannel,
)


class TestWindowFunctions:
    """Tests for window functions."""
    
    def test_hann_window_length(self):
        """Hann window should have correct length."""
        for n in [10, 100, 1024]:
            win = _hann(n)
            assert len(win) == n
    
    def test_hann_window_ends(self):
        """Hann window should be near zero at ends."""
        win = _hann(100)
        assert win[0] < 0.01
        assert win[-1] < 0.01
    
    def test_hann_window_peak(self):
        """Hann window should peak in the middle."""
        win = _hann(101)
        mid_idx = len(win) // 2
        assert win[mid_idx] == pytest.approx(1.0, rel=1e-6)
    
    def test_rect_window_length(self):
        """Rectangular window should have correct length."""
        for n in [10, 100, 1024]:
            win = _rect(n)
            assert len(win) == n
    
    def test_rect_window_values(self):
        """Rectangular window should be all ones."""
        win = _rect(100)
        assert np.allclose(win, np.ones(100))


class TestCoherentGain:
    """Tests for coherent gain calculation."""
    
    def test_rect_window_gain(self):
        """Rectangular window should have coherent gain of 1.0."""
        win = _rect(100)
        cg = _coherent_gain(win)
        assert cg == pytest.approx(1.0, rel=1e-6)
    
    def test_hann_window_gain(self):
        """Hann window should have coherent gain of ~0.5."""
        win = _hann(1000)
        cg = _coherent_gain(win)
        assert cg == pytest.approx(0.5, rel=0.01)


class TestSelectWindowByTime:
    """Tests for time-based window selection."""
    
    def test_basic_selection(self):
        """Basic time window selection should work."""
        signal = np.arange(1000, dtype=float)
        fs = 100.0  # 10 seconds total
        
        segment, i0, i1 = _select_window_by_time(signal, fs, 2.0, 5.0)
        
        assert i0 == 200  # 2.0 * 100
        assert i1 == 500  # 5.0 * 100
        assert len(segment) == 300
    
    def test_clipping_to_bounds(self):
        """Selection should clip to signal bounds."""
        signal = np.arange(100, dtype=float)
        fs = 10.0  # 10 seconds total
        
        segment, i0, i1 = _select_window_by_time(signal, fs, -1.0, 15.0)
        
        assert i0 == 0
        assert i1 == 100
        assert len(segment) == 100
    
    def test_empty_selection_raises(self):
        """Empty selection should raise ValueError."""
        signal = np.arange(100, dtype=float)
        fs = 10.0
        
        with pytest.raises(ValueError, match="no samples selected"):
            _select_window_by_time(signal, fs, 20.0, 25.0)


class TestGoertzelAmplitude:
    """Tests for Goertzel amplitude estimation."""
    
    def test_pure_sine_amplitude(self):
        """Should correctly estimate amplitude of pure sine wave."""
        fs = 1000.0
        f0 = 50.0
        amplitude = 2.5
        t = np.arange(0, 1.0, 1.0/fs)
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        window = _hann(len(signal))
        
        estimated = goertzel_amplitude(signal, fs, f0, window, return_peak=True)
        
        # Should be close to the actual amplitude
        assert estimated == pytest.approx(amplitude, rel=0.05)
    
    def test_rms_vs_peak(self):
        """RMS should be peak / sqrt(2) for sinusoids."""
        fs = 1000.0
        f0 = 100.0
        amplitude = 3.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        window = _hann(len(signal))
        
        peak = goertzel_amplitude(signal, fs, f0, window, return_peak=True)
        rms = goertzel_amplitude(signal, fs, f0, window, return_peak=False)
        
        assert rms == pytest.approx(peak / np.sqrt(2), rel=0.01)
    
    def test_off_frequency_low_amplitude(self):
        """Off-frequency measurement should give low amplitude."""
        fs = 1000.0
        f_signal = 100.0
        f_measure = 200.0  # Measuring at wrong frequency
        amplitude = 5.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = amplitude * np.sin(2 * np.pi * f_signal * t)
        window = _hann(len(signal))
        
        estimated = goertzel_amplitude(signal, fs, f_measure, window, return_peak=True)
        
        # Should be much smaller than the actual amplitude
        assert estimated < amplitude * 0.1


class TestFFTInterpAmplitude:
    """Tests for FFT-based amplitude estimation."""
    
    def test_pure_sine_amplitude(self):
        """Should correctly estimate amplitude of pure sine wave."""
        fs = 1000.0
        f0 = 50.0
        amplitude = 2.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        window = _hann(len(signal))
        
        estimated = fft_interp_amplitude(signal, fs, f0, window, return_peak=True)
        
        assert estimated == pytest.approx(amplitude, rel=0.05)
    
    def test_fractional_frequency(self):
        """Should handle non-bin-centered frequencies."""
        fs = 1000.0
        f0 = 53.7  # Not on a bin center
        amplitude = 1.5
        t = np.arange(0, 1.0, 1.0/fs)
        signal = amplitude * np.sin(2 * np.pi * f0 * t)
        window = _hann(len(signal))
        
        estimated = fft_interp_amplitude(signal, fs, f0, window, return_peak=True)
        
        # Interpolation should still give good estimate
        assert estimated == pytest.approx(amplitude, rel=0.1)


class TestComputeRFFTMultichannel:
    """Tests for multi-channel FFT computation."""
    
    def test_single_channel(self):
        """Should work with single channel."""
        fs = 1000.0
        f0 = 100.0
        t = np.arange(0, 0.5, 1.0/fs)
        signal = np.sin(2 * np.pi * f0 * t)
        
        data_dict = {"ch1": signal.tolist()}
        
        result = compute_rfft_multichannel(
            data_dict=data_dict,
            fs=fs,
            channels=["ch1"],
            nfft=1024
        )
        
        assert "ch1" in result
        assert "freqs" in result["ch1"]
        assert "mags" in result["ch1"]
        assert len(result["ch1"]["freqs"]) == len(result["ch1"]["mags"])
    
    def test_multi_channel(self):
        """Should work with multiple channels."""
        fs = 1000.0
        t = np.arange(0, 0.5, 1.0/fs)
        
        data_dict = {
            "ch1": (1.0 * np.sin(2 * np.pi * 50 * t)).tolist(),
            "ch2": (2.0 * np.sin(2 * np.pi * 100 * t)).tolist(),
        }
        
        result = compute_rfft_multichannel(
            data_dict=data_dict,
            fs=fs,
            channels=["ch1", "ch2"],
            nfft=1024
        )
        
        assert "ch1" in result
        assert "ch2" in result
    
    def test_frequency_axis_correct(self):
        """Frequency axis should be correctly computed."""
        fs = 1000.0
        t = np.arange(0, 1.0, 1.0/fs)
        signal = np.sin(2 * np.pi * 100 * t)
        
        data_dict = {"ch1": signal.tolist()}
        nfft = 1024
        
        result = compute_rfft_multichannel(
            data_dict=data_dict,
            fs=fs,
            channels=["ch1"],
            nfft=nfft
        )
        
        freqs = result["ch1"]["freqs"]
        # Max frequency should be Nyquist
        assert max(freqs) == pytest.approx(fs / 2, rel=0.01)
        # Frequency resolution should be fs / nfft
        df = freqs[1] - freqs[0]
        assert df == pytest.approx(fs / nfft, rel=0.01)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_dc_signal(self):
        """Should handle DC (constant) signals."""
        fs = 1000.0
        signal = np.ones(1000) * 5.0
        window = _hann(len(signal))
        
        # Goertzel at non-zero frequency should give ~0
        amp = goertzel_amplitude(signal, fs, 100.0, window, detrend=True)
        assert amp == pytest.approx(0.0, abs=0.01)
    
    def test_very_short_signal(self):
        """Should handle very short signals."""
        fs = 1000.0
        signal = np.array([1.0, 0.0, -1.0, 0.0])
        window = _hann(len(signal))
        
        # Should not raise
        amp = goertzel_amplitude(signal, fs, 250.0, window)
        assert isinstance(amp, float)
    
    def test_empty_channel_selection(self):
        """Empty channel selection should return empty result."""
        data_dict = {"ch1": [1, 2, 3]}
        
        result = compute_rfft_multichannel(
            data_dict=data_dict,
            fs=1000.0,
            channels=[],
            nfft=1024
        )
        
        # Should return dict without channel entries
        assert isinstance(result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
