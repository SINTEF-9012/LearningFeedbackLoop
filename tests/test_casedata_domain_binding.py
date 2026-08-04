from __future__ import annotations

import numpy as np
import pytest

from backend.agents.domain_config import ChannelRole, get_domain
from backend.inference_streamer import _features_from_channels

# cnc_machining is now defined in domain_packs/cnc.yaml (YAML is the single source).
CNC_MACHINING_DOMAIN = get_domain("cnc_machining")


def test_cnc_domain_resolves_casedata_channels_first():
    channels = [
        "Power_Spindle",
        "Power_Y",
        "Vibration_Severity_X",
        "Chatter_Detection_Amplitude_X",
        "Power_Active",
        "Spindle_Speed_Actual",
        "Feed_Rate_Actual",
    ]

    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.PRIMARY_POWER, channels) == "Power_Spindle"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.SECONDARY_POWER, channels) == "Power_Y"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.PRIMARY_VIBRATION, channels) == "Vibration_Severity_X"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.CHATTER_AMPLITUDE, channels) == "Chatter_Detection_Amplitude_X"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.ACTIVE_POWER, channels) == "Power_Active"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.SPINDLE_SPEED, channels) == "Spindle_Speed_Actual"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.FEED_RATE, channels) == "Feed_Rate_Actual"


def test_cnc_domain_falls_back_to_legacy_aliases():
    channels = [
        "Spindle_Power",
        "X_Axis_Power",
        "Vibration",
        "Chatter_Amp",
        "Active_Power",
        "Spindle_Speed",
        "Feed_Rate",
    ]

    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.PRIMARY_POWER, channels) == "Spindle_Power"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.SECONDARY_POWER, channels) == "X_Axis_Power"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.PRIMARY_VIBRATION, channels) == "Vibration"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.CHATTER_AMPLITUDE, channels) == "Chatter_Amp"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.ACTIVE_POWER, channels) == "Active_Power"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.SPINDLE_SPEED, channels) == "Spindle_Speed"
    assert CNC_MACHINING_DOMAIN.resolve_channel(ChannelRole.FEED_RATE, channels) == "Feed_Rate"


def test_features_from_channels_aligns_with_casedata_columns():
    window = {
        "Power_Spindle": np.array([10.0, 20.0, 30.0]),
        "Power_Y": np.array([1.0, 2.0, 3.0]),
        "Power_Z": np.array([2.0, 4.0, 6.0]),
        "Vibration_Severity_X": np.array([0.1, 0.2, 0.3]),
        "Vibration_Severity_Y": np.array([0.4, 0.5, 0.6]),
        "Chatter_Detection_Amplitude_X": np.array([0.05, 0.6, 0.7]),
        "Chatter_Detection_OnOff_X": np.array([0.0, 1.0, 1.0]),
        "Chatter_Detection_OnOff_Y": np.array([0.0, 0.0, 1.0]),
        "Power_Active": np.array([5.0, 6.0, 7.0]),
        "Spindle_Speed_Actual": np.array([600.0, 600.0, 600.0]),
        "Feed_Rate_Actual": np.array([100.0, 100.0, 100.0]),
    }

    features = _features_from_channels(window, fs=1.0, domain=CNC_MACHINING_DOMAIN)

    assert features["power_spindle_mean"] == pytest.approx(20.0)
    assert features["power_y_mean"] == pytest.approx(2.0)
    assert features["power_z_mean"] == pytest.approx(4.0)
    assert features["vib_severity_x_mean"] == pytest.approx(0.2)
    assert features["vib_severity_y_mean"] == pytest.approx(0.5)
    assert features["power_active_mean"] == pytest.approx(6.0)
    assert features["spindle_speed_mean"] == pytest.approx(600.0)
    assert features["feed_rate_mean"] == pytest.approx(100.0)
    assert features["chatter_ratio"] == pytest.approx(0.5)


def test_features_from_channels_use_num_teeth_for_tooth_pass_features():
    fs = 512.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    window = {
        "Vibration_Severity_X": np.sin(2.0 * np.pi * 80.0 * t),
        "Spindle_Speed_Actual": np.full_like(t, 600.0),
        "Feed_Rate_Actual": np.full_like(t, 100.0),
    }

    features_without_teeth = _features_from_channels(window, fs=fs, domain=CNC_MACHINING_DOMAIN)
    features_with_teeth = _features_from_channels(window, fs=fs, domain=CNC_MACHINING_DOMAIN, num_teeth=4)

    assert features_with_teeth["tp_harmonic_energy"] > features_without_teeth["tp_harmonic_energy"] + 0.2