from __future__ import annotations

import h5py
import pandas as pd
import pytest

from chromstream import parse_channel_hdf5, write_channel_hdf5
from chromstream.objects import ChannelChromatograms, Chromatogram


def make_chromatogram(*, injection_time: str, signal_unit: str) -> Chromatogram:
    return Chromatogram(
        data=pd.DataFrame({"time": [0.0, 0.5, 1.0], "signal": [1.0, 2.0, 1.5]}),
        injection_time=pd.Timestamp(injection_time),
        metadata={"time_unit": "min", "Signal Unit": signal_unit},
        channel="FID_L",
        path=None,
    )


def make_channel() -> ChannelChromatograms:
    channel = ChannelChromatograms(channel="FID_L")
    channel.add_chromatogram(
        2, make_chromatogram(injection_time="2024-01-01T12:10:00Z", signal_unit="mV")
    )
    channel.add_chromatogram(
        0, make_chromatogram(injection_time="2024-01-01T12:00:00Z", signal_unit="pA")
    )
    return channel


def test_write_channel_hdf5_writes_expected_layout(tmp_path):
    output_path = tmp_path / "channel.h5"

    result = write_channel_hdf5(make_channel(), output_path)

    assert result == output_path
    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["schema"] == "chromstream-channel/v0.1.0"
        assert hdf.attrs["name"] == "FID_L"
        assert "inj-0000" in hdf["injections"]
        assert "inj-0002" in hdf["injections"]


def test_channel_round_trips_with_non_contiguous_indices(tmp_path):
    output_path = tmp_path / "channel.h5"

    write_channel_hdf5(make_channel(), output_path)
    parsed = parse_channel_hdf5(output_path)

    assert parsed.channel == "FID_L"
    assert sorted(parsed.chromatograms) == [0, 2]
    assert parsed.chromatograms[0].injection_time == pd.Timestamp(
        "2024-01-01T12:00:00Z"
    )
    assert parsed.chromatograms[2].injection_time == pd.Timestamp(
        "2024-01-01T12:10:00Z"
    )
    assert parsed.chromatograms[0].signal_unit == "pA"
    assert parsed.chromatograms[2].signal_unit == "mV"
    assert parsed.chromatograms[0].data["signal"].tolist() == [1.0, 2.0, 1.5]
    assert parsed.chromatograms[0].path == output_path


def test_channel_to_hdf5_delegates_to_writer(tmp_path):
    output_path = tmp_path / "method.h5"

    result = make_channel().to_hdf5(output_path)

    assert result == output_path
    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["schema"] == "chromstream-channel/v0.1.0"
        assert hdf.attrs["name"] == "FID_L"


def test_write_channel_hdf5_refuses_existing_file_without_overwrite(tmp_path):
    output_path = tmp_path / "channel.h5"
    output_path.touch()

    with pytest.raises(FileExistsError):
        write_channel_hdf5(make_channel(), output_path)


def test_parse_channel_hdf5_rejects_unknown_schema(tmp_path):
    output_path = tmp_path / "channel.h5"
    write_channel_hdf5(make_channel(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "chromstream-channel/v9.9.9"

    with pytest.raises(ValueError, match="Unknown schema"):
        parse_channel_hdf5(output_path)


def test_parse_channel_hdf5_rejects_non_chromstream_schema(tmp_path):
    output_path = tmp_path / "channel.h5"
    write_channel_hdf5(make_channel(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "other-format/v1"

    with pytest.raises(
        ValueError, match="Only parsing of ChromStream HDF5 files is supported"
    ):
        parse_channel_hdf5(output_path)
