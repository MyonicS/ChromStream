from __future__ import annotations

import h5py
import pandas as pd
import pytest

from chromstream import parse_experiment_hdf5, write_experiment_hdf5
from chromstream.objects import ChannelChromatograms, Chromatogram, Experiment


def make_chromatogram(
    *,
    channel: str,
    injection_time: str,
    time_values: list[float],
    signal_values: list[float],
    metadata: dict | None = None,
) -> Chromatogram:
    metadata = {"time_unit": "min", **(metadata or {})}
    return Chromatogram(
        data=pd.DataFrame({"time": time_values, "signal": signal_values}),
        injection_time=pd.Timestamp(injection_time),
        metadata=metadata,
        channel=channel,
        path=None,
    )


def make_experiment() -> Experiment:
    experiment = Experiment(
        name="Hydrogenation Run",
        schema="chromstream-experiment/v0.1.0",
        author="Ada Lovelace",
        creation_date=pd.Timestamp("2024-01-01T12:00:00Z"),
        metadata={"lab": "FHI", "run_id": 42},
    )

    fid_channel = ChannelChromatograms(channel="FID_L")
    fid_channel.add_chromatogram(
        2,
        make_chromatogram(
            channel="FID_L",
            injection_time="2024-01-01T12:10:00Z",
            time_values=[0.0, 0.5, 1.0],
            signal_values=[1.0, 2.0, 1.5],
            metadata={"Signal Unit": "mV"},
        ),
    )
    fid_channel.add_chromatogram(
        0,
        make_chromatogram(
            channel="FID_L",
            injection_time="2024-01-01T12:00:00Z",
            time_values=[0.0, 0.5, 1.0],
            signal_values=[0.0, 1.0, 0.5],
            metadata={"signal_unit": "pA"},
        ),
    )

    tcd_channel = ChannelChromatograms(channel="TCD")
    tcd_channel.add_chromatogram(
        0,
        make_chromatogram(
            channel="TCD",
            injection_time="2024-01-01T12:05:00Z",
            time_values=[0.0, 1.0, 2.0],
            signal_values=[3.0, 2.5, 2.0],
            metadata={"Signal Unit": "a.u."},
        ),
    )

    experiment.add_channel("FID_L", fid_channel)
    experiment.add_channel("TCD", tcd_channel)
    return experiment


def test_parse_experiment_hdf5_round_trips_writer_layout(tmp_path):
    experiment = make_experiment()
    output_path = tmp_path / "experiment.h5"

    write_experiment_hdf5(experiment, output_path)
    parsed = parse_experiment_hdf5(output_path)

    assert parsed.name == "Hydrogenation Run"
    assert parsed.schema == "chromstream-experiment/v0.1.0"
    assert parsed.author == "Ada Lovelace"
    assert parsed.creation_date == pd.Timestamp("2024-01-01T12:00:00Z")
    assert parsed.metadata == {"lab": "FHI", "run_id": 42}
    assert parsed.log is None
    assert parsed.experiment_starttime is None
    assert parsed.experiment_endtime is None

    assert sorted(parsed.channels) == ["FID_L", "TCD"]
    assert sorted(parsed.channels["FID_L"].chromatograms) == [0, 2]
    first_injection = parsed.channels["FID_L"].chromatograms[0]
    second_injection = parsed.channels["FID_L"].chromatograms[2]
    tcd_injection = parsed.channels["TCD"].chromatograms[0]

    assert first_injection.injection_time == pd.Timestamp("2024-01-01T12:00:00Z")
    assert second_injection.injection_time == pd.Timestamp("2024-01-01T12:10:00Z")
    assert first_injection.data.columns.tolist() == ["time", "signal"]
    assert first_injection.data["time"].tolist() == [0.0, 0.5, 1.0]
    assert first_injection.data["signal"].tolist() == [0.0, 1.0, 0.5]
    assert second_injection.data["signal"].tolist() == [1.0, 2.0, 1.5]
    assert first_injection.time_unit == "min"
    assert first_injection.signal_unit == "pA"
    assert tcd_injection.signal_unit == "a.u."
    assert first_injection.path == output_path


def test_parse_experiment_hdf5_rejects_missing_schema(tmp_path):
    output_path = tmp_path / "missing-schema.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        del hdf.attrs["schema"]

    with pytest.raises(
        ValueError, match="Only parsing of ChromStream HDF5 files is supported"
    ):
        parse_experiment_hdf5(output_path)


def test_parse_experiment_hdf5_rejects_unknown_schema(tmp_path):
    output_path = tmp_path / "unknown-schema.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "chromstream-experiment/v9.9.9"

    with pytest.raises(ValueError, match="Unknown schema"):
        parse_experiment_hdf5(output_path)


def test_parse_experiment_hdf5_rejects_non_chromstream_schema(tmp_path):
    output_path = tmp_path / "foreign-schema.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "other-format/v1"

    with pytest.raises(
        ValueError, match="Only parsing of ChromStream HDF5 files is supported"
    ):
        parse_experiment_hdf5(output_path)


def test_parse_experiment_hdf5_rejects_missing_channels_group(tmp_path):
    output_path = tmp_path / "missing-channels.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        del hdf["Channels"]

    with pytest.raises(ValueError, match="Missing required group 'Channels'"):
        parse_experiment_hdf5(output_path)


def test_parse_experiment_hdf5_rejects_missing_signal_dataset(tmp_path):
    output_path = tmp_path / "missing-signal.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        del hdf["Channels"]["FID_L"]["injections"]["inj-0000"]["signal"]

    with pytest.raises(ValueError, match="must contain 'retention_time' and 'signal'"):
        parse_experiment_hdf5(output_path)


def test_parse_experiment_hdf5_rejects_missing_injection_time(tmp_path):
    output_path = tmp_path / "missing-injection-time.h5"
    write_experiment_hdf5(make_experiment(), output_path)

    with h5py.File(output_path, "a") as hdf:
        del hdf["Channels"]["FID_L"]["injections"]["inj-0000"].attrs["injection_time"]

    with pytest.raises(ValueError, match="Missing required attribute 'injection_time'"):
        parse_experiment_hdf5(output_path)
