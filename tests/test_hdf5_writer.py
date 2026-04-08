from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest

from chromstream import write_experiment_hdf5
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
        schema="chromstream-experiment/v1",
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


def test_write_experiment_hdf5_writes_expected_layout(tmp_path):
    experiment = make_experiment()
    output_path = tmp_path / "experiment.h5"

    result = write_experiment_hdf5(experiment, output_path)

    assert result == output_path
    assert output_path.exists()

    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["schema"] == "chromstream-experiment/v1"
        assert hdf.attrs["label"] == "Hydrogenation Run"
        assert hdf.attrs["creation_date"] == "2024-01-01T12:00:00+00:00"
        assert hdf.attrs["author"] == "Ada Lovelace"
        assert hdf.attrs["lab"] == "FHI"
        assert hdf.attrs["run_id"] == 42

        assert "Channels" in hdf
        assert "FID_L" in hdf["Channels"]
        assert "TCD" in hdf["Channels"]
        assert hdf["Channels"]["FID_L"].attrs["name"] == "FID_L"

        first_injection = hdf["Channels"]["FID_L"]["injections"]["inj-0000"]
        second_injection = hdf["Channels"]["FID_L"]["injections"]["inj-0002"]

        assert first_injection.attrs["injection_time"] == "2024-01-01T12:00:00+00:00"

        assert first_injection["retention_time"].attrs["unit"] == "min"
        assert first_injection["retention_time"].attrs["column_name"] == "time"
        assert first_injection["signal"].attrs["unit"] == "pA"
        assert first_injection["signal"].attrs["column_name"] == "signal"

        assert first_injection["retention_time"][:].tolist() == [0.0, 0.5, 1.0]
        assert first_injection["signal"][:].tolist() == [0.0, 1.0, 0.5]
        assert second_injection["signal"][:].tolist() == [1.0, 2.0, 1.5]


def test_write_experiment_hdf5_refuses_existing_file_without_overwrite(tmp_path):
    output_path = tmp_path / "experiment.h5"
    output_path.touch()

    with pytest.raises(FileExistsError):
        write_experiment_hdf5(make_experiment(), output_path)


def test_experiment_to_hdf5_delegates_to_writer(tmp_path):
    experiment = make_experiment()
    output_path = tmp_path / "method.h5"

    result = experiment.to_hdf5(output_path)

    assert result == output_path
    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["label"] == "Hydrogenation Run"
        assert "FID_L" in hdf["Channels"]


def test_write_experiment_hdf5_rejects_reserved_metadata_keys(tmp_path):
    experiment = Experiment(name="Reserved", metadata={"label": "collision"})

    with pytest.raises(ValueError, match="reserved attribute names"):
        write_experiment_hdf5(experiment, tmp_path / "reserved.h5")


def test_write_experiment_hdf5_rejects_nested_metadata_values(tmp_path):
    experiment = Experiment(name="Nested", metadata={"details": {"operator": "Ada"}})

    with pytest.raises(TypeError, match="HDF5-compatible"):
        write_experiment_hdf5(experiment, tmp_path / "nested.h5")


def test_write_experiment_hdf5_skips_creation_date_when_missing(tmp_path):
    experiment = Experiment(name="No Date")
    output_path = tmp_path / "no-date.h5"

    write_experiment_hdf5(experiment, output_path)

    with h5py.File(output_path, "r") as hdf:
        assert "creation_date" not in hdf.attrs
