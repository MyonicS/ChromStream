from __future__ import annotations

import h5py
import pandas as pd

from chromstream import write_chromatogram_hdf5
from chromstream.objects import Chromatogram, Experiment


def make_chromatogram(*, channel: str, injection_time: str) -> Chromatogram:
    return Chromatogram(
        data=pd.DataFrame({"time": [0.0, 1.0], "signal": [1.0, 2.0]}),
        injection_time=pd.Timestamp(injection_time),
        metadata={"time_unit": "min", "Signal Unit": "pA"},
        channel=channel,
        path=None,
    )


def test_reassembles_experiment_with_exact_indices(tmp_path):
    # write several standalone chromatograms across two channels, each carrying
    # its original channel-level injection index
    specs = [
        ("FID_L", 0, "2024-01-01T12:00:00Z"),
        ("FID_L", 2, "2024-01-01T12:10:00Z"),
        ("TCD", 0, "2024-01-01T12:05:00Z"),
    ]
    paths = []
    for channel, index, inj_time in specs:
        path = tmp_path / f"{channel}-{index}.h5"
        write_chromatogram_hdf5(
            make_chromatogram(channel=channel, injection_time=inj_time),
            path,
            injection_index=index,
        )
        paths.append(path)

    experiment = Experiment(name="reassembled")
    # load in a deliberately shuffled order to prove indices come from the files
    for path in reversed(paths):
        experiment.add_chromatogram_hdf5(path)

    assert sorted(experiment.channels) == ["FID_L", "TCD"]
    assert sorted(experiment.channels["FID_L"].chromatograms) == [0, 2]
    assert sorted(experiment.channels["TCD"].chromatograms) == [0]
    assert experiment.channels["FID_L"].chromatograms[2].injection_time == pd.Timestamp(
        "2024-01-01T12:10:00Z"
    )


def test_reassembly_falls_back_to_sequential_without_index(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"chrom-{i}.h5"
        write_chromatogram_hdf5(
            make_chromatogram(
                channel="FID_L", injection_time=f"2024-01-01T12:0{i}:00Z"
            ),
            path,
        )
        # ensure no injection_index attr is present
        with h5py.File(path, "a") as hdf:
            assert "injection_index" not in hdf.attrs
        paths.append(path)

    experiment = Experiment(name="reassembled")
    for path in paths:
        experiment.add_chromatogram_hdf5(path)

    assert sorted(experiment.channels["FID_L"].chromatograms) == [0, 1, 2]


def test_reassembly_channel_override(tmp_path):
    path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(
        make_chromatogram(channel="FID_L", injection_time="2024-01-01T12:00:00Z"),
        path,
        injection_index=0,
    )

    experiment = Experiment(name="reassembled")
    experiment.add_chromatogram_hdf5(path, channel_name="TCD")

    assert list(experiment.channels) == ["TCD"]
    assert experiment.channels["TCD"].chromatograms[0].channel == "FID_L"
