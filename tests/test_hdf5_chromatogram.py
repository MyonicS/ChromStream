from __future__ import annotations

import h5py
import pandas as pd
import pytest

from chromstream import (
    parse_chromatogram_hdf5,
    write_chromatogram_hdf5,
)
from chromstream.objects import Chromatogram
from chromstream.parsers.hdf5 import _parse_chromatogram_hdf5_with_index


def make_chromatogram(*, metadata: dict | None = None) -> Chromatogram:
    metadata = {"time_unit": "min", "Signal Unit": "pA", **(metadata or {})}
    return Chromatogram(
        data=pd.DataFrame({"time": [0.0, 0.5, 1.0], "signal": [1.0, 2.0, 1.5]}),
        injection_time=pd.Timestamp("2024-01-01T12:00:00Z"),
        metadata=metadata,
        channel="FID_L",
        path=None,
    )


def test_write_chromatogram_hdf5_writes_expected_layout(tmp_path):
    output_path = tmp_path / "chrom.h5"

    result = write_chromatogram_hdf5(make_chromatogram(), output_path, injection_index=3)

    assert result == output_path
    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["schema"] == "chromstream-chromatogram/v0.1.0"
        assert hdf.attrs["channel"] == "FID_L"
        assert hdf.attrs["injection_index"] == 3
        assert hdf.attrs["injection_time"] == "2024-01-01T12:00:00+00:00"
        assert hdf["retention_time"].attrs["unit"] == "min"
        assert hdf["retention_time"].attrs["column_name"] == "time"
        assert hdf["signal"].attrs["unit"] == "pA"
        assert hdf["signal"].attrs["column_name"] == "signal"
        assert hdf["retention_time"][:].tolist() == [0.0, 0.5, 1.0]
        assert hdf["signal"][:].tolist() == [1.0, 2.0, 1.5]


def test_chromatogram_full_metadata_round_trips(tmp_path):
    output_path = tmp_path / "chrom.h5"
    original = make_chromatogram(
        metadata={"method": "FID2", "cycle": 7, "threshold": 0.5, "Channel": "FID_L"}
    )

    write_chromatogram_hdf5(original, output_path)
    parsed = parse_chromatogram_hdf5(output_path)

    # every key survives, in particular the 'Signal Unit' key is not mangled and
    # the case-distinct 'Channel' metadata key is preserved separately
    assert parsed.metadata == original.metadata
    assert parsed.metadata["Signal Unit"] == "pA"
    assert parsed.metadata["cycle"] == 7
    assert parsed.metadata["threshold"] == 0.5
    assert parsed.channel == "FID_L"
    assert parsed.injection_time == pd.Timestamp("2024-01-01T12:00:00Z")
    assert parsed.data.columns.tolist() == ["time", "signal"]
    assert parsed.data["signal"].tolist() == [1.0, 2.0, 1.5]
    assert parsed.path == output_path


def test_timestamp_and_path_metadata_normalize_to_strings(tmp_path):
    # Documented caveat: Timestamp/Path metadata round-trips as ISO/str, not the
    # original Python type.
    output_path = tmp_path / "chrom.h5"
    original = make_chromatogram(
        metadata={"Inject Time": pd.Timestamp("2024-01-01T12:00:00Z")}
    )

    write_chromatogram_hdf5(original, output_path)
    parsed = parse_chromatogram_hdf5(output_path)

    assert parsed.metadata["Inject Time"] == "2024-01-01T12:00:00+00:00"


def test_injection_index_preserved(tmp_path):
    output_path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(make_chromatogram(), output_path, injection_index=5)

    index, chrom = _parse_chromatogram_hdf5_with_index(output_path)

    assert index == 5
    assert chrom.channel == "FID_L"


def test_injection_index_absent_returns_none(tmp_path):
    output_path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(make_chromatogram(), output_path)

    index, _ = _parse_chromatogram_hdf5_with_index(output_path)

    assert index is None


def test_chromatogram_to_hdf5_delegates_to_writer(tmp_path):
    output_path = tmp_path / "method.h5"

    result = make_chromatogram().to_hdf5(output_path, injection_index=1)

    assert result == output_path
    with h5py.File(output_path, "r") as hdf:
        assert hdf.attrs["schema"] == "chromstream-chromatogram/v0.1.0"
        assert hdf.attrs["injection_index"] == 1


def test_write_chromatogram_hdf5_rejects_reserved_metadata_keys(tmp_path):
    chrom = make_chromatogram(metadata={"channel": "collision"})

    with pytest.raises(ValueError, match="reserved attribute names"):
        write_chromatogram_hdf5(chrom, tmp_path / "reserved.h5")


def test_write_chromatogram_hdf5_rejects_nested_metadata_values(tmp_path):
    chrom = make_chromatogram(metadata={"details": {"operator": "Ada"}})

    with pytest.raises(TypeError, match="HDF5-compatible"):
        write_chromatogram_hdf5(chrom, tmp_path / "nested.h5")


def test_write_chromatogram_hdf5_refuses_existing_file_without_overwrite(tmp_path):
    output_path = tmp_path / "chrom.h5"
    output_path.touch()

    with pytest.raises(FileExistsError):
        write_chromatogram_hdf5(make_chromatogram(), output_path)


def test_parse_chromatogram_hdf5_rejects_missing_schema(tmp_path):
    output_path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(make_chromatogram(), output_path)

    with h5py.File(output_path, "a") as hdf:
        del hdf.attrs["schema"]

    with pytest.raises(
        ValueError, match="Only parsing of ChromStream HDF5 files is supported"
    ):
        parse_chromatogram_hdf5(output_path)


def test_parse_chromatogram_hdf5_rejects_unknown_schema(tmp_path):
    output_path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(make_chromatogram(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "chromstream-chromatogram/v9.9.9"

    with pytest.raises(ValueError, match="Unknown schema"):
        parse_chromatogram_hdf5(output_path)


def test_parse_chromatogram_hdf5_rejects_non_chromstream_schema(tmp_path):
    output_path = tmp_path / "chrom.h5"
    write_chromatogram_hdf5(make_chromatogram(), output_path)

    with h5py.File(output_path, "a") as hdf:
        hdf.attrs["schema"] = "other-format/v1"

    with pytest.raises(
        ValueError, match="Only parsing of ChromStream HDF5 files is supported"
    ):
        parse_chromatogram_hdf5(output_path)
