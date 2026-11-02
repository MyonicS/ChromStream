from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
from chromstream.parsers import (
    parse_chromatogram_txt,
    parse_chromatogram,
    parse_MTO_asc,
    parse_log_file,
    parse_log_MTO,
    parse_agilent_ch,
    parse_agilent_dot_d,
    parse_agilent_dx,
    parse_chromatograms,
    parse_chromeleon_txt,
    parse_inject_time,
)

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "testdata" / "chroms"
LOG_DATA_DIR = Path(__file__).parent / "testdata" / "logs"


class TestChromatogramParsers:
    """Test chromatogram parsing functions"""

    def test_parse_format_1_txt(self):
        """Test parsing format_1.txt"""
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_format_2_txt(self):
        """Test parsing format_2.txt"""
        file_path = TEST_DATA_DIR / "format_2.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_format_3_txt(self):
        """Test parsing format_3.txt"""
        file_path = TEST_DATA_DIR / "format_3.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_format_4_txt(self):
        """Test parsing format_4.txt"""
        file_path = TEST_DATA_DIR / "format_4.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_format_5_txt(self):
        """Test parsing format_5.txt"""
        file_path = TEST_DATA_DIR / "format_5.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_format_6_txt(self):
        """Test parsing format_6.txt"""
        file_path = TEST_DATA_DIR / "format_6.txt"
        chrom = parse_chromatogram_txt(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_ascii_format_1_asc(self):
        """Test parsing ascii_format_1.asc"""
        file_path = TEST_DATA_DIR / "ascii_format_1.asc"
        chrom1, chrom2, chrom3 = parse_MTO_asc(file_path)

        # Test all three chromatograms returned
        for chrom in [chrom1, chrom2, chrom3]:
            assert chrom is not None
            assert isinstance(chrom.data, pd.DataFrame)
            assert not chrom.data.empty
            assert chrom.injection_time is not None
            assert chrom.channel is not None

        # Verify the channels are correctly assigned
        assert chrom1.channel == "FID_L"
        assert chrom2.channel == "FID_M"
        assert chrom3.channel == "TCD"


class TestLogParsers:
    """Test log file parsing functions"""

    def test_parse_log_1_txt(self):
        """Test parsing Log_1.txt"""
        file_path = LOG_DATA_DIR / "Log_1.txt"
        log_df = parse_log_file(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns

    def test_parse_log_2_txt(self):
        """Test parsing Log_2.txt"""
        file_path = LOG_DATA_DIR / "Log_2.txt"
        log_df = parse_log_file(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns

    def test_parse_log_3_txt(self):
        """Test parsing Log_3.txt"""
        file_path = LOG_DATA_DIR / "Log_3.txt"
        log_df = parse_log_file(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns

    def test_parse_log_4_txt(self):
        """Test parsing Log_4.txt"""
        file_path = LOG_DATA_DIR / "Log_4.txt"
        log_df = parse_log_MTO(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns

    def test_parse_log_5_txt(self):
        """Test parsing Log_5.txt"""
        file_path = LOG_DATA_DIR / "Log_5.txt"
        log_df = parse_log_file(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns

    def test_parse_log_6_txt(self):
        """Test parsing Log_6.txt"""
        file_path = LOG_DATA_DIR / "Log_6.txt"
        log_df = parse_log_file(file_path)

        assert log_df is not None
        assert isinstance(log_df, pd.DataFrame)
        assert not log_df.empty
        assert "Timestamp" in log_df.columns


class TestAgilentParsers:
    """Test Agilent file parsing functions"""

    def test_parse_ch_file_direct(self):
        """Test direct parsing of Agilent .ch files"""
        file_path = TEST_DATA_DIR / "agilent.d" / "FID1A.ch"
        chrom = parse_agilent_ch(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.path is not None
        assert chrom.channel == "FID1A"

    def test_parse_ch_file_with_channel_override(self):
        """Test channel name override for Agilent .ch parser"""
        file_path = TEST_DATA_DIR / "agilent.d" / "FID1A.ch"
        chrom = parse_agilent_ch(file_path, channel_name="CustomFID")

        assert chrom.channel == "CustomFID"

    def test_parse_dx_file(self):
        """Test parsing .dx file"""
        file_path = TEST_DATA_DIR / "test_dx.dx"
        chrom_list = parse_agilent_dx(file_path)

        # Verify we get a list of chromatograms
        assert chrom_list is not None
        assert isinstance(chrom_list, list)
        assert len(chrom_list) > 0

        # Verify each chromatogram has required attributes
        for chrom in chrom_list:
            assert chrom.data is not None
            assert isinstance(chrom.data, pd.DataFrame)
            assert not chrom.data.empty
            assert chrom.injection_time is not None
            assert chrom.channel is not None


class TestAutoChromatogramParser:
    """Test parser dispatch for single chromatogram files."""

    def test_parse_chromeleon_txt_file(self):
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_parse_agilent_ch_file(self):
        file_path = TEST_DATA_DIR / "agilent.d" / "FID1A.ch"
        chrom = parse_chromatogram(file_path)

        assert chrom is not None
        assert isinstance(chrom.data, pd.DataFrame)
        assert not chrom.data.empty
        assert chrom.injection_time is not None
        assert chrom.channel is not None

    def test_reject_non_chromeleon_txt_file(self):
        file_path = LOG_DATA_DIR / "Log_1.txt"
        with pytest.raises(ValueError):
            parse_chromatogram(file_path)


class TestChromparserBackend:
    """Parsing through chromparser keeps the earlier ChromStream conventions."""

    def test_chromeleon_txt_legacy_and_chromparser_metadata(self):
        chrom = parse_chromatogram_txt(TEST_DATA_DIR / "format_2.txt")

        assert list(chrom.data.columns) == ["Time (min)", "Value (pA)"]
        # raw Chromeleon header keys, without the byte order mark
        assert chrom.metadata["URL"] == "none"
        assert chrom.metadata["Channel"] == "FID_right"
        assert chrom.metadata["Signal Unit"] == "pA"
        assert chrom.metadata["Inject Time"] == pd.Timestamp("2023-07-17 15:35:22")
        assert chrom.metadata["time_unit"] == "min"
        # normalised chromparser keys
        assert chrom.metadata["data_vault"] == "ChromeleonLocal"
        assert chrom.metadata["source_format"] == "chromeleon-txt"
        # UTC offset dropped, local wall time kept
        assert chrom.injection_time == pd.Timestamp("2023-07-17 15:35:22")
        assert chrom.injection_time.tzinfo is None

    def test_parse_chromeleon_txt_tuple(self):
        metadata, df = parse_chromeleon_txt(TEST_DATA_DIR / "format_3.txt")

        assert metadata["Channel"] == "FID_Ch1"
        assert list(df.columns) == ["Time (min)", "Value (pA)"]
        assert len(df) == 42000

    @pytest.mark.parametrize(
        ("text", "metadata", "expected"),
        [
            ("7/17/2023 3:35:22 PM +02:00", {}, "2023-07-17 15:35:22"),
            ("17-1-2023 16:45:42 +01:00", {}, "2023-01-17 16:45:42"),
            ("1:43:35 PM", {"Injection Date": "12/19/2023"}, "2023-12-19 13:43:35"),
            ("10:30:07", {"Injection Date": "22-Aug-25"}, "2025-08-22 10:30:07"),
            ("5/6/2024 10:00:00", {}, "2024-06-05 10:00:00"),
            # AM/PM is the US locale: month first (confirmed by a .cmbx archive)
            ("9/7/2023 3:26:11 PM +02:00", {}, "2023-09-07 15:26:11"),
            ("12/19/2023 1:43:35 PM", {}, "2023-12-19 13:43:35"),
        ],
    )
    def test_parse_inject_time(self, text, metadata, expected):
        assert parse_inject_time(text, metadata) == pd.Timestamp(expected)

    def test_parse_inject_time_monthfirst(self):
        parsed = parse_inject_time("5/6/2024 10:00:00", {}, dayfirst=False)
        assert parsed == pd.Timestamp("2024-05-06 10:00:00")

    def test_parse_inject_time_needs_date(self):
        with pytest.raises(ValueError, match="Injection Date"):
            parse_inject_time("10:30:07", {})

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_chromatogram_txt(TEST_DATA_DIR / "missing.txt")
        with pytest.raises(FileNotFoundError):
            parse_chromatogram(TEST_DATA_DIR / "missing.ch")

    def test_agilent_legacy_metadata(self):
        chrom = parse_agilent_ch(TEST_DATA_DIR / "agilent.d" / "FID1A.ch")

        assert list(chrom.data.columns) == ["Time", "Signal"]
        assert chrom.metadata["version"] == "179"
        assert chrom.metadata["Sample Name"] == "Cuatro"
        assert chrom.metadata["Method Name"] == "GLYREACTION.M"
        assert chrom.metadata["date"] == "14-Jan-26, 11:49:50"
        assert chrom.signal_unit == "pA"
        assert chrom.injection_time == pd.Timestamp("2026-01-14 11:49:50")

    def test_agilent_ch_file_like(self):
        path = TEST_DATA_DIR / "agilent.d" / "FID1A.ch"
        with open(path, "rb") as f:
            from_stream = parse_agilent_ch(f, file_name="FID1A.ch")

        from_path = parse_agilent_ch(path)
        assert from_stream.channel == "FID1A"
        assert from_stream.path == "FID1A.ch"
        pd.testing.assert_frame_equal(from_stream.data, from_path.data)

    def test_agilent_dot_d(self):
        chroms = parse_agilent_dot_d(TEST_DATA_DIR / "agilent.d")

        assert [c.channel for c in chroms] == ["FID1A", "FID2B", "TCD3C"]
        assert [Path(c.path).name for c in chroms] == [
            "FID1A.ch",
            "FID2B.ch",
            "TCD3C.ch",
        ]
        assert chroms[2].signal_unit == "25 µV"

    def test_agilent_dx_channel_names(self):
        (chrom,) = parse_agilent_dx(TEST_DATA_DIR / "test_dx.dx")

        assert chrom.channel == "RID1A"
        assert chrom.signal_unit == "nRIU"
        assert chrom.metadata["detector"] == "RID"
        assert chrom.injection_time.tzinfo is None

    @pytest.mark.parametrize(
        ("name", "count"),
        [("format_1.txt", 1), ("agilent.d", 3), ("test_dx.dx", 1)],
    )
    def test_parse_chromatograms(self, name, count):
        chroms = parse_chromatograms(TEST_DATA_DIR / name)
        assert len(chroms) == count

    def test_parse_chromatograms_rejects_unknown(self):
        with pytest.raises(ValueError):  # noqa: PT011
            parse_chromatograms(LOG_DATA_DIR / "Log_1.txt")

    def test_parse_chromatograms_other_format(self, tmp_path):
        """Formats without a dedicated ChromStream parser go through chromparser."""
        import chromparser

        (injection,) = chromparser.read(TEST_DATA_DIR / "agilent.d" / "FID1A.ch")
        (cdf,) = chromparser.write_andi(injection, tmp_path)

        (chrom,) = parse_chromatograms(cdf)
        assert chrom.metadata["source_format"] == "andi-chrom"
        assert chrom.data.columns[0] == "Time (min)"
        assert chrom.signal_unit == "pA"
        assert chrom.injection_time == pd.Timestamp("2026-01-14 11:49:50")
        assert len(chrom.data) == 29200

        # single-chromatogram files also work with the single-file dispatcher
        assert parse_chromatogram(cdf).channel == chrom.channel

    def test_parse_chromatogram_rejects_folder(self):
        with pytest.raises(FileNotFoundError):
            parse_chromatogram(TEST_DATA_DIR / "agilent.d")

    def test_list_metadata_is_scalar_for_hdf5(self, tmp_path):
        import chromparser

        from chromstream.parsers import parse_chromatogram_hdf5
        from chromstream.parsers._chromparser import to_chromatograms

        (injection,) = chromparser.read(TEST_DATA_DIR / "format_1.txt")
        injection.metadata["warnings"] = ["first", "second"]
        injection.metadata["nested"] = {"a": 1}
        (chrom,) = to_chromatograms(injection, "chromeleon-txt", None)

        assert chrom.metadata["warnings"] == "first; second"
        assert chrom.metadata["nested"] == '{"a": 1}'
        chrom.to_hdf5(tmp_path / "chrom.h5")
        assert (
            parse_chromatogram_hdf5(tmp_path / "chrom.h5").metadata["warnings"]
            == "first; second"
        )
