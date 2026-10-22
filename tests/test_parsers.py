from __future__ import annotations

import struct
import zipfile

import numpy as np
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
    parse_agilent_dx,
    parse_chromeleon_cmbx,
    parse_chromeleon_raw,
)
from chromstream.parsers.chromeleon_binary import _decode_svarints

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "testdata" / "chroms"
LOG_DATA_DIR = Path(__file__).parent / "testdata" / "logs"

# Chromeleon 7.3 archive (MS data and audit trail removed) and Chromeleon's text export of its signal
CMBX_FILE = TEST_DATA_DIR / "chromeleon.cmbx"
CMBX_TEXT_EXPORT = TEST_DATA_DIR / "chromeleon_cmbx_FID_left.txt"
CMBX_SIGNAL_RAW = "2_152612018.raw"
# header of a Chromeleon audit trail file (LZMA2-compressed XML, not a signal)
AUDIT_TRAIL_HEADER = b"CpAu" + struct.pack("<HHIII", 20, 2, 0, 0, 0)


def _read_cmbx_member(name: str) -> bytes:
    with zipfile.ZipFile(CMBX_FILE) as archive:
        return archive.read(name)


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


class TestChromeleonBinaryParsers:
    """Test Chromeleon .cmbx archive and raw signal file parsing"""

    def test_parse_cmbx_file(self):
        chrom_list = parse_chromeleon_cmbx(CMBX_FILE)

        assert len(chrom_list) == 1
        chrom = chrom_list[0]
        assert chrom.channel == "FID_left"
        assert list(chrom.data.columns) == ["Time (min)", "Value (pA)"]
        assert len(chrom.data) == 85500
        assert chrom.injection_time == pd.Timestamp("2023-09-07 15:26:11")
        assert chrom.time_unit == "min"
        assert chrom.signal_unit == "pA"
        assert chrom.path == CMBX_FILE
        assert chrom.metadata["Sequence"] == "20230709_SR_ZB_006_PyOil"
        assert chrom.metadata["Instrument Method"] == "Pyrolysis Oils"
        assert chrom.metadata["Detector"] == "GC_left"
        assert chrom.metadata["Generating Data System"] == "Chromeleon 7.3.0.60919"
        assert chrom.metadata["Average Step (s)"] == 0.04

    def test_cmbx_matches_chromeleon_text_export(self):
        """Decoded data must equal Chromeleon's own export of the same signal"""
        chrom = parse_chromeleon_cmbx(CMBX_FILE)[0]
        export = parse_chromatogram_txt(CMBX_TEXT_EXPORT)

        assert len(chrom.data) == int(export.metadata["Data Points"])
        # the export rounds time to 1e-6 min, the signal is stored with 1e-6 pA resolution
        np.testing.assert_allclose(
            chrom.data["Time (min)"], export.data["Time (min)"], rtol=0, atol=5e-7
        )
        np.testing.assert_allclose(
            chrom.data["Value (pA)"], export.data["Value (pA)"], rtol=0, atol=1e-9
        )

    def test_parse_raw_signal_file(self, tmp_path):
        raw_path = tmp_path / CMBX_SIGNAL_RAW
        raw_path.write_bytes(_read_cmbx_member(CMBX_SIGNAL_RAW))

        chrom = parse_chromeleon_raw(raw_path)

        assert chrom.channel == "FID_left"
        # raw files store neither the signal unit nor the instrument time zone
        assert list(chrom.data.columns) == ["Time (min)", "Value"]
        assert chrom.injection_time == pd.Timestamp("2023-09-07 13:26:11.9873556")
        cmbx_chrom = parse_chromeleon_cmbx(CMBX_FILE)[0]
        np.testing.assert_array_equal(
            chrom.data["Value"].to_numpy(), cmbx_chrom.data["Value (pA)"].to_numpy()
        )

    def test_parse_raw_signal_bytes_with_unit_and_timezone(self):
        chrom = parse_chromeleon_raw(
            _read_cmbx_member(CMBX_SIGNAL_RAW),
            file_name=CMBX_SIGNAL_RAW,
            channel_name="FID",
            signal_unit="pA",
            timezone="Europe/Amsterdam",
        )

        assert chrom.channel == "FID"
        assert chrom.path == CMBX_SIGNAL_RAW
        assert chrom.signal_unit == "pA"
        assert list(chrom.data.columns) == ["Time (min)", "Value (pA)"]
        assert chrom.injection_time == pd.Timestamp("2023-09-07 15:26:11.9873556")

    def test_reject_audit_trail_raw_file(self):
        with pytest.raises(ValueError, match="audit trail"):
            parse_chromeleon_raw(AUDIT_TRAIL_HEADER)

    def test_reject_unknown_raw_file(self):
        with pytest.raises(ValueError, match="Not a Chromeleon signal file"):
            parse_chromeleon_raw(b"not a chromeleon file")

    def test_reject_truncated_raw_file(self):
        data = _read_cmbx_member(CMBX_SIGNAL_RAW)
        with pytest.raises(ValueError, match="Invalid length"):
            parse_chromeleon_raw(data[:-100])

    def test_reject_zip_without_chromeleon_header(self):
        with pytest.raises(ValueError, match="header.xml"):
            parse_chromeleon_cmbx(TEST_DATA_DIR / "test_dx.dx")

    def test_missing_cmbx_file(self):
        with pytest.raises(FileNotFoundError):
            parse_chromeleon_cmbx(TEST_DATA_DIR / "does_not_exist.cmbx")

    def test_signed_varint_decoding(self):
        # first byte: [continue][sign][6 bits], then [continue][7 bits]
        values, terminators = _decode_svarints(
            bytes([0x05, 0x45, 0xBD, 0xAF, 0xCC, 0x0E, 0xC0, 0x01, 0x40])
        )

        assert values.tolist() == [5, -5, 15305725, -64, 0]
        assert terminators.tolist() == [False, False, False, False, True]


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

    def test_parse_chromeleon_cmbx_file(self):
        chrom = parse_chromatogram(CMBX_FILE)

        assert chrom.channel == "FID_left"
        assert not chrom.data.empty
        assert chrom.signal_unit == "pA"

    def test_parse_chromeleon_raw_file(self, tmp_path):
        raw_path = tmp_path / CMBX_SIGNAL_RAW
        raw_path.write_bytes(_read_cmbx_member(CMBX_SIGNAL_RAW))
        chrom = parse_chromatogram(raw_path)

        assert chrom.channel == "FID_left"
        assert not chrom.data.empty

    def test_reject_chromeleon_audit_trail_raw_file(self, tmp_path):
        raw_path = tmp_path / "audit.raw"
        raw_path.write_bytes(AUDIT_TRAIL_HEADER)
        with pytest.raises(ValueError, match="audit trail"):
            parse_chromatogram(raw_path)
