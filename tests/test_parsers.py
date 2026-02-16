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
    parse_agilent_dx,
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
