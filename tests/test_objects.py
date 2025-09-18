from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path
from chromstream.objects import ChannelChromatograms, Experiment
from chromstream.parsers import parse_chromatogram_txt

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "testdata" / "chroms"


class TestPlotting:
    """Minimal tests for plotting functionality"""

    def test_chromatogram_plot(self):
        """Test that chromatogram plotting works"""
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        ax = chrom.plot()
        assert ax is not None
        plt.close("all")

    def test_channel_plot(self):
        """Test that channel plotting works"""
        channel = ChannelChromatograms(channel="FID_test")

        # Add chromatograms to the channel
        for i, filename in enumerate(["format_1.txt", "format_2.txt"]):
            file_path = TEST_DATA_DIR / filename
            chrom = parse_chromatogram_txt(file_path)
            channel.add_chromatogram(i, chrom)

        ax = channel.plot()
        assert ax is not None
        plt.close("all")

    def test_experiment_plot(self):
        """Test that experiment plotting works"""
        experiment = Experiment(name="Test")

        # Add chromatograms with different channels
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom1 = parse_chromatogram_txt(file_path)
        chrom1.channel = "FID_left"

        chrom2 = parse_chromatogram_txt(file_path)
        chrom2.channel = "FID_right"

        experiment.add_chromatogram(chrom1)
        experiment.add_chromatogram(chrom2)

        experiment.plot_chromatograms()
        plt.close("all")

    def test_empty_channel_plot_error(self):
        """Test empty channel plotting"""
        empty_channel = ChannelChromatograms(channel="Empty")
        ax = empty_channel.plot()
        assert ax is not None
        plt.close("all")

    def test_empty_experiment_plot_error(self):
        """Test experiment plotting"""
        empty_experiment = Experiment(name="Empty")
        empty_experiment.plot_chromatograms()
        plt.close("all")


class TestChromatogramBasics:
    """Minimal tests for Chromatogram object features"""

    def test_chromatogram_properties(self):
        """Test basic chromatogram properties work"""
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        # Basic properties should be accessible
        assert chrom.data is not None
        assert chrom.injection_time is not None
        assert chrom.metadata is not None
        assert chrom.channel is not None
        assert chrom.path is not None

        # Property methods should work
        time_unit = chrom.time_unit
        signal_unit = chrom.signal_unit
        assert isinstance(time_unit, str)
        assert isinstance(signal_unit, str)

    def test_chromatogram_apply_baseline(self):
        """Test baseline correction works"""
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        def simple_baseline(data):
            return data[data.columns[1]] - data[data.columns[1]].min()

        # Test baseline correction
        original_columns = len(chrom.data.columns)
        chrom.apply_baseline(simple_baseline)

        # Should add a new column by default
        assert len(chrom.data.columns) == original_columns + 1

    def test_chromatogram_integrate_peaks(self):
        """Test peak integration works"""
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        # Simple peak list
        peaks = {"peak1": [0.5, 1.5]}

        result = chrom.integrate_peaks(peaks)
        assert isinstance(result, dict)
        assert "peak1" in result


class TestChannelBasics:
    """Minimal tests for ChannelChromatograms object features"""

    def test_channel_creation(self):
        """Test channel creation and basic operations"""
        channel = ChannelChromatograms(channel="FID_test")

        assert channel.channel == "FID_test"
        assert len(channel.chromatograms) == 0
        assert channel.integrals is None

    def test_channel_add_chromatogram(self):
        """Test adding chromatograms to channel"""
        channel = ChannelChromatograms(channel="FID_test")
        file_path = TEST_DATA_DIR / "format_1.txt"
        chrom = parse_chromatogram_txt(file_path)

        channel.add_chromatogram(0, chrom)

        assert len(channel.chromatograms) == 1
        assert 0 in channel.chromatograms

    def test_channel_apply_baseline(self):
        """Test baseline correction on all chromatograms"""
        channel = ChannelChromatograms(channel="FID_test")

        # Add chromatograms
        for i, filename in enumerate(["format_1.txt", "format_2.txt"]):
            file_path = TEST_DATA_DIR / filename
            chrom = parse_chromatogram_txt(file_path)
            channel.add_chromatogram(i, chrom)

        def simple_baseline(data):
            return data[data.columns[1]] - data[data.columns[1]].min()

        channel.apply_baseline(simple_baseline)

        # Check that baseline was applied to all chromatograms
        for chrom in channel.chromatograms.values():
            assert "_BLcorr" in chrom.data.columns[-1]

    def test_channel_integrate_peaks(self):
        """Test peak integration for channel"""
        channel = ChannelChromatograms(channel="FID_test")

        # Add chromatograms
        for i, filename in enumerate(["format_1.txt", "format_2.txt"]):
            file_path = TEST_DATA_DIR / filename
            chrom = parse_chromatogram_txt(file_path)
            channel.add_chromatogram(i, chrom)

        peaks = {"peak1": [0.5, 1.5]}

        result = channel.integrate_peaks(peaks)
        assert result is not None
        assert len(result) == len(channel.chromatograms)


class TestExperimentBasics:
    """Minimal tests for Experiment object features"""

    def test_experiment_creation(self):
        """Test experiment creation"""
        exp = Experiment(name="Test Experiment")

        assert exp.name == "Test Experiment"
        assert len(exp.channels) == 0
        assert exp.experiment_starttime is None
        assert exp.experiment_endtime is None
        assert exp.log is None

    def test_experiment_add_chromatogram(self):
        """Test adding chromatograms to experiment"""
        exp = Experiment(name="Test")
        file_path = TEST_DATA_DIR / "format_1.txt"

        # Add chromatogram directly
        exp.add_chromatogram(file_path)

        assert len(exp.channels) == 1
        assert len(exp.channel_names) == 1

    def test_experiment_add_channel(self):
        """Test adding channels to experiment"""
        exp = Experiment(name="Test")
        channel = ChannelChromatograms(channel="FID_test")

        exp.add_channel("FID_test", channel)

        assert "FID_test" in exp.channels
        assert "FID_test" in exp.channel_names

    def test_experiment_channel_names(self):
        """Test channel_names property"""
        exp = Experiment(name="Test")

        # Add different channels
        for channel_name in ["FID_left", "FID_right"]:
            file_path = TEST_DATA_DIR / "format_1.txt"
            chrom = parse_chromatogram_txt(file_path)
            chrom.channel = channel_name
            exp.add_chromatogram(chrom)

        channel_names = exp.channel_names
        assert len(channel_names) == 2
        assert "FID_left" in channel_names
        assert "FID_right" in channel_names
