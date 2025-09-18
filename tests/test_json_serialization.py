from __future__ import annotations

import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
from chromstream.objects import Chromatogram, ChannelChromatograms, Experiment

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "testdata" / "chroms"


class TestJSONSerialization:
    """Test JSON serialization and deserialization functionality"""

    def _create_test_chromatogram(self) -> Chromatogram:
        """Create a test chromatogram with sample data"""
        time_data = np.linspace(0, 10, 50)
        rng = np.random.default_rng(42)  # Fixed seed for reproducible tests
        signal_data = np.sin(time_data) + 0.1 * rng.normal(size=50)

        data = pd.DataFrame({"Time (min)": time_data, "Signal (mV)": signal_data})

        metadata = {
            "Channel": "FID",
            "Signal Unit": "mV",
            "time_unit": "min",
            "Sample": "Test Sample",
        }

        return Chromatogram(
            data=data,
            injection_time=pd.Timestamp("2025-09-18 10:30:00"),
            metadata=metadata,
            channel="FID",
            path=Path("/test/path/test.txt"),
        )

    def test_chromatogram_serialization(self):
        """Test Chromatogram to_dict/from_dict round trip"""
        original = self._create_test_chromatogram()

        # Convert to dict and back
        data_dict = original.to_dict()
        reconstructed = Chromatogram.from_dict(data_dict)

        # Verify data integrity
        assert original.channel == reconstructed.channel
        assert original.injection_time == reconstructed.injection_time
        assert original.metadata == reconstructed.metadata
        assert str(original.path) == str(reconstructed.path)

        # Check DataFrame data
        pd.testing.assert_frame_equal(original.data, reconstructed.data)

    def test_channel_chromatograms_serialization(self):
        """Test ChannelChromatograms to_dict/from_dict round trip"""
        channel = ChannelChromatograms(channel="FID")

        # Add test chromatograms
        for i in range(3):
            chrom = self._create_test_chromatogram()
            chrom.injection_time = pd.Timestamp(f"2025-09-18 {10 + i}:30:00")
            channel.add_chromatogram(i, chrom)

        # Add test integrals
        integrals_data = pd.DataFrame(
            {"Peak1": [100.5, 105.2, 98.7], "Peak2": [45.3, 48.1, 44.9]}
        )
        channel.integrals = integrals_data

        # Convert to dict and back
        data_dict = channel.to_dict()
        reconstructed = ChannelChromatograms.from_dict(data_dict)

        # Verify channel data
        assert channel.channel == reconstructed.channel
        assert len(channel.chromatograms) == len(reconstructed.chromatograms)

        # Check chromatograms
        for key in channel.chromatograms:
            orig_chrom = channel.chromatograms[key]
            recon_chrom = reconstructed.chromatograms[key]
            assert orig_chrom.channel == recon_chrom.channel
            assert orig_chrom.injection_time == recon_chrom.injection_time
            pd.testing.assert_frame_equal(orig_chrom.data, recon_chrom.data)

        # Check integrals
        if channel.integrals is not None:
            pd.testing.assert_frame_equal(channel.integrals, reconstructed.integrals)

    def test_experiment_serialization(self):
        """Test Experiment to_dict/from_dict round trip"""
        experiment = Experiment(
            name="Test Experiment",
            experiment_starttime=pd.Timestamp("2025-09-18 10:00:00"),
            experiment_endtime=pd.Timestamp("2025-09-18 15:00:00"),
        )

        # Add test channels
        channel1 = ChannelChromatograms(channel="FID")
        channel1.add_chromatogram(0, self._create_test_chromatogram())
        experiment.add_channel("FID", channel1)

        # Add test log
        rng = np.random.default_rng(42)
        log_data = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2025-09-18 10:00:00", periods=5, freq="1h"),
                "Temperature": rng.normal(350, 5, 5),
                "Pressure": rng.normal(2.5, 0.1, 5),
            }
        )
        experiment.log = log_data

        # Convert to dict and back
        data_dict = experiment.to_dict()
        reconstructed = Experiment.from_dict(data_dict)

        # Verify experiment data
        assert experiment.name == reconstructed.name
        assert experiment.experiment_starttime == reconstructed.experiment_starttime
        assert experiment.experiment_endtime == reconstructed.experiment_endtime
        assert len(experiment.channels) == len(reconstructed.channels)

        # Check log data
        if experiment.log is not None:
            pd.testing.assert_frame_equal(experiment.log, reconstructed.log)

    def test_json_file_io(self):
        """Test JSON file save/load functionality"""
        experiment = Experiment(name="File Test Experiment")
        experiment.add_channel("FID", ChannelChromatograms(channel="FID"))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Save to JSON file
            experiment.to_json(temp_path)

            # Load from JSON file
            reconstructed = Experiment.from_json(temp_path)

            # Verify basic properties
            assert experiment.name == reconstructed.name
            assert len(experiment.channels) == len(reconstructed.channels)

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_json_dict_structure(self):
        """Test that the JSON dict has expected structure"""
        experiment = Experiment(name="Structure Test")
        data_dict = experiment.to_dict()

        # Verify required keys
        required_keys = [
            "name",
            "channels",
            "experiment_starttime",
            "experiment_endtime",
            "log",
        ]
        for key in required_keys:
            assert key in data_dict

        assert isinstance(data_dict["channels"], dict)
