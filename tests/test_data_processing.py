from __future__ import annotations

from chromstream.data_processing import list_baseline_functions


def test_list_baseline_functions_non_verbose():
    output = list_baseline_functions()
    lines = output.splitlines()

    assert lines == [
        "min_subtract",
        "time_window_baseline",
        "time_point_baseline",
        "linear_baseline",
    ]


def test_list_baseline_functions_verbose_includes_docstrings():
    output = list_baseline_functions(verbose=True)

    assert "min_subtract" in output
    assert "Simple minimum subtraction baseline correction" in output

    assert "time_window_baseline" in output
    assert "Use mean of signal in a specific time window as baseline" in output

    assert "time_point_baseline" in output
    assert "Use signal value at a specific time point as baseline" in output

    assert "linear_baseline" in output
    assert "Determines a linear baseline between the signal values" in output
