"""Smoke tests for package imports."""


def test_analysis_cleaning_imports():
    from analysis.cleaning import compute_item_stability, filter_by_confidence, flag_valid_items
    assert callable(compute_item_stability)
    assert callable(filter_by_confidence)
    assert callable(flag_valid_items)


def test_analysis_label_io_imports():
    from analysis.label_io import AXES, load_all_runs
    assert isinstance(AXES, list)
    assert callable(load_all_runs)
