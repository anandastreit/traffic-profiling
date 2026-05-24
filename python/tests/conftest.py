import pytest


@pytest.fixture
def align_setup(tmp_path, monkeypatch):
    """Patch RAW_DIR/DAILY_DIR and create the raw/down input directory."""
    import preprocess
    monkeypatch.setattr(preprocess, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(preprocess, "DAILY_DIR", tmp_path / "daily")
    raw_dir = tmp_path / "raw" / "down"
    raw_dir.mkdir(parents=True)
    out_dir = tmp_path / "daily" / "down" / "allSeries"
    return raw_dir, out_dir
