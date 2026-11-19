from pathlib import Path
from numpy import add
import pytest

from ctscan.io.xtekct import parse_xtekct

def test_parse_xtekct_read_values(tmp_path: Path):
    test_file = tmp_path / "test_scan.xtekct"
    test_file.write_text(
        """
        SOD=69.0
        SDD=1147.123
        Projections=500
        Scanner=Nikon
        """,
        encoding="utf-8",
    )

    result = parse_xtekct(test_file)

    assert result["SOD"] == 69.0
    assert isinstance(result["SOD"], float)
    assert result["SDD"] == 1147.123
    assert result["Projections"] == 500
    assert isinstance(result["Projections"], int)
    assert result["Scanner"] == "Nikon"


def test_parse_xtekct_missing_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.xtekct"

    with pytest.raises(FileNotFoundError):
        parse_xtekct(missing_file)

def test_parse_xtekct_duplicate_keys(tmp_path: Path):
    test_file = tmp_path / "duplicate_scan.xtekct"

    test_file.write_text(
        """
        SOD=69.0
        SDD=1147.123
        SOD=75.0
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate configuration key"):
        parse_xtekct(test_file)

def test_parse_xtekct_empty_file(tmp_path: Path):
    test_file = tmp_path / "empty_scan.xtekct"
    test_file.write_text("",encoding="utf-8")
    with pytest.raises(ValueError, match="No configuration entries found"):
        parse_xtekct(test_file)