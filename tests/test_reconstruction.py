
import pytest

from ctscan.reconstruction import calculate_initial_cor


def test_calculate_initial_cor():
    config = {
        "SrcToObject": 235.161392211914,
        "SrcToDetector": 1147.123,
        "DetectorPixelSizeX": 0.2,
        "DetectorPixelsX": 2048,
        "ObjectOffsetX": 0.505766076888359,
    }

    cor = calculate_initial_cor(config)

    assert cor == pytest.approx(1035.8357, abs=0.001)

def test_missing_geometry_parameter():
    config = {
        "SrcToObject": 235.161392211914,
        "SrcToDetector": 1147.123,
    }

    with pytest.raises(ValueError, match="Missing XTekCT parameters"):
        calculate_initial_cor(config)

def test_invalid_source_distance():
    config = {
        "SrcToObject": 0,
        "SrcToDetector": 1147.123,
        "DetectorPixelSizeX": 0.2,
        "DetectorPixelsX": 2048,
        "ObjectOffsetX": 0.505766076888359,
    }

    with pytest.raises(ValueError, match="must be positive"):
        calculate_initial_cor(config)


from types import SimpleNamespace

from ctscan.reconstruction import find_opposite_pairs


def test_find_opposite_pairs():
    angle_data = SimpleNamespace(
        projection_numbers=[1, 2, 3, 4],
        angles=[0.0, 90.0, 180.0, 270.0],
    )

    pairs = find_opposite_pairs(angle_data)

    assert len(pairs) == 2

    assert pairs[0]["projection_a"] == 1
    assert pairs[0]["projection_b"] == 3

    assert pairs[1]["projection_a"] == 2
    assert pairs[1]["projection_b"] == 4


def test_find_opposite_pairs_across_zero_degrees():
    angle_data = SimpleNamespace(
        projection_numbers=[1, 2],
        angles=[350.0, 170.0],
    )

    pairs = find_opposite_pairs(angle_data)

    assert len(pairs) == 1
    assert pairs[0]["angular_error"] == 0.0


def test_find_opposite_pairs_ignores_missing_opposite():
    angle_data = SimpleNamespace(
        projection_numbers=[1, 2, 3],
        angles=[0.0, 90.0, 180.0],
    )

    pairs = find_opposite_pairs(angle_data)

    assert len(pairs) == 1
    assert pairs[0]["projection_a"] == 1
    assert pairs[0]["projection_b"] == 3

