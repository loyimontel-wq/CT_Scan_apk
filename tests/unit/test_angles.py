import numpy as np
import pytest
from ctscan.io.angles import read_angles


def test_read_ang_format(tmp_path):

    angle_file = tmp_path / "sample.ang"

    angle_file.write_text(
        "Proj    Angle(deg)\n"
        "1:      -000.010\n"
        "2:      -000.201\n"
        "3:      -000.351\n"
        "4:      -000.515\n"
    )

    result = read_angles(angle_file)

    np.testing.assert_array_equal(
        result.projection_numbers,
        [1, 2, 3, 4]
    )

    np.testing.assert_allclose(
        result.angles,
        [-0.010, -0.201, -0.351, -0.515]
    )

    assert result.missing_numbers == ()
    assert result.duplicate_numbers == ()
    assert result.malformed_rows == ()

def test_read_txt_format(tmp_path):

    angle_file = tmp_path / "sample.txt"

    angle_file.write_text(
        "Projections    Frames    Exposure(ms)\n"
        "3000           1         4000.00\n"
        "Projection     Angle(deg) Time(s)\n"
        "1              000.116    024.260\n"
        "2              000.236    028.260\n"
        "3              000.356    032.261\n"
    )

    result = read_angles(angle_file)

    np.testing.assert_array_equal(
        result.projection_numbers,
        [1, 2, 3]
    )

    np.testing.assert_allclose(
        result.angles,
        [0.116, 0.236, 0.356]
    )

def test_detect_missing_projection_numbers(tmp_path):

    angle_file = tmp_path / "missing.ang"

    angle_file.write_text(
        "Proj Angle(deg)\n"
        "1 -0.010\n"
        "2 -0.201\n"
        "4 -0.515\n"
    )

    result = read_angles(angle_file)

    assert result.missing_numbers == (3,)


def test_detect_duplicate_projection_numbers(tmp_path):

    angle_file = tmp_path / "duplicate.ang"

    angle_file.write_text(
        "Proj Angle(deg)\n"
        "1 -0.010\n"
        "2 -0.201\n"
        "2 -0.351\n"
        "3 -0.515\n"
    )

    result = read_angles(angle_file)

    assert result.duplicate_numbers == (2,)
    assert len(result.angles) == 4




def test_missing_header_raises_error(tmp_path):

    angle_file = tmp_path / "invalid.ang"

    angle_file.write_text(
        "This is not an angle file\n"
        "1 0.010\n"
        "2 0.020\n"
    )

    with pytest.raises(ValueError):
        read_angles(angle_file)

def test_detect_malformed_angle_rows(tmp_path):

    angle_file = tmp_path / "malformed.ang"

    angle_file.write_text(
        "Proj Angle(deg)\n"
        "1 -0.010\n"
        "2 invalid\n"
        "3 -0.351\n"
    )

    result = read_angles(angle_file)

    np.testing.assert_array_equal(
        result.projection_numbers,
        [1, 3]
    )

    assert result.malformed_rows == (3,)

def test_projection_numbers_can_start_above_one(tmp_path):

    angle_file = tmp_path / "offset.ang"

    angle_file.write_text(
        "Proj Angle(deg)\n"
        "5 -0.010\n"
        "6 -0.201\n"
        "7 -0.351\n"
    )

    result = read_angles(angle_file)

    np.testing.assert_array_equal(
        result.projection_numbers,
        [5, 6, 7]
    )

    assert result.missing_numbers == ()

def test_preserve_projection_order(tmp_path):

    angle_file = tmp_path / "out_of_order.ang"

    angle_file.write_text(
        "Proj Angle(deg)\n"
        "1 -0.010\n"
        "3 -0.351\n"
        "2 -0.201\n"
    )

    result = read_angles(angle_file)

    np.testing.assert_array_equal(
        result.projection_numbers,
        [1, 3, 2]
    )

    np.testing.assert_allclose(
        result.angles,
        [-0.010, -0.351, -0.201]
    )