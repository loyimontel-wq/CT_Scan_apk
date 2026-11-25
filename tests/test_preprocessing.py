
import numpy as np
import pytest

from ctscan.preprocessing import log_transform


def test_log_transform_calculates_attenuation():
    projection = np.array(
        [[60000, 30000, 15000]],
        dtype=np.uint16,
    )

    attenuation, invalid_mask = log_transform(projection)

    expected = np.array(
        [[0.0, 0.6931472, 1.3862944]],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        attenuation,
        expected,
        rtol=1e-5,
    )

    assert not np.any(invalid_mask)


def test_log_transform_identifies_zero_pixels():
    projection = np.array(
        [[60000, 0, 30000]],
        dtype=np.uint16,
    )

    attenuation, invalid_mask = log_transform(projection)

    assert invalid_mask.tolist() == [[False, True, False]]
    assert attenuation[0, 1] == 0.0


def test_log_transform_returns_float32():
    projection = np.array(
        [[60000, 30000]],
        dtype=np.uint16,
    )

    attenuation, _ = log_transform(projection)

    assert attenuation.dtype == np.float32


def test_log_transform_accepts_custom_reference_intensity():
    projection = np.array([[30000]], dtype=np.uint16)

    attenuation, _ = log_transform(projection, i0=30000)

    assert attenuation[0, 0] == pytest.approx(0.0)


def test_log_transform_rejects_non_2d_input():
    projection = np.array([60000, 30000], dtype=np.uint16)

    with pytest.raises(ValueError):
        log_transform(projection)


@pytest.mark.parametrize("i0", [0, -1, np.inf, np.nan])
def test_log_transform_rejects_invalid_reference_intensity(i0):
    projection = np.array([[30000]], dtype=np.uint16)

    with pytest.raises(ValueError):
        log_transform(projection, i0=i0)


def test_log_transform_preserves_negative_attenuation():
    projection = np.array([[61000]], dtype=np.uint16)

    attenuation, _ = log_transform(projection)

    assert attenuation[0, 0] < 0
