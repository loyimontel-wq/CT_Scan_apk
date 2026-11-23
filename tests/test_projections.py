import numpy as np
from pathlib import Path

from ctscan.io.angles import AngleData
from ctscan.io.projections import (ProjectionError,ProjectionSet,ScanReport)
from ctscan.io.projections import (match_projections,load_projection,iter_projections,validate_scan)
import pytest, tifffile

from ctscan.io.projections import find_projection_files

def test_find_projection_files(tmp_path):
    (tmp_path / "scan_0001.tif").touch()
    (tmp_path / "scan_0002.tif").touch()
    (tmp_path / "scan_0003.tiff").touch()

    files = find_projection_files(tmp_path)

    assert len(files) == 3
    assert all(path.is_file() for path in files)

def test_find_projection_files_case_insensitive(tmp_path):
    (tmp_path / "scan_0001.TIF").touch()
    (tmp_path / "scan_0002.TIFF").touch()

    files = find_projection_files(tmp_path)

    assert len(files) == 2

def test_find_projection_files_ignores_subdirectories(tmp_path):
    (tmp_path / "scan_0001.tif").touch()

    subfolder = tmp_path / "reconstruction"
    subfolder.mkdir()
    (subfolder / "volume.tif").touch()

    files = find_projection_files(tmp_path)

    assert len(files) == 1
    assert files[0].name == "scan_0001.tif"

def test_find_projection_files_ignores_other_extensions(tmp_path):
    (tmp_path / "scan_0001.tif").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "image.png").touch()

    files = find_projection_files(tmp_path)

    assert len(files) == 1
    assert files[0].suffix == ".tif"

def test_find_projection_files_missing_folder(tmp_path):
    folder = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        find_projection_files(folder)

def test_find_projection_files_path_is_file(tmp_path):
    file = tmp_path / "not_a_folder.txt"
    file.touch()

    with pytest.raises(NotADirectoryError):
        find_projection_files(file)

def test_find_projection_files_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_projection_files(tmp_path)

def make_angle_data(numbers, angles):
    return AngleData(
        projection_numbers=np.array(numbers),
        angles=np.array(angles),
        missing_numbers=(),
        duplicate_numbers=(),
        malformed_rows=(),
    )

def test_match_projections_normal(tmp_path):
    files = []

    for number in [1, 2, 3]:
        file = tmp_path / f"scan_{number:04d}.tif"
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2, 3],
        [10.0, 20.0, 30.0]
    )

    result = match_projections(files, angle_data)

    assert len(result.files) == 3
    assert result.projection_numbers.tolist() == [1, 2, 3]
    assert result.angles.tolist() == [10.0, 20.0, 30.0]

def test_match_projections_missing_tiff(tmp_path):
    files = []

    for number in [1, 3]:
        file = tmp_path / f"scan_{number:04d}.tif"
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2, 3],
        [10.0, 20.0, 30.0]
    )

    result = match_projections(files, angle_data)

    assert result.missing_tiff_numbers == (2,)

def test_match_projections_unmatched_tiff(tmp_path):
    files = []

    for number in [1, 2, 4]:
        file = tmp_path / f"scan_{number:04d}.tif"
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2, 3],
        [10.0, 20.0, 30.0]
    )

    result = match_projections(files, angle_data)

    assert result.unmatched_tiff_numbers == (4,)

def test_match_projections_duplicate_tiff(tmp_path):
    files = []

    for name in ["scan_0001.tif", "scan_0002.tif", "other_0002.tif"]:
        file = tmp_path / name
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2],
        [10.0, 20.0]
    )

    result = match_projections(files, angle_data)

    assert result.duplicate_tiff_numbers == (2,)
    assert result.projection_numbers.tolist() == [1]

def test_match_projections_duplicate_angle(tmp_path):
    files = []

    for number in [1, 2, 3]:
        file = tmp_path / f"scan_{number:04d}.tif"
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2, 2, 3],
        [10.0, 20.0, 21.0, 30.0]
    )

    result = match_projections(files, angle_data)

    assert result.duplicate_angle_numbers == (2,)
    assert result.projection_numbers.tolist() == [1, 3]

def test_match_projections_preserves_angle_association(tmp_path):
    files = []

    for number in [3, 1, 2]:
        file = tmp_path / f"scan_{number:04d}.tif"
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [3, 1, 2],
        [30.0, 10.0, 20.0]
    )

    result = match_projections(files, angle_data)

    assert result.projection_numbers.tolist() == [1, 2, 3]
    assert result.angles.tolist() == [10.0, 20.0, 30.0]

def test_match_projections_rejects_mismatched_angle_data(tmp_path):
    file = tmp_path / "scan_0001.tif"
    file.touch()

    angle_data = make_angle_data(
        [1, 2],
        [10.0]
    )

    with pytest.raises(ValueError, match="same length"):
        match_projections([file], angle_data)

def test_match_projections_rejects_empty_files():
    angle_data = make_angle_data([1], [10.0])

    with pytest.raises(ValueError, match="No projection files"):
        match_projections([], angle_data)

def test_match_projections_rejects_empty_angle_data(tmp_path):

    file = tmp_path / "scan_0001.tif"
    file.touch()

    angle_data = make_angle_data([], [])

    with pytest.raises(ValueError, match="No angle records"):
        match_projections([file], angle_data)

def test_match_projections_rejects_empty_files():
    angle_data = make_angle_data([1], [10.0])

    with pytest.raises(ValueError, match="No projection files"):
        match_projections([], angle_data)

def test_match_projections_rejects_empty_angle_data(tmp_path):

    file = tmp_path / "scan_0001.tif"
    file.touch()

    angle_data = make_angle_data([], [])

    with pytest.raises(ValueError, match="No angle records"):
        match_projections([file], angle_data)

def test_match_projections_continues_with_invalid_filenames(tmp_path):
    valid_file_1 = tmp_path / "scan_0001.tif"
    invalid_file = tmp_path / "calibration.tif"
    valid_file_2 = tmp_path / "scan_0002.tif"

    valid_file_1.touch()
    invalid_file.touch()
    valid_file_2.touch()

    angle_data = make_angle_data(
        [1, 2],
        [10.0, 20.0]
    )

    result = match_projections(
        [valid_file_1, invalid_file, valid_file_2],
        angle_data
    )

    assert result.projection_numbers.tolist() == [1, 2]
    assert result.angles.tolist() == [10.0, 20.0]
    assert result.invalid_files == (invalid_file,)

def test_match_projections_all_filenames_invalid(tmp_path):
    files = []

    for name in ["calibration.tif", "background.tif"]:
        file = tmp_path / name
        file.touch()
        files.append(file)

    angle_data = make_angle_data(
        [1, 2],
        [10.0, 20.0]
    )

    result = match_projections(files, angle_data)

    assert result.files == ()
    assert result.projection_numbers.size == 0
    assert result.angles.size == 0
    assert result.invalid_files == tuple(files)

def test_load_projection(tmp_path):
    image = np.full((2048, 2048), 1200, dtype=np.uint16)

    file = tmp_path / "scan_0001.tif"
    tifffile.imwrite(file, image)

    result = load_projection(file)

    assert result.shape == (2048, 2048)
    assert result.dtype == np.uint16
    assert np.array_equal(result, image)

def test_load_projection_checks_shape(tmp_path):
    image = np.zeros((512, 512), dtype=np.uint16)

    file = tmp_path / "scan_0001.tif"
    tifffile.imwrite(file, image)

    with pytest.raises(ValueError, match="Expected image shape"):
        load_projection(file, expected_shape=(2048, 2048))

def test_load_projection_checks_dtype(tmp_path):
    image = np.zeros((2048, 2048), dtype=np.uint8)

    file = tmp_path / "scan_0001.tif"
    tifffile.imwrite(file, image)

    with pytest.raises(ValueError, match="Expected image dtype"):
        load_projection(file, expected_dtype=np.dtype("uint16"))

def test_load_projection_rejects_non_2d_image(tmp_path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    file = tmp_path / "scan_0001.tif"
    tifffile.imwrite(file, image)

    with pytest.raises(ValueError, match="Expected a 2D projection"):
        load_projection(file)

def test_load_projection_missing_file(tmp_path):
    file = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError):
        load_projection(file)

def test_iter_projections(tmp_path):
    from ctscan.io.projections import iter_projections, ProjectionSet

    files = []

    for number in range(1, 4):
        file = tmp_path / f"scan_{number:04d}.tif"
        image = np.full((10, 10), number, dtype=np.uint16)
        tifffile.imwrite(file, image)
        files.append(file)

    projection_set = ProjectionSet(
        files=tuple(files),
        projection_numbers=np.array([1, 2, 3]),
        angles=np.array([0.1, 0.2, 0.3]),
        missing_tiff_numbers=(),
        unmatched_tiff_numbers=(),
        duplicate_tiff_numbers=(),
        duplicate_angle_numbers=(),
        invalid_files=(),
    )

    results = list(iter_projections(projection_set))

    assert len(results) == 3

    number, angle, image = results[0]

    assert number == 1
    assert angle == 0.1
    assert image.shape == (10, 10)
    assert image.dtype == np.uint16

def test_iter_projections_rejects_inconsistent_lengths():
    from ctscan.io.projections import iter_projections, ProjectionSet

    projection_set = ProjectionSet(
        files=(Path("scan_0001.tif"), Path("scan_0002.tif")),
        projection_numbers=np.array([1]),
        angles=np.array([0.1, 0.2]),
        missing_tiff_numbers=(),
        unmatched_tiff_numbers=(),
        duplicate_tiff_numbers=(),
        duplicate_angle_numbers=(),
        invalid_files=(),
    )

    with pytest.raises(ValueError, match="same length"):
        list(iter_projections(projection_set))

def test_iter_projections_rejects_empty_set():
    from ctscan.io.projections import iter_projections, ProjectionSet

    projection_set = ProjectionSet(
        files=(),
        projection_numbers=np.array([], dtype=np.int64),
        angles=np.array([], dtype=np.float64),
        missing_tiff_numbers=(),
        unmatched_tiff_numbers=(),
        duplicate_tiff_numbers=(),
        duplicate_angle_numbers=(),
        invalid_files=(),
    )

    with pytest.raises(ValueError, match="No matched projections"):
        list(iter_projections(projection_set))

def test_projection_error():
    error = ProjectionError(
        projection_number=1628,
        file=Path("scan_1628.tif"),
        reason="Incorrect image dimensions",
    )

    assert error.projection_number == 1628
    assert error.file == Path("scan_1628.tif")
    assert error.reason == "Incorrect image dimensions"

def test_scan_report():
    angle_data = AngleData(
        projection_numbers=np.array([1, 2]),
        angles=np.array([0.1, 0.2]),
        missing_numbers=(),
        duplicate_numbers=(),
        malformed_rows=(),
    )

    projection_set = ProjectionSet(
        files=(Path("scan_0001.tif"), Path("scan_0002.tif")),
        projection_numbers=np.array([1, 2]),
        angles=np.array([0.1, 0.2]),
        missing_tiff_numbers=(),
        unmatched_tiff_numbers=(),
        duplicate_tiff_numbers=(),
        duplicate_angle_numbers=(),
        invalid_files=(),
    )

    report = ScanReport(
        scan_directory=Path("test_scan"),
        total_tiff_files=2,
        angle_data=angle_data,
        projection_set=projection_set,
        successfully_loaded=2,
        failed_projections=(),
        image_shape=(2048, 2048),
        image_dtype=np.dtype("uint16"),
        is_valid=True,
    )

    assert report.total_tiff_files == 2
    assert report.successfully_loaded == 2
    assert report.failed_projections == ()
    assert report.image_shape == (2048, 2048)
    assert report.image_dtype == np.dtype("uint16")
    assert report.is_valid is True

    assert report.angle_data is angle_data
    assert report.projection_set is projection_set

def test_validate_scan_rejects_missing_directory(tmp_path):
    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n1:\t0.100\n"
    )

    with pytest.raises(FileNotFoundError, match="Scan directory"):
        validate_scan(tmp_path / "missing", angle_file)

def test_validate_scan_rejects_file_as_directory(tmp_path):
    scan_file = tmp_path / "scan"
    scan_file.write_text("not a directory")

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n1:\t0.100\n"
    )

    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_scan(scan_file, angle_file)

def test_validate_scan_rejects_missing_angle_file(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    with pytest.raises(FileNotFoundError, match="Angle file"):
        validate_scan(scan_directory, tmp_path / "missing.ang")

def test_validate_scan_success(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    # Create three TIFF projection images
    for number in range(1, 4):
        image = np.full((64, 64), number, dtype=np.uint16)

        tifffile.imwrite(
            scan_directory / f"scan_{number:04d}.tif",
            image,
        )

    # Create the corresponding angle file
    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "3:\t0.300\n"
    )

    report = validate_scan(scan_directory, angle_file)

    assert report.total_tiff_files == 3
    assert report.successfully_loaded == 3

    assert report.failed_projections == ()

    assert report.image_shape == (64, 64)
    assert report.image_dtype == np.dtype("uint16")

    assert report.is_valid is True

def test_validate_scan_strict_stops_at_first_failed_projection(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    # Projection 1: valid TIFF
    image = np.full((64, 64), 1, dtype=np.uint16)
    tifffile.imwrite(scan_directory / "scan_0001.tif", image)

    # Projection 2: deliberately invalid TIFF
    (scan_directory / "scan_0002.tif").write_text("not a valid TIFF")

    # Projection 3: valid TIFF, but should never be reached
    image = np.full((64, 64), 3, dtype=np.uint16)
    tifffile.imwrite(scan_directory / "scan_0003.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "3:\t0.300\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=True,
    )

    assert report.successfully_loaded == 1
    assert len(report.failed_projections) == 1

    assert report.failed_projections[0].projection_number == 2
    assert report.failed_projections[0].file.name == "scan_0002.tif"

    assert report.is_valid is False

def test_validate_scan_tolerant_continues_after_failed_projection(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    # Projection 1: valid TIFF
    image = np.full((64, 64), 1, dtype=np.uint16)
    tifffile.imwrite(scan_directory / "scan_0001.tif", image)

    # Projection 2: deliberately invalid TIFF
    (scan_directory / "scan_0002.tif").write_text("not a valid TIFF")

    # Projection 3: valid TIFF
    image = np.full((64, 64), 3, dtype=np.uint16)
    tifffile.imwrite(scan_directory / "scan_0003.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "3:\t0.300\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=False,
    )

    assert report.successfully_loaded == 2
    assert len(report.failed_projections) == 1

    assert report.failed_projections[0].projection_number == 2

    assert report.is_valid is False

def test_validate_scan_strict_rejects_duplicate_tiff_numbers(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    # Two TIFF files have the same projection number.
    image = np.full((64, 64), 1, dtype=np.uint16)
    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)
    tifffile.imwrite(scan_directory / "duplicate_0002.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=True,
    )

    assert report.successfully_loaded == 0
    assert report.is_valid is False
    assert report.projection_set.duplicate_tiff_numbers == (2,)

def test_validate_scan_tolerant_handles_duplicate_tiff_numbers(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)
    tifffile.imwrite(scan_directory / "duplicate_0002.tif", image)
    tifffile.imwrite(scan_directory / "scan_0003.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "3:\t0.300\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=False,
    )

    assert report.successfully_loaded == 2
    assert report.projection_set.duplicate_tiff_numbers == (2,)
    assert report.is_valid is False

def test_validate_scan_strict_rejects_duplicate_angle_numbers(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "2:\t0.300\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=True,
    )

    assert report.successfully_loaded == 0
    assert report.is_valid is False
    assert report.projection_set.duplicate_angle_numbers == (2,)

def test_validate_scan_tolerant_handles_duplicate_angle_numbers(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)
    tifffile.imwrite(scan_directory / "scan_0003.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "2:\t0.300\n"
        "3:\t0.400\n"
    )

    report = validate_scan(
        scan_directory,
        angle_file,
        strict=False,
    )

    assert report.successfully_loaded == 2
    assert report.projection_set.duplicate_angle_numbers == (2,)
    assert report.is_valid is False

def test_validate_scan_no_matching_projections(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "10:\t0.100\n"
        "11:\t0.200\n"
    )

    report = validate_scan(scan_directory, angle_file)

    assert len(report.projection_set.files) == 0
    assert report.successfully_loaded == 0
    assert report.image_shape is None
    assert report.image_dtype is None
    assert report.is_valid is False

def test_validate_scan_reports_invalid_filenames(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0002.tif", image)
    tifffile.imwrite(scan_directory / "projection_unknown.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
    )

    report = validate_scan(scan_directory, angle_file)

    assert report.successfully_loaded == 2
    assert len(report.projection_set.invalid_files) == 1
    assert report.projection_set.invalid_files[0].name == "projection_unknown.tif"
    assert report.is_valid is True

def test_validate_scan_allows_missing_tiff_projection(tmp_path):
    scan_directory = tmp_path / "scan"
    scan_directory.mkdir()

    image = np.full((64, 64), 1, dtype=np.uint16)

    tifffile.imwrite(scan_directory / "scan_0001.tif", image)
    tifffile.imwrite(scan_directory / "scan_0003.tif", image)

    angle_file = tmp_path / "scan.ang"
    angle_file.write_text(
        "Proj\tAngle(deg)\n"
        "1:\t0.100\n"
        "2:\t0.200\n"
        "3:\t0.300\n"
    )

    report = validate_scan(scan_directory, angle_file)

    assert report.successfully_loaded == 2
    assert report.projection_set.missing_tiff_numbers == (2,)
    assert report.is_valid is True























































