from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from ctscan.io.angles import AngleData, read_angles
import glob, os, re, tifffile
import numpy as np


@dataclass
class ProjectionSet:
    files: tuple[Path, ...]
    projection_numbers: np.ndarray
    angles: np.ndarray
    missing_tiff_numbers: tuple[int, ...]
    unmatched_tiff_numbers: tuple[int, ...]
    duplicate_tiff_numbers: tuple[int, ...]
    duplicate_angle_numbers: tuple[int, ...]
    invalid_files: tuple[Path, ...]

@dataclass
class ProjectionError:
    projection_number: int
    file: Path
    reason: str

@dataclass
class ScanReport:
    scan_directory: Path
    total_tiff_files: int
    angle_data: AngleData
    projection_set: ProjectionSet
    successfully_loaded: int
    failed_projections: tuple[ProjectionError, ...]
    image_shape: tuple[int, int] | None
    image_dtype: np.dtype | None
    is_valid: bool


def match_projections(
    files: list[str | Path],
    angle_data: AngleData,
) -> ProjectionSet:

    if not files:
        raise ValueError("No projection files were provided.")

    if len(angle_data.projection_numbers) == 0:
        raise ValueError("No angle records were provided.")

    if len(angle_data.projection_numbers) != len(angle_data.angles):
        raise ValueError(
            "Projection numbers and angles must have the same length."
        )

    pattern = re.compile(r"_(\d+)\.(?:tif|tiff)$", re.IGNORECASE)

    file_by_number = {}
    tiff_numbers = []
    invalid_files = []

    for file in files:
        path = Path(file)
        match = pattern.search(path.name)

        if match is None:
            invalid_files.append(path)
            continue
        number = int(match.group(1))
        tiff_numbers.append(number)

        if number not in file_by_number:
            file_by_number[number] =  path

    tiff_counts = Counter(tiff_numbers)
    angle_counts = Counter(map(int,angle_data.projection_numbers))
    duplicate_tiff = tuple(sorted(number for number, count in tiff_counts.items() if count > 1))
    duplicate_angle = tuple(sorted(number for number, count in angle_counts.items() if count > 1))

    tiff_ids = set(tiff_counts)
    angle_ids = set(angle_counts)

    missing_tiff = tuple(sorted(angle_ids - tiff_ids))
    unmatched_tiff = tuple(sorted(tiff_ids - angle_ids))

    angle_by_number = {int(number): angle for number, angle in zip(angle_data.projection_numbers,angle_data.angles) if angle_counts[int(number)]==1}
    valid_numbers = sorted((tiff_ids & angle_ids) - set(duplicate_tiff) - set(duplicate_angle))


    return ProjectionSet(
        files= tuple(file_by_number[number] for number in valid_numbers),
        projection_numbers= np.asarray(valid_numbers, dtype=np.int64),
        angles= np.asarray([angle_by_number[number] for number in valid_numbers], dtype=np.float64),
        missing_tiff_numbers=missing_tiff,
        unmatched_tiff_numbers=unmatched_tiff,
        duplicate_tiff_numbers=duplicate_tiff,
        duplicate_angle_numbers=duplicate_angle,
        invalid_files=tuple(invalid_files)
        
    )

def find_projection_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    extensions = {".tif", ".tiff"}

    files = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.name.casefold()
    )

    if not files:
        raise FileNotFoundError(f"No TIFF files found in: {folder}")

    return files

def load_projection(file: str | Path, expected_shape: tuple[int,int] | None = None, expected_dtype: np.dtype | None = None, ) -> np.ndarray:
    path = Path(file)

    if not path.exists():
        raise FileNotFoundError(f"Projection file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Projection path is not a file: {path}")

    image = tifffile.imread(path)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D projection, got {image.ndim} dimensions: {path.name}")
    if expected_shape is not None and image.shape != expected_shape:
        raise ValueError(f"Expected image shape {expected_shape}, got {image.shape}: {path.name}")
    if expected_dtype is not None and image.dtype != expected_dtype:
        raise ValueError(f"Expected image dtype {expected_dtype}, got{image.dtype}: {path.name} ")

    return image

def iter_projections(projection_set: ProjectionSet,) -> Iterator[tuple[int, float, np.ndarray]]:
    if not projection_set.files:
        raise ValueError("No matched projections are available.")
    if not (len(projection_set.files) == len(projection_set.projection_numbers) == len(projection_set.angles)):
        raise ValueError("Projection files, numbers and angles must have the same length.")

    expected_shape = None
    expected_dtype = None

    for number, angle, file in zip(projection_set.projection_numbers, projection_set.angles,projection_set.files,):
        if expected_shape is None:
            image = load_projection(file)
            expected_shape = image.shape
            expected_dtype = image.dtype
        else:
            load_projection(file,expected_shape=expected_shape,expected_dtype=expected_dtype,)
        yield int(number),float(angle),image

def validate_scan(scan_directory: str | Path, angle_file: str | Path, strict: bool = True,) -> ScanReport:
    scan_directory = Path(scan_directory)
    angle_file = Path(angle_file)

    if not scan_directory.exists():
        raise FileNotFoundError(f"Scan directory does not exist {scan_directory}")
    if not scan_directory.is_dir():
        raise NotADirectoryError(f"Scan path is not a directory: {scan_directory}")
    if not angle_file.exists():
        raise FileNotFoundError(f"Angle file does not exist: {angle_file}")
    if not angle_file.is_file():
        raise ValueError(f"Angle path is not a file: {angle_file}")

    files = find_projection_files(scan_directory)
    angle_data =read_angles(angle_file)
    projection_set = match_projections(files, angle_data)

    has_duplicates = bool(projection_set.duplicate_tiff_numbers or projection_set.duplicate_angle_numbers)

    if strict and has_duplicates:
        return ScanReport(scan_directory=scan_directory,total_tiff_files=len(files),angle_data=angle_data,
                          projection_set=projection_set,successfully_loaded=0,failed_projections=(),image_shape=None,image_dtype=None,is_valid=False,)
    successfully_loaded = 0
    failed_projections = []

    image_shape = None
    image_dtype = None

    # 7. Load and validate projections one at a time
    for number, angle, image in iter_projections(projection_set):
        successfully_loaded += 1

        if image_shape is None:
            image_shape = image.shape
            image_dtype = image.dtype