from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from io import AngleData
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

def match_projections(files: list[str | Path], angle_data: AngleData,) -> ProjectionSet:
    pattern = re.compile(r"_(\d+)\.(?:tif|tiff)$", re.IGNORECASE)
    file_by_number = {}
    tiff_numbers = []

    for file in files:
        path = Path(file)
        match = pattern.search(path.name)

        if match is None:
            raise ValueError(f"Cannot extracvt projection ID from: {path.name}")
        number = int(match.group(1))
        tiff_numbers.append(number)

        if number not in file_by_number:
            file_by_number[number] =  path

    tiff_counts = Counter(tiff_numbers)
    angle_counts = Counter(map(int,angle_data.projecton_numbers))
    duplicate_tiff = tuple(sorted(number for number, count in tiff_counts.items() if count > 1))
    duplicate_angle = tuple(sorted(number for number, count in angle_counts.items() if count > 1))

    tiff_ids = set(tiff_counts)
    angle_ids = set(angle_counts)

    missing_tiff = tuple(sorted(angle_ids - tiff_ids))
    unmatched_tiff = tuple(sorted(tiff_ids - angle_ids))

    angle_by_number = {int(number): angle for number, angle in zip(angle_data.projection_numbers,angle_data.angles) if angle_counts[int(number)]==1}
    valid_numbers = sorted((tiff_ids & angle_ids) - set(duplicate_tiff) - set(duplicate_angle))


    return ProjectionSet(files= tuple(file_by_number[number] for number in valid_numbers),
        projection_numbers= np.asarray(valid_numbers, dtype=np.int64),
        angles= np.asarray([angle_by_number[number] for number in valid_numbers], dtype=np.float64),
        missing_tiff_numbers=missing_tiff,
        unmatched_tiff_numbers=unmatched_tiff,
        duplicate_tiff_numbers=duplicate_tiff
        duplicate_angle_numbers=duplicate_angle,
    )