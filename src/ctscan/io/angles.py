

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass
class AngleData:
    projection_numbers: np.ndarray
    angles: np.ndarray
    missing_numbers: tuple[int, ...]
    duplicate_numbers: tuple[int, ...]
    malformed_rows: tuple[int, ...]

def read_angles(path: str | Path) -> AngleData:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Angle file not found: {path}")
    projection_labels = {
        "proj",
        "projection",
        "projectionnumber",
        "projectionno",
    }
    angle_labels = {
        "angle",
        "angledeg",
        "angledegrees",
        "projectionangle",
        "projectionangledeg",
        "projectionangledegrees",
    }
    angles = []
    projection_numbers = []
    malformed_rows = []

    projection_column = None
    angle_column = None
    header_found = False

    
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number,line in enumerate(file, start=1):
            parts = line.split()
            if not parts:
                continue
            if not header_found:
                header = [ re.sub(r"[^a-z]","",part.lower()) for part in parts]
                projection_matches = [ i for i, value in enumerate(header) if value in projection_labels]
                angle_matches = [i for i, value in enumerate(header) if value in angle_labels]
                if not projection_matches or not angle_matches:
                    continue                
                if len(projection_matches) != 1 or len(angle_matches) != 1:
                    raise ValueError(f"Ambiguous projection or angle columns " f"in {path} at line {line_number}")
                projection_column = projection_matches[0]
                angle_column = angle_matches[0]
                header_found = True

                continue
            if parts[0].startswith(("#",":")):
                continue
            normalised_parts = [re.sub(r"[^a-z]","",part.lower()) for part in parts]
            if (any(value in projection_labels for value in normalised_parts) 
                and any(value in angle_labels for value in normalised_parts)):
                continue
            required_column = max(projection_column, angle_column)

            if len(parts) <= required_column:
                continue

            try:
                projection_number = int(parts[projection_column].rstrip(":,;"))
                angle = float(parts[angle_column].rstrip(",;"))
            except ValueError:
                malformed_rows.append(line_number)
                continue
            if not np.isfinite(angle):
                malformed_rows.append(line_number)
                continue
            projection_numbers.append(projection_number)
            angles.append(angle)
        
        
    if not header_found:
        raise ValueError(
            f"Could not identify projection and angle columns in: {path}"
        )

    if not angles:
        raise ValueError(f"No valid projection angles found in: {path}")

    counts = Counter(projection_numbers)
    duplicates = tuple(sorted(number for number, count in counts.items() if count > 1))
    recorded_numbers = set(projection_numbers)

    missing = tuple(number
        for number in range(min(recorded_numbers),max(recorded_numbers) + 1)
        if number not in recorded_numbers
    )

    return AngleData(
        projection_numbers=np.asarray(projection_numbers, dtype=np.int64),
        angles=np.asarray(angles, dtype=np.float64),
        missing_numbers=missing,
        duplicate_numbers=duplicates,
        malformed_rows=tuple(malformed_rows),
    )
        
           
        