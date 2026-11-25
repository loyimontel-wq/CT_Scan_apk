import math 

def calculate_initial_cor(xtekct: dict) -> float:
    required = ["SrcToObject",
                "SrcToDetector",
                "DetectorPixelSizeX",
                "DetectorPixelsX",
                "ObjectOffsetX",
                ]

    missing = [key for key in required if key not in xtekct]
    if missing: 
        raise ValueError(f"Missing XTekCT parameters: {', '.join(missing)}")
    sod = float(xtekct["SrcToObject"])
    sdd = float(xtekct["SrcToDetector"])
    pixel_size = float(xtekct["DetectorPixelSizeX"])
    detector_pixels = int(xtekct["DetectorPixelsX"])
    object_offset = float(xtekct["ObjectOffsetX"])

    values = [sod, sdd,pixel_size, object_offset]

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Geometry parameters must be finite.")
    if sod <= 0 or sdd <= 0 or pixel_size <= 0:
        raise ValueError("Distances and pixel size must be positive.")
    if detector_pixels <= 0:
        raise ValueError("Detector width must be positive.")

    magnification = sdd /sod
    offset_pixels = (object_offset * magnification / pixel_size)
    detector_centre = (detector_pixels -1 )/2


    return detector_centre + offset_pixels



def find_opposite_pairs(
    angle_data,
    tolerance_deg: float = 0.1,
) -> list[dict]:

    if not math.isfinite(tolerance_deg) or tolerance_deg < 0:
        raise ValueError("Tolerance must be finite and non-negative.")

    projection_numbers = angle_data.projection_numbers
    angles = angle_data.angles

    if len(projection_numbers) != len(angles):
        raise ValueError("Projection numbers and angles must have equal lengths.")

    if len(set(projection_numbers)) != len(projection_numbers):
        raise ValueError("Duplicate projection numbers found.")

    if not all(math.isfinite(float(angle)) for angle in angles):
        raise ValueError("All projection angles must be finite.")

    candidates = []

    # Find all possible opposite-angle pairs.
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):

            angle_difference = (
                float(angles[j]) - float(angles[i])
            ) % 360.0

            error = abs(angle_difference - 180.0)

            if error <= tolerance_deg:
                candidates.append(
                    (
                        error,
                        projection_numbers[i],
                        projection_numbers[j],
                        float(angles[i]),
                        float(angles[j]),
                    )
                )

    # Sort candidates by angular accuracy.
    candidates.sort(key=lambda pair: pair[0])

    pairs = []
    used_projections = set()

    # Ensure each projection is used only once.
    for error, number_a, number_b, angle_a, angle_b in candidates:

        if number_a in used_projections or number_b in used_projections:
            continue

        pairs.append(
            {
                "projection_a": number_a,
                "projection_b": number_b,
                "angle_a": angle_a,
                "angle_b": angle_b,
                "angular_error": error,
            }
        )

        used_projections.add(number_a)
        used_projections.add(number_b)

    return pairs
