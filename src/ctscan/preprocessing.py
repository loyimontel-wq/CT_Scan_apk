import numpy as np 

def log_transform(projection: np.ndarray, i0: float = 60000.0,) -> tuple[np.ndarray, np.ndarray]:
    if projection.ndim != 2:
        raise ValueError(" projection must be a 2D image.")
    if not np.isfinite(i0) or i0 <= 0:
        raise ValueError("Reference instensity i0 must be postive and finite.")
    image = projection.astype(np.float32)
    invalid_mask = ~np.isfinite(image) | (image <= 0)
    attenuation = np.zeros_like(image, dtype=np.float32)
    valid_pixels = ~invalid_mask
    transmission = image[valid_pixels] /i0
    attenuation[valid_pixels] = -np.log(transmission)
    return attenuation, invalid_mask
