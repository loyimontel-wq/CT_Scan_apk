import glob, os, re, tifffile
import numpy as np


def bin2d(a, b):
    if b == 1:
        return a
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return a[:h, :w].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def find_projection_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")) +
                   glob.glob(os.path.join(folder, "*.tif")))

    files = [f for f in files
             if not re.search(r"(flat|dark|white|black|ref|stack)", os.path.basename(f), re.I)]
    return files



def load_projections(files, binning=1, step=1, flat=None, dark=None, white_level=60000.0):
    """Load every `step`-th projection, bin, and convert to line integrals p = -ln(I/I0).
    Returns array (n_proj, nv, nu) float32 and the indices used."""
    idx = np.arange(0, len(files), step)
    flat_b = bin2d(tifffile.imread(flat).astype(np.float32), binning) if flat else None
    dark_b = bin2d(tifffile.imread(dark).astype(np.float32), binning) if dark else None
    out = None
    for n, i in enumerate(idx):
        im = bin2d(tifffile.imread(files[i]).astype(np.float32), binning)
        if flat_b is not None:
            d = dark_b if dark_b is not None else 0.0
            im = (im - d) / np.maximum(flat_b - d, 1.0)
        else:
            im = im / white_level
        im = np.clip(im, 1e-4, None)
        if out is None:
            out = np.empty((len(idx),) + im.shape, np.float32)
        out[n] = -np.log(im)
        if n % 100 == 0:
            print(f"  loaded {n + 1}/{len(idx)}", end="\r")
    print()
    return out, idx