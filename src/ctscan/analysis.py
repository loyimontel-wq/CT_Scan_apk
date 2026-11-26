
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scan_folder = Path(
    "../CT_scan/old stuff/runnit/"
    "Lunga_OlivChloriFluoriApatite130kV _469 "
    "[2019-04-22 22.55.02]"
)

file_a = scan_folder / (
    "Lunga_OlivChloriFluoriApatite130kV _469_0794.tif"
)

file_b = scan_folder / (
    "Lunga_OlivChloriFluoriApatite130kV _469_2295.tif"
)

projection_a = tifffile.imread(file_a).astype(np.float32)
projection_b = tifffile.imread(file_b).astype(np.float32)

I0 = 60000.0
row_centres = [512, 1024, 1536]
half_band = 2

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for ax, centre in zip(axes, row_centres):

    start = centre - half_band
    end = centre + half_band + 1

    profile_a = projection_a[start:end, :].mean(axis=0)
    profile_b = projection_b[start:end, :].mean(axis=0)

    valid_a = profile_a > 0
    valid_b = profile_b > 0

    attenuation_a = np.full(profile_a.shape, np.nan)
    attenuation_b = np.full(profile_b.shape, np.nan)

    attenuation_a[valid_a] = -np.log(profile_a[valid_a] / I0)
    attenuation_b[valid_b] = -np.log(profile_b[valid_b] / I0)

    mirrored_b = attenuation_b[::-1]

    ax.plot(attenuation_a, label="Projection 794")
    ax.plot(mirrored_b, label="Mirrored projection 2295")

    ax.set_title(f"Detector row {centre} (5-row average)")
    ax.set_xlabel("Detector column (pixels)")
    ax.set_ylabel("Attenuation")
    ax.legend()
    ax.grid(True)

plt.tight_layout()

output_file = Path("opposite_profiles_794_2295.png")
plt.savefig(output_file, dpi=150)

print(f"\nPlot saved to: {output_file.resolve()}")

