import tifffile
from pathlib import Path
from collections import Counter

folder = Path(
    "../CT_scan/old stuff/runnit/"
    "Lunga_OlivChloriFluoriApatite130kV _469 "
    "[2019-04-22 22.55.02]"
)

files = [
    p for p in folder.iterdir()
    if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
]

image_info = Counter()
errors = []

for path in files:
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            image_info[(page.shape, str(page.dtype), len(tif.pages))] += 1
    except Exception as error:
        errors.append((path.name, str(error)))

print("Total TIFF files:", len(files))
print("\nImage formats:")

for info, count in image_info.items():
    print(f"Shape: {info[0]}, dtype: {info[1]}, pages: {info[2]}, count: {count}")

print("\nFiles with reading errors:", len(errors))

for name, error in errors[:10]:
    print(name, error)


    