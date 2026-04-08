from pathlib import Path
from urllib.request import Request, urlopen
import zipfile
import time
from ultralytics.data.converter import convert_coco


VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

DATASET_ROOT = Path("datasets")
COCO_DIR = DATASET_ROOT / "coco"
IMAGES_DIR = COCO_DIR / "images"
VAL_ZIP_PATH = IMAGES_DIR / "val2017.zip"

ANNOTATIONS_ZIP_PATH = COCO_DIR / "annotations_trainval2017.zip"

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.1f}{u}"
        size /= 1024

def print_progress(prefix: str, done: int, total: int | None, width: int = 32, start_time: float | None = None) -> None:
    if total and total > 0:
        ratio = min(done / total, 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        pct = ratio * 100
        speed = ""
        eta = ""
        if start_time:
            elapsed = max(time.time() - start_time, 1e-6)
            bps = done / elapsed
            speed = f" {human_size(int(bps))}/s"
            if bps > 0 and done < total:
                eta_sec = int((total - done) / bps)
                eta = f" ETA {eta_sec}s"
        msg = f"\r{prefix} [{bar}] {pct:6.2f}% {human_size(done)}/{human_size(total)}{speed}{eta}"
    else:
        msg = f"\r{prefix} {human_size(done)}"
    print(msg, end="", flush=True)

def download_with_progress(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"Zip already exists, skipping download: {dst}")
        return

    req = Request(url, headers={"User-Agent": "Python-stdlib-coco-downloader/1.0"})
    print(f"Downloading: {url}")
    with urlopen(req) as resp, open(dst, "wb") as f:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        done = 0
        start = time.time()

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            print_progress("Download", done, total, start_time=start)

    print()
    print(f"Saved to: {dst}")

def extract_with_progress(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, path=out_dir)
            print_progress("Extract ", i, total)

    print()
    print(f"Extracted to: {out_dir}")

if __name__ == "__main__":
    download_with_progress(VAL_URL, VAL_ZIP_PATH)
    extract_with_progress(VAL_ZIP_PATH, IMAGES_DIR)
    # Rename extracted images folder to 'val'
    extracted_folder = IMAGES_DIR / "val2017"
    target_folder = IMAGES_DIR / "val"
    if extracted_folder.exists() and not target_folder.exists():
        extracted_folder.rename(target_folder)

    download_with_progress(ANNOTATIONS_URL, ANNOTATIONS_ZIP_PATH)
    extract_with_progress(ANNOTATIONS_ZIP_PATH, COCO_DIR)

    print("Images folder:", IMAGES_DIR / "val")
    print("Annotation file:", DATASET_ROOT / "annotations" / "instances_val2017.json")