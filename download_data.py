import kagglehub
import shutil
import os

DEST = os.path.join(os.path.dirname(__file__), "data")

print("Downloading Elliptic dataset...")
path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
print(f"Downloaded to: {path}")

os.makedirs(DEST, exist_ok=True)
for f in os.listdir(path):
    src = os.path.join(path, f)
    dst = os.path.join(DEST, f)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    print(f"Copied: {f}")

print(f"\nFiles in data/: {os.listdir(DEST)}")
