"""
Split side-by-side subplot figures into stacked individual PNGs.
Outputs: reports/ae_errors_top.png, ae_errors_bottom.png,
         reports/gnn_prog_top.png, gnn_prog_bottom.png
"""
from PIL import Image
import os

REPORTS = os.path.join(os.path.dirname(__file__), '..', 'reports')

def split_horizontal(src_name, out_top, out_bot, label=""):
    path = os.path.join(REPORTS, src_name)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {src_name}")
        return
    img = Image.open(path)
    w, h = img.size
    mid = w // 2
    left  = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))
    left.save(os.path.join(REPORTS, out_top))
    right.save(os.path.join(REPORTS, out_bot))
    print(f"  {src_name} ({w}x{h}) -> {out_top} + {out_bot}")

print("Splitting figures...")
split_horizontal("autoencoder_errors.png",       "ae_errors_left.png",    "ae_errors_right.png")
split_horizontal("gnn_experiments_progression.png", "gnn_prog_left.png",  "gnn_prog_right.png")
print("Done.")
