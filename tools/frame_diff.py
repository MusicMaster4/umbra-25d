"""Pixel differences between two folders of rendered frames (needs numpy and Pillow; umbra itself does not).

    python -m umbra render scene.py --range 8 11 --keep-frames --out-dir a
    python -m umbra render scene.py --range 8 11 --keep-frames --out-dir b --safe
    python tools/frame_diff.py a/.frames/scene_final b/.frames/scene_final [-v]

Frames are matched by number (f_00241.tif against f_00241.png is fine). For each folder compared with the first:
how many frames are identical, the largest difference in 8-bit levels, the share of pixels that differ at all and
by more than one level, and the lowest PSNR.

Reading the numbers: EEVEE is not bit-reproducible. The same frame rendered twice by the same code usually differs
by one level on a few pixels in a million (up to 0.03 % in scenes with dithered fades), and unlit looks (blueprint,
paper) come out identical. Also, what Blender draws for a frame depends on the frame the same process rendered just
before it, so compare frames that were rendered in film order on both sides."""
import glob
import os
import sys

import numpy as np
from PIL import Image


def _frames(d):
    return {os.path.splitext(os.path.basename(p))[0]: p for e in ("png", "tif") for p in glob.glob(os.path.join(d, "f_*." + e))}


def compare(a_dir, b_dir):
    """[(frame name, (max level difference, share of pixels differing, share differing by > 1 level, psnr dB))]"""
    fa, fb = _frames(a_dir), _frames(b_dir)
    rows = []
    for nm in sorted(set(fa) & set(fb)):
        a = np.asarray(Image.open(fa[nm]).convert("RGB")).astype(np.int16)
        b = np.asarray(Image.open(fb[nm]).convert("RGB")).astype(np.int16)
        if a.shape != b.shape:
            rows.append((nm, None))
            continue
        d = np.abs(a - b)
        mse = float((d.astype(np.float64) ** 2).mean())
        px = d.max(axis=2)
        rows.append((nm, (int(d.max()), float((px > 0).mean()), float((px > 1).mean()),
                          99.0 if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse)))))
    return rows


def main(argv):
    verbose = "-v" in argv
    dirs = [x for x in argv if x != "-v"]
    if len(dirs) < 2:
        sys.exit(__doc__)
    for b_dir in dirs[1:]:
        rows = compare(dirs[0], b_dir)
        ok = [r for _, r in rows if r is not None]
        if not ok:
            print(f"{b_dir}: no frames in common with {dirs[0]}")
            continue
        print(f"{b_dir}: frames={len(ok)} identical={sum(1 for r in ok if r[0] == 0)} max_diff={max(r[0] for r in ok)} "
              f"px_differ={100 * max(r[1] for r in ok):.3f}% px_over_1_level={100 * max(r[2] for r in ok):.4f}% "
              f"min_psnr={min(r[3] for r in ok):.2f}dB")
        if verbose:
            for nm, r in rows:
                print("   ", nm, "different size" if r is None else "max=%d differ=%.4f%% over1=%.4f%% psnr=%.2f" % (r[0], 100 * r[1], 100 * r[2], r[3]))


if __name__ == "__main__":
    main(sys.argv[1:])
