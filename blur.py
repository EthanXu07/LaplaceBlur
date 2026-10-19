"""Laplace Blur - Python port of index.html.

The JS version packs RGBA into 32-bit ints and averages all channels at once
with shift & mask tricks. Channels never interact there, so this port runs
the same integer math per channel with numpy, vectorized across rows/columns.

Usage:
    python blur.py input.png [-o output.png] [-n passes]
"""
import argparse
import time

import numpy as np
from PIL import Image


def _step(t, v):
    # Per-channel equivalent of: (x01 + t >> 1 & x7f) + (v >> 1 & x7f)
    return ((t + 1) >> 1) + (v >> 1)


def exp_blur_7r(img):
    """One blur pass. img: (h, w, 3) uint8 array. Returns a new uint8 array."""
    buf = img.astype(np.int32)
    h, w, _ = buf.shape

    # 1st: every row, left to right
    t = np.zeros((h, 3), np.int32)
    for x in range(w):
        buf[:, x] = t = _step(t, buf[:, x])

    # last row, right to left, halved
    t = np.zeros(3, np.int32)
    for x in range(w - 1, -1, -1):
        t = _step(t, buf[h - 1, x])
        buf[h - 1, x] = t >> 1

    # 2nd-3rd: remaining rows right to left, mixed with the row below (bottom up)
    if h > 1:
        rows = buf[:h - 1]
        acc = np.empty_like(rows)
        t = np.zeros((h - 1, 3), np.int32)
        for x in range(w - 1, -1, -1):
            acc[:, x] = t = _step(t, rows[:, x])
        for y in range(h - 2, -1, -1):
            buf[y] = ((buf[y + 1] + 1) >> 1) + (acc[y] >> 1)

    # 4th: every column, top to bottom
    t = np.zeros((w, 3), np.int32)
    for y in range(h):
        buf[y] = t = _step(t, buf[y])

    return buf.astype(np.uint8)


def load(path):
    # The JS canvas is opaque (alpha: false), so flatten transparency onto black.
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def main():
    p = argparse.ArgumentParser(description="Laplace Blur")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="blurred.png")
    p.add_argument("-n", "--passes", type=int, default=1,
                   help="times to apply the blur (like clicking n times)")
    args = p.parse_args()

    img = load(args.input)
    for _ in range(args.passes):
        start = time.perf_counter()
        img = exp_blur_7r(img)
        print(f"{(time.perf_counter() - start) * 1000:.0f}ms")
    Image.fromarray(img).save(args.output)


if __name__ == "__main__":
    main()
