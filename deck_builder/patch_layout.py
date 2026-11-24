"""Patch positions inside a 60 x 220 layer. Patches are (x, y, w, h) with 0-based start cells."""
import numpy as np

NX, NY = 60, 220


def overlap_frac(a, b):
    """Intersection area divided by the area of `a` (patches of one size)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3])


def layer_rng(seed, layer, stream=0):
    return np.random.default_rng([seed, layer, stream])


def place_train(rng, n, size, max_overlap, max_tries=100_000):
    """Greedy seeded random placement with a pairwise overlap limit. May return fewer than n."""
    placed = []
    for _ in range(max_tries):
        if len(placed) == n:
            break
        cand = (int(rng.integers(0, NX - size + 1)), int(rng.integers(0, NY - size + 1)), size, size)
        if all(overlap_frac(cand, p) <= max_overlap for p in placed):
            placed.append(cand)
    return placed


def place_spread(rng, n, size):
    """n non-overlapping patches evenly spread along y, each at a seeded random x offset."""
    ys = np.round(np.linspace(0, NY - size, n)).astype(int)
    xs = rng.integers(0, NX - size + 1, size=n)
    return [(int(x), int(y), size, size) for x, y in zip(xs, ys)]


def resample_x(rng, patch):
    return (int(rng.integers(0, NX - patch[2] + 1)), patch[1], patch[2], patch[3])


# 60x60: full x width; (60, 120) cut for 120x60 is rotated afterwards.
POS_60 = [(0, 0, 60, 60), (0, 80, 60, 60), (0, 160, 60, 60)]
POS_120 = [(0, 0, 60, 120), (0, 100, 60, 120)]
