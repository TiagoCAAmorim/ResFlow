"""Defaults shared by the scripts."""

# Seed used by every script unless --seed is given: the same inputs and the same default seed give the same
# patches, layouts and BHP schedules on any machine (numpy's default_rng is stable across versions).
DEFAULT_SEED = 0


def out_dir_in_use(out):
    """True if `out` exists and is not an empty folder; the scripts refuse to write there. An empty folder is fine."""
    return out.exists() and (not out.is_dir() or any(out.iterdir()))
