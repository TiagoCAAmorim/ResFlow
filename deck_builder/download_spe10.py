"""Step 0: download the input files into --cache-dir (once) and print the path of the permeability file.

    python download_spe10.py --cache-dir $DATA/inputs

- SPE10 permeability: spe.org (por_perm_case2a.zip, only the permeability member is extracted), then the OPM
  data repository. Pass the printed path to --perm-file of layer_correlation.py and generate_spe10_patches.py.
- Relative permeability (UNISIM-IV Nominal.rar) and PVT (UNISIM-II-D zip), used by build_decks.py, which
  reads them from the same --cache-dir. Use --no-fluids to skip them.
Files already in --cache-dir are not downloaded again. Extracting KR from the RAR (in build_decks.py) needs
unrar, unar or 7z.
"""
import argparse
from pathlib import Path

import fluid_files
import perm_io

HERE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=str(HERE / "_inputs"), help="where the file is kept (default: %(default)s)")
    p.add_argument("--no-fluids", action="store_true", help="download only the permeability file")
    args = p.parse_args(argv)
    if not args.no_fluids:
        for f in fluid_files.fetch_fluid_sources(args.cache_dir):
            print(f"fluid source: {Path(f).resolve()}")
    print(perm_io.fetch_spe10_perm(args.cache_dir).resolve())


if __name__ == "__main__":
    main()
