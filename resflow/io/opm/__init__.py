"""OPM Flow results -> ResFlow raw folders (same files as the IMEX ``.sr3`` converter).

Public entry points: :func:`convert_opm_run` (one run) and :func:`organize_opm_dataset` (a deck folder, used by
``resflow convert --format opm``).
"""
from .convert import OpmRunDiscarded, convert_opm_run, organize_opm_dataset

__all__ = ["OpmRunDiscarded", "convert_opm_run", "organize_opm_dataset"]

