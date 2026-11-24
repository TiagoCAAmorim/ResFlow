#!/usr/bin/env python
"""Download the ResFlow OPM demo data (raw simulation folders of 6 decks) from Zenodo.

    python scripts/download_demo.py --out _data/demo                # fetch the files, check the md5s, unzip
    python scripts/download_demo.py --out _data/demo --from-dir D   # use a local copy of the record instead
    python scripts/download_demo.py --list                          # list the files of the record

The record holds one zip per deck (`<set>__<deck>.zip`, Zenodo has no folders) and one `<set>_decks_index.csv` per set.
Result: <out>/spe10/test/raw/<deck>/..., <out>/spe10_fluvial/test/raw/<deck>/..., <out>/<set>/decks_index.csv.
The zips are kept in <out>/_download (delete them afterwards); files already downloaded with the right md5 are skipped.
Only the standard library is used (the Zenodo REST files API).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD = "24509172"     # Zenodo record of the demo data (DOI 10.5281/zenodo.24509172)
API = "https://zenodo.org/api/records/{rid}"
WANTED = ("spe10__", "spe10_fluvial__")          # prefixes of the deck zips
CHUNK = 1 << 20


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def record_id(value: str) -> str:
    """'1234567', 'https://zenodo.org/records/1234567' or '10.5281/zenodo.1234567' -> '1234567'."""
    value = value.strip().rstrip("/")
    if value.upper() == "TBD" or not value:
        sys.exit("The Zenodo record is not set yet (ZENODO_RECORD = 'TBD' in this script): pass --record <id or DOI> "
                 "or use --from-dir.")
    tail = value.replace("zenodo.", "/").split("/")[-1]
    if not tail.isdigit():
        sys.exit(f"cannot read a record id from {value!r}")
    return tail


def list_record(rid: str) -> list[dict]:
    """[{'key', 'size', 'md5', 'url'}] from the record's files list."""
    req = urllib.request.Request(API.format(rid=rid), headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        meta = json.load(r)
    files = meta.get("files")
    if isinstance(files, dict):                      # newer API: {"entries": {...}}
        files = list(files.get("entries", {}).values())
    out = []
    for f in files or []:
        chk = str(f.get("checksum", ""))
        out.append(dict(key=f["key"], size=int(f.get("size", 0)), md5=chk.split(":", 1)[-1] if chk else "",
                        url=(f.get("links") or {}).get("self") or f"https://zenodo.org/records/{rid}/files/{f['key']}?download=1"))
    if not out:
        sys.exit(f"record {rid} lists no files")
    return out


def from_dir(src: Path) -> list[dict]:
    out = [dict(key=p.name, size=p.stat().st_size, md5="", url=str(p)) for p in sorted(src.iterdir()) if p.is_file()]
    if not out:
        sys.exit(f"no files in {src}")
    return out


def fetch(url: str, dest: Path, tries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, CHUNK)
            tmp.replace(dest)
            return
        except OSError as e:
            if k == tries - 1:
                sys.exit(f"download failed for {dest.name}: {e}")


def unzip(zpath: Path, out: Path) -> None:
    root = out.resolve()
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f"{zpath.name}: corrupt member {bad}")
        for n in z.namelist():
            if not (root / n).resolve().is_relative_to(root):
                sys.exit(f"{zpath.name}: unsafe path {n!r}")
        z.extractall(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="_data/demo", help="folder to write to (default: _data/demo)")
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record id or DOI (default: the constant in this script)")
    ap.add_argument("--from-dir", help="use a local copy of the record (same files) instead of downloading")
    ap.add_argument("--list", action="store_true", help="only list the files and sizes")
    a = ap.parse_args(argv)

    entries = from_dir(Path(a.from_dir)) if a.from_dir else list_record(record_id(a.record))
    print(f"{len(entries)} files, {sum(e['size'] for e in entries) / 1e6:.0f} MB")
    if a.list:
        for e in entries:
            print(f"{e['size']:>12}  {e['key']}")
        return 0
    out = Path(a.out)
    cache = out / "_download"
    bad = 0
    for e in entries:
        key = e["key"]
        if key.endswith(".zip") and key.startswith(WANTED):
            dest = cache / key
        elif key.endswith("_decks_index.csv"):
            dest = None
        else:
            continue                                 # README and other files are not needed
        src = Path(e["url"]) if a.from_dir else None
        tmp = dest or (cache / key)
        if not (tmp.exists() and (not e["md5"] or md5_of(tmp) == e["md5"])):
            if src is not None:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, tmp)
            else:
                fetch(e["url"], tmp)
            if e["md5"] and md5_of(tmp) != e["md5"]:
                print(f"MD5 MISMATCH: {key}", file=sys.stderr)
                tmp.unlink()
                bad += 1
                continue
        if dest is not None:
            unzip(dest, out)
            print(f"  {key}: unzipped")
        else:
            idx = out / key.removesuffix("_decks_index.csv") / "decks_index.csv"
            idx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp, idx)
    if bad:
        print(f"{bad} file(s) failed the md5 check and were removed; run again", file=sys.stderr)
        return 1
    decks = sorted(p.name for p in out.glob("*/test/raw/*") if p.is_dir())
    print(f"done: {len(decks)} decks in {out}: " + ", ".join(decks))
    print(f"next: resflow build --root {out}/spe10   (and {out}/spe10_fluvial)   [zips can be deleted: {cache}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
