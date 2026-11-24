#!/usr/bin/env python
"""Create a DRAFT Zenodo deposition of the demo data and upload its files. It never publishes.

    export ZENODO_TOKEN=...            # personal access token with the deposit:write scope (never stored or printed)
    python scripts/upload_zenodo.py --dir _data/zenodo/resflow_opm_demo [--sandbox]
    python scripts/upload_zenodo.py --dir _data/zenodo/resflow_opm_demo --dry-run     # list files, no network

Steps: create the draft, upload every file of --dir except .zenodo.json through the bucket API, set the metadata from
.zenodo.json, print the URL of the draft. Check the draft on the website and press Publish there (the DOI is created then).
Entries of related_identifiers that still contain 'TBD' are left out (add them on the website). --sandbox uses
sandbox.zenodo.org (separate account and token). Only the standard library is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def call(method: str, url: str, token: str, data=None, content_type="application/json", length=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type}
    if length is not None:
        headers["Content-Length"] = str(length)
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"Zenodo answered {e.code} to {method} {url.split('?')[0]}: {e.read().decode(errors='replace')[:500]}")
    return json.loads(body) if body else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="_data/zenodo/resflow_opm_demo", help="staging folder (default: %(default)s)")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org instead of zenodo.org")
    ap.add_argument("--dry-run", action="store_true", help="list the files and the metadata, do nothing else (no network, no token)")
    a = ap.parse_args(argv)

    d = Path(a.dir)
    meta_file = d / ".zenodo.json"
    if not meta_file.is_file():
        sys.exit(f"{meta_file} not found")
    files = sorted(p for p in d.iterdir() if p.is_file() and p.name != ".zenodo.json")
    meta = json.loads(meta_file.read_text())
    rel = meta.get("related_identifiers", [])
    kept = [r for r in rel if "TBD" not in json.dumps(r)]
    if len(kept) != len(rel):
        print(f"note: {len(rel) - len(kept)} related identifier(s) with TBD left out; add them on the website")
    meta["related_identifiers"] = kept
    base = "https://sandbox.zenodo.org" if a.sandbox else "https://zenodo.org"

    print(f"{len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB, to {base}")
    for p in files:
        print(f"{p.stat().st_size:>12}  {p.name}")
    print("metadata:", ", ".join(meta))
    if a.dry_run:
        print("dry run: nothing sent")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set the environment variable ZENODO_TOKEN to your Zenodo access token (deposit:write scope).")
    dep = call("POST", f"{base}/api/deposit/depositions", token, data=b"{}")
    bucket = dep["links"]["bucket"]
    print("draft", dep["id"], "created")
    for p in files:
        size = p.stat().st_size
        with open(p, "rb") as f:
            call("PUT", f"{bucket}/{p.name}", token, data=f, content_type="application/octet-stream", length=size)
        print("  uploaded", p.name)
    call("PUT", f"{base}/api/deposit/depositions/{dep['id']}", token, data=json.dumps({"metadata": meta}).encode())
    print("metadata set")
    print(f"Draft (not published): {dep['links'].get('html', base + '/deposit/' + str(dep['id']))}")
    print("Check it and press Publish on the website.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
