"""Convert a CMG IMEX (MODSI, black-oil, corner-point) .dat deck into an OPM Flow (ECLIPSE METRIC) deck.

Scope: the decks built by build_decks.py (corner-point grid of one or more layers, dead/live black oil
with BOT/VOT tables, SWT/SLT relperm, BHP-controlled wells changed by DATE / TARGET BHP).

Usage:
    python cmg_to_opm.py path/to/case.dat [-o outdir] [--wi-scale 1.0] [--no-restart]
Default outdir: <deck folder>_opm/<case>/.

Wells: no crossflow (WELSPECS item 10 = NO; pass --crossflow to allow it): a well whose flow would reverse is shut
until its next control. The SUMMARY also requests the per-connection vectors CTFAC, COPT, CWPT, CWIT, CGPT, CPR.
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

KGCM2_TO_BAR = 0.980665
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# ---------------------------------------------------------------- low-level parsing
def strip_comments(text):
    """Drop CMG '**' comments (whole-line or trailing)."""
    return "\n".join(line.split("**")[0] for line in text.splitlines())


def expand_tokens(tokens):
    """Expand CMG/ECLIPSE 'n*value' repeats into a flat list of floats."""
    out = []
    for t in tokens:
        if "*" in t:
            n, v = t.split("*")
            out.extend([float(v)] * int(n))
        else:
            out.append(float(t))
    return out


def read_numbers(path):
    return np.array(expand_tokens(strip_comments(Path(path).read_text()).split()))


def find_include(lines, keyword_regex, base):
    """Path of the INCLUDE that directly follows a line matching keyword_regex."""
    for i, line in enumerate(lines):
        if re.match(keyword_regex, line.strip(), re.I):
            for nxt in lines[i + 1:i + 4]:
                m = re.match(r"\s*INCLUDE\s+'([^']+)'", nxt, re.I)
                if m:
                    return (base / m.group(1)).resolve()
    return None


def all_includes(lines, base):
    return [(base / m.group(1)).resolve() for line in lines
            if (m := re.match(r"\s*INCLUDE\s+'([^']+)'", line, re.I))]


# ---------------------------------------------------------------- grid
def parse_corners(path, nx, ny, nz):
    """CMG CORNERS -> (X, Y, Z) arrays each of length 8*nx*ny*nz, in ECLIPSE ZCORN order."""
    text = strip_comments(Path(path).read_text())
    body = text.split("CORNERS", 1)[1]
    vals = np.array(expand_tokens(body.split()))
    n = 8 * nx * ny * nz
    if vals.size != 3 * n:
        raise ValueError(f"CORNERS has {vals.size} values, expected {3 * n}")
    return vals[:n], vals[n:2 * n], vals[2 * n:]


def corners_to_coord_zcorn(X, Y, Z, nx, ny, nz, tol=1e-6):
    """Build COORD (pillars) and ZCORN from per-corner x,y,z. Raises if pillars are not straight."""
    shape = (nz, 2, ny, 2, nx, 2)
    X, Y, Z = (a.reshape(shape) for a in (X, Y, Z))
    coord = np.zeros((ny + 1, nx + 1, 6))
    for J in range(ny + 1):
        for I in range(nx + 1):
            # corners touching pillar (I, J): cells (I-1|I, J-1|J) at their matching corner
            pts_top, pts_bot = [], []
            for cj, jj in ((J - 1, 1), (J, 0)):
                for ci, ii in ((I - 1, 1), (I, 0)):
                    if 0 <= ci < nx and 0 <= cj < ny:
                        pts_top.append((X[0, 0, cj, jj, ci, ii], Y[0, 0, cj, jj, ci, ii], Z[0, 0, cj, jj, ci, ii]))
                        pts_bot.append((X[-1, 1, cj, jj, ci, ii], Y[-1, 1, cj, jj, ci, ii], Z[-1, 1, cj, jj, ci, ii]))
            top, bot = np.mean(pts_top, axis=0), np.mean(pts_bot, axis=0)
            coord[J, I] = [*top, *bot]
            # check every corner on this pillar lies on the straight line top->bot
            for cj, jj in ((J - 1, 1), (J, 0)):
                for ci, ii in ((I - 1, 1), (I, 0)):
                    if 0 <= ci < nx and 0 <= cj < ny:
                        for k in range(nz):
                            for f in (0, 1):
                                z = Z[k, f, cj, jj, ci, ii]
                                t = (z - top[2]) / (bot[2] - top[2]) if bot[2] != top[2] else 0.0
                                xe, ye = top[0] + t * (bot[0] - top[0]), top[1] + t * (bot[1] - top[1])
                                if abs(xe - X[k, f, cj, jj, ci, ii]) > tol or abs(ye - Y[k, f, cj, jj, ci, ii]) > tol:
                                    raise ValueError(f"Pillar ({I},{J}) is not straight at k={k}; grid needs a manual conversion")
    return coord.reshape(-1), Z.reshape(-1)


# ---------------------------------------------------------------- PVT / rock-fluid includes
def parse_pvt(path):
    text = strip_comments(Path(path).read_text())
    lines = text.splitlines()

    def table_after(header_regex, ncols):
        """Yield successive numeric tables following each header line."""
        tables, i = [], 0
        while i < len(lines):
            if re.match(header_regex, lines[i].strip(), re.I):
                rows, i = [], i + 1
                while i < len(lines):
                    tok = lines[i].split()
                    if tok and len(tok) == ncols and all(re.fullmatch(r"[-+0-9.eE]+", t) for t in tok):
                        rows.append([float(t) for t in tok])
                        i += 1
                    elif not tok:
                        i += 1
                    else:
                        break
                tables.append(np.array(rows))
            else:
                i += 1
        return tables

    pvt = table_after(r"PVT\b", 6)[0]        # p, Rs, Bo, Bg, viso, visg
    bot = table_after(r"BOT\b", 2)
    vot = table_after(r"VOT\b", 2)

    def scalar(key):
        pattern = r"\s+".join(key.split())     # tolerate any spacing between the words
        return float(re.search(rf"^\s*{pattern}\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))

    return dict(pvt=pvt, bot=bot, vot=vot,
                bwi=scalar("BWI"), cw=scalar("CW"), vwi=scalar("VWI"), cvw=scalar("CVW"), refpw=scalar("REFPW"),
                dens_o=scalar("DENSITY OIL"), dens_w=scalar("DENSITY WATER"), dens_g=scalar("DENSITY GAS"))


def parse_kr(path):
    text = strip_comments(Path(path).read_text())
    def block(name):
        m = re.search(rf"^\s*{name}\s*$(.*?)(?=^\s*(?:SLT|LIST|SWT|\Z))", text, re.I | re.M | re.S)
        return np.array([[float(x) for x in l.split()] for l in m.group(1).splitlines() if l.split()])
    return block("SWT"), block("SLT")


# ---------------------------------------------------------------- output formatting
def fmt(vals, per_line=6, prec=10):
    vals = list(vals)
    return "\n".join("  " + "  ".join(f"{v:.{prec}g}" for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line))


def ecl_date(y, m, d):
    return f"{d} {MONTHS[m - 1]} {y}"


# ---------------------------------------------------------------- schedule parsing
def parse_wells_and_schedule(lines):
    """Return (start_date, wells dict, events list).

    events: (date, kind, well, bhp_kg_cm2) with kind in {'date','target','shut','open','stop'}; DATEs without events are kept as ('date', ...).
    """
    run_idx = next(i for i, l in enumerate(lines) if l.strip().upper() == "RUN")
    body = [l.strip() for l in lines[run_idx + 1:] if l.strip()]
    wells, events, start, cur, current_well = {}, [], None, None, None
    i = 0
    while i < len(body):
        l, up = body[i], body[i].upper()
        if up.startswith("DATE"):
            y, m, d = map(int, l.split()[1:4])
            cur = date(y, m, d)
            start = start or cur
            events.append((cur, "date", None, None))
        elif up.startswith("WELL "):
            name = re.search(r"'([^']+)'", l).group(1)
            wells[name] = dict(name=name)
        elif up.startswith("PRODUCER"):
            name = re.search(r"'([^']+)'", l).group(1)
            wells[name]["type"] = "PROD"
        elif up.startswith("INJECTOR"):
            name = re.search(r"'([^']+)'", l).group(1)
            wells[name]["type"] = "INJ"
            fluid = re.match(r"INCOMP\s+(\w+)", body[i + 1], re.I)
            wells[name]["fluid"] = fluid.group(1).upper() if fluid else "WATER"
        elif up.startswith("OPERATE"):
            t = up.split()
            wells[name]["bhp"] = float(t[3])      # OPERATE MIN|MAX BHP value
        elif up.startswith("GEOMETRY"):
            t = l.split()
            wells[name]["rw"], wells[name]["geofac"], wells[name]["wfrac"], wells[name]["skin"] = map(float, t[2:6])
        elif up.startswith("PERF"):
            t = body[i + 1].split()
            k1, _, k2 = t[2].partition(":")        # 'k' or 'k1:k2' (a range of layers)
            for k in range(int(k1), int(k2 or k1) + 1):
                wells[name].setdefault("perfs", []).append((int(t[0]), int(t[1]), k))
            i += 1
        elif up.startswith("BHPDEPTH"):
            depth = float(l.split()[2])
            for w in wells.values():
                w["ref_depth"] = depth
        elif up.startswith("TARGET BHP"):
            w = re.search(r"'([^']+)'", l).group(1)
            events.append((cur, "target", w, float(body[i + 1].split()[0])))
            i += 1
        elif up.startswith("STOP"):
            events.append((cur, "stop", None, None))
        elif up.startswith(("SHUTIN", "OPEN")):
            target = re.search(r"'([^']+)'", l).group(1)
            if up.startswith("SHUTIN") and target == "*" and cur == start:
                pass                                   # initial SHUTIN '*': all wells start shut (handled in convert)
            else:
                events.append((cur, "shut" if up.startswith("SHUTIN") else "open", target, None))
        elif not (up.startswith(("GROUP", "INCOMP")) or not re.match(r"[A-Z]", up)):
            print(f"WARNING: unhandled schedule keyword ignored: {l}", file=sys.stderr)
        i += 1
    return start, wells, events


# ---------------------------------------------------------------- main conversion
def convert(dat_path, out_dir=None, wi_scale=1.0, restart=True, crossflow=False):
    dat_path = Path(dat_path).resolve()
    base = dat_path.parent
    text = strip_comments(dat_path.read_text())
    lines = text.splitlines()
    name = dat_path.stem
    out_dir = Path(out_dir) if out_dir else base.parent / f"{base.name}_opm" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = map(int, re.search(r"^\s*CORNER\s+(\d+)\s+(\d+)\s+(\d+)", text, re.I | re.M).groups())
    title = re.search(r"TITLE1\s+'([^']*)'", text, re.I).group(1)
    por = float(re.search(r"^\s*POR\s+CON\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    permk = float(re.search(r"^\s*PERMK\s+CON\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    cpor = float(re.search(r"^\s*CPOR\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    prpor = float(re.search(r"^\s*PRPOR\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    refdepth = float(re.search(r"^\s*REFDEPTH\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    refpres = float(re.search(r"^\s*REFPRES\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    pb = float(re.search(r"^\s*PB\s+CON\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))
    dwoc = float(re.search(r"^\s*DWOC\s+([-+0-9.eE]+)", text, re.I | re.M).group(1))

    corner_line = next(i for i, l in enumerate(lines) if re.match(r"\s*CORNER\s", l, re.I))
    corners_inc = (base / re.match(r"\s*INCLUDE\s+'([^']+)'", lines[corner_line + 1], re.I).group(1)).resolve()
    permi_inc = find_include(lines, r"PERMI\s+ALL", base)
    permj_inc = find_include(lines, r"PERMJ\s+ALL", base)
    pvt_inc = next(p for p in all_includes(lines, base) if p.name.upper().startswith("PVT"))
    kr_inc = next(p for p in all_includes(lines, base) if p.name.upper().startswith("KR"))

    # ---- GRID
    X, Y, Z = parse_corners(corners_inc, nx, ny, nz)
    coord, zcorn = corners_to_coord_zcorn(X, Y, Z, nx, ny, nz)
    permi, permj = read_numbers(permi_inc), read_numbers(permj_inc)
    ncell = nx * ny * nz
    assert permi.size == ncell and permj.size == ncell, "perm size mismatch"

    # TRANSI/J/K IJK -> MULTX/Y/Z on a BOX (IMEX and ECLIPSE both apply it to the +I/+J/+K face)
    mult = []
    for axis in "IJK":
        m = re.search(rf"^\s*TRANS{axis}\s+IJK\s*\n((?:[ \t]*\d+(?::\d+)?[ \t]+\d+(?::\d+)?[ \t]+\d+(?::\d+)?[ \t]+[-+0-9.eE]+[ \t]*\n)+)",
                      text, re.I | re.M)
        if m:
            for row in m.group(1).strip().splitlines():
                *rng, val = row.split()
                lo_hi = [tuple(int(v) for v in r.split(":")) * (2 if ":" not in r else 1) for r in rng]
                (i1, i2), (j1, j2), (k1, k2) = lo_hi
                n = (i2 - i1 + 1) * (j2 - j1 + 1) * (k2 - k1 + 1)
                mult.append(f"BOX\n  {i1} {i2} {j1} {j2} {k1} {k2} /\nMULT{'XYZ'['IJK'.index(axis)]}\n  {n}*{val} /\nENDBOX\n")
    (out_dir / "GRID.INC").write_text(
        f"COORD\n{fmt(coord, 6, 12)}\n/\n\nZCORN\n{fmt(zcorn, 8, 10)}\n/\n\n"
        f"PORO\n  {ncell}*{por}\n/\n\nPERMX\n{fmt(permi, 8, 8)}\n/\n\nPERMY\n{fmt(permj, 8, 8)}\n/\n\nPERMZ\n  {ncell}*{permk}\n/\n\n" + "\n".join(mult))

    # ---- PROPS
    pv = parse_pvt(pvt_inc)
    P = pv["pvt"]
    nrs = len(P)
    pvto = []
    for i in range(nrs):
        p_b, rs, bo, mu = P[i, 0] * KGCM2_TO_BAR, P[i, 1], P[i, 2], P[i, 4]
        rec = [f"  {rs:.6g}  {p_b:.6g}  {bo:.6g}  {mu:.6g}"]
        if i < len(pv["bot"]):
            bt, vt = pv["bot"][i], pv["vot"][i]
            assert abs(bt[0, 0] - P[i, 0]) < 1e-9 and abs(bt[0, 1] - bo) < 1e-9, f"BOT {i} does not start at the saturated point"
            for (pp, b), (_, v) in zip(bt[1:], vt[1:]):
                rec.append(f"        {pp * KGCM2_TO_BAR:.6g}  {b:.6g}  {v:.6g}")
        else:
            # CMG gives no BOT/VOT for the highest-Rs rows; OPM needs one undersaturated node per record.
            # Extrapolate 50 bar using the relative Bo/viscosity slopes of the last tabulated branch.
            bt, vt = pv["bot"][-1], pv["vot"][-1]
            dp_ref = (bt[-1, 0] - bt[0, 0]) * KGCM2_TO_BAR
            c_o = (bt[0, 1] - bt[-1, 1]) / bt[0, 1] / dp_ref
            c_mu = (vt[-1, 1] - vt[0, 1]) / vt[0, 1] / dp_ref
            rec.append(f"        {p_b + 50:.6g}  {bo * (1 - c_o * 50):.6g}  {mu * (1 + c_mu * 50):.6g}")
        pvto.append("\n".join(rec) + " /")
    pvdg = "\n".join(f"  {r[0] * KGCM2_TO_BAR:.6g}  {r[3]:.6g}  {r[5]:.6g}" for r in P)

    swt, slt = parse_kr(kr_inc)
    swof = "\n".join(f"  {r[0]:.6g}  {r[1]:.6g}  {r[2]:.6g}  {r[3] * KGCM2_TO_BAR:.6g}" for r in swt)
    sgof_rows = slt[::-1]           # Sl descending -> Sg ascending
    sgof = "\n".join(f"  {1 - r[0]:.6g}  {r[1]:.6g}  {r[2]:.6g}  {r[3] * KGCM2_TO_BAR:.6g}" for r in sgof_rows)

    cw_bar = pv["cw"] / KGCM2_TO_BAR
    (out_dir / "PROPS.INC").write_text(f"""\
-- Converted from {pvt_inc.name} and {kr_inc.name}. Pressures kg/cm2 -> bar.
DENSITY
  {pv['dens_o']:.7g}  {pv['dens_w']:.7g}  {pv['dens_g']:.7g} /

PVTW
  {pv['refpw'] * KGCM2_TO_BAR:.6g}  {pv['bwi']:.6g}  {cw_bar:.6g}  {pv['vwi']:.6g}  {pv['cvw']:.6g} /

ROCK
  {prpor * KGCM2_TO_BAR:.6g}  {cpor / KGCM2_TO_BAR:.6g} /

PVTO
{chr(10).join(pvto)}
/

PVDG
{pvdg}
/

STONE2

SWOF
{swof}
/

SGOF
{sgof}
/
""")

    # ---- SOLUTION
    (out_dir / "SOLUTION.INC").write_text(f"""\
-- IMEX: VERTICAL BLOCK_CENTER WATER_OIL, REFDEPTH/REFPRES, DWOC {dwoc:g}, PB CON {pb:g} kg/cm2
EQUIL
  {refdepth:g}  {refpres * KGCM2_TO_BAR:.6g}  {dwoc:g}  0  0  0  1  0  0 /

PBVD
  0     {pb * KGCM2_TO_BAR:.6g}
  20000 {pb * KGCM2_TO_BAR:.6g} /
""")

    # ---- SCHEDULE
    start, wells, events = parse_wells_and_schedule(lines)
    sched = (["RPTRST\n  BASIC=2 /\n"] if restart else []) + ["TUNING\n  1*  1  0.001 /\n/\n/\n"]
    sched.append("WELSPECS\n" + "\n".join(
        f"  '{w['name']}'  'FIELD'  {w['perfs'][0][0]}  {w['perfs'][0][1]}  {w['ref_depth']:g}  "
        f"{'OIL' if w['type'] == 'PROD' else 'WATER'}  1*  1*  1*  {'YES' if crossflow else 'NO'} /"
        for w in wells.values()) + "\n/\n")
    compdat = []
    for w in wells.values():
        dia = 2 * w["rw"]
        for (i, j, k) in w["perfs"]:
            compdat.append(f"  '{w['name']}'  {i}  {j}  {k}  {k}  OPEN  1*  1*  {dia:.6g}  1*  {w['skin']:g} /")
    sched.append("COMPDAT\n" + "\n".join(compdat) + "\n/\n")
    if wi_scale != 1.0:
        sched.append("-- WI multiplier calibrated to CMG\nWPIMULT\n" + "\n".join(f"  '{w}'  {wi_scale:.6g} /" for w in wells) + "\n/\n")

    def control(w, bhp_kg, status):
        bhp = bhp_kg * KGCM2_TO_BAR
        if w["type"] == "PROD":
            return f"WCONPROD\n  '{w['name']}'  {status}  BHP  5*  {bhp:.6g} /\n/\n"
        return f"WCONINJE\n  '{w['name']}'  {w['fluid']}  {status}  BHP  2*  {bhp:.6g} /\n/\n"

    # all wells start shut (SHUTIN '*'); the first TARGET for a well opens it
    for w in wells.values():
        sched.append(control(w, w["bhp"], "SHUT"))
    opened = set()
    for d, kind, wname, bhp in events:
        if kind == "date":
            if d != start:
                sched.append(f"DATES\n  {ecl_date(d.year, d.month, d.day)} /\n/\n")
        elif kind in ("shut", "open"):
            names = list(wells) if wname == "*" else [wname]
            sched.append("WELOPEN\n" + "".join(f"  '{n}'  {kind.upper()} /\n" for n in names) + "/\n")
        elif kind == "target":
            opened.add(wname)
            sched.append(control(wells[wname], bhp, "OPEN"))
    sched.append("END\n")
    (out_dir / "SCHEDULE.INC").write_text("\n".join(sched))

    data = f"""\
-- Converted from CMG IMEX deck {dat_path.name} by cmg_to_opm.py
RUNSPEC
TITLE
  {title}

DIMENS
  {nx} {ny} {nz} /

OIL
WATER
GAS
DISGAS
METRIC
UNIFOUT

TABDIMS
  1 1 50 50 /

EQLDIMS
  1 100 50 1 1 /

WELLDIMS
  {max(len(wells), 1)} {max(len(wells[w]['perfs']) for w in wells)} 1 {len(wells)} /

START
  {ecl_date(start.year, start.month, start.day)} /

GRID
INIT
INCLUDE
  'GRID.INC' /

PROPS
INCLUDE
  'PROPS.INC' /

SOLUTION
INCLUDE
  'SOLUTION.INC' /

SUMMARY
FOPR
FWPR
FWIR
FOPT
FWPT
FWIT
FPR
WBHP
/
WOPR
/
WWPR
/
WWIR
/
WOPT
/
WWPT
/
WWIT
/
-- per-connection vectors (all connections of all wells), read by the OPM -> training-data converter
CTFAC
'*' /
/
COPT
'*' /
/
CWPT
'*' /
/
CWIT
'*' /
/
CGPT
'*' /
/
CPR
'*' /
/

SCHEDULE
INCLUDE
  'SCHEDULE.INC' /
"""
    data_path = out_dir / f"{name.upper()}.DATA"
    data_path.write_text(data)
    return data_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dat")
    ap.add_argument("-o", "--out-dir")
    ap.add_argument("--no-restart", action="store_true", help="do not write grid restart output every report step (much faster)")
    ap.add_argument("--crossflow", action="store_true",
                    help="allow crossflow in the wells (WELSPECS item 10 = YES). Default: no crossflow, a well whose "
                         "flow would reverse is shut until its next control")
    ap.add_argument("--wi-scale", type=float, default=1.0, help="multiplier on Peaceman well index (WPIMULT)")
    a = ap.parse_args()
    print(convert(a.dat, a.out_dir, a.wi_scale, restart=not a.no_restart, crossflow=a.crossflow))
