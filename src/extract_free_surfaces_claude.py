#!/usr/bin/env python3
"""
extract_free_surface.py

Collect the free-surface profile h(r) from every VTU file written by
rod_climbing_giesekus.py and write two CSV files:

  free_surface_profiles.csv   one row per (output file, surface node)
  climb_heights.csv           one row per output file: height at the rod wall,
                              at the outer wall, and a few summary numbers

Usage
-----
    python extract_free_surface.py [OUTPUT_DIR] [options]

OUTPUT_DIR is the OUTPUT_DIRECTORY of the simulation (default:
rod_climbing_output); it must contain domain/domain_000000.vtu, ...

The VTU file names do not contain Omega, so the script reconstructs the
continuation values s = Omega/Omega_target exactly as solve_continuation()
generates them:

    s_i = s_initial + (1 - s_initial) * i / (n_steps - 1),  i = 0..n_steps-1

and, when SUPG is used (default), the FIRST value is solved and written twice
(SUPG off, then SUPG on).  So with the default settings there are
n_steps + 1 files, files 0 and 1 are both s = s_initial, and by default the
superseded SUPG-off file (file 0) is dropped from the CSVs.  Use
--keep-duplicate to keep it.

All numbers are DIMENSIONLESS (lengths in units of the rod radius a) unless
--rod-radius / --omega-target are given, in which case extra dimensional
columns are added.

Requires: numpy, meshio.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# numpy >= 2 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

try:
    import meshio
except ImportError:  # pragma: no cover
    sys.exit("This script needs meshio:  pip install meshio")


# --------------------------------------------------------------------------- #
# Continuation bookkeeping
# --------------------------------------------------------------------------- #

def continuation_values(s_initial: float, n_steps: int):
    """Same linear sequence as RodClimbingGiesekusProblem.solve_continuation."""
    if n_steps <= 1:
        return [1.0]
    return np.geomspace(s_initial, 1, n_steps)


def file_labels(n_files: int, s_initial: float, n_steps: int, supg_duplicate: bool):
    """
    Return a list of (s, is_supg_off_duplicate) for each file index, or raise if
    the number of files does not match the expected continuation.
    """
    s_vals = continuation_values(s_initial, n_steps)
    labels = []
    if supg_duplicate:
        labels.append((s_vals[0], True))      # solved with SUPG off, then superseded
    for s in s_vals:
        labels.append((s, False))
    if len(labels) != n_files:
        raise SystemExit(
            f"Found {n_files} VTU files but the continuation settings "
            f"(--s-initial {s_initial}, --n-steps {n_steps}, "
            f"supg duplicate={'yes' if supg_duplicate else 'no'}) predict "
            f"{len(labels)}.\n"
            "Check --s-initial/--n-steps, use --no-supg-duplicate if USE_SUPG was "
            "False, and make sure WRITE_EVERY_STEP was True and the output "
            "directory only contains one run."
        )
    return labels


# --------------------------------------------------------------------------- #
# Free-surface extraction
# --------------------------------------------------------------------------- #

# VTK quad9: corners 0-3, edge midpoints 4 (0-1), 5 (1-2), 6 (2-3), 7 (3-0)
_QUAD9_EDGES = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]


def free_surface_nodes(mesh):
    """
    Return the node indices of the free surface ordered from the rod contact line
    (smallest r) to the outer-wall contact line (largest r).

    The boundary edges of the quad mesh are walked from the top-left corner to
    the top-right corner without going down a wall.  This does not rely on node
    ordering or on the (ALE-perturbed) z coordinates.
    """
    pts = mesh.points
    quads = None
    for block in mesh.cells:
        if block.type == "quad9":
            quads = block.data
    if quads is None:
        raise RuntimeError("Expected quad9 cells in the VTU file "
                           "(this script was written for pyoomph C2 output).")

    # count edges (by corner pair) to find the boundary
    count = {}
    for q in quads:
        for a, m, b in _QUAD9_EDGES:
            key = frozenset((q[a], q[b]))
            count[key] = count.get(key, 0) + 1

    adj = {}
    for q in quads:
        for a, m, b in _QUAD9_EDGES:
            if count[frozenset((q[a], q[b]))] == 1:
                for u, v in ((q[a], q[m]), (q[m], q[b])):
                    adj.setdefault(u, []).append(v)
                    adj.setdefault(v, []).append(u)

    bnodes = np.array(sorted(adj))
    r = pts[:, 0]
    z = pts[:, 1]
    rmin, rmax = r[bnodes].min(), r[bnodes].max()
    tol = 1e-9 * (rmax - rmin)

    left = bnodes[np.abs(r[bnodes] - rmin) < tol]     # rod wall nodes
    right = bnodes[np.abs(r[bnodes] - rmax) < tol]    # outer wall nodes
    start = left[np.argmax(z[left])]                  # rod contact line
    end = right[np.argmax(z[right])]                  # outer-wall contact line

    # walk along the boundary from start, never stepping onto the rod wall
    path = [start]
    prev = None
    cur = start
    while cur != end:
        if cur == start:
            # leave the contact-line corner along the interface, not down the rod wall
            nxt = [n for n in adj[cur] if abs(r[n] - rmin) >= tol]
        else:
            nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            raise RuntimeError("Boundary walk got stuck; unexpected mesh topology.")
        prev, cur = cur, nxt[0]
        path.append(cur)
        if len(path) > len(bnodes) + 1:
            raise RuntimeError("Boundary walk did not reach the outer wall.")

    path = np.array(path)
    # sanity: r must increase monotonically along a graph-like free surface
    order = np.argsort(r[path])
    if not np.array_equal(order, np.arange(len(path))):
        path = path[order]
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output_dir", nargs="?", default="rod_climbing_output")
    ap.add_argument("--s-initial", type=float, default=0.1,
                    help="OMEGA_INITIAL/OMEGA_TARGET (default 0.02)")
    ap.add_argument("--n-steps", type=int, default=11,
                    help="N_OMEGA_STEPS (default 8)")
    ap.add_argument("--no-supg-duplicate", action="store_true",
                    help="USE_SUPG was False: the first s value is written only once")
    ap.add_argument("--keep-duplicate", action="store_true",
                    help="also write the superseded SUPG-off file (first s value)")
    ap.add_argument("--rod-radius", type=float, default=None,
                    help="ROD_RADIUS: adds dimensional r and h columns")
    ap.add_argument("--omega-target", type=float, default=None,
                    help="OMEGA_TARGET: adds a dimensional Omega column")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the CSVs (default: OUTPUT_DIR)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.output_dir, "domain", "domain_*.vtu")))
    if not files:
        sys.exit(f"No VTU files found in {args.output_dir}/domain/")

    labels = file_labels(len(files), args.s_initial, args.n_steps,
                         supg_duplicate=not args.no_supg_duplicate)

    out_dir = args.out_dir or args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    prof_path = os.path.join(out_dir, "free_surface_profiles.csv")
    clim_path = os.path.join(out_dir, "climb_heights.csv")

    a = args.rod_radius
    prof_cols = ["file_index", "file", "omega_factor", "r", "h"]
    clim_cols = ["file_index", "file", "omega_factor",
                 "h_rod", "h_outer", "h_rod_minus_h_outer", "h_mean_area_weighted",
                 "h_max", "h_min", "n_surface_nodes"]
    if args.omega_target is not None:
        prof_cols.insert(3, "omega")
        clim_cols.insert(3, "omega")
    if a is not None:
        prof_cols += ["r_dimensional", "h_dimensional"]
        clim_cols += ["h_rod_dimensional", "h_outer_dimensional"]

    n_written = 0
    with open(prof_path, "w", newline="") as fp, open(clim_path, "w", newline="") as fc:
        wp = csv.writer(fp)
        wc = csv.writer(fc)
        wp.writerow(prof_cols)
        wc.writerow(clim_cols)

        for idx, (f, (s, is_dup)) in enumerate(zip(files, labels)):
            if is_dup and not args.keep_duplicate:
                continue

            mesh = meshio.read(f)
            nodes = free_surface_nodes(mesh)
            r = mesh.points[nodes, 0]
            h = mesh.points[nodes, 1]       # interface height; initial flat level is z = 0

            fname = os.path.basename(f)
            omega = None if args.omega_target is None else s * args.omega_target

            for ri, hi in zip(r, h):
                row = [idx, fname, f"{s:.10g}"]
                if omega is not None:
                    row.append(f"{omega:.10g}")
                row += [f"{ri:.10g}", f"{hi:.10g}"]
                if a is not None:
                    row += [f"{ri * a:.10g}", f"{hi * a:.10g}"]
                wp.writerow(row)

            h_rod, h_out = h[0], h[-1]
            h_mean = _trapz(h * r, r) / _trapz(r, r)
            row = [idx, fname, f"{s:.10g}"]
            if omega is not None:
                row.append(f"{omega:.10g}")
            row += [f"{h_rod:.10g}", f"{h_out:.10g}", f"{h_rod - h_out:.10g}",
                    f"{h_mean:.10g}", f"{h.max():.10g}", f"{h.min():.10g}", len(nodes)]
            if a is not None:
                row += [f"{h_rod * a:.10g}", f"{h_out * a:.10g}"]
            wc.writerow(row)
            n_written += 1

    print(f"Processed {n_written} of {len(files)} files.")
    print(f"  profiles : {prof_path}")
    print(f"  climbs   : {clim_path}")
    print("Dimensionless units (lengths in rod radii) unless --rod-radius was given.")


if __name__ == "__main__":
    main()
