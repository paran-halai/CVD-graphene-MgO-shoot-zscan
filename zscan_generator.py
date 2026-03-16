"""
Z-scan generator for C21 cluster above MgO slab.

Places a carbon-21 cluster at varying heights above an MgO surface
to construct an energy vs distance curve. Generates VASP POSCAR files
for single-point energy calculations at each distance.

The surface top is identified by filtering atoms with fractional z < 0.5,
which correctly handles periodic boundary wrapping in slab models.

Author: Paran
"""

import numpy as np
import os


def read_poscar(path):
    """Read a VASP POSCAR/CONTCAR file and return structure data."""
    with open(path) as f:
        lines = [l.rstrip() for l in f if l.strip()]

    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale

    symbols = lines[5].split()
    counts = list(map(int, lines[6].split()))

    idx = 7
    selective = False
    if lines[idx].lower().startswith("s"):
        selective = True
        idx += 1

    coord_type = lines[idx].strip().lower()
    idx += 1

    n_atoms = sum(counts)
    coords, flags = [], []
    for i in range(n_atoms):
        parts = lines[idx + i].split()
        coords.append([float(p) for p in parts[:3]])
        if selective and len(parts) >= 6:
            flags.append(parts[3:6])
        else:
            flags.append(["T", "T", "T"])

    coords = np.array(coords)
    if coord_type.startswith("c"):
        cart = coords * scale
        frac = cart @ np.linalg.inv(lattice)
    else:
        frac = coords
        cart = frac @ lattice

    return {
        "lattice": lattice, "symbols": symbols, "counts": counts,
        "frac": frac, "cart": cart, "flags": flags
    }


def write_poscar(path, lattice, symbols, counts, frac, flags, title):
    """Write a VASP POSCAR file with selective dynamics."""
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write("1.0\n")
        for v in lattice:
            f.write(f"  {v[0]:20.16f}  {v[1]:20.16f}  {v[2]:20.16f}\n")
        f.write("  ".join([""] + symbols) + "\n")
        f.write("  ".join([""] + [str(c) for c in counts]) + "\n")
        f.write("Selective dynamics\n")
        f.write("Direct\n")
        for i, r in enumerate(frac):
            f.write(f"  {r[0]:20.16f}  {r[1]:20.16f}  {r[2]:20.16f}"
                    f"  {flags[i][0]}  {flags[i][1]}  {flags[i][2]}\n")


def find_surface_top(frac_z, cart_z):
    """
    Find the true surface top of a slab model.

    In slab calculations with vacuum, some atoms at the bottom of the
    slab may wrap through the periodic boundary, appearing at high z
    values (frac_z ~ 1.0). These are filtered out by only considering
    atoms with fractional z < 0.5.
    """
    mask = frac_z < 0.5
    return cart_z[mask].max()


def generate_zscan(slab_file, c21_file, output_dir,
                   gap_start=1.5, gap_end=8.0, gap_step=0.2):
    """
    Generate POSCAR files with C21 placed at varying heights above MgO.

    Parameters
    ----------
    slab_file : str
        Path to the optimised MgO slab POSCAR.
    c21_file : str
        Path to the C21 cluster POSCAR.
    output_dir : str
        Directory for output POSCAR files.
    gap_start, gap_end : float
        Distance range in Angstroms (bottom of C21 to top of slab).
    gap_step : float
        Step size in Angstroms.
    """
    slab = read_poscar(slab_file)
    c21 = read_poscar(c21_file)

    lattice = slab["lattice"]
    inv_lat = np.linalg.inv(lattice)

    # identify true surface top, accounting for periodic wrapping
    slab_zmax = find_surface_top(slab["frac"][:, 2], slab["cart"][:, 2])

    # C21 geometry
    c21_zmin = c21["cart"][:, 2].min()
    c21_com = c21["cart"].mean(axis=0)

    # centre C21 over the slab in x, y
    cell_centre = np.array([lattice[0, 0] / 2, lattice[1, 1] / 2])
    shift_xy = cell_centre - c21_com[:2]

    os.makedirs(output_dir, exist_ok=True)
    gaps = np.round(np.arange(gap_start, gap_end + 1e-3, gap_step), 1)

    print(f"Slab surface top: {slab_zmax:.4f} A")
    print(f"Generating {len(gaps)} POSCAR files ({gap_start}-{gap_end} A)\n")

    for gap in gaps:
        # shift C21 so its lowest atom sits 'gap' above the surface
        shift_z = (slab_zmax + gap) - c21_zmin
        shift = np.array([shift_xy[0], shift_xy[1], shift_z])

        c21_cart = c21["cart"] + shift
        c21_frac = c21_cart @ inv_lat
        c21_frac[:, 0] %= 1.0
        c21_frac[:, 1] %= 1.0

        # combine slab + C21
        combined_symbols = slab["symbols"][:]
        combined_counts = slab["counts"][:]
        for s, c in zip(c21["symbols"], c21["counts"]):
            if s in combined_symbols:
                combined_counts[combined_symbols.index(s)] += c
            else:
                combined_symbols.append(s)
                combined_counts.append(c)

        combined_frac = np.vstack([slab["frac"], c21_frac])

        # freeze slab atoms, allow C21 to move
        combined_flags = [["F", "F", "F"]] * sum(slab["counts"])
        combined_flags += [["T", "T", "T"]] * len(c21_frac)

        fname = os.path.join(output_dir, f"POSCAR_{gap:.1f}A.vasp")
        title = f"MgO + C21 at {gap:.1f} A above surface"
        write_poscar(fname, lattice, combined_symbols, combined_counts,
                     combined_frac, combined_flags, title)
        print(f"  {gap:.1f} A  ->  {fname}")

    print(f"\nDone. {len(gaps)} files written to {output_dir}/")


if __name__ == "__main__":
    # --- edit these paths ---
    slab_file = "POSCAR_MgO.vasp"
    c21_file = "POSCAR_C21_only.vasp"
    output_dir = "zscan_output"

    generate_zscan(slab_file, c21_file, output_dir,
                   gap_start=1.5, gap_end=8.0, gap_step=0.2)
