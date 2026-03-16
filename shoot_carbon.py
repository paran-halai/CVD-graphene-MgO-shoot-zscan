"""
Carbon atom shooting script for graphene growth simulations.

Adds a single carbon atom to an existing carbon cluster on an MgO
surface. The new atom is placed at a bonding distance (1.6-1.9 A)
from an existing carbon atom, with constraints to prevent bonding
directly to the MgO substrate.

Designed for use with VASP CONTCAR/POSCAR files containing an
MgO slab with a carbon cluster (species order: Mg, O, C).
"""

import numpy as np
import random
import os
import sys


def read_poscar(path):
    """Read a VASP POSCAR/CONTCAR file and return structure data."""
    with open(path) as f:
        lines = [l.rstrip() for l in f.readlines()]

    scale = float(lines[1])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale

    symbols = lines[5].split()
    counts = list(map(int, lines[6].split()))

    idx = 7
    selective = False
    if lines[idx].strip().lower().startswith("s"):
        selective = True
        idx += 1
    idx += 1  # skip Direct/Cartesian line

    n_atoms = sum(counts)
    frac, flags = [], []
    for i in range(n_atoms):
        parts = lines[idx + i].split()
        frac.append([float(p) for p in parts[:3]])
        if selective and len(parts) >= 6:
            flags.append(parts[3:6])
        else:
            flags.append(["T", "T", "T"])

    frac = np.array(frac)
    cart = frac @ lattice

    return {
        "lattice": lattice, "symbols": symbols, "counts": counts,
        "frac": frac, "cart": cart, "flags": flags
    }


def min_image_distance(r1, r2, lattice):
    """Minimum image convention distance between two Cartesian points."""
    inv_lat = np.linalg.inv(lattice)
    d_frac = (r1 - r2) @ inv_lat
    d_frac -= np.round(d_frac)
    return np.linalg.norm(d_frac @ lattice)


def shoot_carbon(data, min_bond=1.6, max_bond=1.9, min_clash=1.2,
                 max_attempts=500000):
    """
    Place a new carbon atom near the existing cluster.

    The placement uses random fractional coordinates with the
    following constraints:
      - Must be within bonding distance (1.6-1.9 A) of an existing C
      - Must not clash with any atom (< 1.2 A)
      - Must be above the MgO surface
      - Blocks placement directly below the cluster centre of mass
        to prevent the cluster growing into the substrate

    Returns fractional coordinates of the new atom, or None on failure.
    """
    lattice = data["lattice"]
    all_cart = data["cart"]

    # carbon atoms are the third species (Mg, O, C)
    c_idx = data["symbols"].index("C")
    c_start = sum(data["counts"][:c_idx])
    c_end = c_start + data["counts"][c_idx]
    c_cart = all_cart[c_start:c_end]
    c_frac = data["frac"][c_start:c_end]

    # centre of mass in fractional coords
    com = c_frac.mean(axis=0)

    # find true slab top (filter periodic boundary wrapping)
    slab_frac_z = data["frac"][:c_start, 2]
    slab_cart_z = all_cart[:c_start, 2]
    slab_top = slab_cart_z[slab_frac_z < 0.5].max()
    slab_top_frac = slab_top / lattice[2, 2]

    for attempt in range(max_attempts):
        sx = random.uniform(0, 1)
        sy = random.uniform(0, 1)
        sz = random.uniform(slab_top_frac + 0.02, 0.9)

        # block placement directly under the cluster COM
        # (prevents growth into the MgO substrate)
        in_x = (com[0] / 2) < sx < (com[0] + com[0] / 2)
        in_y = (com[1] / 2) < sy < (com[1] + com[1] / 2)
        if in_x and in_y and sz < com[2]:
            continue

        new_frac = np.array([sx, sy, sz])
        new_cart = new_frac @ lattice

        if new_cart[2] < slab_top + 0.5:
            continue

        # check for clashes with all atoms
        clash = False
        for atom in all_cart:
            if min_image_distance(new_cart, atom, lattice) < min_clash:
                clash = True
                break
        if clash:
            continue

        # check bonding distance to carbon atoms
        for c_atom in c_cart:
            d = min_image_distance(new_cart, c_atom, lattice)
            if min_bond <= d <= max_bond:
                return new_frac

    return None


def write_new_poscar(data, new_frac, output_file):
    """Write POSCAR with the additional carbon atom."""
    lattice = data["lattice"]
    symbols = data["symbols"]
    counts = data["counts"][:]
    frac = data["frac"].copy()
    flags = data["flags"][:]

    c_idx = symbols.index("C")
    old_n = counts[c_idx]
    counts[c_idx] += 1

    insert_pos = sum(counts[:c_idx]) + old_n
    frac = np.insert(frac, insert_pos, new_frac, axis=0)
    flags.insert(insert_pos, ["T", "T", "T"])

    with open(output_file, "w") as f:
        f.write(f"MgO + C{counts[c_idx]}\n")
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

    return counts[c_idx]


if __name__ == "__main__":
    input_file = "CONTCAR"
    output_file = "POSCAR_new"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        sys.exit(1)

    data = read_poscar(input_file)
    c_idx = data["symbols"].index("C")
    n_carbon = data["counts"][c_idx]
    print(f"Current system: C{n_carbon} on MgO ({sum(data['counts'])} atoms)")

    new_frac = shoot_carbon(data)

    if new_frac is not None:
        new_cart = new_frac @ data["lattice"]
        new_n = write_new_poscar(data, new_frac, output_file)
        print(f"Added carbon atom -> C{new_n}")
        print(f"  Position: ({new_cart[0]:.3f}, {new_cart[1]:.3f}, {new_cart[2]:.3f}) A")
        print(f"  Written to: {output_file}")
    else:
        print("Failed to place atom. Run again for a different random attempt.")
