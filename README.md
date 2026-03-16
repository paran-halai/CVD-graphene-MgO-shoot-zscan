Python scripts for simulating the initial stages of graphene growth on MgO via chemical vapour deposition (CVD), using VASP with machine learning force fields (MLFF).

Developed as part of the CHE700 research project at Queen Mary University of London.

## Scripts

### `zscan_generator.py`

Generates VASP POSCAR files for an energy vs distance (E vs z) scan of a carbon-21 cluster above an MgO slab. Used to determine the optimal adsorption height before running molecular dynamics.

The script correctly identifies the true surface top by filtering out atoms that wrap through the periodic boundary — a common issue in slab models where bottom-layer atoms appear at high z values due to periodic boundary conditions.

**Usage:**
```python
from zscan_generator import generate_zscan

generate_zscan(
    slab_file="POSCAR_MgO.vasp",
    c21_file="POSCAR_C21_only.vasp",
    output_dir="zscan_output",
    gap_start=1.5,  # Angstrom
    gap_end=8.0,
    gap_step=0.2
)
```

### `shoot_carbon.py`

Adds a single carbon atom to an existing carbon cluster on MgO. Used iteratively to grow the cluster from C21 to C30, following a cycle of: shoot → optimise → MD → optimise → shoot.

The atom is placed at a C-C bonding distance (1.6–1.9 Å) from an existing carbon, with constraints to prevent the cluster growing into the MgO substrate.

**Usage:**
```bash
cp path/to/CONTCAR ./CONTCAR
python shoot_carbon.py
# produces POSCAR_new with one additional carbon
```

## VASP Settings

### Single-point energy (E vs z scan)
```
IBRION = -1
NSW    = 0
```

### Geometry optimisation
```
IBRION = 2
NSW    = 100
POTIM  = 0.5
EDIFFG = -0.05
```

### Molecular dynamics (300 K, 20 ps)
```
IBRION = 0
NSW    = 10000
POTIM  = 2
TEBEG  = 300
TEEND  = 300
SMASS  = 1
```

All calculations used ENCUT = 400 eV, KPOINTS 2×2×1, PBE with D3 dispersion (IVDW = 11), and VASP 6.4.3 with MLFF.

## Requirements

- Python 3.6+
- NumPy
- VASP 6.4.x with MLFF support
- G. Kresse and J. Hafner, *Phys. Rev. B*, 1993, **47**, 558–561
- Vienna Ab Initio Simulation Package, https://www.vasp.at/
