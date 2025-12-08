# pyhamiltonian
N-particle Hamiltonian constructor in arbitrary-dimensional lattice space.

# Purpose
Generating Hamiltonians for N-particles while correctly accounting for couplings, many-body interactions, and dimensionality is difficult to generalize _a priori_. This constructor facilitates construction of Hamiltonians with arbitrary lattice dimension and number of particles. Furthermore, this class contains plotting features to observe eigenstate population features for arbitrary dimensions or numbers of particles.
Note: at present, this constructor assumes indistinguishable bosons with "hard core" exclusion.

# Installation Instructions
To install, run the following commands in a Python environment of your choice.
```aiignore
git clone https://github.com/ZacaroniMacaroni/pyhamiltonian.git
cd pyhamiltonian
pip install -e .
```

# Future Features
- [ ] Add support for arbitrary many-body interactions.
- [ ] Add support for fermionic creation/annihilation operators.
- [ ] Add support for arbitrary lattice symmetries.
- [ ] Develop a GUI for interactive visualization of Hamiltonian and eigenstate decomposition.
- [ ] Add multidimensional N-particle plotting features.