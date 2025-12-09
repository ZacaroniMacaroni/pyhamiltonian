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

# Usage
For simple Hamiltonian initialization and construction, see the example code below.
```aiignore
from pyhamiltonian import ManyBodyHamiltonian, Lattice

# Lattice Initialization
nlen = 2  # Number of sites along each lattice dimension
ndim = 2  # Number of lattice dimensions
base_energy = 0.  # Base site energies
disorder_type = 'uniform'  # Energy disorder distribution type
disorder_strength = 0.5  # Strength of energy disorder

lattice = Lattice(nlen, ndim, base_energy, disorder_strength, disorder_type)

# Hamiltonian initialization
nparticles = 2  # Number of particles
base_coupling = 1.  # Base couplings between sites
coupling_type = 'nearest neighbor'

hamiltonian = ManyBodyHamiltonian(lattice, nparticles, base_coupling, coupling_type)

# Array construction
hamiltonian.construct_hamiltonian()
```

# Future Features
- [ ] Add support for arbitrary many-body interactions.
- [ ] Add support for fermionic creation/annihilation operators.
- [ ] Add support for arbitrary lattice symmetries.
- [ ] Develop a GUI for interactive visualization of Hamiltonian and eigenstate decomposition.
- [ ] Add multidimensional N-particle plotting features.