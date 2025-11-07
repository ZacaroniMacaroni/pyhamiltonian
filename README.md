# pyhamiltonian
N-particle Hamiltonian constructor in d-dimensional lattice space.

# Purpose
Generating Hamiltonians for N-particles while correctly accounting for couplings, many-body interactions, and dimensionality is difficult to generalize _a priori_. This constructor is intended to facilitate Hamiltonian construction for arbitrary lattice dimension, number of particles, and particle interactions via a straightforward UI. (We will restrict ourselves to the bosonic case for this implementation.)

# Directory Structure
- hamiltonian-construction
    - lattice-generation
    - many-body-transformation
- ui
- analysis

# Development Plan
1. Generate N-dimensional Hamiltonian for 1-particle case
2. Develop visualization of coupling using networkx
3. Extend to 2-particle case and include many-body interactions
4. Generalize to N-particle case
5. Refine UI and input structure
6. Add eigenstate decomposition visualization

# Contributions (by member)
Sivan S. will develop single-particle lattice indexing and Hamiltonian generation (via subclass of generic Hamiltonian) as well as a visualization class to show coupling relations.
Zach F. will develop a generic Hamiltonian class and prepare methods for transforming from 1-particle basis to N-particle basis Hamiltonian (via generation of new N-particle Hamiltonian subclass).
Both members will contribute to UI design and eigenstate decomposition classes.
