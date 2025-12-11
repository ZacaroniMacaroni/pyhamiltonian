# tests/test_many_body.py
import numpy as np

from src.hamiltonian_construction.hamiltonians import Lattice, ManyBodyHamiltonian


def test_many_body_states_are_combinations():
    # 1D lattice of size 4, 2 particles
    lattice = Lattice(nlen=4, ndim=1, disorder_strength=0.)
    mb = ManyBodyHamiltonian(lattice, nparticles=2, base_coupling=1.0)

    from itertools import combinations
    expected_states = list(combinations(lattice.sites, 2))

    # state_list should contain all combinations
    assert set(mb.state_list) == set(expected_states)
    assert len(mb.state_list) == len(expected_states)


def test_many_body_energies_are_sums_of_site_energies():
    lattice = Lattice(nlen=3, ndim=1, base_energy=2.0, disorder_strength=0.)
    mb = ManyBodyHamiltonian(lattice, nparticles=2, base_coupling=1.0)

    # Check the first few states manually
    for state, energy in zip(mb.state_list, mb.energies):
        # Each state is a tuple of coords, e.g. ((0,), (1,))
        expected_energy = sum(lattice.get_energy_of_site(site) for site in state)
        assert np.isclose(energy, expected_energy)


def test_many_body_nearest_neighbor_couplings():
    # 1D lattice: sites 0,1,2. 2 particles.
    lattice = Lattice(nlen=3, ndim=1, disorder_strength=0.)
    mb = ManyBodyHamiltonian(lattice, nparticles=2, base_coupling=1.0,
                             coupling_type='nearest neighbor')
    H = mb.construct_hamiltonian()

    # States (in order of combinations):
    # index 0: (0,1)
    # index 1: (0,2)
    # index 2: (1,2)

    # Coupling between (0,1) and (0,2): particle hops from 1 -> 2 (distance 1)
    assert np.isclose(H[0, 1], 1.0)
    assert np.isclose(H[1, 0], 1.0)

    # Coupling between (0,1) and (1,2): particle hops from 0 -> 2 (distance 2) -> no coupling
    assert np.isclose(H[0, 2], 0.0)
    assert np.isclose(H[2, 0], 0.0)

    # Coupling between (0,2) and (1,2): particle hops from 0 -> 1 (distance 1)
    assert np.isclose(H[1, 2], 1.0)
    assert np.isclose(H[2, 1], 1.0)
