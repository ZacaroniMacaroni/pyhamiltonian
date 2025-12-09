# tests/test_single_particle.py
import numpy as np

from hamiltonian_construction.hamiltonians import Lattice, SingleParticleHamiltonian


def test_single_particle_hamiltonian_shape_and_diag():
    # Simple 1D lattice with no disorder
    nlen = 4
    lattice = Lattice(nlen=nlen, ndim=1, base_energy=2.0, disorder_strength=0.)
    H_single_particle = SingleParticleHamiltonian(lattice, base_coupling=1.0,
                                    coupling_type='nearest neighbor')
    H = H_single_particle.construct_hamiltonian()

    # Hamiltonian should be n_sites x n_sites
    assert H.shape == (nlen, nlen)

    # Diagonal should match lattice.site_energies
    assert np.allclose(np.diag(H), lattice.site_energies)

def test_single_particle_uniform_disorder_diagonal_matches_lattice():
    np.random.seed(0)
    nlen = 10
    base_energy = 0.5
    disorder_strength = 2.0

    # Create lattice with uniform disorder
    lattice = Lattice(nlen=nlen, ndim=1, base_energy=base_energy,
                      disorder_strength=disorder_strength,
                      disorder_type="uniform")
    
    # Create single particle Hamiltonian
    H_single_particle = SingleParticleHamiltonian(lattice, base_coupling=0.5,
                                    coupling_type='nearest neighbor')
    H = H_single_particle.construct_hamiltonian()

    # Diagonal should match lattice.site_energies
    assert np.allclose(np.diag(H), lattice.site_energies)

    # Check that disorder is within expected range
    lo = base_energy - disorder_strength / 2
    hi = base_energy + disorder_strength / 2
    assert np.all(np.diag(H) >= lo)
    assert np.all(np.diag(H) <= hi)

    # Ensure not all diagonal elements are the same and reflects disorder
    assert not np.allclose(np.diag(H), np.diag(H)[0])


def test_single_particle_gaussian_disorder_diagonal_matches_lattice():
    np.random.seed(1)
    nlen = 20
    base_energy = -0.3
    strength = 0.7

    lattice = Lattice(
        nlen=nlen,
        ndim=1,
        base_energy=base_energy,
        disorder_strength=strength,
        disorder_type="gaussian",
    )
    Hsp = SingleParticleHamiltonian(
        lattice,
        base_coupling=1.0,
        coupling_type="nearest neighbor",
    )
    H = Hsp.construct_hamiltonian()

    # Diagonal of H must match the lattice site energies
    assert np.allclose(np.diag(H), lattice.site_energies)

    # Gaussian disorder should give a non-zero standard deviation
    assert np.std(lattice.site_energies) > 0.0

    # Mean should be close to base_energy
    assert abs(np.mean(lattice.site_energies) - base_energy) < 0.1