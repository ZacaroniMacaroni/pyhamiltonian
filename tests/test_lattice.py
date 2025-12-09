# tests/test_lattice.py
import numpy as np
import pytest

from src.hamiltonian_construction.hamiltonians import Lattice


def test_square_lattice_sites_1d():
    lattice = Lattice(nlen=4, ndim=1, disorder_strength=0.)
    # Expect coordinates: (0,), (1,), (2,), (3,)
    expected_sites = [(0,), (1,), (2,), (3,)]
    assert lattice.sites == expected_sites
    assert len(lattice.sites) == 4


def test_square_lattice_sites_2d():
    lattice = Lattice(nlen=3, ndim=2, disorder_strength=0.)
    # Cartesian product of {0,1,2} x {0,1,2}
    expected_sites = [(0, 0), (0, 1), (0, 2),
                      (1, 0), (1, 1), (1, 2),
                      (2, 0), (2, 1), (2, 2)]
    assert set(lattice.sites) == set(expected_sites)
    assert len(lattice.sites) == 9

def test_no_disorder_has_constant_energy():
    base_energy = 1.5
    lattice = Lattice(nlen=3, ndim=1, base_energy=base_energy, disorder_strength=0.)
    assert np.all(lattice.site_energies == base_energy)
    # Energy lookup should agree
    for site in lattice.sites:
        assert lattice.get_energy_of_site(site) == base_energy


def test_uniform_disorder_range_is_correct():
    np.random.seed(0)  # make test deterministic
    strength = 2.0
    base = 0.5
    lattice = Lattice(nlen=10, ndim=1,
                      base_energy=base,
                      disorder_strength=strength,
                      disorder_type="uniform")
    # Disorder should be within [base - strength/2, base + strength/2]
    lo = base - strength / 2
    hi = base + strength / 2
    assert np.all(lattice.site_energies >= lo)
    assert np.all(lattice.site_energies <= hi)
    assert not np.allclose(lattice.site_energies, lattice.site_energies[0])  # ensure not all same


def test_gaussian_disorder_mean_is_reasonable():
    np.random.seed(0)
    strength = 1.0
    base = 0.0
    lattice = Lattice(nlen=1000, ndim=1,
                      base_energy=base,
                      disorder_strength=strength,
                      disorder_type="gaussian")
    # For large sample, mean should be near base (0)
    mean_energy = np.mean(lattice.site_energies)
    assert abs(mean_energy - base) < 0.1  # loose tolerance


def test_invalid_lattice_type_raises():
    with pytest.raises(ValueError):
        Lattice(nlen=3, ndim=2, lattice_type='weird', disorder_strength=0.)


def test_not_implemented_lattice_types_raise():
    with pytest.raises(NotImplementedError):
        Lattice(nlen=3, ndim=2, lattice_type='triangular')
