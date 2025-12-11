# tests/test_base_hamiltonian.py
import numpy as np
import pytest

from src.hamiltonian_construction.hamiltonians import BaseHamiltonian, Lattice


def test_base_hamiltonian_requires_lattice_instance():
    with pytest.raises(TypeError):
        BaseHamiltonian(lattice="not_a_lattice")


def test_construct_hamiltonian_raises_if_states_not_set():
    lattice = Lattice(nlen=2, ndim=1, disorder_strength=0.)
    bh = BaseHamiltonian(lattice)

    # state_list and energies are None by default
    with pytest.raises(RuntimeError):
        bh.construct_hamiltonian()
