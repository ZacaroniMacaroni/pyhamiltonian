import itertools

import numpy as np

class BaseHamiltonian:
    """
    Base Hamiltonian class.
    """
    def __init__(self, nlen, ndim, base_coupling=1., coupling_type='nearest neighbor',
                 minimum_coupling=0.):
        self.nlen = nlen
        self.ndim = ndim

        # Generate lattice basis
        self.lattice_basis = list(itertools.product([x for x in range(self.nlen)],
                                                    repeat=self.ndim))

        # Add coupling information
        self.base_coupling = base_coupling
        if coupling_type == 'nearest neighbor' or coupling_type == 'dipole':
            self.coupling_type = coupling_type
        else:
            raise ValueError(f"Coupling_type {coupling_type} not recognized. Accepted "
                             f"values are 'nearest neighbor' and 'dipole'.")
        self.minimum_coupling = minimum_coupling

    def diagonalize(self, *args):
        raise NotImplementedError("Need to implement diagonalize method")

    def visualize_couplings(self, *args):
        raise NotImplementedError("Need to implement visualize_couplings method")

    def construct_hamiltonian(self):
        raise NotImplementedError("Need to implement construct_hamiltonian method")


class SingleParticleHamiltonian(BaseHamiltonian):
    """
    Single particle Hamiltonian
    """
    def __init__(self, nlen, ndim, energy=0., list_energies=None, base_coupling=1.,
                 coupling_type='nearest neighbor', minimum_coupling=0.):
        super().__init__(nlen, ndim, base_coupling, coupling_type, minimum_coupling)

        # Generate state list for indexing diagonal of Hamiltonian.
        self.state_list = self.lattice_basis

        # Generate diagonal energies
        if list_energies is not None:
            if len(list_energies) != len(self.state_list):
                raise ValueError(
                    f"Length of list_energies ({len(list_energies)}) does not match "
                    f"length of state_list ({len(self.state_list)})"
                )
            self.energies = list_energies
        else:
            self.energies = [energy for _ in range(len(self.state_list))]

    def construct_hamiltonian(self):
        raise NotImplementedError("Need to implement construct_hamiltonian method")


class ManyBodyHamiltonian(BaseHamiltonian):
    """
    Many-Body Hamiltonian
    """
    def __init__(self, nlen, ndim, nparticles, base_coupling=1., coupling_type='nearest neighbor',
                 minimum_coupling=0.):
        super().__init__(nlen, ndim, base_coupling, coupling_type, minimum_coupling)

        # Generate many-body state list for indexing diagonal of Hamiltonian.
        self.state_list = list(itertools.combinations(self.lattice_basis, nparticles))

