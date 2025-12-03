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
    
    def _assign_couplings(self, state_1, state_2):
        dist = np.linalg.norm(np.array(state_1) - np.array(state_2))
        if self.coupling_type == 'nearest neighbor':
            if dist <= 1.:
                return self.base_coupling
            else:
                return 0.
        elif self.coupling_type == 'dipole':
            if self.base_coupling / dist**3 >= self.minimum_coupling:
                return self.base_coupling / dist**3
            else:
                return 0.
        else:
            raise ValueError(f"Coupling_type {self.coupling_type} not recognized. "
                             f"Accepted values are 'nearest neighbor' and 'dipole'.")

    def construct_hamiltonian(self):
        raise NotImplementedError("Need to implement construct_hamiltonian method")


class ManyBodyHamiltonian(BaseHamiltonian):
    """
    Many-Body Hamiltonian
    """
    def __init__(self, nlen, ndim, nparticles, energy=0., list_energies=None,
                 base_coupling=1., coupling_type='nearest neighbor',
                 minimum_coupling=0.):
        super().__init__(nlen, ndim, base_coupling, coupling_type, minimum_coupling)

        # Generate many-body state list for indexing diagonal of Hamiltonian.
        self.state_list = list(itertools.combinations(self.lattice_basis, nparticles))

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
    
    def _assign_couplings(self, state_1, state_2):
        # Check if exactly one particle changes site between the two states.
        # Both states are tuples of lattice sites (each site itself is a tuple)
        # share_site should be True only if all but one particle do not change sites
        # i.e., the number of common sites equals total particles minus one.
        common_sites = set(state_1).intersection(state_2)
        share_site = (len(common_sites) == (len(state_1) - 1))
        if share_site:
            # Only include the tuples that are different between the two states
            diff_12 = sorted(set(state_1) - set(state_2))
            diff_21 = sorted(set(state_2) - set(state_1))

            # Compute distance using only differing tuples
            dist = np.linalg.norm(np.array(diff_12) - np.array(diff_21))
            if self.coupling_type == 'nearest neighbor':
                if dist <= 1.:
                    return self.base_coupling
                else:
                    return 0.
            elif self.coupling_type == 'dipole':
                if self.base_coupling / dist ** 3 >= self.minimum_coupling:
                    return self.base_coupling / dist ** 3
                else:
                    return 0.
            else:
                raise ValueError(f"Coupling_type {self.coupling_type} not recognized. "
                                 f"Accepted values are 'nearest neighbor' and 'dipole'.")
        else:
            return 0.

