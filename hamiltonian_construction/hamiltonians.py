import itertools
import numpy as np

class Lattice:
    """
    Class to handle Lattice generation and static disorder.
    """
    def __init__(self, nlen, ndim, lattice_type='square', base_energy=0., disorder_strength=0.):
        self.nlen = nlen
        self.ndim = ndim
        self.lattice_type = lattice_type.lower()
        self.base_energy = base_energy
        self.disorder_strength = disorder_strength
        
        # 1. Generate Geometry (Sites)
        self.sites = self._generate_basis()
        
        # 2. Generate On-Site Energies (Static Disorder)
        self.site_energies = self._generate_site_energies()
        
        # Helper map for O(1) energy lookups by coordinate
        self._coord_to_energy = {coord: en for coord, en in zip(self.sites, self.site_energies)}

class BaseHamiltonian:
    """
    Base Hamiltonian class.
    """
    def __init__(self, lattice, base_coupling=1., coupling_type='nearest neighbor',
                 minimum_coupling=0.):
        
        # Take a Lattice object 
        if not isinstance(lattice, Lattice):
            raise TypeError("The 'lattice' argument must be an instance of the Lattice class.")
        
        self.lattice = lattice
        
        # Add coupling information
        self.base_coupling = base_coupling
        if coupling_type == 'nearest neighbor' or coupling_type == 'dipole':
            self.coupling_type = coupling_type
        else:
            raise ValueError(f"Coupling_type {coupling_type} not recognized. Accepted "
                             f"values are 'nearest neighbor' and 'dipole'.")
        self.minimum_coupling = minimum_coupling

        self.state_list = None
        self.energies = None

    def _assign_couplings(self, state_1, state_2):
        raise NotImplementedError("Need to implement _assign_couplings method")

    def construct_hamiltonian(self):
        """
        Constructs the Hamiltonian matrix.
        """
        if self.state_list is None or self.energies is None:
            raise RuntimeError("State list and Energies must be defined before construction.")

        M = len(self.state_list)
        H = np.zeros((M, M))

        # 1. Set the diagonal elements (On-site / State energies)
        np.fill_diagonal(H, self.energies)

        # 2. Set the off-diagonal elements (Couplings)
        for i in range(M):
            for j in range(i + 1, M):
                state_i = self.state_list[i]
                state_j = self.state_list[j]

                # Calculate coupling strength
                coupling = self._assign_couplings(state_i, state_j)
                
                # Assign to both H[i, j] and H[j, i] (Hermitian)
                H[i, j] = coupling
                H[j, i] = coupling
        
        return H


class SingleParticleHamiltonian(BaseHamiltonian):
    """
    Single particle Hamiltonian.
    """
    def __init__(self, lattice, base_coupling=1., coupling_type='nearest neighbor', 
                 minimum_coupling=0.):
        super().__init__(lattice, base_coupling, coupling_type, minimum_coupling)

        # 1. Define States
        # In 1-body case, the states are simply the lattice sites themselves
        self.state_list = self.lattice.sites

        # 2. Define Energies
        # In 1-body case, state energy is exactly the site energy (including disorder)
        self.energies = self.lattice.site_energies
    
    def _assign_couplings(self, state_1, state_2):
        # Calculate Euclidean distance between coordinates
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


class ManyBodyHamiltonian(BaseHamiltonian):
    """
    Many-Body Hamiltonian.
    """
    def __init__(self, lattice, nparticles, base_coupling=1., 
                 coupling_type='nearest neighbor', minimum_coupling=0.):
        super().__init__(lattice, base_coupling, coupling_type, minimum_coupling)

        self.nparticles = nparticles

        # 1. Define States
        # A state is a combination of N distinct lattice sites (Fermions/Hard-core Bosons logic implied by combinations)
        self.state_list = list(itertools.combinations(self.lattice.sites, nparticles))

        # 2. Define Energies
        # The energy of a many-body state is the SUM of the energies of the occupied sites
        self.energies = self._calculate_many_body_energies()

    def _calculate_many_body_energies(self):
        """
        Sums the individual site energies (from the Lattice) for every site
        occupied in a specific Many-Body state.
        """
        mb_energies = []
        for state in self.state_list:
            # 'state' is a tuple of coordinates, e.g., ((0,0), (0,1))
            total_energy = sum(self.lattice.get_energy_of_site(site) for site in state)
            mb_energies.append(total_energy)
        return np.array(mb_energies)
    
    def _assign_couplings(self, state_1, state_2):
        # Logic: Exactly one particle hops from one site to another
        common_sites = set(state_1).intersection(state_2)
        share_site = (len(common_sites) == (len(state_1) - 1))
        
        if share_site:
            diff_12 = sorted(set(state_1) - set(state_2))
            diff_21 = sorted(set(state_2) - set(state_1))

            # Compute distance between the old site and the new site
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
            return 0.