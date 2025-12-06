import itertools
import numpy as np

class Lattice:
    """
    Class to handle Lattice generation and static disorder.
    """
    def __init__(self, nlen, ndim, lattice_type='square', base_energy=0., disorder_strength=0., disorder_type='uniform'):
        self.nlen = nlen
        self.ndim = ndim
        self.lattice_type = lattice_type.lower()
        self.base_energy = base_energy
        self.disorder_strength = disorder_strength
        self.disorder_type = disorder_type.lower()
        
        # 1. Generate Geometry (Sites)
        self.sites = self._generate_basis()
        
        # 2. Generate On-Site Energies (Static Disorder)
        self.site_energies = self._generate_site_energies()
        
        # Helper map for O(1) energy lookups by coordinate
        self._coord_to_energy = {coord: en for coord, en in zip(self.sites, self.site_energies)}
    
    def _generate_basis(self):
        if self.lattice_type == 'square':
            return self._generate_square_lattice()
        elif self.lattice_type in ['triangular', 'hexagonal']:
            raise NotImplementedError(f"Logic for {self.lattice_type} lattice is not yet implemented.")
        else:
            raise ValueError(f"Lattice type {self.lattice_type} not recognized. "
                             f"Currently only 'square' lattice is implemented.")

    def _generate_square_lattice(self):
        """Generates coordinates for a hyper-cubic (square) lattice."""
        return list(itertools.product([x for x in range(self.nlen)], repeat=self.ndim))
    
    def _generate_site_energies(self):
        """
        Generates on-site energies with static disorder.
        """
        num_sites = len(self.sites)
        if self.disorder_strength == 0.:
            return np.full(num_sites, self.base_energy)
        
        if self.disorder_type == 'uniform':
            disorder = np.random.uniform(-self.disorder_strength / 2,
                                         self.disorder_strength / 2,
                                         size=num_sites)
        elif self.disorder_type == 'gaussian':
            disorder = np.random.normal(0., self.disorder_strength, size=num_sites)
        else:
            raise ValueError(f"Disorder type {self.disorder_type} not recognized. "
                             f"Accepted values are 'uniform' and 'gaussian'.")
        
        return self.base_energy + disorder
        

    def get_energy_of_site(self, coordinate):
        """Returns the energy of a specific site coordinate."""
        return self._coord_to_energy[coordinate]


class BaseHamiltonian:
    """
    Base Hamiltonian class.
    """
    def __init__(self, lattice, base_coupling=1., interaction_strength=0., 
                 coupling_type='nearest neighbor', minimum_coupling=0.):
        
        if not isinstance(lattice, Lattice):
            raise TypeError("The 'lattice' argument must be an instance of the Lattice class.")
        
        self.lattice = lattice
        self.base_coupling = base_coupling
        self.interaction_strength = interaction_strength # Added for particle-particle interaction
        self.minimum_coupling = minimum_coupling
        
        if coupling_type in ['nearest neighbor', 'dipole']:
            self.coupling_type = coupling_type
        else:
            raise ValueError(f"Coupling_type {coupling_type} not recognized.")

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
        # Single particles don't have particle-particle interactions, so interaction_strength=0
        super().__init__(lattice, base_coupling=base_coupling, interaction_strength=0., 
                         coupling_type=coupling_type, minimum_coupling=minimum_coupling)

        self.state_list = self.lattice.sites
        self.energies = self.lattice.site_energies
    
    def _assign_couplings(self, state_1, state_2):
        dist = np.linalg.norm(np.array(state_1) - np.array(state_2))
        
        # FIX: Added tolerance (1e-9) to handle floating point imprecision
        if self.coupling_type == 'nearest neighbor':
            if dist <= 1.0 + 1e-9: 
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
    Models Hard-Core Bosons (implied by itertools.combinations).
    """
    def __init__(self, lattice, nparticles, base_coupling=1., interaction_strength=0.,
                 coupling_type='nearest neighbor', minimum_coupling=0.):
        super().__init__(lattice, base_coupling=base_coupling, interaction_strength=interaction_strength,
                         coupling_type=coupling_type, minimum_coupling=minimum_coupling)

        self.nparticles = nparticles
        
        # Combinations imply Hard-Core Bosons (no two particles on the same site)
        self.state_list = list(itertools.combinations(self.lattice.sites, nparticles))
        self.energies = self._calculate_many_body_energies()

    def _calculate_many_body_energies(self):
        """
        Sums site energies and ADDS particle-particle interaction energies.
        """
        mb_energies = []
        for state in self.state_list:
            # 1. Sum of on-site potential energies (Disorder)
            site_energy_sum = sum(self.lattice.get_energy_of_site(site) for site in state)
            
            # 2. Add Interaction Energy (e.g. Nearest Neighbor Repulsion)
            interaction_energy = 0.
            if self.interaction_strength != 0:
                # Check every pair of particles in this state
                for i in range(self.nparticles):
                    for j in range(i + 1, self.nparticles):
                        dist = np.linalg.norm(np.array(state[i]) - np.array(state[j]))
                        # FIX: Added tolerance here as well
                        if dist <= 1.0 + 1e-9:
                            interaction_energy += self.interaction_strength
            
            mb_energies.append(site_energy_sum + interaction_energy)
            
        return np.array(mb_energies)
    
    def _assign_couplings(self, state_1, state_2):
        common_sites = set(state_1).intersection(state_2)
        share_site = (len(common_sites) == (len(state_1) - 1))
        
        if share_site:
            diff_12 = sorted(set(state_1) - set(state_2))
            diff_21 = sorted(set(state_2) - set(state_1))

            dist = np.linalg.norm(np.array(diff_12) - np.array(diff_21))
            
            # FIX: Added tolerance (1e-9)
            if self.coupling_type == 'nearest neighbor':
                if dist <= 1.0 + 1e-9:
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