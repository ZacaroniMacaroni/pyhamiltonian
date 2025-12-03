import itertools

class BaseHamiltonian:
    """
    Base Hamiltonian class.
    """
    def __init__(self):
        pass

    def diagonalize(self, *args):
        raise NotImplementedError("Need to implement diagonalize method for "
                                  "Hamiltonian.")

    def visualize_couplings(self, *args):
        raise NotImplementedError("Need to implement visualize_couplings method for "
                                  "Hamiltonian.")


class SingleParticleHamiltonian(BaseHamiltonian):
    """
    Single particle Hamiltonian
    """
    def __init__(self, nlen, ndim, energy=0., base_coupling=1., list_energies=None):
        super().__init__()
        self.len = nlen
        self.dim = ndim

        # Generate state list for indexing diagonal of Hamiltonian.
        self.state_list = list(itertools.product([x for x in range(nlen)], repeat=ndim))

        # Generate diagonal energies
        if list_energies is not None:
            if nlen(list_energies) != nlen(self.state_list):
                raise ValueError(
                    f"Length of list_energies ({nlen(list_energies)}) does not match "
                    f"length of state_list ({nlen(self.state_list)})"
                )
            self.energies = list_energies
        else:
            self.energies = [energy for _ in range(nlen(self.state_list))]

        self.base_coupling = base_coupling

    def construct_hamiltonian(self):
        raise NotImplementedError("Need to implement construct_hamiltonian method")
