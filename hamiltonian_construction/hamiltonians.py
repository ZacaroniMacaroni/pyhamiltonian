import itertools

class BaseHamiltonian:
    """
    Base Hamiltonian class.
    """
    def __init__(self):
        pass

    def diagonalize(self, *args):
        raise NotImplementedError("Need to implement diagonalize method for Hamiltonian.")

    def visualize_couplings(self, *args):
        raise NotImplementedError("Need to implement visualize_couplings method for Hamiltonian.")


class SingleParticleHamiltonian(BaseHamiltonian):
    """
    Single particle Hamiltonian
    """
    def __init__(self, len, dim):
        super().__init__()
        self.len = len
        self.dim = dim

        # Generate state list for indexing diagonal of Hamiltonian.
        self.state_list = list(itertools.product([x for x in range(len)], repeat=dim))