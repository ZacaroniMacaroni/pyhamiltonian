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