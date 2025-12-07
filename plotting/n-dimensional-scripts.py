import matplotlib.pyplot as plt
import numpy as np

from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian


class NDimPlot:
    """
    Plotter for single-particle N-dimensional systems.
    """
    def __init__(self, hamiltonian):
        self.lattice = hamiltonian.lattice
        if not isinstance(hamiltonian, SingleParticleHamiltonian):
            raise TypeError("Hamiltonian must be an instance of "
                            "SingleParticleHamiltonian.")
        else:
            self.hamiltonian = hamiltonian

    def plot_energy_spectrum(self):
        """
        Returns a plot of the energy spectrum of the Hamiltonian, in order of increasing
        energy.
        """
        eigvals = sorted(np.linalg.eigvals(self.hamiltonian.construct_hamiltonian()))
        fig, ax = plt.subplots()
        ax.set_xlabel('Eigenstate Index')
        ax.set_ylabel('Eigenvalue')
        ax.plot(eigvals)
        return fig, ax

