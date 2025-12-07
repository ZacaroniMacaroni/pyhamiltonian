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
        ax.set_xticks(range(len(eigvals)))
        ax.plot(eigvals, ls='', marker='o')
        return fig, ax

    def _project_eigvec(self, eigvec, selected_dim):
        """
        Projects eigenvector onto a subset of 1-2 dimensions for plotting.

        Parameters:
            1. eigvec: np.ndarray
                       Eigenvector to be projected.
            2. selected_dim: Tuple[int]
                             Tuple of indices of the projected dimensions.

        Returns:
            1. projected_eigvec: np.ndarray
                                 [PLACEHOLDER TEXT]
        """
        raise NotImplementedError("Need to implement _project_eigvec method.")

    def plot_eigenstate_population(self, idx, selected_dim):
        """
        Returns a plot of the population distribution of a specific eigenstate in the
        projected dimensions.

        Parameters:
            1. idx: int
                    Index of the eigenstate to plot.
            2. selected_dim: Tuple[int]
                             Tuple of indices of the projected dimensions.
                             Note: Tuple must have length <= 2 to provide an axis for
                             population.
        """
        if (selected_dim is None and self.hamiltonian.lattice.dimension > 2) or \
                (len(selected_dim) > 2):
            raise ValueError("Must project to <= 2 dimensions to plot population "
                             "distribution.")
        elif (selected_dim is not None) and (len(selected_dim) == 0):
            raise ValueError("Must select at least one dimension to plot population "
                             "distribution.")
        eigvals, eigvecs = np.linalg.eigh(self.hamiltonian.construct_hamiltonian())

        # Need to reorder eigenvectors such that they are sorted based on
        # increasing eigenvalue
        idx_sorted = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx_sorted]
        selected_eigvec = eigvecs[:, idx]

        # Project eigenvector onto selected dimensions
        projected_eigvec = self._project_eigvec(selected_eigvec, selected_dim)

        # Plot population distribution
        pass