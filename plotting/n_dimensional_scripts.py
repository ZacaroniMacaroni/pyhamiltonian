import itertools

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

    def _project_eigpop(self, eigpop, selected_dim):
        """
        Projects eigenvector onto a subset of 1-2 dimensions for plotting.

        Parameters:
            1. eigpop: np.ndarray
                       Population vector to be projected.
            2. selected_dim: Tuple[int]
                             Tuple of indices of the projected dimensions.

        Returns:
            1. projected_eigpop: np.ndarray
                                 [PLACEHOLDER TEXT]
        """
        projected_eigpop = np.zeros((self.hamiltonian.lattice.nlen,) * len(selected_dim))
        # For selected dimension(s), group states containing each possible coordinate
        list_coords_selected = list(itertools.product(
            range(self.hamiltonian.lattice.nlen),
            repeat=len(selected_dim)
        ))
        list_coords_unselected = list(itertools.product(
            range(self.hamiltonian.lattice.nlen),
            repeat=self.hamiltonian.lattice.ndim-len(selected_dim)
        ))
        for coords_selected in list_coords_selected:
            for coords_unselected in list_coords_unselected:
                # Convert coords_unselected to list to use insert method
                coords = list(coords_unselected)
                for i in range(len(selected_dim)):
                    coords.insert(selected_dim[i], coords_selected[0])
                projected_eigpop[*coords_selected] \
                    += eigpop[self.hamiltonian.state_list.index(tuple(coords))]
        return projected_eigpop

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
        eigpop = np.abs(selected_eigvec)**2

        # Project eigenvector onto selected dimensions
        projected_eigpop = self._project_eigpop(eigpop, selected_dim)

        # Plot population distribution
        pass