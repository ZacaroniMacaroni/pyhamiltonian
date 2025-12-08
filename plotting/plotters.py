import itertools

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian

class BasePlot:
    """
    Base plotter class.
    """
    def __init__(self, hamiltonian):
        """
        Parameters:
            1. hamiltonian: SingleParticleHamiltonian or ManyBodyHamiltonian
                            Hamiltonian object to be used for plotting.
        """
        self.nlen = hamiltonian.lattice.nlen
        self.ndim = hamiltonian.lattice.ndim
        self.hamiltonian = hamiltonian

    def plot_energy_spectrum(self):
        """
        Returns a plot of the energy spectrum of the Hamiltonian, in order of increasing
        energy.
        """
        eigvals = sorted(np.linalg.eigvalsh(self.hamiltonian.construct_hamiltonian()))
        fig, ax = plt.subplots()
        ax.set_xlabel('Eigenstate Index')
        ax.set_ylabel('Eigenvalue')
        ax.plot(eigvals, ls='', marker='o')
        fig.show()


class NDimPlot(BasePlot):
    """
    Plotter for single-particle N-dimensional systems.
    """
    def __init__(self, hamiltonian):
        """
        Parameters:
            1. hamiltonian: SingleParticleHamiltonian
                            Single particle Hamiltonian object to be used for plotting
        """
        if not isinstance(hamiltonian, SingleParticleHamiltonian):
            raise TypeError("Hamiltonian must be an instance of "
                            "SingleParticleHamiltonian.")
        else:
            super().__init__(hamiltonian)

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
                                 Eigenstate population projected onto selected
                                 dimensions.
        """
        projected_eigpop = np.zeros((self.nlen,) * len(selected_dim))
        # For selected dimension(s), group states containing each possible coordinate
        list_coords_selected = list(itertools.product(
            range(self.nlen),
            repeat=len(selected_dim)
        ))
        list_coords_unselected = list(itertools.product(
            range(self.nlen),
            repeat=self.ndim-len(selected_dim)
        ))
        for coords_selected in list_coords_selected:
            for coords_unselected in list_coords_unselected:
                # Convert coords_unselected to list to use insert method
                coords = list(coords_unselected)
                for i in range(len(selected_dim)):
                    coords.insert(selected_dim[i], coords_selected[i])
                projected_eigpop[*coords_selected] \
                    += eigpop[self.hamiltonian.state_list.index(tuple(coords))]
        return projected_eigpop

    def plot_eigenstate_population(self, idx, selected_dim=None):
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
        # Initial checks on selected_dim
        if (selected_dim is None and self.ndim > 2) or \
                (len(selected_dim) > 2):
            raise ValueError("Must project to <= 2 dimensions to plot population "
                             "distribution.")
        elif (selected_dim is not None) and (len(selected_dim) == 0):
            raise ValueError("Must select at least one dimension to plot population "
                             "distribution.")
        elif selected_dim is None:
            selected_dim = tuple(range(self.ndim))

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
        if len(selected_dim) == 1:
            fig, ax = plt.subplots()
            ax.set_xlabel('Position')
            ax.set_ylabel('Population')
            ax.plot(projected_eigpop)
            fig.show()
        elif len(selected_dim) == 2:
            # For 3-D plot, render the population as a surface
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel('Position Coordinate {}'.format(selected_dim[0]))
            ax.set_ylabel('Position Coordinate {}'.format(selected_dim[1]))
            ax.set_zlabel('Population')

            X, Y = np.meshgrid(range(self.nlen), range(self.nlen))
            Z = projected_eigpop

            surf = ax.plot_surface(X, Y, Z, cmap='viridis', vmin=0)
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1, label='Population')
            fig.show()


class NParticlePlot(BasePlot):
    """
    Plotter for 1-D N-particle systems.
    """
    def __init__(self, hamiltonian):
        """
        Parameters:
            1. hamiltonian: SingleParticleHamiltonian or ManyBodyHamiltonian
                            Hamiltonian object to be used for plotting (must be 1-D for
                            current implementation of plotter).
        """
        if hamiltonian.lattice.ndim != 1:
            raise ValueError("NParticlePlot only works for 1-D systems.")
        else:
            super().__init__(hamiltonian)

    def _project_eigpop(self, eigpop):
        """
        Projects N-particle eigenvector population onto site basis.

        Parameters:
            1. eigpop: np.ndarray
                       Population vector to be projected.

        Returns:
            1. projected_eigpop: np.ndarray
                                 Eigenstate population projected onto site basis.
        """
        projected_eigpop = np.zeros(self.nlen)
        for site in range(self.nlen):
            for state in self.hamiltonian.state_list:
                if (site,) in state:
                    projected_eigpop[site] \
                        += eigpop[self.hamiltonian.state_list.index(state)]
        return projected_eigpop

    def plot_eigenstate_population(self, idx):
        """
        Returns a plot of the population distribution of a specific eigenstate, taking
        the probability of finding any particle at a given site as the site population.

        Parameters:
            1. idx: int
                    Index of the eigenstate to plot.
        """
        eigvals, eigvecs = np.linalg.eigh(self.hamiltonian.construct_hamiltonian())

        # Need to reorder eigenvectors such that they are sorted based on
        # increasing eigenvalue
        idx_sorted = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx_sorted]
        selected_eigvec = eigvecs[:, idx]
        eigpop = np.abs(selected_eigvec)**2

        # Project N-particle states onto site basis
        projected_eigpop = self._project_eigpop(eigpop)

        # Plot population distribution
        fig, ax = plt.subplots()
        ax.set_xlabel('Position')
        ax.set_ylabel('Population')
        ax.plot(projected_eigpop)
        fig.show()