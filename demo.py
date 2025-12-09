from src.hamiltonian_construction.hamiltonians import SingleParticleHamiltonian, \
    ManyBodyHamiltonian, Lattice
from src.plotting.plotters import NDimPlot, NParticlePlot

# Sample 2-D Hamiltonian
lat = Lattice(50, 2, disorder_type='uniform', disorder_strength=0.5)
ham = SingleParticleHamiltonian(lat, 1.)
nd_plot = NDimPlot(ham)
nd_plot.plot_energy_spectrum()
nd_plot.plot_eigenstate_population(0, (0, 1))

# Comparison of 1-particle and 2-particle localization
lat_1d = Lattice(50, 1, disorder_type='uniform', disorder_strength=0.5)
ham_1p = ManyBodyHamiltonian(lat_1d, 1, 1.)
plot_1p = NParticlePlot(ham_1p)
plot_1p.plot_energy_spectrum()
plot_1p.plot_eigenstate_population(0)

mbham = ManyBodyHamiltonian(lat_1d, 2, 1.)
plot_2p = NParticlePlot(mbham)
plot_2p.plot_energy_spectrum()
plot_2p.plot_eigenstate_population(0)
