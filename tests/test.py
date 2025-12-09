from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian, \
    ManyBodyHamiltonian, Lattice
from plotting.plotters import NDimPlot, NParticlePlot

lat = Lattice(50, 2, disorder_type='uniform', disorder_strength=0.5)

ham = SingleParticleHamiltonian(lat, 1.)
print(ham.construct_hamiltonian())
nd_plot = NDimPlot(ham)
nd_plot.plot_energy_spectrum()
nd_plot.plot_eigenstate_population(0, (0, 1))

lat_1d = Lattice(50, 1, disorder_type='uniform', disorder_strength=0.5)
mbham = ManyBodyHamiltonian(lat_1d, 2, 1.)
print(mbham.construct_hamiltonian())
np_plot = NParticlePlot(mbham)
np_plot.plot_energy_spectrum()
np_plot.plot_eigenstate_population(0)
