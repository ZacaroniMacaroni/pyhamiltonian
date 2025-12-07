from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian, \
    ManyBodyHamiltonian, Lattice
from plotting.n_dimensional_scripts import NDimPlot

lat = Lattice(2, 2)

ham = SingleParticleHamiltonian(lat, 1.)
print(ham.state_list)
print(ham.construct_hamiltonian())

nd_plot = NDimPlot(ham)
fig, ax = nd_plot.plot_energy_spectrum()
fig.show()

mbham = ManyBodyHamiltonian(lat, 2, 1.)
print(mbham.state_list)
print(mbham.construct_hamiltonian())
