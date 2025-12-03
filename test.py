from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian, \
    ManyBodyHamiltonian

ham = SingleParticleHamiltonian(3, 2)
print(ham.state_list)

mbham = ManyBodyHamiltonian(3, 2, 2)
print(mbham.state_list)
