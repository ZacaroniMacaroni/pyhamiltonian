from hamiltonian_construction.hamiltonians import SingleParticleHamiltonian, \
    ManyBodyHamiltonian

ham = SingleParticleHamiltonian(2, 2)
print(ham.state_list)
print(ham.construct_hamiltonian())

mbham = ManyBodyHamiltonian(2, 2, 2)
print(mbham.state_list)
print(mbham.construct_hamiltonian())
