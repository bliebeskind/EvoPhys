#! /usr/bin/env python

from mpi4py import MPI
from EvoPhys import PhysPopulation

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert size == 15

pop = PhysPopulation(N=1000)

dt_params = [.1,1,10,100,1000]
local_dt = dt_params[rank/3]
rep = rank%3
pop.dt = local_dt

#N_params = [5,10,50,100,500,1000]
#local_N = N_params[rank/4]
#rep = rank%4
#pop.N = local_N

outfile = open("dtSweep_%s_%i.csv" % (str(local_dt),rep),'w')
outfile.write(",".join(["gen","w_bar","w_var","k1_var","k2_var","f_var","numk1","numk2","numf"]) + "\n")

for i in range(1000):
	pop.procreate()
	pop.select()
	
	numk1 = len(set((i.k1 for i in pop.population)))
	numk2 = len(set((i.k2 for i in pop.population)))
	numf = len(set((i.f for i in pop.population)))
	
	outfile.write(",".join(map(str,
		[i,
			pop.mean_fitness,
			pop.fitness_var,
			pop.var_k1,
			pop.var_k2,
			pop.var_f,
			numk1,
			numk2,
			numf])) + "\n")
	
outfile.close()


