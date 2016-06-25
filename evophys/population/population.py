'''
Genetic algorithm simulation
other documentation....
'''
import numpy as np
import pandas as pd
from scipy.stats import norm


class PhysPopulation():
	'''
	A population of user-specified biophysical models
	'''

	def __init__(self,model,modelParams={},params_to_mut=[],N=100,dt=10,selection_strength=1,mutAllParams=True,startRandom=False,paramBounds=None):
		
		# Biophysical model
		# Create target model on startup by simulating under binding model and populating self.binding curve
		self.model = model
		self._model_params = modelParams
		self._model_inst = self.model(**modelParams)
		self.target = self._model_inst.output
		
		self.params_to_mut = params_to_mut # params in model to explore. Must specify so some can be left static
		self._validate_params()
		self.param_means = {i:None for i in self._model_inst.paramD}
		self.param_vars = {i:None for i in self._model_inst.paramD}

		
		# Population parameters
		self.population = []
		self.mean_fitness = 0
		self.fitness_var = 0
		
		# Genetic algorithm params
		self.N = N
		self.selection_strength = selection_strength
		self.mutAllParams = mutAllParams
		
		# Brownian motion params
		self.dt = dt
		
		# Initialization
		self.param_bounds = paramBounds
		self._initialize_pop(startRandom)
		
		
	def _validate_params(self):
		for i in self.params_to_mut:
			assert i in self._model_inst.paramD, "Parameter %s not found in supplied model" % i
		
	def _update_biophys_params(self):
		'''Update means and variances for population K1 and K2s'''
		paramVals = {p:[] for p in self._model_inst.paramD}
		ws = []
		for i in self.population:
			for param in self._model_inst.paramD.keys():
				paramVals[param].append(i.paramD[param])
			ws.append(i.w)
			
		for p in self._model_inst.paramD:
			self.param_means[p] = sum(paramVals[p])/len(paramVals[p])
			self.param_vars[p] = np.var(paramVals[p])
			
		self.mean_fitness = sum(ws)/len(ws)
		self.fitness_var = np.var(ws)
		
	def _initialize_pop(self,startRandom=False):
		'''Create starting population of size self.N filled with Adair models, each sampled'''
		
		self.population = []
		paramVals = {p:[] for p in self._model_inst.paramD}
		ws = []
		
		for i in range(self.N):
			if startRandom:
				rando = np.random.uniform
				for param in self._model_inst.paramD:
					assert self.param_bounds and param in self.param_bounds, "Must supply bounds for parameter random start"
				kwargs = {p: np.random.uniform(self.param_bounds[p][0],self.param_bounds[p][1]) for p in self.param_bounds}
				newModel = self.model(**kwargs)
			else:
				newModel = self.model(**self._model_params)
			
			newModel.w = self._get_w(self.target,newModel.output)
			
			for param in self._model_inst.paramD:
				paramVals[param].append(newModel.paramD[param])
			ws.append(newModel.w)
			
			self.population.append(newModel)
			
		for p in self._model_inst.paramD:
			self.param_means[p] = sum(paramVals[p])/len(paramVals[p])
			self.param_vars[p] = np.var(paramVals[p])

		self.mean_fitness = sum(ws)/len(ws)
		self.fitness_var = np.var(ws)

	def _get_w(self,target,comp):
		'''
		Compute fitness function. Fitness is modeled as a normal with x-mu = rmsd between
		target model output and focal, and var = self.selection_strength. 
		'''
		assert len(comp) == len(target), "Different vector lengths: target: %i, comp: %i" % (len(target),len(comp))
		rmsd = np.sqrt(sum(map(lambda (x,y): (x-y)**2, zip(target,comp)))/len(comp))
		return np.exp(-(rmsd**2)/2*self.selection_strength)

	def brownian(self,paramD):
		'''Take a single brownian step over params vector'''
		out = {}
		if not self.mutAllParams:
			paramToMut = np.random.choice(self.params_to_mut)
		for i,j in paramD.iteritems():
			if i not in self.params_to_mut:
				out[i] = j # set to old value
				continue
			elif not self.mutAllParams and i != paramToMut:
				out[i] = j # set to old value
				continue
			else:
				new_param = j + norm.rvs(loc=0,scale=2*self.dt)
				if new_param < 0:
					new_param = 0
				out[i] = new_param
		return out
		
	def procreate(self):
		newpop = []
		paramVals = {p:[] for p in self._model_inst.paramD}
		ws = []
		
		for model in self.population:
			kwargs = self.brownian(model.paramD)
			newModel = self.model(**kwargs)
			
			newModel.w = self._get_w(self.target,newModel.output)
			
			for param in self._model_inst.paramD:
				paramVals[param].append(newModel.paramD[param])
			ws.append(newModel.w)
			
			newpop.append(newModel)
			
		self.population = newpop
			
		for p in self._model_inst.paramD:
			self.param_means[p] = sum(paramVals[p])/len(paramVals[p])
			self.param_vars[p] = np.var(paramVals[p])
			
		self.mean_fitness = sum(ws)/len(ws)
		self.fitness_var = np.var(ws)
		
	def select(self):
		fitness_vec = [m.w for m in self.population]
		w_sum = sum(fitness_vec)
		w_norm = [i/w_sum for i in fitness_vec]
		new_pop = np.random.choice(self.population,self.N,p=w_norm)
		self.population = new_pop
		self._update_biophys_params()
		
		
		
class WrightFisherSim(SSB):

	def __init__(self,num_generations=1000,N=10e5,mu=10e-5,std=10,fitness_function="normal",omega=1,k1=1e7,k2=1e8,f=10):
	
		SSB.__init__(self,k1,k2,f)
		self.sim() # get starting curve
		self.target = self.binding_curve
		self.mu = mu
		self.std = std
		self.num_generations = num_generations
		self.N = N
		self.params = {"k1":self.k1, "k2":self.k2, "f":self.f}
		self.fitness = 1
		self.fitness_function = fitness_function
		self.omega = omega
		self.current_gen = 1
		
	def _num_gens_till_mut(self):
		return random.expovariate(4 * self.N * self.mu)
		
	def _fitness_function_linear(self,target,comp):
		'''Compute rmsd between two input binding curves and return (2 - rmsd)/2. 2 is maximum
		distance for the SSB model, so this should scale between 0 and 1'''
		assert len(comp) == len(target)
		return (2 - np.sqrt(sum(map(lambda (x,y): (x-y)**2, zip(target,comp)))/len(target))) / 2
		
	def _fitness_function_normal(self,target,comp,omega):
		'''Compute rmsd between two input binding curves and return a normally 
		distributed fitness function with variance parameter omega'''
		assert len(comp) == len(target)
		rmsd = np.sqrt(sum(map(lambda (x,y): (x-y)**2, zip(target,comp)))/len(target))
		return np.exp(-(rmsd**2)/2*omega)
		
	def _selection_coefficient(self,w):
		return (w/self.fitness) - 1
		
	def _prob_fix(self,s):
		return (1 - np.exp(-(2 * s))) / (1 - np.exp(-(4 * self.N * s)))

	def runSim(self,outfile):
	
		out = open(outfile,'a')
		out.write(",".join(["generation","k1","k2","f","fitness","mutType","fixed","\t",
								"s","probFix","rando"]) + "\n")
	
		gens_since_mut = 1
		for gen in np.arange(self.num_generations):
		
			# pull num generations until mutation from exponential(4Nmu)
			mutGens = self._num_gens_till_mut()
			
			# if gens generations < num until mut. write current state
			if mutGens <= gens_since_mut:
				
				# if mutation, get kind of mutation from uniform over how many parameters to disturb (1,2,3)
				mutKind = random.choice([1,2,3])
				
				# if num parameters to disturb < 3, choose random
				mutParams = random.sample(self.params.keys(),mutKind)
				newParamD = {}
				
				# mutate parameters using normal (mean=0,std)
				for p in mutParams:
					newParamD[p] = self.params[p] + random.normalvariate(0,self.std)
				if len(newParamD.keys()) < 3:
					for i in self.params.keys(): # fill in with old params
						if i not in newParamD:
							newParamD[i] = self.params[i]
				newModel = SSB(k1=newParamD["k1"],k2=newParamD["k2"],f=newParamD["f"])
				newModel.sim()
				
				# Calculate fitness, choosing linear or normally distributed fitness function
				if self.fitness_function == "linear":
					newModelFitness = self._fitness_function_linear(self.target,newModel.binding_curve)
				elif self.fitness_function == "normal":
					newModelFitness = self._fitness_function_normal(self.target,newModel.binding_curve,self.omega)
				else:
					raise Exception("Fitness function type: %s not found" % fitness)
				
				# calculate selection coefficient
				newModelS = self._selection_coefficient(newModelFitness)
				
				# calc prob of fixation
				probFix = self._prob_fix(newModelS)
				
				# draw random(0,1), if < prob of fixation, set new model as current, with parameters and fitness values.
				rando = random.uniform(0,1)
				fixes = rando < probFix
				if fixes:
					self.model = newModel.binding_curve
					self.fitness = newModelFitness
					self.k1 = newModel.k1
					self.k2 = newModel.k2
					self.f = newModel.f
					self.binding_curve = newModel.binding_curve
					self.params = {"k1":self.k1, "k2":self.k2, "f":self.f}
				
			else:
				mutKind = None
				fixes = None
				gens_since_mut += 1
			
			if fixes:
				out.write(",".join(map(str,[self.current_gen,self.k1,self.k2,self.f,self.fitness,mutKind,fixes,"\t",
											newModelS,probFix,rando,"\t",newModel.k1,newModel.k2,newModel.f]))+ "\n")
			print self.current_gen
			self.current_gen += 1
			
		out.close()