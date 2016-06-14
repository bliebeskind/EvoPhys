'''
Genetic algorithm simulation
other documentation....
'''
import numpy as np
import pandas as pd
from scipy.stats import norm


class PhysPopulation():

	def __init__(self,model,modelParams={},params_to_mut=[],N=100,dt=10,selection_strength=1,startRandom=False,paramBounds=None):
		
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
		self.cost_function = self._distance_function
		
		
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
		
		if startRandom:
			rando = np.random.uniform
		for i in range(self.N):
			if startRandom:
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

	def _distance_function(self,target,comp):
		'''Compute root mean squared distance between two input binding curves'''
		assert len(comp) == len(target), "Different vector lengths: target: %i, comp: %i" % (len(target),len(comp))
		return np.sqrt(sum(map(lambda (x,y): (x-y)**2, zip(target,comp)))/len(comp))

	def _get_w(self,target,comp):
		return self.selection_strength * (1 - self._distance_function(target,comp))

	def brownian(self,paramD):
		'''Take a single brownian step over params vector'''
		out = {}
		for i,j in paramD.iteritems():
			if i not in self.params_to_mut:
				out[i] = j
				continue
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