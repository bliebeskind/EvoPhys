#! /usr/bin/env python


import numpy as np
import pandas as pd
from scipy.stats import norm


class SSB:

    def __init__(self,k1=1e7,k2=1e8,f=10):
        self.k1 = k1
        self.k2 = k2
        self.f = f
        self.w = None
        self.conc_range = (-12,-3)
        self.binding_curve = []
        self.xvals = []
        self.step = .01
        self.step_func = lambda x: 10**x

        self.function = lambda x: (k1*x + k2*x + f*2*k1*k2*x**2) / (2 * (1 + k1*x + k2*x + f*k1*k2*x**2))
        
        self.sim()

    def sim(self):
        self.binding_curve = []
        self.xvals = []
        for val in np.arange(self.conc_range[0],self.conc_range[1],self.step):
            if self.step_func:
                self.xvals.append(self.step_func(val))
                self.binding_curve.append(self.function(self.step_func(val)))
            else:
                self.xvals.append(val)
                self.binding_curve.append(self.function(val))
			
class PhysPopulation(SSB):
    
    def __init__(self,N=100,selection_strength=1,k1=1e7,k2=1e8,f=10,startRandom=False):
        
        # Biophysical model
        # Create target model on startup by simulating under binding model and populating self.binding curve
        SSB.__init__(self,k1,k2,f)
        self.sim() # get starting curve
        self.target = self.binding_curve
        self.mean_k1 = 0
        self.mean_k2 = 0
        self.mean_f = 0
        self.var_k1 = 0
        self.var_k2 = 0
        self.var_f = 0
        
        # Population parameters
        self.population = []
        self.mean_fitness = 0
        self.fitness_var = 0
        
        # Genetic algorithm params
        self.N = N
        self.selection_strength = selection_strength
        self.cost_function = self._distance_function
        self._initialize_pop(startRandom)
        
        # Brownian motion params
        self.dt = 10
        
    def _update_biophys_params(self,sim=False):
        '''Update means and variances for population K1 and K2s'''
        k1s,k2s,fs,ws = [],[],[],[]
        for i in self.population:
            if sim:
                i.sim()
                i.w = self._get_w(i,self.target)
            k1s.append(i.k1)
            k2s.append(i.k2)
            fs.append(i.f)
            ws.append(i.w)
            
        self.mean_k1 = sum(k1s)/len(k1s)
        self.mean_k2 = sum(k2s)/len(k2s)
        self.mean_f = sum(fs)/len(fs)
        self.mean_fitness = sum(ws)/len(ws)
        
        self.var_k1 = np.var(k1s)
        self.var_k2 = np.var(k2s)
        self.var_f = np.var(fs)
        self.fitness_var = np.var(ws)
        
    def _initialize_pop(self,startRandom=False,K_lower=1e5,K_upper=1e9,f_lower=0,f_upper=100):
        '''Create starting population of size self.N filled with Adair models, each sampled'''
        
        self.population = []
        k1s,k2s,fs,ws = [],[],[],[]
        
        if startRandom:
            rando = np.random.uniform
        for i in range(self.N):
            if startRandom:
                rando = np.random.uniform
                newModel = SSB(rando(K_lower,K_upper),rando(K_lower,K_upper),rando(f_lower,f_upper))
            else:
                newModel = SSB(self.k1,self.k2,self.f)
            newModel.w = self._get_w(self.target,newModel.binding_curve)
            
            k1s.append(newModel.k1)
            k2s.append(newModel.k2)
            fs.append(newModel.f)
            ws.append(newModel.w)
            
            self.population.append(newModel)
            
        self.mean_k1 = sum(k1s)/len(k1s)
        self.mean_k2 = sum(k2s)/len(k2s)
        self.mean_f = sum(fs)/len(fs)
        self.mean_fitness = sum(ws)/len(ws)
        
        self.var_k1 = np.var(k1s)
        self.var_k2 = np.var(k2s)
        self.var_f = np.var(fs)
        self.fitness_var = np.var(ws)

    def _distance_function(self,target,comp):
        '''Compute root mean squared distance between two input binding curves'''
        assert len(comp) == len(target), "Different vector lengths: target: %i, comp: %i" % (len(target),len(comp))
        return np.sqrt(sum(map(lambda (x,y): (x-y)**2, zip(target,comp)))/len(comp))
    
    def _get_w(self,target,comp):
        return self.selection_strength * (1 - self._distance_function(target,comp))
    
    def brownian(self,params=[]):
        '''Take a single brownian step over params vector'''
        out = []
        for i in params:
            new_param = i + norm.rvs(loc=0,scale=2*self.dt)
            if new_param < 0:
                new_param = 0
            out.append(new_param)
        return out
        
    def procreate(self):
        newpop = []
        k1s,k2s,fs,ws = [],[],[],[]
        
        for model in self.population:
            new_k1,new_k2,new_f = self.brownian([model.k1,model.k2,model.f])
            model = SSB(new_k1,new_k2,new_f)
            
            model.w = self._get_w(self.target,model.binding_curve)
            
            k1s.append(model.k1)
            k2s.append(model.k2)
            fs.append(model.f)
            ws.append(model.w)
            
            newpop.append(model)
            
        self.population = newpop
            
        self.mean_k1 = sum(k1s)/len(k1s)
        self.mean_k2 = sum(k2s)/len(k2s)
        self.mean_f = sum(fs)/len(fs)
        self.mean_fitness = sum(ws)/len(ws)
        
        self.var_k1 = np.var(k1s)
        self.var_k2 = np.var(k2s)
        self.var_f = np.var(fs)
        self.fitness_var = np.var(ws)
        
    def select(self):
        fitness_vec = [m.w for m in self.population]
        w_sum = sum(fitness_vec)
        w_norm = [i/w_sum for i in fitness_vec]
        new_pop = np.random.choice(self.population,self.N,p=w_norm)
        self.population = new_pop
        self._update_biophys_params(sim=False)