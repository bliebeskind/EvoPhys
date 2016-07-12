import random
import numpy as np
from itertools import combinations

class SSBmut:

	def __init__(self,mutations=None,nk_prob=.1,anyF_prob=.45):
		if mutations == None:
			self.mutations = {"k":lambda x:x + random.normalvariate(0,10),
								"f":lambda x:x + random.normalvariate(0,.1), 
								"nk":lambda x:x*2 if x < 8 else x}
		else:
			self.mutations = mutations
		self._nk_prob = nk_prob
		self._anyF_prob = anyF_prob
		self.param_vec = None
		self.probs_vec = None
		self.n_params = None # when called, check this, and if the same, don't update other attributes
		self.mut_defs = None
		
	def _update_probs(self):
		# divvy up probabilities - right now nk_prob remains static, 
		# with total F and K probs getting smaller with more binding sites
		self._ks = sorted(filter(lambda x: x.startswith("k"),self.paramD.keys()))
		self._fs = sorted(filter(lambda x: x.startswith("f"),self.paramD.keys()))
		self.param_vec = ["nk"] + self._ks + self._fs
		if len(self._fs) > 0:
			fprob = self._anyF_prob/float(len(self._fs))
			self._anyK_prob = np.round(1 - self._nk_prob - self._anyF_prob, 10) # have to round to get rid of floating point errors
		else:
			fprob = 0
			self._anyK_prob = np.round(1 - self._nk_prob, 10)
		assert 0 <= self._anyK_prob <= 1, "nk_prob + anyF_prob must be between 0 and 1"
		kprob = self._anyK_prob/float(len(self._ks))
		self.probs_vec = [self._nk_prob] + [kprob]*len(self._ks) + [fprob]*len(self._fs)
		
		#testit
		testsum = sum(i for i in self.probs_vec)
		np.testing.assert_almost_equal(testsum,1.0,err_msg="Probabilities don't add up: \n%s" % 
										str(dict(zip(self.param_vec,self.probs_vec))))
	
	def _update_defs(self):
		# Define mutation types - Weiner proc for f and k, and multiplication by 2 for nk
		k_defs = {i:self.mutations["k"] for i in self._ks} # defs for k params
		f_defs = {i:self.mutations["f"] for i in self._fs} # defs for f params
		self.mut_defs = k_defs.copy()
		self.mut_defs.update(f_defs)
		self.mut_defs["nk"] = self.mutations["nk"] # def for nk
		
	def _construct_output_params(self):
		'''
		Tandem duplication model.
			Molecule:
		   k1-<f12>-k2
			
			becomes: 
				
			   f12
			k1 	- 	k2
	f13 = 1	|	X	|   f24 = 1
			k3	_ 	k4
			   f12
			
			with f14 and f23 also = 1
		'''
		
		# set new ks
		currNumKs = len(self._ks)
		self._new_ks = ["k"+str(i) for i in range(currNumKs+1,self.out_paramD["nk"]+1)]
		for k in self._new_ks: # tandem duplication: assign new site to parent site parameter
			parentK = "k" + str(int(k[1:])-len(self._new_ks))
			try:
				self.out_paramD[k] = self.paramD[parentK]
			except KeyError, e:
				print parentK, self._new_ks, self.paramD
				raise Exception("%s" % e)
		
		# set new fs
		for i in range(2,self.out_paramD["nk"]+1):
			for tup in combinations(range(1,self.out_paramD["nk"]+1),i): # set fs, build up ks
				fstring = "f"+"".join(map(lambda x: str(x),tup))
				if fstring in self.paramD:
					continue
				maybe_fParent = "f" + "".join(map(lambda x: str(x-self.paramD["nk"]),map(int,fstring[1:]))) # e.g. f12 if on f34
				if maybe_fParent in self.paramD:
					self.out_paramD[fstring] = self.paramD[maybe_fParent]
				else:
					self.out_paramD[fstring] = 1

	def mutate(self,paramD):
		self.paramD = paramD
		#self.out_paramD = {}
		if len(paramD) != self.n_params: # should only happen the first time
			self._update_probs()
			self._update_defs()
		self.param_mutated = np.random.choice(self.param_vec,p=self.probs_vec)
		self.param_mutated
		self.out_paramD = {}
		if self.param_mutated == "nk":
			mut_func = self.mut_defs["nk"]
			self.out_paramD = self.paramD.copy()
			self.out_paramD["nk"] = int(mut_func(self.paramD["nk"]))
			if self.out_paramD["nk"] != self.paramD["nk"]:
				self._construct_output_params()
				self._update_defs()
		else:
			self.out_paramD = self.paramD.copy()
			mut_func = self.mut_defs[self.param_mutated]
			self.out_paramD[self.param_mutated] = mut_func(self.paramD[self.param_mutated])
		return self.out_paramD