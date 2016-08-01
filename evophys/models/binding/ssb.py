'''
Site-specific Binding model
other documentation....
'''

import random
import numpy as np
from itertools import combinations


class SSBdynam:

	def __init__(self,paramD={"nk":2,"k1":1e7,"k2":1e8,"f12":10}):
		
		assert type(paramD) is dict, "provide parameter dictionary"
		self.paramD = paramD
		assert self.paramD["nk"] > 0, "nk must be greater than 0"
		self._construct_attributes()
		self._construct_function()

		self.w = None
		
		self.conc_range = (-12,-3)
		self.xvals = []
		
		self.output = []
		
		self.step = .1
		self.step_func = lambda x: 10**x
		
		self.sim()
		
	def _construct_f(self,i,tup):
		return "".join(["{f"+"".join(map(str,tup[j-2:]))+"}*" for j in range(i,2,-1)])
		
	def _construct_attributes(self):
		for i,j in self.paramD.iteritems():
			if i == "nk": j = int(j)
			setattr(self,i,j)
		
	def _construct_function(self):
		'''Construct the equilibrium binding equation'''
		kpart = "+".join(["{k"+str(i)+"}*x" for i in range(1,self.nk+1)])
		numerator = "("+kpart
		denominator = "("+str(self.nk)+"*"+"(1+"+kpart
		for i in range(2,self.nk+1):
			for tup in combinations(range(1,self.nk+1),i):
				numerator += "+"
				denominator += "+"
				numerator += "{}*".format(i) + "%s" % (self._construct_f(i,tup) if i > 2 else '') + "{f" + "".join(map(str,tup)) \
								+"}*" + "*".join(map(lambda x: "{k"+str(x)+"}",tup)) + "*x**%d" % i
				denominator += "%s" % (self._construct_f(i,tup) if i > 2 else '') + "{f" + "".join(map(str,tup)) +"}*" \
								+ "*".join(map(lambda x: "{k"+str(x)+"}",tup)) + "*x**{}".format(i)
		numerator += ")"
		denominator += "))"
		self._func_to_format = "lambda x: " + numerator + " / " + denominator
		self.function_string = self._func_to_format.replace("}",'').replace("{",'').split(":")[1].strip()
		self.function = eval(self._func_to_format.format(**self.paramD))

	def sim(self):
		'''Populate self.output with output from self.function'''
		self.output = []
		self.xvals = []
		for val in np.arange(self.conc_range[0],self.conc_range[1],self.step):
			if self.step_func:
				self.xvals.append(self.step_func(val))
				self.output.append(self.function(self.step_func(val)))
			else:
				self.xvals.append(val)
				self.output.append(self.function(val))
		
	
	
	
	
	
	
				
