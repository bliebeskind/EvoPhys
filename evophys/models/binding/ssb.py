'''
Single Site Binding model
other documentation....
'''

import numpy as np


class SSB:

    def __init__(self,k1=1e7,k2=1e8,f=10):
		self.k1 = k1
		self.k2 = k2
		self.f = f
		self.w = None
		
		self.paramD = {"k1":self.k1,"k2":self.k2,"f":self.f}
		
		self.conc_range = (-12,-3)
		self.xvals = []
		
		self.output = []
		
		self.step = .01
		self.step_func = lambda x: 10**x

		self.function = lambda x: (k1*x + k2*x + f*2*k1*k2*x**2) / (2 * (1 + k1*x + k2*x + f*k1*k2*x**2))
		
		self.sim()

    def sim(self):
        self.output = []
        self.xvals = []
        for val in np.arange(self.conc_range[0],self.conc_range[1],self.step):
            if self.step_func:
                self.xvals.append(self.step_func(val))
                self.output.append(self.function(self.step_func(val)))
            else:
                self.xvals.append(val)
                self.output.append(self.function(val))