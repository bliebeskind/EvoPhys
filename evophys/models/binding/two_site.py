import numpy as np
from evophys.models.binding.binding_model import BindingModel


class TwoSiteNoCoop(BindingModel):
    """
    This simple class represents a two-site binding model with no cooperativity.
    This class is a little at odds with the existing SSBdynam, and is more of a 
        testing ground for MCMC methods; will probably be obsolete soon. 
    """

    def __init__(self,paramD = {"k1": 1e-6,"k2": 1e-8}):
        self.paramD = paramD
        self.k1 = paramD['k1']
        self.k2 = paramD['k2']
        self.f = 1
        self.w = None
        self.conc_range = (-12,-1)
        self.step = .01
        self.function = lambda x: (self.k1*x + self.k2*x + 2*self.f*self.k1*self.k2*(x**2)) / float((2 * (1 + self.k1*x + self.k2*x + self.f*self.k1*self.k2*(x**2))))
        self.sim()

    def sim(self):
        self.xvals = [10**val for val in range(self.conc_range[0],self.conc_range[1])]
        self.binding_curve = self.get_binding_curve(self.paramD.values())
            
    def get_paramD(self):
        return self.paramD

    def get_output(self):
        return self.output

    def get_binding_curve(self, params):
        """
        For some input parameter values, compute and return the predicted binding curve.
        """
        (k1, k2) = params
        f=1
        func = lambda x : (k1*x + k2*x + f*2*k1*k2*(x**2)) / (2 * (1 + k1*x + k2*x + f*k1*k2*(x**2)))
        curve = [func(x) for x in self.xvals]
        return curve