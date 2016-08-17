import numpy as np
from evophys.models.binding.binding_model import BindingModel


class TwoSiteNoCoop(BindingModel):
    """
    This simple class represents a two-site binding model with no cooperativity.
    This class is a little at odds with the existing SSBdynam, and is more of a 
        testing ground for MCMC methods; will probably be obsolete soon. 
    """

    def __init__(self,paramD = {"k1": 1e6,"k2": 1e8}):
        self.paramD = paramD
        self.k1 = paramD['k1']
        self.k2 = paramD['k2']
        self.f = 1
        self.w = None
        self.conc_range = (-12,-3)
        self.binding_curve = []
        self.xvals = []
        self.step = .01
        self.function = lambda x: (self.k1*x + self.k2*x + self.f*2*self.k1*self.k2*x**2) / float((2 * (1 + self.k1*x + self.k2*x + self.f*self.k1*self.k2*x**2)))
        
        self.sim()

    def sim(self):
        self.binding_curve = []
        self.xvals = []
        for val in np.arange(self.conc_range[0],self.conc_range[1],self.step):
            self.xvals.append(val)
            self.binding_curve.append(self.function(val))

    def get_paramD(self):
        return self.paramD

    def get_output(self):
        return self.output

    def get_binding_curve(self, params):
        """
        For some input parameter values, compute the predicted binding curve.
        """
        (k1, k2) = params
        f=1
        func = lambda x : (k1*x + k2*x + f*2*k1*k2*x**2) / (2 * (1 + k1*x + k2*x + f*k1*k2*x**2))
        curve = [func(x) for x in self.xvals]
        return curve