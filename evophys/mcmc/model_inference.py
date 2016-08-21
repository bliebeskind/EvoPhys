import emcee
import numpy as np
from evophys.models.binding.binding_model import BindingModel

class ModelInference(object):
	def __init__(self, input_model):
		assert(isinstance(input_model, BindingModel))
		self.input_model = input_model
		self.input_model_param_names = self.input_model.paramD.keys()
		self.model_class = self.input_model.__class__ 

		self.y_data = None
		self.x_data = None
		self.posterior_samples = None

	def load_data(self, input = None):
		self.x_data = self.input_model.xvals
		if not input:
			self.y_data = self.input_model.output
		else:
			self.y_data = input


	def __ln_prior(self, theta):
		return 0.0 # flat priors for now

	def __ln_likelihood(self, theta, x, y):
		assert(len(theta) == len(self.input_model_param_names))
		d = {}
		for index, param_key in enumerate(self.input_model_param_names):
			d[param_key] = theta[index]
		
		self.input_model.get_binding_curve()	
		inv_sigma2 = 1.0/(3) # TODO add sigma to the param list
		return -0.5*(np.sum((y-m.output)**2*inv_sigma2 - np.log(inv_sigma2)))

		
	def __ln_posterior(self, theta, x, y):
		lp = self.__ln_prior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + self.__ln_likelihood(theta, x, y)

	def sample(self, iterations=1000, nwalkers=10, burnin=100):
		# Set up the sampler.
		ndim = len(self.input_model_param_names)
		pos = [ 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__ln_posterior, args=(self.x_data, self.y_data))
		res = self.sampler.run_mcmc(pos, iterations, rstate0=np.random.get_state())

		samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))

		self.posterior_samples = {}
		for index, param_key in enumerate(self.input_model_param_names):
			self.posterior_samples[param_key] = samples[:,i]

