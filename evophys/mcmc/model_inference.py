import emcee
import numpy as np
from evophys.models.binding.binding_model import BindingModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""
Parameter inference for binding models using Markov chain Monte Carlo methods.
"""
class ModelInference(object):
	def __init__(self, input_model):
		# input model must inherit BindingModel
		assert(isinstance(input_model, BindingModel))
		self.input_model = input_model
		self.input_model_param_names = self.input_model.paramD.keys()
		self.model_class = self.input_model.__class__ 

		self.y_data = None
		self.x_data = None
		self.posterior_samples = None

		self.sampling_finished = False

	def load_data(self, input = None):
		''' Load in the binding data to be used for model fitting. '''
		self.x_data = self.input_model.xvals
		if input == None:
			self.y_data = self.input_model.output
		else:
			self.y_data = input


	def __ln_prior(self, theta):
		return 0.0 # flat priors for now

	def __ln_likelihood(self, theta, x, y):
		assert(len(theta) == len(self.input_model_param_names)) + 1
		d = {}
		for index, param_key in enumerate(self.input_model_param_names):
			d[param_key] = theta[index]
		sigma = theta[-1]

		pred = self.input_model.get_binding_curve(theta[:-1])	

		inv_sigma2 = 1.0/(sigma**2) 

		# Here, we're assuming Gaussian noise around binding curve
		return -0.5*(np.sum((y-pred)**2*inv_sigma2 - np.log(inv_sigma2)))

		
	def __ln_posterior(self, theta, x, y):
		lp = self.__ln_prior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + self.__ln_likelihood(theta, x, y)

	def sample(self, iterations=1000, nwalkers=10, burnin=100):
		''' Posterior sampling with adaptive MCMC. '''

		# Set up the sampler.
		ndim = len(self.input_model_param_names) + 1 # add one for sigma parameter
		pos = [ 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
		self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__ln_posterior, args=(self.x_data, self.y_data))
		res = self.sampler.run_mcmc(pos, iterations, rstate0=np.random.get_state())

		samples = self.sampler.chain[:, burnin:, :].reshape((-1, ndim))

		# extract posterior samples from numpy array and construct a DataFrame
		self.posterior_samples = {} 
		for index, param_key in enumerate(self.input_model_param_names):
			self.posterior_samples[param_key] = samples[burnin:,index]
		self.posterior_samples["sigma"] = samples[burnin:,-1]

		self.posterior_samples = pd.DataFrame(self.posterior_samples)
		self.sampling_finished = True

	def convergence_plot(self, logarithmic = True):
		# this is probably a little silly, to have these very particular plotting methods...
		
		if not self.sampling_finished:
			raise Exception("Must run .sample() before any output results can be viewed.")

		num_plots = len(self.input_model_param_names) + 1

		plt.clf()
		fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(8, 9))
		for i in range(num_plots):			
			axes[i].plot(self.sampler.chain[:, :, i].T, color="k", alpha=0.4)
			if logarithmic: axes[i].set_yscale("log")
			try:
				axes[i].set_ylabel(self.input_model_param_names[i])
			except IndexError:
				axes[i].set_ylabel("sigma")

	def correlation_plot(self, logarithmic=True):
		# plot pairwise parameter correlations with a scatterplot matrix

		if not self.sampling_finished:
			raise Exception("Must run .sample() before any output results can be viewed.")

		if not logarithmic: 
			sns.pairplot(self.posterior_samples)
		else:
			df = np.log10(self.posterior_samples.iloc[:,:-1]).dropna()
			g = sns.PairGrid(df, diag_sharey=False)
			g.map_lower(sns.kdeplot, cmap="Blues_d")
			g.map_upper(plt.scatter,alpha=.1)
			g.map_diag(sns.kdeplot, lw=3)


