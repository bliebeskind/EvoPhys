from abc import ABCMeta, abstractmethod

class BindingModel(object):
	"""
	Abstract base class for all binding models. 
	"""
	__metaclass__ = ABCMeta

	@abstractmethod
	def __init__(self, parameter_dictionary):
		"""
		A binding model must instantiate with parameter dictionary.
		"""
		pass

	@abstractmethod
	def sim(self):
		"""
		A binding model must implement a method
		to simulate a binding curve. This method has
		no return value, stores resulting curve as a side effect.
		"""
		pass

	@abstractmethod
	def get_paramD(self):
		"""
		A binding model must have a public method to return
		the parameter dictionary.
		"""
		pass

	@abstractmethod
	def get_output(self):
		"""
		A binding model must have a public method to return
		the simulated binding curve.
		"""
		pass