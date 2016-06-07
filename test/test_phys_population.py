import unittest
from evophys.models.binding.phys_population import PhysPopulation


class SimplisticTest(unittest.TestCase):
	
	def test_GA(self):
		pop = PhysPopulation(N=100,startRandom=True)
		initial_fitness = pop.mean_fitness

		for i in range(15):
			pop.select()

		final_fitness = pop.mean_fitness

		# asserts that this class initalizes correctly
		# assert that fitness should in fact improve
		self.assertTrue(initial_fitness < final_fitness)


if __name__ == '__main__':
    unittest.main()

