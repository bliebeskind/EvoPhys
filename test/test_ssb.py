import unittest
from evophys.models.binding.ssb import SSBdynam


class SimplisticTest(unittest.TestCase):
	
	def test_binding_curve(self):
		s = SSBdynam()
		s.sim()

		# asserts that this class initalizes correctly
		self.assertTrue( len(s.get_output()) > 0 )


if __name__ == '__main__':
    unittest.main()

