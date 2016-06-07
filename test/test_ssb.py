import unittest
from evophys.models.binding.ssb import SSB


class SimplisticTest(unittest.TestCase):
	
	def test_binding_curve(self):
		s = SSB()
		self.assertTrue( len(s.binding_curve) > 0 )


if __name__ == '__main__':
    unittest.main()

