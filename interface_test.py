import unittest
from car import Car
import interface

eps = 1e-8

class TestInterface(unittest.TestCase):

    def test_input(self):
        car = Car(2.0, 3.0, 0.1)
        car2 = Car(4.0, 6.0, 0.1)
        car3 = Car(99.0, 99.0, 0.1)
        car2.move(2.0, 2.0, 1.0)
        inarr = interface.build_nn_input([car, car2, car3], 1)
        self.assertEqual(inarr.shape, (3, 8))
        self.assertEqual(inarr[0][0], 3.0)
        self.assertEqual(inarr[0][1], 4.0)
        self.assertEqual(inarr[0][2], 2.0)
        self.assertEqual(inarr[0][3], 2.0)
        self.assertEqual(inarr[1][0], -3.0)
        self.assertEqual(inarr[1][1], -4.0)
        self.assertEqual(inarr[1][2], -2.0)
        self.assertEqual(inarr[1][3], -2.0)


if __name__ == '__main__':
    unittest.main()