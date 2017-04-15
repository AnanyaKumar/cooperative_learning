import unittest
from car import Car

eps = 1e-8

class TestCar(unittest.TestCase):

    def test_init(self):
        car = Car(2.0, 3.0, 0.5)
        self.assertEqual(car.start_x, 2.0)
        self.assertEqual(car.start_y, 3.0)
        self.assertEqual(car.pos_x, 2.0)
        self.assertEqual(car.pos_y, 3.0)
        self.assertEqual(car.vel_x, 0.0)
        self.assertEqual(car.vel_y, 0.0)
        self.assertEqual(car.max_x, 2.0)
        self.assertTrue(car.is_alive)
        self.assertTrue(abs(car.get_reward() - 0.0) < eps)

    def test_motion(self):
        car = Car(2.0, 3.0, 0.5)

        car.move(1.0, -2.0, 1.0)
        self.assertTrue(abs(car.vel_x - 1.0) < eps)
        self.assertTrue(abs(car.vel_y - (-2.0)) < eps)
        self.assertTrue(abs(car.pos_x - 2.5) < eps)
        self.assertTrue(abs(car.pos_y - 2.0) < eps)
        self.assertEqual(car.start_x, 2.0)
        self.assertEqual(car.start_y, 3.0)
        self.assertTrue(abs(car.max_x - 2.5) < eps)

        car.move(2.0, 2.0, 1.0)
        self.assertTrue(abs(car.vel_x - 3.0) < eps)
        self.assertTrue(abs(car.vel_y - 0.0) < eps)
        self.assertTrue(abs(car.pos_x - 4.5) < eps)
        self.assertTrue(abs(car.pos_y - 1.0) < eps)
        self.assertTrue(abs(car.max_x - 4.5) < eps)

        car.move(-10.0, 2.0, 1.0)
        self.assertTrue(abs(car.max_x - 4.5) < eps)

    def test_die(self):
        car = Car(2.0, 3.0, 0.5)
        old_reward = car.get_reward()
        car.die()
        new_reward = car.get_reward()
        self.assertTrue(new_reward < old_reward)


if __name__ == '__main__':
    unittest.main()
    