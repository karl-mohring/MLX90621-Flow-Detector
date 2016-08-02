from unittest import TestCase
from MLX90621 import Pixel


class TestPixel(TestCase):
    def test_is_adjacent(self):
        """
        Test pixel adjacency discovery
        :return:
        """
        test = Pixel(x=1, y=1, temperature=25)
        x_adjacent = Pixel(x=2, y=1, temperature=25)
        y_adjacent = Pixel(x=1, y=0, temperature=25)
        xy_adjacent = Pixel(x=2, y=2, temperature=25)
        not_adjacent = Pixel(x=3, y=1, temperature=25)

        self.assertTrue(test.is_adjacent(x_adjacent))
        self.assertTrue(test.is_adjacent(y_adjacent))
        self.assertTrue(test.is_adjacent(xy_adjacent))
        self.assertFalse(test.is_adjacent(not_adjacent))

        self.assertFalse(y_adjacent.is_adjacent(not_adjacent))
