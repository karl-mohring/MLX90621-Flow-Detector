from unittest import TestCase
from MLX90621 import Blob,  Pixel


class TestBlob(TestCase):
    def test_blob_size(self):
        """
        Test blob size calculations after adding a new pixel
        :return:
        """
        blob = Blob()

        blob.add_pixel(Pixel(1, 2, 3))
        self.assertEqual(blob.size, 1)

        blob.add_pixel(Pixel(4, 5, 6))
        self.assertEqual(blob.size, 2)

    def test_blob_centroid(self):
        """
        Test blob centroid calculations after adding a new pixel
        :return:
        """
        blob = Blob()

        blob.add_pixel(Pixel(1, 3, 0))
        self.assertEqual(blob.centroid, (1, 3))

        blob.add_pixel(Pixel(3, 3, 0))
        self.assertEqual(blob.centroid, (2, 3))

        blob.add_pixel(Pixel(1, 1, 0))
        self.assertEqual(blob.centroid, (5/3, 7/3))

        blob.add_pixel(Pixel(3, 1, 0))
        self.assertEqual(blob.centroid, (2, 2))
