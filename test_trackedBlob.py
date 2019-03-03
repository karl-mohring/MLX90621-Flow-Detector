from unittest import TestCase
from MLX90621 import Blob, TrackedBlob, Pixel


class TestTrackedBlob(TestCase):
    def setUp(self):
        self.first_blob = Blob()
        self.first_blob.add_pixel(Pixel(3, 1, 10))
        self.first_blob.add_pixel(Pixel(3, 2, 10))
        self.first_blob.add_pixel(Pixel(4, 2, 10))
        self.first_blob.add_pixel(Pixel(4, 1, 10))

        self.second_blob = Blob()
        self.second_blob.add_pixel(Pixel(1, 0, 10))
        self.second_blob.add_pixel(Pixel(1, 1, 10))
        self.second_blob.add_pixel(Pixel(2, 1, 10))
        self.second_blob.add_pixel(Pixel(2, 0, 10))

        self.third_blob = Blob()
        self.third_blob.add_pixel(Pixel(5, 1, 10))
        self.third_blob.add_pixel(Pixel(5, 2, 8))
        self.third_blob.add_pixel(Pixel(6, 2, 8))
        self.third_blob.add_pixel(Pixel(6, 1, 1))
        self.third_blob.add_pixel(Pixel(7, 1, 9))
        self.third_blob.add_pixel(Pixel(6, 3, 10))

    def test_update_blob(self):
        tracked_blob = TrackedBlob(self.first_blob)
        tracked_blob.update_blob(self.second_blob)

        self.assertEqual(tracked_blob.blob.centroid, self.second_blob.centroid)
        self.assertEqual(tracked_blob.travel[0], -2)

    def test_update_blob_2(self):
        tracked_blob = TrackedBlob(self.second_blob)
        tracked_blob.update_blob(self.first_blob)
        self.assertEqual(tracked_blob.blob.centroid, self.first_blob.centroid)
        self.assertEqual(tracked_blob.travel[0], 2)
        self.assertEqual(tracked_blob.travel[1], 1)

    def test_get_distance_factor(self):
        tracked_blob = TrackedBlob(self.first_blob)
        distance = tracked_blob.get_difference_factor(self.second_blob)

        # Distance:
        # Position = 3 (2 + 1)
        # Area = 0
        # Aspect Ratio = 0
        # Averaged = 0
        self.assertEqual(distance, 3)

        distance = tracked_blob.get_difference_factor(self.third_blob)
        # Distance:
        # Position = 2.33 + 0.166
        # Area = 2 * 5
        # Aspect Ratio = 0
        # Averaged = 2.333 * 10
        self.assertAlmostEqual(distance, 35.8333333)

