import logging
import sys
from Queue import Queue
from threading import Thread

import datetime
from SinkNode.Reader import SerialReader
from matplotlib import pyplot as plt
import numpy as np

# # Serial grabber
# draw_queue = Queue()
# reader = SerialReader.SerialReader(port="COM13", baud_rate=115200, start_delimiter='!', outbox=draw_queue, logger_level=logging.INFO)
# reader.start()
#
# # Graphing parameters
# cmap = 'inferno'
# zvals = np.zeros((4, 16))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# img = ax.imshow(zvals, interpolation='nearest', cmap=cmap, vmin=20, vmax=32)
# fig.show()
#
# while True:
#     try:
#         frame = draw_queue.get()
#         zvals = np.array([frame["row0"][::-1], frame["row1"][::-1], frame["row2"][::-1], frame["row3"][::-1]])
#         draw_queue.task_done()
#
#         img.set_data(zvals)
#         fig.canvas.flush_events()
#         fig.canvas.draw()
#
#     except KeyboardInterrupt:
#         sys.exit()



class Pixel(object):
    pass


class MLX90621:
    def __init__(self, port='COM13', baud=115200, min_blob_size=4, running_average_size=50, show_output=False, cmap='inferno'):
        self.port = port
        self.baud_rate = baud
        self.read_queue = Queue()
        self.reader = SerialReader.SerialReader(port=self.port, baud_rate=self.baud_rate, start_delimiter='!', outbox=self.read_queue)

        self.current_average_size = 1
        self.running_average_size = running_average_size
        self.min_blob_size = min_blob_size
        self.pixel_std_deviations = np.zeros((4, 16))
        self.pixel_averages = np.zeros((4, 16))

        self.running_average = np.zeros((4, 16))
        self.current_frame = np.zeros((4, 16))
        self.read_thread = Thread(target=self._read_loop)
        self.is_running = False

        self.cmap = cmap
        self.min_display_value = 25
        self.max_display_value = 32
        self.save_output = False
        self.show_output = show_output
        if self.show_output:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.axis("off")
            self.img = self.ax.imshow(self.current_frame, interpolation='nearest', cmap=self.cmap, vmin=self.min_display_value, vmax=self.max_display_value)
            plt.colorbar(self.img)
            self.fig.show()

    def start(self):
        """
        Start receiving data from the sensor.
        Reading and display threads are started.
        :return: None
        """
        self.is_running = True
        self.reader.start()
        self.read_thread.start()

    def _read_loop(self):
        """
        Reading loop for processing and displaying frames.
        Incoming frames are parsed into 'current_frame' in a 16x4 array
        The image is horizontally flipped from the sensor, so we flip it back.
        :return: None
        """
        while self.is_running:
            frame = self.read_queue.get()
            self.current_frame = np.array([frame["row0"][::-1], frame["row1"][::-1], frame["row2"][::-1], frame["row3"][::-1]])
            self.read_queue.task_done()

            if self.show_output:
                self.img.set_data(self.current_frame)
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()

                if self.save_output:
                    self.fig.savefig(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S.%f ") + "mlx.jpg")

    def stop(self):
        """
        Stop receiving data from the sensor.
        Shut down any reading and display threads.
        :return: None
        """
        self.is_running = False
        self.reader.stop()

    def _add_weighted_frame_to_background(self, frame):
        """
        Add a frame to the background image.
        Active pixels that only appear for the short term are averaged out.
        Long-term active pixels gradually become part of the background as more frames are added to the average.
        The impact of the running average is defined in 'running_average_size' (default: 50)
        This means that each new frame added to the background has a weighting of 1/50.

        Until the 'running_average_size' number of frames has been added, the weighting of the frame is inversely
        proportional to the number of frames averaged.
            e.g: The second frame is worth 1/2, third frame is worth 1/3, and so on...
                ...49th frame is worth 1/49, 50th+ frame is worth 1/50

        :param frame: New frame to add to the background
        :return: None
        """
        # If this is the first frame, just replace the average with the frame
        if self.current_average_size == 1:
            self.pixel_averages = frame

        else:
            self.pixel_averages = (self.pixel_averages * (self.current_average_size - 1) + frame) / self.current_average_size

        # Keep increasing the average each time until it hits the limit
        if self.current_average_size < self.running_average_size:
            self.current_average_size += 1

    def add_frame_to_average(self, frame):
        """
        Add a new frame into the background calculations
        Up to 'running_average_size' frames are included in the calculations with the oldest frames removed when
            new frames are added in.
        The running average of each pixel is stored in the 'pixel_averages' array, while the standard deviations of each
            pixel are stored in 'pixel_std_deviations'.
        This method is the true way of calculating the running average, as opposed to using a weighted average
        :param frame: Frame to be added to the calculations
        :return: None
        """

        # Remove the oldest frame if the frame limit has been reached
        if self.running_average_size <= self.running_average.shape[2]:
            self.running_average = np.delete(self.running_average, 0)

        # Add new frame at the end of the background array
        self.running_average = np.dstack(self.running_average, frame)

        self.pixel_std_deviations = np.std(self.running_average, axis=2)
        self.pixel_averages = np.average(self.running_average, axis=2)

    def find_active_pixels(self, frame):
        """

        :param frame:
        :return:
        """

        blobs = []

        for row in xrange(0, 4):
            for column in xrange(0, 16):
                if abs(self.pixel_averages[row][column] - frame[row][column]) > (self.pixel_std_deviations[row][column] * 3):

                    # Pixel is active; add to existing blob or make new blob
                    if len(blobs) > 0:
                        for blob in blobs:
                            is_neighbour = blob.is_neighbour(row, column)
                            if is_neighbour:
                                blob.add_pixel(row, column, frame[row][column])
                                break

        # Iterate through pixels to see which ones have deviated

        pass

    def find_blobs(self, frame):
        pass

    class Pixel:
        def __init__(self, x, y, temperature):
            self.x = x
            self.y = y
            self.temperature = temperature

    class Blob:
        def __init__(self):
            self.pixels = []
            self.uniformity
            self.size
            self.centroid

        def is_neighbour(self, x, y):
            """
            Find if a pixel is adjacent to an existing pixel in the blob.
            Diagonal adjacency is also considered here.

            :param x: row of the pixel
            :param y: column of the pixel
            :return: True if the pixel is adjacent to an existing blob pixel
            """
            is_neighbour = False

            for pixel in self.pixels:
                if abs(pixel.x - x) <= 1 and abs(pixel.y - y) <= 1:
                    is_neighbour = True
                    break

            return is_neighbour

        def get_centroid(self):
            pass

        def get_size(self):
            pass

        def add_pixel(self, x, y, value):
            pixel = Pixel(x, y, value)
            self.add_pixel(pixel)

        def add_pixel(self, pixel):
            self.pixels.append(pixel)




if __name__ == '__main__':
    sensor = MLX90621(show_output=True, cmap='inferno')
    sensor.start()
    while True:
        try:
            pass

        except KeyboardInterrupt:
            sensor.stop()
