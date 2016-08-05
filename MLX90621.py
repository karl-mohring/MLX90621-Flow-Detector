# coding=utf-8
import logging

from SinkNode.Reader import SerialReader
from Queue import Queue
from threading import Thread
import datetime
from matplotlib import pyplot as plt
import numpy as np


class Pixel:
    """
    Pixel class for the MLX90621
    Contains temperature and position information
    """

    def __init__(self, x, y, temperature):
        """
        Create a pixel object
        :param x: x-coordinate of the pixel
        :param y: y-coordinate of the pixel
        :param temperature: Temperature value in °C
        :return: None
        """
        self.x = x
        self.y = y
        self.temperature = temperature

    def __str__(self):
        return "X= {}, Y= {}, Temp= {}".format(self.x, self.y, self.temperature)

    def is_adjacent(self, other_pixel):
        """
        Check if a pixel is adjacent to the current pixel
        :param other_pixel: Other pixel for adjacency testing
        :return: True if pixel is adjacent
        """
        return abs(self.x - other_pixel.x) <= 1 and abs(self.y - other_pixel.y) <= 1


class Blob:
    """
    Contiguous group of pixels that represent a hot or cold object
    """

    def __init__(self):
        """
        Create a new blob
        :return: None
        """
        self.pixels = []
        self.min = [None, None]
        self.max = [None, None]
        self.width = 0
        self.height = 0
        self.area = 0
        self.centroid = 0.0
        self.aspect_ratio = 0.0
        self.average_temperature = 0.0

    def add_pixel(self, pixel):
        """
        Add a pixel to the blob.
        Blob characteristics are recalculated when new pixels are added

        :param pixel: Pixel to be added to the blob
        :return: None
        """

        if len(self.pixels) < 1:
            self.min = [pixel.x, pixel.y]
            self.max = [pixel.x, pixel.y]

        else:
            if pixel.x > self.max[0]:
                self.max[0] = pixel.x
            if pixel.x < self.min[0]:
                self.min[0] = pixel.x
            if pixel.y > self.max[1]:
                self.max[1] = pixel.y
            if pixel.y < self.min[1]:
                self.min[1] = pixel.y

        self.pixels.append(pixel)
        self.area = len(self.pixels)
        self.average_temperature = (self.average_temperature * (self.area - 1) + pixel.temperature)/self.area
        self._recalculate_centroid()
        self.width = (self.max[0] - self.min[0]) + 1
        self.height = (self.max[1] - self.min[1]) + 1
        self.aspect_ratio = self.width/self.height

    def _recalculate_centroid(self):
        """
        Recalculate the centre of the blob.
        The result is stored in the 'centroid' attribute.
        Intended for internal use
        :return: None
        """
        total_x = 0.0
        total_y = 0.0

        for pixel in self.pixels:
            total_x += pixel.x
            total_y += pixel.y

        self.centroid = (total_x / self.area, total_y / self.area)


class TrackedBlob:
    def __init__(self, blob):
        """
        Create a new tracked blob
        :param blob: Blob to start tracking
        :return: Tracked blob object
        """
        self.blob = blob
        self.predicted_position = None
        self.travel = (0, 0)

    def __str__(self):
        return "Tracked blob at {},{}. Size: {} px. Travel: {},{} px".format(self.blob.centroid[0], self.blob.centroid[1], self.blob.area, self.travel[0], self.travel[1])

    def update_blob(self, blob):
        """
        Update the position and features of the currently tracked blob.
        Travel distance of the blob is calculated in the update

        :param blob: Blob to replace the current tracked blob
        :return: None
        """
        movement = np.subtract(blob.centroid, self.blob.centroid)
        self.predicted_position = np.add(blob.centroid, movement)
        self.travel = np.add(movement, self.travel)
        self.blob = blob

    def get_difference_factor(self, other_blob):
        """
        Get the degree of difference between the currently tracked blob and another blob.
        A low difference factor means the blobs are similar and could be the same entity.
        The difference factor should be thresholded to distinguish between the same blob across mutliple frames and
            a new blob that does not match any existing tracked blob.

        Features of the blob are weighted to give more emphasis on the blobs shape and features, rather than position.

        :param other_blob: Blob used for a comparison in difference factor calculations.
        :return:
        """
        difference_factor = 0.0

        # Predicted position - 2x penalty
        if self.predicted_position is not None:
            difference_factor += abs(self.predicted_position[0] - other_blob.centroid[0]) * 2.0
            difference_factor += abs(self.predicted_position[1] - other_blob.centroid[1]) * 2.0

        else:
            # Relative Position - 1x penalty
            difference_factor += abs(float(self.blob.centroid[0] - other_blob.centroid[0]))
            difference_factor += abs(self.blob.centroid[1] - other_blob.centroid[1])

        # Area - 2x penalty
        difference_factor += abs(self.blob.area - other_blob.area) * 1.0

        # Aspect ratio - 10x penalty
        difference_factor += abs(self.blob.aspect_ratio - other_blob.aspect_ratio) * 10.0

        # Average temperature - 10x penalty
        difference_factor += float(abs(float(self.blob.average_temperature) - float(other_blob.average_temperature))) * 10.0

        return difference_factor


class MLX90621:
    LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"

    def __init__(self, reader=SerialReader.SerialReader(port='COM14', baud_rate=115200, start_delimiter='!'),
                 min_blob_size=0, running_average_size=100, colormap=None, log_level=logging.WARNING):
        """
        Create an image processing object for the MLX90621 IR thermopile array sensor.

        :param reader: Reader that collects the raw temperature data. default:SerialReader
        :param min_blob_size: The minimum area in pixels needed for a blob to be counted. default: no minimum
        :param running_average_size: The number of frames collected to calculate the running temperature average for
            each pixel. default: 50 frames
        :param colormap: Colour map used to graph the thermal array. default: None (but 'inferno' looks nice)
        :return: MLX90621 sensor processor object
        """
        self.logger = logging.getLogger(__name__)
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(logging.Formatter(self.LOGGER_FORMAT))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(log_level)
        self.logger.debug("MLX90621 tracking object created")

        # Set up incoming data cache
        self.read_queue = Queue()
        self.reader = reader
        self.reader.set_outbox(self.read_queue)

        # Runtime variables
        self.current_average_size = 1  # TODO - Possible redundant variable; get rid of it?
        self.running_average_size = running_average_size
        self.running_average = None
        self.pixel_std_deviations = np.zeros((4, 16))
        self.pixel_averages = np.zeros((4, 16))
        self.current_frame = np.zeros((4, 16))
        self.distance_threshold = 40
        self.left_passes = 0
        self.right_passes = 0
        self.leftward_travel_direction = True
        self.net_passes = 0

        self.min_blob_size = min_blob_size

        self.read_thread = Thread(target=self._read_loop)
        self.is_running = False

        # Display variables
        self.colourmap = colormap
        self.min_display_value = 25
        self.max_display_value = 32
        self.save_output = False
        if self.colourmap is not None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.axis("off")
            self.img = self.ax.imshow(self.current_frame, interpolation='nearest', cmap=self.colourmap,
                                      vmin=self.min_display_value, vmax=self.max_display_value)
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
        self.logger.info("Starting...")

    def _read_loop(self):
        """
        Reading loop for processing and displaying frames.
        Incoming frames are parsed into 'current_frame' in a 16x4 array
        The image is horizontally flipped from the sensor, so we flip it back.
        :return: None
        """

        tracked_blobs = []

        while self.is_running:
            # Get new frames as they come in
            frame = self.read_queue.get()
            self.logger.debug("Received frame")
            self.current_frame = np.array([frame["row0"][::-1], frame["row1"][::-1], frame["row2"][::-1], frame["row3"][::-1]])
            self.read_queue.task_done()

            # Let the running average build before tracking starts
            add_to_average = True
            if self.running_average is not None and len(self.running_average.shape) > 2:
                if self.running_average.shape[2] >= self.running_average_size:
                    # Process the new frame - track blobs and add the frame to the running average
                    current_blobs = self.find_blobs(self.current_frame)
                    current_blobs = self.remove_small_blobs(current_blobs)

                    # Do not add the frame to the average if there are active blobs
                    # TODO - May have to change this to a blob cooldown so persistent blobs are added after a delay
                    num_blobs = len(current_blobs)
                    self.logger.debug("Frame contains {} blobs".format(num_blobs))
                    if num_blobs:
                        add_to_average = False

                    tracked_blobs = self.track_blobs(tracked_blobs, current_blobs)

                else:
                    self.logger.info("Building background frames: {}/{}".format(self.running_average.shape[2], self.running_average_size))

            if add_to_average:
                self.add_frame_to_average(self.current_frame)

            # Display current frame if you're into that kind of thing
            if self.colourmap is not None:
                self.img.set_data(self.current_frame)
                self.fig.canvas.flush_events()
                self.fig.canvas.draw()

                if self.save_output:
                    self.fig.savefig(datetime.datetime.now().strftime("%Y-%m-%d %H%M%S.%f ") + "mlx.jpg")

    def track_blobs(self, tracked_blobs, current_blobs):
        new_tracked_blobs = []
        left_passes = 0
        right_passes = 0

        # If there are no tracked blobs, then all new blobs become tracked
        if len(tracked_blobs) == 0:
            if len(current_blobs) > 0:
                for blob in current_blobs:
                    new_tracked_blobs.append(TrackedBlob(blob))

        # If there are tracked blobs, update them if any new blobs match. Process any leftover, non-matching blobs.
        else:
            if len(current_blobs) > 0:
                for blob in current_blobs:
                    closest_distance = 999
                    closest_blob_index = None

                    # Find out if the blob is pre-existing
                    for x in xrange(0, len(tracked_blobs)):
                        distance = tracked_blobs[x].get_difference_factor(blob)
                        if distance < closest_distance:
                            closest_blob_index = x
                            closest_distance = distance
                        self.logger.debug("Distance to tracked blob {}: {}".format(x, distance))

                    # If the blob is already being tracked, update the blob and re-add to the track list
                    if closest_distance < self.distance_threshold:
                        self.logger.debug("Blob matches tracked blob with {} difference".format(closest_distance))
                        tracked_blob = tracked_blobs.pop(closest_blob_index)
                        tracked_blob.update_blob(blob)
                        blob = tracked_blob

                    # Blobs that are not currently tracked are added to the list
                    else:
                        blob = TrackedBlob(blob)
                        self.logger.debug("New blob detected")

                    new_tracked_blobs.append(blob)

            # Process old blobs that are no longer being tracked
            if len(tracked_blobs) > 0:
                for blob in tracked_blobs:
                    if blob.travel[1] > 8:
                        right_passes += 1
                    elif blob.travel[1] < -8:
                        left_passes += 1

        self.add_left_passes(left_passes)
        self.add_right_passes(right_passes)

        return new_tracked_blobs

    def add_left_passes(self, num_passes):
        if num_passes > 0:
            self.left_passes += num_passes
            self.update_net_passes()

    def add_right_passes(self, num_passes):
        if num_passes > 0:
            self.right_passes += num_passes
            self.update_net_passes()

    def update_net_passes(self):
        if self.leftward_travel_direction:
            self.net_passes = self.left_passes - self.right_passes
        else:
            self.net_passes = self.right_passes - self.left_passes
        self.logger.info("Net movements: {} \t {} Left\t {} Right".format(self.net_passes, self.left_passes, self.right_passes))

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

        # TODO - This method may be redundant because the running average calculations are more useful
        # If this is the first frame, just replace the average with the frame
        if self.current_average_size == 1:
            self.pixel_averages = frame

        else:
            self.pixel_averages = (self.pixel_averages * (
                self.current_average_size - 1) + frame) / self.current_average_size

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

        if self.running_average is None:
            self.running_average = frame
            self.pixel_averages = frame

        else:

            # Remove the oldest frame if the frame limit has been reached
            if len(self.running_average.shape) > 2 and self.running_average_size <= self.running_average.shape[2]:
                self.running_average = np.delete(self.running_average, 0, axis=2)

            # Add new frame at the end of the background array
            self.running_average = np.dstack((self.running_average, frame))

            self.pixel_std_deviations = np.std(self.running_average, axis=2)
            self.pixel_averages = np.average(self.running_average, axis=2)

    def find_active_pixels(self, frame):
        """
        Find the active pixels in the given frame.
        'Active' means the pixel temperature deviates from the average temperature for that pixel.
        The pixel must deviate by at least 3 standard deviations to be active.
        :param frame: The 4x16 array of pixels
        :return: List of active Pixel objects
        """
        active_pixels = []

        # Iterate through all pixels to calculate which ones are active
        for row in xrange(0, 4):
            for column in xrange(0, 16):

                # A pixel is active if it deviates from its average by 3 std deviations
                if abs(self.pixel_averages[row][column] - frame[row][column]) > (
                            self.pixel_std_deviations[row][column] * 3):
                    pixel = Pixel(row, column, frame[row][column])
                    active_pixels.append(pixel)

        return active_pixels

    def find_blobs(self, frame):
        """
        Find and group active pixels from a frame into blobs.
        A blob is a connected group of adjacent pixels.

        A breadth-first search algorithm finds all members of a blob before starting on the next blob entry.
        Active pixels - contains Pixels that have not yet been assigned to a blob
        Current blob - contains Pixels that are found to be adjacent to other pixels in the blob
        Blob queue - basically a holding area for pixels that are adjacent to the current blob (thus part of the blob),
            but have not been used to search for further adjacent pixels.

        An pixel is pulled from the active pixel list and used to seed the blob.
        Adjacent active pixels to the seed are added to the blob and used to expand the adjacency search.
        The blob is complete when no more active pixels are adjacent to its own.

        :param frame: A 4x16 frame containing pixel information for blob extraction
        :return: A list of blobs that contain a cluster of Pixels
        """
        blobs = []
        blob_queue = Queue()
        active_pixels = self.find_active_pixels(frame)

        # Assign every active pixel to a blob
        while len(active_pixels) > 0:

            # First pixel in a new blob
            blob_queue.put(active_pixels.pop())
            current_blob = Blob()

            # Grow the current blob - search for adjacent pixels
            while not blob_queue.empty():
                current_pixel = blob_queue.get()

                # Add any adjacent pixels to the queue for searching
                non_adjacent_pixels = []
                for x in active_pixels[:]:
                    if current_pixel.is_adjacent(x):
                        blob_queue.put(x)

                    # Non-adjacent pixels get sent back to the active pixels array
                    else:
                        non_adjacent_pixels.append(x)

                # Update the active pixel list to only include non-assigned pixels
                active_pixels = non_adjacent_pixels
                current_blob.add_pixel(current_pixel)
                blob_queue.task_done()

            blobs.append(current_blob)

        return blobs

    def remove_small_blobs(self, blob_list):
        big_blobs = [blob for blob in blob_list if blob.area > self.min_blob_size]
        return big_blobs


if __name__ == '__main__':
    reader = SerialReader.SerialReader(port='COM13', baud_rate=115200, start_delimiter='!')
    sensor = MLX90621(reader=reader, colormap='inferno', log_level=logging.INFO)
    sensor.start()
    while True:
        try:
            pass

        except KeyboardInterrupt:
            sensor.stop()
