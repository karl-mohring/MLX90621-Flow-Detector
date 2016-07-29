import logging
from threading import Thread
from Queue import Queue
import json

LOGGER_FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"

__author__ = 'Leenix'


class Reader(object):
    """
    Parent class.
    Reads data from a source, which is specified by the child class.
    Data is converted into JSON format, then placed in a queue for processing.
    """

    def __init__(self, outbox=None, logger_level=logging.FATAL, reader_id=__name__, logger_format=LOGGER_FORMAT):
        self.outbox = outbox

        self.is_running = False
        self.read_thread = Thread(target=self._read_loop)

        self.logger = logging.getLogger(reader_id)
        log_handler = logging.StreamHandler()
        log_handler.setFormatter(logging.Formatter(logger_format))
        self.logger.addHandler(log_handler)
        self.logger.setLevel(logger_level)

    def stop(self):
        """
        Halt the reading process.
        Halting the read does not affect the read queue.
        :return:
        """
        self.logger.info("Stopping reader...")
        self.is_running = False

    def start(self):
        """
        Read in packets of data and convert them to JSON format
        JSON packets are placed in the read queue to await further processing
        :return:
        """
        self.is_running = True
        self.read_thread.start()
        self.logger.info("Starting reader...")

    def _read_loop(self):
        """
        Read in data from a source or stream
        Data is packetised and converted into JSON format
        :return:
        """
        while self.is_running:
            raw_entry = self.read_entry()
            self.logger.debug("Raw entry: " + str(raw_entry))

            processed_entry = self.convert_to_json(raw_entry)
            self.logger.debug("Processed entry: {}".format(processed_entry))

            if processed_entry is not None:
                self.outbox.put(processed_entry)

    def read_entry(self):
        """
        Read a single data entry in from the source stream
        No data formatting is performed. Data is just cut into a manageable chunk
        :return:
        """
        raise Exception("Method [read_entry] not implemented")

    # TODO - change reader to use a processor class as well - or leave as-is
    def convert_to_json(self, entry_line):
        """
        Convert the entry line to JSON
        Entry lines should already be in a JSON string; extract it.

        :param entry_line: JSON-formatted string
        :return: JSON object of the entry string
        """

        entry = ""
        try:
            entry = json.loads(entry_line.replace("\r\n", "|"))

        except ValueError:
            self.logger.warning("Entry could not be converted to JSON: {}".format(entry_line))

        return entry

    def set_outbox(self, queue):
        """
        Set the output destination of parsed entries
        :param queue: Queue object that will hold parsed entries
        :return:
        """
        assert isinstance(queue, Queue)
        self.outbox = queue

