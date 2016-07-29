import sys
import logging
from Queue import Queue
import serial
from serial import SerialException
from SinkNode.Reader import Reader


class SerialReader(Reader):
    """
    Serial reader that sorts the incoming stream into packets.
    """

    def __init__(self, port, baud_rate, start_delimiter=None, stop_delimiter='\n',
                 outbox=None, logger_level=logging.FATAL):

        self.port = port
        self.baud_rate = baud_rate
        self.ser = serial.Serial()
        self.start_delimiter = start_delimiter
        self.stop_delimiter = stop_delimiter

        super(SerialReader, self).__init__(outbox=outbox, logger_level=logger_level)
        self.logger.name = "SerialLogger"

    def start(self):
        """
        Start the read loop for Serial input.
        Received packets are entered into the read_queue.

        :return: None
        """

        try:
            self.logger.debug("Opening serial port [{}] with {} baud".format(self.port, self.baud_rate))
            self.ser.baudrate =self.baud_rate
            self.ser.setPort(self.port)
            self.ser.timeout = 1
            self.ser.open()

        except SerialException:
            self.logger.fatal("Serial port [{}] cannot be opened :(".format(self.port))
            sys.exit()

        self.logger.debug("Starting reader thread")

        super(SerialReader, self).start()

    def stop(self):
        """
        Halt all reading operations
        The read queue is not cleared by stopping the reader

        :return: None
        """

        self.logger.debug("Stopping reader thread")
        self.ser.close()
        super(SerialReader, self).stop()

    def read_entry(self):
        """
        Reading loop to be run by thread.
        Loop will run until manually stopped

        :return: Entry line
        """
        recording_entry = False
        received = ""

        # Wait for the 'packet start' signal to start recording the entry
        while self.is_running and not recording_entry:
            c = self.ser.read()
            if c == self.start_delimiter:
                recording_entry = True

        # Entry started - record until it stops
        # TODO - hide the serial read inside the is_running check
        c = self.ser.read()
        while self.is_running and c != self.stop_delimiter:
                received += c
                c = self.ser.read()

        return received

if __name__ == '__main__':
    read_queue = Queue()
    logger = logging.getLogger("SerialReader")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    reader = SerialReader("COM4", 57600, '#', '$', logger.name)

    reader.set_queue(read_queue)
    reader.start()

    while True:
        packet = read_queue.get()
        logger.info("Packet: {}".format(packet))
        read_queue.task_done()


