__author__ = 'Leenix'

from SinkNode.Reader import *
import socket
import logging

MAX_CONNECT_REQUESTS = 5
BUFFER_SIZE = 1024


class SocketReader(Reader):
    def __init__(self, start_delimiter=' ', stop_delimiter='\n', server_address='localhost', listening_port=8888, outbox=None, logger_level=logging.FATAL, logger_format=LOGGER_FORMAT, allow_reuse=True):
        super(SocketReader, self).__init__(outbox=outbox, logger_level=logger_level, logger_format=logger_format)
        self.logger.name = 'SocketReader'

        self.listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if allow_reuse:
            self.listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);

        self.listening_port = listening_port
        self.server_address = server_address
        self.start_delimiter = start_delimiter
        self.stop_delimiter = stop_delimiter

    def start(self):
        self.listening_socket.bind((self.server_address, self.listening_port))
        self.listening_socket.listen(MAX_CONNECT_REQUESTS)
        super(SocketReader, self).start()

    def read_entry(self):
        client, address = self.listening_socket.accept()
        self.logger.debug('Connection started [{}]'.format(address))
        received_data = client.recv(BUFFER_SIZE)
        return received_data

