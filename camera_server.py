import socket
import pickle
import struct
import traceback

import cv2
import numpy as np

from src.camera import OCamS1CGNU
from src.camera import ThermoCam160B
from src.camera import RaspberryPiCamera2
from src.camera import get_gstreamer_pipeline


def get_ocams(source: str) -> OCamS1CGNU:
    ocams = OCamS1CGNU()
    ocams.initialize(source, 640, 480, 45)
    ocams.set_exposure(0)
    return ocams


def get_picam(sensor_id: int) -> RaspberryPiCamera2:
    picam = RaspberryPiCamera2()
    picam.initialize(sensor_id, get_gstreamer_pipeline(sensor_id))
    return picam


def get_ircam(source: str) -> ThermoCam160B:
    ircam = ThermoCam160B()
    ircam.initialize(source, 160, 120, 9)
    return ircam


def get_server_socket(port: int) -> socket.socket:
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host_ip, port))
    return server_socket


def crop_to_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]  # frame height and width
    s = min(w, h)
    gap_w = int((w - s) / 2)
    gap_h = int((h - s) / 2)
    frame = frame[gap_h:gap_h+s, gap_w:gap_w+s, ...]
    return frame


class CameraStremingServer():

    def __init__(self):
        self.host_name = None
        self.host_ip = None
        self.port = None
        self.server_socket = None
        #self.camera = None
        self.picam = None
        self.ircam = None
        self.ocams = None

    def initialize_server(self, port: int):
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.server_socket.bind((self.host_ip, self.port))
        self.server_socket.bind(('0.0.0.0', self.port))

    def initialize_camera(self):
        # FIXME: Refactoring
        # TODO:
        #self.camera = get_picam('0')
        self.picam = get_picam('0')
        self.ircam = get_ircam('/dev/cam/ThermoCam160B')
        self.ocams = get_ocams('/dev/cam/oCamS-1CGN-U')
        

    def stream(self, client_socket: socket.socket):
        while True:
            # FIXME: Refactoring
            # TODO:
            self.picam.grab()
            self.ircam.grab()
            self.ocams.grab()

            picam_frame = self.picam.retrieve()

            ircam_frame = self.ircam.retrieve()
            ircam_frame = ThermoCam160B.normalize(ircam_frame)
            ircam_frame = cv2.resize(ircam_frame, dsize=(640, 480))
            ircam_frame = cv2.cvtColor(ircam_frame, cv2.COLOR_GRAY2BGR)

            ocams_frame = self.ocams.retrieve()

            frame = cv2.hconcat([picam_frame, ircam_frame, ocams_frame])

            #frame = self.camera.read()
            #frame = ThermoCam160B.normalize(frame)
            #frame = crop_to_square(frame)
            data = pickle.dumps(frame)  # to byte
            message_size = struct.pack('L', len(data))
            client_socket.sendall(message_size + data)

    def run(self):
        self.server_socket.listen(1)
        print(f'Listening at {self.host_ip}:{self.port}')

        client_socket, addr = self.server_socket.accept()
        print(f'Connection established with: {addr}')

        try:
            while True:
                self.stream(client_socket)
        except:
            traceback.print_exc()
        finally:
            if client_socket:
                client_socket.close()
            self.camera.release()


if __name__ == '__main__':
    server = CameraStremingServer()
    server.initialize_server(port=5000)
    server.initialize_camera()
    server.run()
