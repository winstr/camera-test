import os
from pathlib import Path

import cv2
import numpy as np


class oCamS_1CGN_U():

    def __init__(self, device_path, img_width=640, img_height=480):
        self.device_path = device_path
        self.img_width = img_width
        self.img_height = img_height
        self.dummy = np.zeros((img_height, img_width), np.uint8)
        self.create_cap()

    def get_image(self):
        if not self.cap.isOpened():
            left_img = self.dummy
            right_img = self.dummy
        else:
            ret, buffer = self.cap.read()
            if not ret:
                left_img = self.dummy
                right_img = self.dummy
            else:
                buffer = buffer.reshape(self.img_height, self.img_width, 2)
                left_img, right_img = cv2.split(buffer)
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BAYER_BG2BGR)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BAYER_BG2BGR)
            return left_img, right_img
    
    def create_cap(self):
        self.cap = cv2.VideoCapture(self.device_path)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)

    def release_cap(self):
        self.cap.release()


if __name__ == "__main__":
    device_path = "/dev/video0"
    imgdir_name = "out"
    shutter_spd = 500  # millisecond, ms

    cam = oCamS_1CGN_U("/dev/video0")
    out = str(Path(__file__).parents[0].absolute() / imgdir_name)
    idx = 0

    if not os.path.isdir(out):
        os.makedirs(out)

    while True:
        left_img, right_img = cam.get_image()
        
        cv2.imshow("REC.", cv2.hconcat([left_img, right_img]))
        cv2.imwrite(os.path.join(out, f"{str(idx).zfill(4)}L.jpg"), left_img)
        cv2.imwrite(os.path.join(out, f"{str(idx).zfill(4)}R.jpg"), right_img)

        if cv2.waitKey(1) == ord("q"):
            break
        idx += 1

    cv2.destroyAllWindows()
    cam.release_cap()