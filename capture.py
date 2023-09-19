import os

import cv2

from src.utils.camera import oCamS_1CGN_U


class ChessboardCapturer():

    def __init__(self, num_corners_horizontal, num_corners_vertical, output_dir="out"):
        if os.path.isdir(output_dir):
            raise FileExistsError(output_dir)
        self.output_dir = output_dir
        self.chessboard = (num_corners_horizontal, num_corners_vertical)

    def draw_chessboard_corners(self, frame):
        ret, corners = cv2.findChessboardCorners(frame, self.chessboard)
        if ret:
            cv2.drawChessboardCorners(frame, self.chessboard, corners, ret)
    
    def save_image(filepath, image, overwrite=True):
        if os.path.isfile(filepath) and not overwrite:
            return
        cv2.imwrite(filepath, image)

    def capture(self, camera):
        frame_idx = 0
        while True:
            frame = camera.get_frame()


def draw_chessboard_corner(frame, chessboard):
    ret, corners = cv2.findChessboardCorners(frame, chessboard)
    if ret:
        cv2.drawChessboardCorners(frame, chessboard, corners, ret)


def draw_chessboard_corner_stereo(frame, chessboard):
    left_frame, right_frame = cv2.split(frame)
    left_ret, left_corners = cv2.findChessboardCorners(left_frame, chessboard)
    right_ret, right_corners = cv2.findChessboardCorners(right_frame, chessboard)
    if left_ret and right_ret:
        cv2.drawChessboardCorners(left_frame, chessboard, left_corners, left_ret)
        cv2.drawChessboardCorners(right_frame, chessboard, right_corners, right_ret)


def capture_chessboard_oCamS_1CGN_U(chessboard, output_dir, lens):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    stereo_cam = oCamS_1CGN_U(frame_exposure=250)
    i = 0
    while True:
        frame = stereo_cam.get_frame(lens)
        frame_ = frame.copy()

        if lens == "all":
            draw_chessboard_corner_stereo(frame_, chessboard)
            cv2.imshow(f"Preview - {lens}", cv2.hconcat([frame_[0], frame_[1]]))
        else:
            draw_chessboard_corner(frame_, chessboard)
            cv2.imshow(f"Preview - {lens}", frame_)

        key = cv2.waitKey(10)
        if key == ord("s"):
            if lens == "all":
                left_img_path = f"{str(i).zfill(3)}_left.jpg"
                right_img_path = f"{str(i).zfill(3)}_right.jpg"
                cv2.imwrite(left_img_path, frame[0])
                cv2.imwrite(right_img_path, frame[1])
            else:
                img_name = f"{str(i).zfill(3)}_{lens}.jpg"
                img_path = f"{output_dir}/{img_name}"
                cv2.imwrite(img_path, frame)
            i += 1
        elif key == ord("q"):
            break
    del(stereo_cam)


if __name__ == "__main__":
    capture_chessboard_oCamS_1CGN_U(
        chessboard=(8, 6), output_dir="out", lens=["left", "right"])