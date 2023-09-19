import os

import cv2

from src.utils.camera import oCamS_1CGN_U


def draw_chessboard_corner(frame, chessboard):
    ret, corners = cv2.findChessboardCorners(frame, chessboard)
    if ret:
        cv2.drawChessboardCorners(frame, chessboard, corners, ret)


def capture_chessboard_oCamS_1CGN_U(chessboard, output_dir, lens):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    stereo_cam = oCamS_1CGN_U(frame_exposure=250)
    i = 0
    while True:
        frame = stereo_cam.get_frame(lens)
        frame_ = frame.copy()
        draw_chessboard_corner(frame_, chessboard)
        cv2.imshow(f"Preview - {lens}", frame_)
        key = cv2.waitKey(10)
        if key == ord("s"):
            img_path = ""
            while True:
                img_name = f"{str(i).zfill(3)}_{lens}.jpg"
                img_path = f"{output_dir}/{img_name}"
                if not os.path.isfile(img_path):
                    break
                else:
                    i += 1
            cv2.imwrite(img_path, frame)
            i += 1
        elif key == ord("q"):
            break
    del(stereo_cam)


if __name__ == "__main__":
    capture_chessboard_oCamS_1CGN_U(
        chessboard=(8, 6), output_dir="out", lens="right")