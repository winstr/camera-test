from pathlib import Path
import pickle
import cv2

def load_calibration_data(filename):
    with open(filename, 'rb') as f:
        calibration_data = pickle.load(f)
    return calibration_data

def undistort_image(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist)
    return dst


if __name__ == "__main__":
    device_path = "/dev/video0"
    img_width = 640
    img_height = 480
    exposure = 500

    calib_data_dir = str(Path(__file__).parent.absolute())

    cap = cv2.VideoCapture(device_path)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_height)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    left_calib_data = load_calibration_data(f"{calib_data_dir}/left_cam_calib.pkl")
    left_mtx = left_calib_data['mtx']
    left_dist = left_calib_data['dist']

    right_calib_data = load_calibration_data(f"{calib_data_dir}/right_cam_calib.pkl")
    right_mtx = right_calib_data['mtx']
    right_dist = right_calib_data['dist']

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img_right, img_left = cv2.split(img)
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BAYER_BG2GRAY)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BAYER_BG2GRAY)

        corrected_img_left = undistort_image(img_left, left_mtx, left_dist)
        corrected_img_right = undistort_image(img_right, right_mtx, right_dist)

        concat_1 = cv2.hconcat([img_left, img_right])
        concat_2 = cv2.hconcat([corrected_img_left, corrected_img_right])
        concat = cv2.vconcat([concat_1, concat_2])
        cv2.imshow("result", concat)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()