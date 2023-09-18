import cv2
import pickle
import numpy as np


with open('calibration_data.pkl', 'rb') as f:
    calibration_data = pickle.load(f)

M1 = calibration_data['M1']
d1 = calibration_data['d1']
M2 = calibration_data['M2']
d2 = calibration_data['d2']
R = calibration_data['R']
T = calibration_data['T']
E = calibration_data['E']
F = calibration_data['F']

blockSize = 15
minDisparity = -39
numDisparities = 144
stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    P1=8 * 3 * blockSize ** 2,
    P2=32 * 3 * blockSize ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
)

DEVICE = "/dev/video0"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

cap = cv2.VideoCapture(DEVICE)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(M1, d1, M2, d2, (IMAGE_WIDTH, IMAGE_HEIGHT), R, T, alpha=0)
map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.CV_32FC1)

while cap.isOpened():
    ret_pair, buffer = cap.read()
    if not ret_pair:
        break

    buffer = buffer.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 2)
    left_image, right_image = cv2.split(buffer)

    left_image = cv2.cvtColor(left_image, cv2.COLOR_BAYER_GB2BGR)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BAYER_GB2BGR)

    cv2.imshow("", cv2.hconcat([left_image, right_image]))

    left_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # 깊이맵 생성
    #disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
    #cv2.imshow("disparity", disparity)

    #depth_map = (disparity - minDisparity) / numDisparities
    #depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)  # Normalize for visualization

    # 깊이맵 표시
    #cv2.imshow('Depth Map', depth_map)
    
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()