import cv2


cap = cv2.VideoCapture("/dev/video2")
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, buffer = cap.read()
    if not ret:
        break

    buffer = buffer.reshape(480, 640, 2)
    left, right = cv2.split(buffer)

    left = cv2.cvtColor(left, cv2.COLOR_BAYER_GB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BAYER_GB2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(right, left)

    cv2.imshow('', disparity)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destryAllWindows()
cap.release()