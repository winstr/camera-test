import cv2

MIN_DEG = 20000
MAX_DEG = 65535

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.normalize(frame, frame, MIN_DEG, MAX_DEG, cv2.NORM_MINMAX)
    frame = cv2.resize(frame, dsize=(640, 480))
    cv2.imshow("", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()