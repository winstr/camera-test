import cv2


if __name__ == "__main__":
    cam = cv2.VideoCapture("/dev/cam/ThermoCam160B")
    fourcc = cv2.VideoWriter.fourcc('Y', '1', '6', ' ')

    #cam.set(cv2.CAP_PROP_FOURCC, fourcc)
    cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    while cam.isOpened():
        rt, frame = cam.read()
        if not rt:
            break
        cv2.normalize(frame, frame, 20000, 65535, cv2.NORM_MINMAX)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    cam.release()
