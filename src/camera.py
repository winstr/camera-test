import cv2


def get_gstreamer_pipeline(csi_port_id, width, height, fps):
    return (
        f"nvarguscamerasrc sensor_id={csi_port_id} ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv flip-method=2 ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink")


if __name__ == "__main__":

    csi_cam_0 = get_gstreamer_pipeline(0, 400, 300, 30)
    csi_cam_1 = get_gstreamer_pipeline(1, 400, 300, 30)

    cap_0 = cv2.VideoCapture(csi_cam_0, cv2.CAP_GSTREAMER)
    cap_1 = cv2.VideoCapture(csi_cam_1, cv2.CAP_GSTREAMER)

    while True:
        active_0, frame_0 = cap_0.read()
        active_1, frame_1 = cap_1.read()

        if active_0:
            frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
            cv2.imshow("csi_0: IR", frame_0)
        if active_1:
            cv2.imshow("csi_1: RGB", frame_1)
        if not active_0 and not active_1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_0.release()
    cap_1.release()
    cv2.destroyAllWindows()
