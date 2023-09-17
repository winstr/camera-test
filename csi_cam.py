#gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1' ! nvvidconv flip-method=0 ! nvegltransform ! nveglglessink

import cv2

# GStreamer 파이프라인 정의
gstreamer_pipeline = (
    "nvarguscamerasrc sensor_id=1 ! "
    "video/x-raw(memory:NVMM), "
    "width=(int)960, height=(int)540, "
    "format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=2 ! "
    "video/x-raw, width=(int)960, height=(int)540, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)

cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    cv2.imshow('Camera Streaming', frame)

    # 'q'키를 눌러서 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
