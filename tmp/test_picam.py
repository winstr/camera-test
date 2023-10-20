import cv2
from picamera2 import Picamera2


picam = Picamera2(0)
picam.configure(
    picam.create_video_configuration(
        main={"size": (1640, 1232),
              "format": "RGB888"},
        controls={"NoiseReductionMode": 0,
                  "FrameDurationLimits": (111111, 111111)}))

picam.start()

while True:
    frame = picam.capture_array()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('res', frame)
    if cv2.waitKey(1) == ord('q'):
        break

picam.stop()
picam.close()