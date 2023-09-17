import cv2
import numpy as np

print(cv2.__version__)

if __name__ == '__main__':
    try:
        cap = cv2.VideoCapture("/dev/video3")
    except:
        print("Please Check your Camera")
    else:
        print("start capture")
        width = 640
        height = 480
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        while(True):
            try:
                ret, buff = cap.read()
            except:
                print("Fail to read Frame")
                break
            else:
                if buff is None:
                    break
                reshape_buffer = buff.reshape(height,width,2)
                bayer_right, bayer_left  = cv2.split(reshape_buffer)
                cap_right = cv2.cvtColor(bayer_left, cv2.COLOR_BAYER_GB2BGR)
                cap_left = cv2.cvtColor(bayer_right, cv2.COLOR_BAYER_GB2BGR)
                cv2.imshow('left frame', cap_right)
                cv2.imshow('right frame', cap_left)
                key = cv2.waitKey(5)
                if key & 0xFF == ord('q'):
                    break
        cap.release()