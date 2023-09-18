import cv2


if __name__ == '__main__':

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    try:
        cap = cv2.VideoCapture("/dev/video2")
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
                bayer_left, bayer_right  = cv2.split(reshape_buffer)
                cap_right = cv2.cvtColor(bayer_left, cv2.COLOR_BAYER_GB2BGR)
                cap_left = cv2.cvtColor(bayer_right, cv2.COLOR_BAYER_GB2BGR)

                kp1, desc1 = orb.detectAndCompute(cap_left, None)
                kp2, desc2 = orb.detectAndCompute(cap_right, None)
                matches = bf.match(desc1, desc2)
                matches = sorted(matches, key=lambda x: x.distance)
                matched_img = cv2.drawMatches(cap_left, kp1, cap_right, kp2, matches[:50], None, flags=2)
                cv2.imshow("matched", matched_img)

                #cv2.imshow("Left", cap_left)
                #cv2.imshow("Right", cap_right)

                #cap_left = cv2.cvtColor(cap_left, cv2.COLOR_BGR2GRAY)
                #cap_right = cv2.cvtColor(cap_right, cv2.COLOR_BGR2GRAY)
                #disparity = stereo.compute(cap_right, cap_left)
                #normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                #cv2.imshow("disparity", normalized_disparity)

                key = cv2.waitKey(5)
                if key & 0xFF == ord('q'):
                    break
        cap.release()