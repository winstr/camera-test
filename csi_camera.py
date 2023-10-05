def gstreamer_pipeline(
        csi_device,
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0):
    '''
    usage:
        cap = cv2.VideoCapture(gstreamer_pipeline(device=0), cv2.CAP_GSTREAMER)
        ...

    params:
        - csi_device(int): csi camera port number. ex) 0 -> /dev/video0
        - capture_width(int): ...
        - capture_height(int): ...
        - display_width(int): ...
        - display_height(int): ...
        - framerate(int): ...
        - flip_method(int):
            0: No rotation/flip (default).
            1: Rotate clockwise by 90 degrees.
            2: Flip horizontally.
            3: Rotate clockwise by 180 degrees.
            4: Rotate clockwise by 90 degrees then flip horizontally.
            5: Rotate clockwise by 270 degrees.
            6: Rotate clockwise by 270 degrees then flip horizontally.
            7: Rotate clockwise by 90 degrees.
    '''
    return (
        f'nvarguscamerasrc sensor-id={csi_device} ! video/x-raw(memory:NVMM), '
        f'width=(int){capture_width}, '
        f'height=(int){capture_height}, '
        f'format=(string)NV12, framerate=(fraction){framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! video/x-raw, '
        f'width=(int){display_width}, '
        f'height=(int){display_height}, '
        'format=(string)BGRx ! videoconvert ! '
        'video/x-raw, format=(string)BGR ! appsink')
