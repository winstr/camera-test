from overrides import overrides
from camera_fusion.capture import CameraCapture


class RaspberryPiCameraModule2(CameraCapture):
    """ Raspberry Pi Camera Module v.2 (IMX 219)

    capture_modes = {0: CaptureMode(3264, 2464,  21),  # default
                     1: CaptureMode(3264, 1848,  28),
                     2: CaptureMode(1920, 1080,  30),
                     3: CaptureMode(1280,  720,  60),
                     4: CaptureMode(1280,  720, 120)}
    """

    @overrides
    def connect(self,
                cap_source:str,
                cap_width:int,
                cap_height:int,
                cap_fps:int,
                dst_width:int,
                dst_height:int) -> None:

        """ gstreamer_pipeline = (
            f'nvarguscamerasrc sensor-id={cap_source} wbmode=3 tnr-mode=2 '
            f'tnr-strength=1 ee-mode=2 ee-strength=1 ! '
            f'video/x-raw(memory:NVMM), width={cap_w}, height={cap_h}, '
            f'format=NV12, '
            f'framerate={cap_fps}/1 ! nvvidconv flip-method=0 ! '
            f'video/x-raw, width={dst_w}, height={dst_h}, format=BGRx ! '
            f'videoconvert ! '
            f'video/x-raw, format=BGR ! videobalance contrast=1.5 '
            f'brightness=-.2 saturation=1.2 ! appsink') """

        gstreamer_pipeline = (
            f'nvarguscamerasrc sensor-id={cap_source} wbmode=3 tnr-mode=2 '
            f'tnr-strength=1 ee-mode=2 ee-strength=1 ! '
            f'video/x-raw(memory:NVMM), '
            f'width={cap_width}, height={cap_height}, '
            f'format=NV12, framerate={cap_fps}/1 ! nvvidconv flip-method=0 ! '
            f'video/x-raw, width={dst_width}, height={dst_height}, '
            f'format=BGRx ! videoconvert ! video/x-raw, format=BGR ! '
            f'videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! '
            f'appsink')

        super().connect(gstreamer_pipeline)
        self.cap_source = cap_source