import time
from typing import Any, Optional, Tuple

import cv2


class Camera:
    def __init__(self, device_id: int = 0, width: Optional[int] = None, height: Optional[int] = None, window_name: str = "LiveView"):
        self.device_id = device_id
        self.window_name = window_name
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {device_id}")
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.start_time = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def get_frame(self) -> Tuple[Optional[Any], Optional[float]]:
        """Returns frame and relative timestamp [s]."""
        if self.start_time is None:
            self.start_time = time.perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        timestamp = time.perf_counter() - self.start_time
        return frame, timestamp

    def show(self, frame) -> int:
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key

    def release(self) -> None:
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def set_mouse_callback(self, callback) -> None:
        cv2.setMouseCallback(self.window_name, callback)
