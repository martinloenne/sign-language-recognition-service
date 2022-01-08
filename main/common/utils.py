import numpy as np
import cv2
import time
import datetime
import base64

from datetime import datetime as dt
from pathlib import Path


def log(message):
    _log("Info", message)


def warning(message):
    _log("WARNING", message)


def _log(msg_type, message):
    t_string = dt.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"{t_string}    {msg_type}    {message}")


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.running = False

    def start(self):
        self.start_time = time.time()
        self.end_time = None
        self.running = True

    def end(self):
        self.end_time = time.time()
        self.running = False

    def get_elapsed_time(self):
        return datetime.timedelta(seconds=round(self.get_elapsed_seconds()))

    def get_elapsed_time_milliseconds(self):
        return self.get_elapsed_seconds() * 1000

    def get_elapsed_seconds(self):
        if self.start_time is None:
            raise Exception("Timer not started.")
        elif self.end_time is None:
            current_time = time.time()
            return current_time - self.start_time
        else:
            return self.end_time - self.start_time


def calc_percentage(count, total):
    return round(((count / total) * 100), 2)


def base64_to_img(src):
    encoded_data = src.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception('Base64 not an image.')
    return img


def base64_array_to_img_array(src):
    frames = []

    try:
        for b64 in src:
            frames.append(base64_to_img(b64))
    except Exception as ex:
        warning(ex)
        raise DataInvalidException('Request not valid - must be an array of Base64 images.')

    return frames


def extract_frames_from_video(video_path):
    sequence = []

    # Create OpenCV capture using video as src
    capture = cv2.VideoCapture(video_path)
    # Check if capture opened successfully
    if (capture.isOpened() == False):
        raise Exception("Error opening video file.")
    # Read frames until video is completed
    while capture.isOpened():
        # Capture every frame
        ret, frame = capture.read()
        if ret:
            sequence.append(frame)

        # Break the loop when no more frames
        else:
            break
    # When all frames are extracted, release the video capture object
    capture.release()
    # Close all the OpenCV window frames
    cv2.destroyAllWindows()

    return sequence


def get_relative_path(file, relative_path):
    mod_path = Path(file).parent
    return str((mod_path / relative_path).resolve())


class DataInvalidException(Exception):
    def __init__(self, message):
        super().__init__(message)