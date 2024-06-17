#!/usr/bin/python3
import os
import time

import copy
import queue
import threading
from datetime import datetime

from typing import Optional

import cv2
import numpy

from vmbpy import Frame, Camera, Log, Stream, FrameStatus, PixelFormat, VmbFeatureError, VmbCameraError

from camera_handler import print_feature, set_nearest_value
from config import FRAME_WIDTH, FRAME_HEIGHT, MAX_FRAME_HEIGHT, MAX_FRAME_WIDTH, DEVICE_FRAMERATE, DEVICE_EXPOSURE

video_writer = cv2.VideoWriter()
video_start_time = time.time()
video_frame_count = 0
video_end_recording_time = time.time()
video_framerate = DEVICE_FRAMERATE
start_saving_flag = False


def add_camera_id(frame: Frame, cam_id: str) -> Frame:
    # Helper function inserting 'cam_id' into given frame. This function
    # manipulates the original image buffer inside frame object.
    image_mat = frame.as_opencv_image()
    image_text = 'Cam: {}'.format(cam_id)
    cv2.putText(image_mat, image_text, org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame


# Function to resize the image frame
# mainly for display pusposes, but can also be used for computation if need be
def resize_if_required(frame: Frame) -> numpy.ndarray:
    # Helper function resizing the given frame, if it has not the required dimensions.
    # On resizing, the image data is copied and resized, the image inside the frame object
    # is untouched.
    try:
        if frame is not None:
            cv_frame = frame.as_opencv_image()

            if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
                cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                cv_frame = cv_frame[..., numpy.newaxis]

            return cv_frame
    except Exception as e:
        print(e)

    return None


# This class is responsible to initialize the camera and start grabbing frames.
# It also has video writer method that directly writes the video to an avi file
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        # This method is executed within VmbC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:

            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                self.try_put_frame(self.frame_queue, frame_cpy)

        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    # Initializes the camera before starting frame capture
    def setup_camera(self):
        set_nearest_value(self.cam, 'Height', MAX_FRAME_HEIGHT, self.log)
        set_nearest_value(self.cam, 'Width', MAX_FRAME_WIDTH, self.log)
        if DEVICE_FRAMERATE:
            set_nearest_value(self.cam, 'AcquisitionFrameRateEnable', True, self.log)
            set_nearest_value(self.cam, 'AcquisitionFrameRate', DEVICE_FRAMERATE, self.log)
        # Try to enable automatic exposure time setting
        try:
            self.cam.ExposureAuto.set('Off')
            set_nearest_value(self.cam, 'ExposureTime', DEVICE_EXPOSURE, self.log)

        except (AttributeError, VmbFeatureError):
            self.log.info('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(self.cam.get_id()))

        try:
            self.cam.set_pixel_format(PixelFormat.Mono8)
        except (AttributeError, VmbFeatureError):
            self.log.info('Camera {}: Failed to set pixel format to Mono8.'.format(self.cam.get_id()))

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))

        try:
            with self.cam:
                self.setup_camera()
                self.print_features()

                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()

                finally:
                    self.cam.stop_streaming()

        except VmbCameraError:
            pass

        finally:
            self.try_put_frame(self.frame_queue, None)

        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))

    def print_features(self):
        print('Print all features of camera \'{}\':'.format(self.cam.get_id()))
        for feature in self.cam.get_all_features():
            print_feature(feature)

    def try_put_frame(self, q: queue.Queue, frame: Optional[Frame]):
        global start_saving_flag, video_writer
        try:
            q.put_nowait((self.cam.get_id(), frame))
            if start_saving_flag:
                FrameProducer.write_video_to_file(resize_if_required(frame))
            else:
                if video_writer is not None:
                    FrameProducer.save_video_to_file(video_writer)
                    video_writer = None
                    start_saving_flag = False

        except queue.Full:
            pass

    @staticmethod
    def request_create_video(directory, recording_time, frame_rate):
        global video_writer, video_frame_count, video_end_recording_time, video_start_time, \
            video_framerate, start_saving_flag
        if video_writer is not None:
            start_saving_flag = False
            return

        video_framerate = frame_rate
        try:
            os.makedirs(directory, exist_ok=True)
            # Get the current date and time
            current_time = datetime.now()
            video_end_recording_time = time.time() + recording_time

            # Format the date and time as a string to use in the filename
            date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            filename = os.path.join(directory, f"{date_string}_original.avi")
            fourcc = cv2.VideoWriter.fourcc(*'GREY')
            frameSize = (FRAME_WIDTH, FRAME_HEIGHT)
            print(f'framerate is {DEVICE_FRAMERATE}')
            video_writer = cv2.VideoWriter(filename, fourcc, video_framerate, frameSize, False)
            video_start_time = time.time()
            video_frame_count = 0
            start_saving_flag = True
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def write_video_to_file(images=None):
        global video_writer, video_end_recording_time, video_frame_count, start_saving_flag
        if video_writer is None:
            return
        try:
            time_left = video_end_recording_time - time.time()
            # print(f" {time_left}  {time.time()}")
            if time_left > 0:
                if images is not None:
                    # Calculate the expected elapsed time for the current frame
                    expected_elapsed_time = video_frame_count / video_framerate

                    # Calculate the actual elapsed time since the start of the loop
                    actual_elapsed_time = time.time() - video_start_time

                    # Check if the actual elapsed time matches the expected elapsed time
                    if actual_elapsed_time >= expected_elapsed_time:
                        video_writer.write(images)
                        # cv2.imshow('video', self.images['marked'])
                    time.sleep(max(0, expected_elapsed_time - actual_elapsed_time))

                else:
                    print("Error: Frame data is empty.")
            else:
                FrameProducer.save_video_to_file(video_writer)
                video_writer = None
                start_saving_flag = False
        except Exception as e:
            print(e)

    @staticmethod
    def save_video_to_file(video_instance: cv2.VideoWriter):
        video_instance.release()
        cv2.destroyWindow('video')
