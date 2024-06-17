import os
import signal
import sys
import queue
import threading
import time
from datetime import datetime
from pynput import keyboard

import cv2
import numpy
from vmbpy import Log, CameraEvent, Camera, LOG_CONFIG_INFO_CONSOLE_ONLY, VmbSystem

from angle_math import calc_angle, calculate_diagonal_distance, calculate_area
from aruco_detect import draw_aruco, barcode_data_map, detect_aruco, try_crop
from camera_handler import get_camera
from config import CAMERA_DEVICE_ID, images_directory, FRAME_WIDTH, FRAME_QUEUE_SIZE, FRAME_HEIGHT
from frame_server import resize_if_required, FrameProducer
from image_handler import ImageHandler, display_window, destroy_window, create_dummy_frame, create_text_frame
from utils import char_to_val, get_user_input, log_timestamp, timestamps, subtract_starting_time, sum_timestamps

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

calculated_angle = 0
compensate_angle = 0
compensated_angle = 0
collinearity = 0

cropping_region = 0, 0, FRAME_WIDTH, FRAME_HEIGHT


def print_preamble():
    print('////////////////////////////////////////')
    print('/// PACCAR Frame Twist project /////////')
    print('////////////////////////////////////////\n')
    print(flush=True)


def looper(img_handler: ImageHandler):
    global cropping_region
    global calculated_angle, compensate_angle, compensated_angle, collinearity

    img_handler.images['marked'] = cv2.cvtColor(img_handler.images['original'], cv2.COLOR_GRAY2BGR)
    log_timestamp('start time')

    img_handler.preprocess()
    log_timestamp('preprocessing')

    corners, ids = detect_aruco(img_handler.images['thresholded'], cropping_region)
    log_timestamp('detect_aruco')

    if corners is not None and ids is not None:
        cropping_region = try_crop(img_handler.images['thresholded'], corners, ids)
        log_timestamp('try_crop')
        ret = draw_aruco(img_handler.images['marked'], img_handler.images['thresholded'], corners, ids)
        log_timestamp('draw_aruco')

        if ret is True:
            calculated_angle, collinearity = calc_angle(img_handler.images['marked'], barcode_data_map)
            log_timestamp('calc_angle')
            compensated_angle = calculated_angle - compensate_angle
            # Display the angle
            angle_text = f"{compensated_angle:.2f} degrees"
            cv2.putText(img_handler.images['marked'], angle_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
            # print(f"timestamp: {datetime.now()} Angle: {compensated_angle}")
    log_timestamp('end time')


class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue, curr_frame=None):
        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self._compensate_r = 0
        self._compensate_l = 0
        self._capture_continuous = None
        self._capture_once = False
        self._snapshot = False
        self._video = False
        self._alive = True
        self._imageHandler = ImageHandler(curr_frame, FRAME_WIDTH, FRAME_HEIGHT)
        self.key_listener = keyboard.Listener(on_press=self.key_event_handler)

    @staticmethod
    def key_event_handler(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    # Function to address all the user controls on the press of keys
    def userinput(self):
        global calculated_angle, compensate_angle
        input_key = get_user_input()
        if input_key == char_to_val("q"):
            # If the 'q' key is pressed, break from the loop
            self._alive = False
        elif input_key == char_to_val("r"):
            compensate_angle = calculated_angle
        elif input_key == char_to_val("c"):
            self._capture_once = True
        elif input_key == char_to_val("x"):
            self._capture_continuous = not self._capture_continuous
        elif input_key == char_to_val("p"):
            self._snapshot = True
        elif input_key == char_to_val("v"):
            framerate = (get_camera(CAMERA_DEVICE_ID).get_feature_by_name('AcquisitionFrameRate').get())
            FrameProducer.request_create_video(images_directory, 60, framerate)

    def run(self):
        IMAGE_CAPTION = 'MAIN-WINDOW'
        CROPPED_CAPTION = 'CROP WITH THRESHOLDING'
        ANGLE_CAPTION = 'CALCULATED ANGLE'
        frames = {}
        self.key_listener.start()

        self.log.info('\'FrameConsumer\' started.')
        print("waiting for key")

        while self._alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()

                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame

                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                time_init = time.time()
                cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                self._imageHandler.images['original'] = numpy.concatenate(cv_images, axis=1)
                if self._video is True:
                    # self._imageHandler.saveVideo(True)
                    pass
                if self._snapshot:
                    self._snapshot = False
                    self._imageHandler.saveImages('all', images_directory)
                if self._capture_continuous or self._capture_once:
                    looper(self._imageHandler)
                    current_fps = 1 / (time.time() - time_init)
                    fps_text = f"{current_fps:.2f}"
                    print("{}  Angle: {:.2f}\t Collinearity of points: {}\t FPS: {:.2f}\t compute time: {:.2f}".format(
                        datetime.now(), compensated_angle,
                        collinearity, current_fps,
                        sum_timestamps()))

                    # Show the output frame
                    display_window(IMAGE_CAPTION, self._imageHandler.images['marked'], 0.5)
                    # Update the cropping region
                    x_min, y_min, x_max, y_max = cropping_region
                    # Display the cropped frame with detected markers
                    self._imageHandler.images['cropped'] = ImageHandler.select_roi(
                        self._imageHandler.images['thresholded'], x_min, y_min, x_max, y_max)
                    display_window(CROPPED_CAPTION, self._imageHandler.images['cropped'], 0.5)
                    angle_window = create_text_frame(f"{compensated_angle:.2f}")
                    display_window(ANGLE_CAPTION, angle_window)

                    if self._capture_once:
                        self._capture_once = False
                else:
                    destroy_window(CROPPED_CAPTION)
                    destroy_window(ANGLE_CAPTION)
                    display_window(IMAGE_CAPTION, create_dummy_frame())

            # If there are no frames available, show dummy image instead
            else:
                display_window(IMAGE_CAPTION, create_dummy_frame())

            self.userinput()
        self.key_listener.stop()
        cv2.destroyAllWindows()

        self.log.info('\'FrameConsumer\' terminated.')


class Application:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.producer = {}
        self.producer_lock = threading.Lock()
        self.consumer = FrameConsumer(self.frame_queue)
        self._alive = True

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producer_lock:
                self.producer[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producer[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producer_lock:
                producer = self.producer.pop(cam.get_id())
                producer.stop()
                producer.join()

    def exit_sequence(self):
        # Stop all FrameProducer threads
        with self.producer_lock:
            # Initiate concurrent shutdown
            for producer in self.producer.values():
                producer.stop()

            # Wait for shutdown to complete
            for producer in self.producer.values():
                producer.join()

    def run(self):
        log = Log.get_instance()
        # self.consumer = FrameConsumer(self.frame_queue)

        vmb = VmbSystem.get_instance()
        vmb.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('\'Application\' started.')

        with vmb:
            # Construct FrameProducer threads for all detected cameras
            cam = get_camera(CAMERA_DEVICE_ID)

            self.producer[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            # Start FrameProducer threads
            with self.producer_lock:
                self.producer[cam.get_id()].start()

            vmb.unregister_all_camera_change_handlers()
            # Run the frame consumer to display the recorded images
            vmb.register_camera_change_handler(self)
            self.consumer.run()
            vmb.unregister_camera_change_handler(self)

            self.exit_sequence()

        self._alive = False
        log.info('\'Application\' terminated.')

    @property
    def alive(self):
        return self._alive


app = None

if __name__ == '__main__':
    app = Application()
    print_preamble()
    app.run()
