import os
import time
from datetime import datetime

import cv2
import numpy
import numpy as np


# Function to create dummy frames when the actual frame from camera is not available
# due to race conditions or sync between producer-consumer
def create_dummy_frame() -> np.ndarray:
    cv_frame = np.zeros((50, 640, 1), dtype=np.uint8)
    cv_frame[:] = 0
    text = 'No stream available. Please connect a camera.'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (50 + text_size[1]) // 2
    cv2.putText(cv_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return cv_frame


# Function to create a simple text based window
# used to display angle on a different window (for testing and diagnosis)
def create_text_frame(text: str):
    cv_frame = np.full((50, 640, 1), 255, dtype=np.uint8)
    cv_frame[:] = 0
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (50 + text_size[1]) // 2
    cv2.putText(cv_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    return cv_frame


# Common function to create a frame and display it with given window name
def display_window(window_name: str, image, scale: float = 1):
    try:
        if scale != 1 and scale > 0:
            # Get the original dimensions of the image
            original_height, original_width = image.shape[:2]
            # Calculate the new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            # image = image[..., numpy.newaxis]
        cv2.imshow(window_name, image)
    except Exception as e:
        print(e)


def destroy_window(window_name: str):
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
        cv2.destroyWindow(window_name)


class ImageHandler:
    def __init__(self, image, frame_width, frame_height):
        self._start_time = None
        self._frame_count = None
        self._framerate = None
        self._end_recording_time = None
        self._video_instance = None
        self.images = {}
        self._frame_width = frame_width
        self._frame_height = frame_height
        self.images['original'] = image
        self.images['marked'] = image
        self.images['cropped'] = None
        self.images['gray'] = None
        self.images['thresholded'] = None
        self.images['sharpened'] = None
        self.images['dilated'] = None

    @staticmethod
    def select_roi(image, x_min: int, y_min: int, x_max: int = 0, y_max: int = 0, x_pixels: int = 0, y_pixels: int = 0):
        # Convert x, y, width, and height to integers
        if x_pixels == 0 or y_pixels == 0:
            cropped_frame = image[y_min:y_max, x_min:x_max]
        else:
            cropped_frame = image[y_min:y_min + y_pixels, x_min:x_min + x_pixels]

        return cropped_frame

    def preprocess(self):
        # self.images['cropped'] = self.select_roi(self.images['original'], 0, 0,
        #                                          x_pixels=self._frame_width, y_pixels=1200)

        # Apply the sharpening kernel to the image
        # img_blur = cv2.blur(src=frame_img, ksize=(1,1))
        # self.images['sharpened'] = self.sharpen_image(self.images['original'])
        # self.images['dilated'] = self.erode_and_dilate_image(self.images['sharpened'])

        # Split the image into left, center, and right sections
        # left_section = self.images['sharpened'][:, :self._frame_width // 3]
        # center_section = self.images['sharpened'][:, self._frame_width // 3:2 * (self._frame_width // 3)]
        # right_section = self.images['sharpened'][:, 2 * (self._frame_width // 3):]
        # merged_image = np.concatenate((left_section, right_section), axis=1)

        self.images['thresholded'] = self.threshold_image(self.images['original'], 20, 255)
        # print(f"t1: {get_time_in_msec(time_start, t1)} t2: {get_time_in_msec(t1, t2)} t3: {get_time_in_msec(t2, t3)}")

    def convert_to_gray(self):
        return cv2.cvtColor(self.images['original'], cv2.COLOR_BGR2GRAY)

    def threshold_image(self, image, threshold, max_val):
        ret, thresholded = cv2.threshold(image, threshold, max_val, cv2.THRESH_BINARY)
        return thresholded

    def sharpen_image(self, image):
        # Define a sharpening kernel
        kernel2d = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel=kernel2d)
        return sharpened

    def erode_and_dilate_image(self, image):
        kernel2d = np.ones((5, 5), np.uint8)
        # erode = cv2.erode(image, kernel2d, iterations=1)
        # dilated = cv2.dilate(erode, kernel2d, iterations=1)
        dilated = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel2d)
        return dilated

    def createVideo(self, directory, framerate, recording_time):
        try:
            os.makedirs(directory, exist_ok=True)
            # Get the current date and time
            current_time = datetime.now()
            self._end_recording_time = time.time() + recording_time
            self._framerate = framerate

            # Format the date and time as a string to use in the filename
            date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

            filename = os.path.join(directory, f"{date_string}_original.avi")
            fourcc = cv2.VideoWriter.fourcc(*'GREY')
            frameSize = (self._frame_width, self._frame_height)
            print(f'framerate is {self._framerate}')
            self._video_instance = cv2.VideoWriter(filename, fourcc, self._framerate, frameSize, False)
            self._start_time = time.time()
            self._frame_count = 0
        except Exception as e:
            print(e)
            return None

    def saveVideo(self, saveFlag):
        if self._video_instance is None:
            return
        try:
            time_left = self._end_recording_time - time.time()
            print(f" {time_left}  {time.time()}")
            if saveFlag and (time_left > 0):
                if self.images['original'] is not None:
                    # Calculate the expected elapsed time for the current frame
                    expected_elapsed_time = self._frame_count / self._framerate

                    # Calculate the actual elapsed time since the start of the loop
                    actual_elapsed_time = time.time() - self._start_time

                    # Check if the actual elapsed time matches the expected elapsed time
                    if actual_elapsed_time >= expected_elapsed_time:
                        self._video_instance.write(self.images['marked'])
                        # cv2.imshow('video', self.images['marked'])
                    time.sleep(max(0, expected_elapsed_time - actual_elapsed_time))

                else:
                    print("Error: Frame data is empty.")

            else:

                self._video_instance.release()
                cv2.destroyWindow('video')
                self._video_instance = None
        except Exception as e:
            print(e)

    def saveImages(self, image_type, directory):
        # Get the current date and time
        current_time = datetime.now()

        # Format the date and time as a string to use in the filename
        date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        image_folder = os.path.join(directory, date_string)
        os.makedirs(image_folder, exist_ok=True)

        if (image_type == 'all' or image_type == 'original') and self.images['original'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_original.jpg")
            cv2.imwrite(filename, self.images['original'])

        if (image_type == 'all' or image_type == 'marked') and self.images['marked'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_marked.jpg")
            cv2.imwrite(filename, self.images['marked'])

        if (image_type == 'all' or image_type == 'cropped') and self.images['cropped'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_cropped.jpg")
            cv2.imwrite(filename, self.images['cropped'])

        if (image_type == 'all' or image_type == 'gray') and self.images['gray'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_gray.jpg")
            cv2.imwrite(filename, self.images['gray'])

        if (image_type == 'all' or image_type == 'thresholded') and self.images['thresholded'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_thresholded.jpg")
            cv2.imwrite(filename, self.images['thresholded'])

        if (image_type == 'all' or image_type == 'sharpened') and self.images['sharpened'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_sharpened.jpg")
            cv2.imwrite(filename, self.images['sharpened'])

        if (image_type == 'all' or image_type == 'dilated') and self.images['dilated'] is not None:
            # Construct the filename with the date and time
            filename = os.path.join(image_folder, f"{date_string}_dilated.jpg")
            cv2.imwrite(filename, self.images['dilated'])
