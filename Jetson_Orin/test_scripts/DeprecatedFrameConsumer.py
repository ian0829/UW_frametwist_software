from frame_server import *
from angle_math import *
from utils import get_user_input, char_to_val


class FrameConsumerOld:
    def __init__(self):

        self.log = Log.get_instance()
        self._server = FrameServer(0)
        self._alive = True
        self._compensate_r = 0
        self._compensate_l = 0
        self._capture_continuous = None
        self._capture_once = False
        self._server.start()

    def userinput(self):
        input_key = get_user_input()
        # If the 'q' key is pressed, break from the loop
        if input_key == char_to_val("q"):
            self._alive = False
        elif input_key == char_to_val("r"):
            self._compensate_r = angle_left
            self._compensate_l = angle_right
        elif input_key == char_to_val("c"):
            self._capture_once = True
        elif input_key == char_to_val("x"):
            self._capture_continuous = not self._capture_continuous

    def run(self):

        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
        self.log.info('\'FrameConsumer\' started.')
        print("waiting for key")
        curr_frame = None
        while self._alive:
            curr_frame = self._server.wait_for_frame(curr_frame)
            if self._capture_continuous:
                curr_frame = looper(curr_frame)
            elif self._capture_once:
                looper(curr_frame)
                self._capture_once = False
            else:
                # If there are no frames available, show dummy image instead
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            self.userinput()
        self._server.stop()
        # Release the video stream and close all windows
        cv2.destroyAllWindows()
