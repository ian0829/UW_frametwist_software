# Python program to illustrate
# saving an operated video
import os
import time
from datetime import datetime

# organize imports
import numpy as np
import cv2

frame_width = 768
frame_height = 688
directory = os.path.join(os.path.expanduser("~"), "Documents", "images")

# This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)

os.makedirs(directory, exist_ok=True)
# Get the current date and time
current_time = datetime.now()
framerate = 30.0

# Format the date and time as a string to use in the filename
date_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")

filename = os.path.join(directory, f"{date_string}_original.avi")
fourcc = cv2.VideoWriter.fourcc(*'GREY')
frameSize = (frame_width, frame_height)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(filename, fourcc, framerate, frameSize, False)
#

# loop runs if capturing has been initialized.
while (True):
    # reads frames from a camera
    # ret checks return at each frame
    ret, frame = cap.read()

    # output the frame
    out.write(frame)

    # The original input frame is shown in the window
    cv2.imshow('Original', frame)

    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
out.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
