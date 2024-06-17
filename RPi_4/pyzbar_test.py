import cv2
from pyzbar.pyzbar import decode

import time
import numpy as np
from threading import Thread

from picamera2 import MappedArray, Picamera2, Preview
picam2 = Picamera2()
frame_width = 1280
frame_height = 720

# Specify the coordinates of the region to crop (x, y, width, height)
x, y, width, height = 275, 225, 550, 150  # Modify these values based on your requirements
    
# QR codes to specifically detect
target_barcode_1 = "3300051001_1345_2"
target_barcode_2 = "3300051001_1345_3"

# Dictionary to store information about detected barcodes
barcode_data_map = {
                    target_barcode_1:None,
                    target_barcode_2:None
                    }

timestamps = {}

def get_time_in_msec(start, end):
    return (end - start)*1000

# Function to calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Function to calculate the angle between two centroids
def angle_between_centroids(centroid1, centroid2):
    # Calculate the angle using the tangent function
    delta_x = centroid2[0] - centroid1[0]
    delta_y = centroid2[1] - centroid1[1]

    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)

    return -angle_deg
# Function to calculate the centroid of a set of points
def calculate_centroid(points):
    x = int(sum(p[0] for p in points) / len(points))
    y = int(sum(p[1] for p in points) / len(points))
    return x, y

colour = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2

# Loop over the frames from the video stream
def draw_barcodes(frame, gray):
    t2_start = time.time()
    
    # Use the pyzbar library to decode barcodes in the frame
    barcodes = decode(gray)
    t2_1 = time.time()
    timestamps['t2_1'] = get_time_in_msec(t2_start, t2_1)
    
    # Loop over the detected barcodes
    for barcode in barcodes:
        # Extract the bounding box location of the barcode
        (x, y, w, h) = barcode.rect
        
        # Draw a rectangle around the barcode
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the data from the barcode
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        #print(barcode_data, barcode.rect)

        # Display barcode information and coordinates
        text = f"{barcode_type}: {barcode_data}  ({x}, {y})"
        #cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate the centroid (position) of the QR code
        barcode_corners = np.array(barcode.polygon, dtype=np.int32)
        centroid = calculate_centroid(barcode_corners)
        
        # Display the position of the QR code
        position_text = f"Position: {centroid}"
        cv2.putText(frame, position_text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
        
        # Update centroid information in the dictionary based on barcode data
        if barcode_data in barcode_data_map:
            barcode_data_map[barcode_data] = centroid
    t2_2 = time.time()
    timestamps['t2_2'] = get_time_in_msec(t2_1, t2_2)
    
        
def calc_angle(frame):
    # Calculate the angle between detected barcodesp
    barcode_keys = list(barcode_data_map.keys())
    if len(barcode_keys) >= 2:
        barcode1 = barcode_data_map[target_barcode_2]
        barcode2 = barcode_data_map[target_barcode_1]
        # Check if both barcodes have been detected and have centroids
        angle = 0
        if barcode1 is not None and barcode2 is not None:    
            # Draw a line between the centroids of the two barcodes
            cv2.line(frame, barcode1, barcode2, (255, 0, 0), 2)
            # Draw a horizontally parallel line spanning the entire width of the screen
            horizontal_line_start_point = (0, barcode1[1])
            horizontal_line_end_point = (frame.shape[1], barcode1[1])
            cv2.line(frame, horizontal_line_start_point, horizontal_line_end_point, (0, 0, 255), 2)

            angle = angle_between_centroids(barcode1, barcode2)

            # Display the angle
            angle_text = f"{angle:.2f} degrees"
           # cv2.putText(frame, angle_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            #barcode_data_map.clear()
        return angle

def capture_and_convert():
    global curr_frame, curr_gray
    
    time_start = time.time()
    
    frame_img = picam2.capture_array("main")
        
    t1 = time.time()
    # Crop the specified region
    #frame_img = frame_img[y:y+height, x:x+width]
    # Define a sharpening kernel
    kernel2d = np.array([[0, -1, 0],
                       [-1,  9, -1],
                       [0, -1, 0]])

    # Apply the sharpening kernel to the image
    #img_blur = cv2.blur(src=frame_img, ksize=(1,1))
    sharpened_image = cv2.filter2D(frame_img, -1, kernel=kernel2d)
        
    t2 = time.time()
        
    gray_img = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
        
    t3 = time.time()
        
    ret, curr_gray = cv2.threshold(gray_img, 220, 255, cv2.THRESH_BINARY)
    t4 = time.time()
    curr_frame =  frame_img
    fps = 1/(time.time() - time_start)
    fps_text = f"{fps:.2f}"
    print(f"fps:", fps_text )
    print(f"t1: {get_time_in_msec(time_start, t1)} t2: {get_time_in_msec(t1, t2)} t3: {get_time_in_msec(t2, t3)} t4: {get_time_in_msec(t3, t4)}")   
        
def capture_thread(stop):
    while True:
        capture_and_convert()
        if stop():
            break
    
def camera_setup():
    capture_config = picam2.create_video_configuration(main={"size": (frame_width, frame_height),"format": 'RGB888'}, display=None)
    hires_config = config = picam2.create_still_configuration(raw={'format': 'SBGGR12', 'size': (4056, 3040)})

    picam2.configure(capture_config)

    barcodes = []
    #picam2.post_callback = draw_barcodes
    picam2.start()
    time.sleep(1)

curr_frame = None
curr_gray = None
camera_setup()
stop_threads = False
camera_thread = Thread(target = capture_thread, args =(lambda : stop_threads, ))
camera_thread.start()
time.sleep(1)
# Capture the first frame to get the initial QR code coordinates
#prev_frame, prev_gray = capture_and_convert()
#prev_barcodes = decode(prev_gray)

while True:
    time_start = time.time()
    
    #curr_frame, curr_gray = capture_and_convert()
    t1 = time.time()
    timestamps['preprocessing'] = get_time_in_msec(time_start, t1)
    
    draw_barcodes(curr_frame, curr_gray)
    t2 = time.time()
    timestamps['barcode identification'] = get_time_in_msec(t1, t2)
    
    curr_angle = calc_angle(curr_frame)
    t3 = time.time()
    timestamps['angle calc'] = get_time_in_msec(t2, t3)
    
    
    # Show the output frame
    cv2.imshow("gray", curr_gray)
    cv2.imshow("detections", curr_frame)
    #cv2.putText(curr_frame, fps_text, (curr_frame.shape[1]-280, curr_frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #print(f"angle: {curr_angle:.2f},")
    
    
    t4 = time.time()
    timestamps['image display'] = get_time_in_msec(t3, t4)
    
    # Concatenate all timestamps into a single string
    timestamps_string = ', '.join([f"{key}: {value}" for key, value in timestamps.items()])
    # Print the single line with all timestamps
    #print(f"time taken---{timestamps_string}")
    
    # Update the previous frame and barcodes for the next iteration
    prev_frame = curr_frame.copy()
    prev_gray = curr_gray
       
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        stop_threads = True
        camera_thread.join()
        break
    
# Release the video stream and close all windows
picam2.stop()
cv2.destroyAllWindows()
