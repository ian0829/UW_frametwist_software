import cv2
#from pyzbar.pyzbar import decode
import cv2.aruco as aruco
import time
import numpy as np
from threading import Thread
from frame_server import FrameServer

from picamera2 import MappedArray, Picamera2, Preview

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW']
tracker_type = tracker_types[2]


if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
# Create an ArUco parameters object
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

frame_width = 1280
frame_height = 720

####-------------------------------------------------
#calib_path = ""
#camera_matrix = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')
#camera_distortion = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')
####-------------------------------------------------

# Specify the coordinates of the region to crop (x, y, width, height)
x, y, width, height = 275, 225, 550, 150  # Modify these values based on your requirements
    
# QR codes to specifically detect
target_aruco = "10"
center_aruco = "99"

# Dictionary to store information about detected barcodes
barcode_data_map = {f"{target_aruco}_endleft":None,
                    f"{target_aruco}_endright":None,
                    center_aruco: None}
known_barcodes = {center_aruco: None,
                  target_aruco: None}

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

def draw_aruco(frame, gray):
    t2_start = time.time()
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    ####-------------------------------------------------
    #corners, ids, _ = detector.detectMarkers(image=gray, cameraMatrix=camera_matrix, distCoeff=camera_distortion)
    ####-------------------------------------------------
    aruco.drawDetectedMarkers(frame, corners)
    
    t2_1 = time.time()
    timestamps['aruco detect'] = get_time_in_msec(t2_start, t2_1)
    
    # Draw the detected markers
    if ids is not None:
        for i in range(len(ids)):  
            chosen_corner = corners[i][0]
            #print(chosen_corner)
            centroid = calculate_centroid(chosen_corner)
            # Display the position of the QR code
            #print(centroid)
            position_text = f"pos: {centroid}"
            display_position = tuple([int(chosen_corner[3][0]-90),int(chosen_corner[3][1]+20)])
            
            cv2.putText(frame, position_text, display_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
            if str(ids[i][0]) in known_barcodes:
                print(f"centroid x:{centroid[0]} and y: {centroid[1]}")
                if(centroid[0] > 550 and centroid[0] < 650):
                    barcode_data_map[center_aruco] = centroid
                elif (centroid[0] < 550):
                    barcode_data_map[f"{target_aruco}_endleft"] = centroid
                elif (centroid[0] > 650):
                    barcode_data_map[f"{target_aruco}_endright"] = centroid
                else:
                    print("Do NOTHING")
                
    t2_2 = time.time()
    timestamps['bounding box'] = get_time_in_msec(t2_1, t2_2)   
        
def calc_angle(frame):
    angle_l = None
    angle_r = None
    # Calculate the angle between detected barcodesp
    barcode_keys = list(barcode_data_map.keys())
    if len(barcode_keys) >= 3:
        center_barcode = barcode_data_map[center_aruco]
        barcode_right = barcode_data_map[f"{target_aruco}_endright"]
        barcode_left = barcode_data_map[f"{target_aruco}_endleft"]
        # Check if both barcodes have been detected and have centroids
        if center_barcode is not None and barcode_right is not None and barcode_left:
            # Draw a line between the centroids of the two barcodes
            cv2.line(frame, center_barcode, barcode_left, (255, 0, 0), 2)
            cv2.line(frame, center_barcode, barcode_right, (255, 0, 0), 2)
            
            # Draw a horizontally parallel line spanning the entire width of the screen
            horizontal_line_start_point = (0, center_barcode[1])
            horizontal_line_end_point = (frame.shape[1], center_barcode[1])
            cv2.line(frame, horizontal_line_start_point, horizontal_line_end_point, (0, 0, 255), 2)

            angle_l = angle_between_centroids(barcode_left, center_barcode)
            angle_r = angle_between_centroids(center_barcode, barcode_right)
            #barcode_data_map.clear()
        return angle_l, angle_r

def preprocess_image():
    global curr_frame, curr_gray
    frame_img = curr_frame
    time_start = time.time()
    # Crop the specified region
    #frame_img = frame_img[y:y+height, x:x+width]
    # Define a sharpening kernel
    kernel2d = np.array([[0, -1, 0],
                       [-1,  9, -1],
                       [0, -1, 0]])

    # Apply the sharpening kernel to the image
    #img_blur = cv2.blur(src=frame_img, ksize=(1,1))
    sharpened_image = cv2.filter2D(frame_img, -1, kernel=kernel2d)
        
    t1 = time.time()
        
    gray_img = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    
    '''
    hsv = cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask1,mask2)
    inv_mask = cv2.bitwise_not(mask)
    new_red = cv2.bitwise_and(frame_img, frame_img, mask=mask)
    background = cv2.bitwise_and(gray_img, gray_img, mask=inv_mask)
    background = np.stack((background,)*3, axis=-1)
    added_img = cv2.add(new_red, background)
    
    cv2.imshow("redimg", added_img)
    '''
    
    t2 = time.time()
        
    ret, curr_gray = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY)
    t3 = time.time()
    curr_frame =  frame_img
    #print(f"t1: {get_time_in_msec(time_start, t1)} t2: {get_time_in_msec(t1, t2)} t3: {get_time_in_msec(t2, t3)}")   
    
def camera_setup():
    capture_config = picam2.create_video_configuration(main={"size": (frame_width, frame_height),"format": 'RGB888'}, display=None)
    #hires_config = picam2.create_still_configuration(raw={'format': 'SBGGR12', 'size': (4056, 3040)})
    #preview_config = picam2.create_preview_configuration(main={"size": (frame_width, frame_height),"format": 'RGB888'})
    picam2.configure(capture_config)

    barcodes = []
    picam2.start()
    time.sleep(1)

picam2 = Picamera2()
camera_setup()

server = FrameServer(picam2)
server.start()

curr_frame = None
curr_gray = None
angle_left = None
angle_right = None
compensate_r = 0
compensate_l = 0
time.sleep(1)
#tracker.init(curr_frame, )

while True:
    time_init = time.time()
    
    curr_frame = server.wait_for_frame(curr_frame)
    
    preprocess_image()
    t1 = time.time()
    timestamps['preprocessing'] = get_time_in_msec(time_init, t1)
    
    draw_aruco(curr_frame, curr_gray)
    t2 = time.time()
    timestamps['barcode identification'] = get_time_in_msec(t1, t2)
    
    angle_left, angle_right = calc_angle(curr_frame)
    compensated_left = angle_left - compensate_l
    compensated_right = angle_right - compensate_r
    
    # Display the angle
    angle_text = f"{compensated_left:.2f} degrees"
    cv2.putText(curr_frame, angle_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    angle_text = f"{compensated_right:.2f} degrees"
    cv2.putText(curr_frame, angle_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    t3 = time.time()
    timestamps['angle calc'] = get_time_in_msec(t2, t3)
    
    fps = 1/(time.time() - time_init)
    fps_text = f"{fps:.2f}"
    print(f"fps:", fps_text )
    
    
    # Show the output frame
    cv2.imshow("gray", curr_gray)
    cv2.imshow("detections", curr_frame)
    #cv2.putText(curr_frame, fps_text, (curr_frame.shape[1]-280, curr_frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #print(f"left angle: {angle_left:.2f},")
    #print(f"right angle: {angle_right:.2f},")
    
    
    t4 = time.time()
    timestamps['image display'] = get_time_in_msec(t3, t4)
    
    # Concatenate all timestamps into a single string
    timestamps_string = ', '.join([f"{key}: {value}" for key, value in timestamps.items()])
    # Print the single line with all timestamps
    #print(f"time taken---{timestamps_string}")
    
    key = cv2.waitKey(1) & 0xFF
    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        server.stop()
        picam2.stop()
        break
    elif key == ord("r"):
        compensate_r = angle_left
        compensate_l = angle_right
        
    
# Release the video stream and close all windows
cv2.destroyAllWindows()

