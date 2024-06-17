# from pyzbar.pyzbar import decode
import time

import cv2
import cv2.aruco as aruco

from angle_math import calculate_centroid, calculate_bounding_box, calculate_diagonal_distance
from config import FRAME_WIDTH, FRAME_HEIGHT, right_aruco, left_aruco, center_aruco, TOTAL_MARKER_COUNT
from image_handler import ImageHandler, display_window
from utils import log_timestamp

# Dictionary to store information about detected barcodes
barcode_data_map = {}
known_barcodes = {center_aruco: None,
                  left_aruco: None,
                  right_aruco: None}

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
# Create an ArUco parameters object
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

timestamps = {}


def try_crop(frame, corners, ids):
    # Revert to full-frame detection if no markers are detected
    cropping_region = (0, 0, FRAME_WIDTH, FRAME_HEIGHT)

    try:

        if ids is not None and len(ids) >= TOTAL_MARKER_COUNT:
            # Use the first three detected markers for cropping
            marker_positions = [calculate_centroid(corner[0]) for corner in corners[:3]]
            marker_size = int(max([calculate_diagonal_distance(corner[0]) for corner in corners[:3]]))
            # Calculate the bounding box based on the markers
            cropping_region = calculate_bounding_box(marker_positions, roi_offset=marker_size)

        return cropping_region
    except Exception as e:
        print(e)
        return cropping_region


def detect_aruco(image, crop_coordinates=None):
    offset_x = 0
    offset_y = 0
    # Detect ArUco markers
    if crop_coordinates is not None:
        x_min, y_min, x_max, y_max = crop_coordinates
        offset_x = x_min
        offset_y = y_min
        image = ImageHandler.select_roi(image, x_min, y_min, x_max, y_max)

    corners, ids, _ = detector.detectMarkers(image)
    # Adjust the coordinates of the detected markers by adding the offset
    if corners is not None:
        for corner_set in corners:
            for corner in corner_set[0]:  # Iterate over each corner in the set
                corner[0] += offset_x  # Add the x-offset to the x-coordinate
                corner[1] += offset_y  # Add the y-offset to the y-coordinate

    return corners, ids


def draw_aruco(frame, gray, corners, ids):
    try:
        global barcode_data_map

        aruco.drawDetectedMarkers(frame, corners)

        # Draw the detected markers
        if ids is not None:
            for i in range(len(ids)):
                aruco_id = ids[i][0]  # Get the marker ID
                chosen_corner = corners[i][0]

                # print(chosen_corner)
                centroid = calculate_centroid(chosen_corner)
                # Display the position of the QR code
                # print(centroid)
                position_text = f"pos: {centroid}"
                display_position = tuple([int(chosen_corner[3][0] - 90), int(chosen_corner[3][1] + 20)])

                cv2.putText(frame, position_text, display_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)
                if str(aruco_id) in known_barcodes:
                    if (str(aruco_id) == left_aruco and centroid[0] < (FRAME_WIDTH / 2)) or (
                            str(aruco_id) == right_aruco and centroid[0] > (FRAME_WIDTH / 2)):
                        barcode_data_map[f"{aruco_id}_{centroid[0]}"] = {'id': aruco_id,
                                                                         'corners': chosen_corner,
                                                                         'centroid': centroid
                                                                         }
            log_timestamp('bounding box')
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
