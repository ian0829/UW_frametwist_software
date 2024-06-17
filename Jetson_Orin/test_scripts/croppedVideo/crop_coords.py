import cv2
import cv2.aruco as aruco
import numpy as np


def detect_aruco_markers(image):
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    arucoParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParams)
    (corners, ids, rejected) = detector.detectMarkers(image)
    return corners, ids


def cal_crop_coords(cur_frame):
    crop_coordinates = None

    corners, ids = detect_aruco_markers(cur_frame)
    if ids is not None and len(corners) > 0:
        all_corners = np.concatenate(corners)
        min_x = np.min(all_corners[:, :, 0]) - 50
        max_x = np.max(all_corners[:, :, 0]) + 50
        min_y = np.min(all_corners[:, :, 1]) - 50
        max_y = np.max(all_corners[:, :, 1]) + 50
        # 50, Just an arbitrary number; have a larger image so the program detects the whole frame less frequently.
        crop_coordinates = (int(min_y), int(max_y), int(min_x), int(max_x))

    print(crop_coordinates)

    return crop_coordinates
