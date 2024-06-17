import math
import cv2
import numpy as np
from itertools import combinations
from config import left_aruco, center_aruco, FRAME_WIDTH, FRAME_HEIGHT, COLINEARITY_THRESHOLD, right_aruco


# Function to draw a triangle using points in the frame
def draw_triangle(image, points, color=(0, 255, 0), thickness=2):
    # Convert points to integer tuples
    points = np.array(points, dtype=np.int32)

    # Draw the triangle
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)

    # Fill the triangle
    cv2.fillPoly(image, [points], color=color)


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
    try:
        x_pos = int(sum(p[0] for p in points) / len(points))
        y_pos = int(sum(p[1] for p in points) / len(points))
        return x_pos, y_pos
    except Exception as e:
        print(e)
        return 0, 0


def calculate_diagonal_distance(points):
    try:
        # Ensure there are exactly 4 points
        if len(points) != 4:
            raise ValueError("There must be exactly 4 corner points to calculate the marker size.")

        # Calculate the distance between adjacent points
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # Calculate distances between adjacent points
        d1 = distance(points[0], points[1])
        d2 = distance(points[1], points[2])
        d3 = distance(points[2], points[3])
        d4 = distance(points[3], points[0])

        # Determine the width and height (assuming marker is rectangular)
        width = max(d1, d3)
        height = max(d2, d4)

        # Calculate the diagonal (hypotenuse) of the rectangle
        diagonal = math.sqrt(width ** 2 + height ** 2)

        return diagonal
    except Exception as e:
        print(e)
        return None


# Function to calculate area for all the coordinates
def calculate_area(coords):
    try:
        xmin, ymin, xmax, ymax = coords
        width = xmax - xmin
        height = ymax - ymin

        area = height * width

        return area
    except Exception as e:
        print(e)
        return None


# Function to check if the dictionary has the detected marker
def has_aruco_marker(barcode_data_map, marker_id):
    for data in barcode_data_map.values():
        if str(data['id']) == marker_id:
            return True
    return False


# Function to calculate angle between the extreme points in the list of detected marker positions
def calc_angle(frame, barcode_data_map):
    angle = 0
    collinearity = 0
    # Calculate the angle between detected barcodes
    try:

        # barcode_keys = [info['id'] for info in barcode_data_map.values()]
        barcode_positions = [info['centroid'] for info in barcode_data_map.values()]
        sorted_points = sorted(barcode_positions, key=lambda point: point[1])

        if (len(barcode_data_map) >= 2 and has_aruco_marker(barcode_data_map, right_aruco) and
                has_aruco_marker(barcode_data_map, left_aruco)):
            # Extract the center ArUco marker
            # center_marker_key = next(key for key, value in
            #                          barcode_data_map.items() if value['id'] == int(center_aruco))
            # center_marker = barcode_data_map[center_marker_key]

            angle = angle_between_centroids(sorted_points[0], sorted_points[-1])
            if angle < -90:
                angle = angle + 180

            # Draw a horizontally parallel line spanning the entire width of the screen
            horizontal_line_start_point = (0, sorted_points[0][1])
            horizontal_line_end_point = (frame.shape[1], sorted_points[-1][1])
            # cv2.line(frame, horizontal_line_start_point, horizontal_line_end_point, (0, 0, 255), 2)

            barcode_data_map.clear()
            draw_triangle(frame, sorted_points)
            ret, collinearity = check_collinearity_threshold(sorted_points)
            return angle, collinearity
    except Exception as e:
        print(e)
    return angle, collinearity


# Function to calculate the cropping region using the list of markers
def calculate_bounding_box(markers, roi_offset: int = 50, guard_x=FRAME_WIDTH, guard_y=FRAME_HEIGHT):
    try:
        # Calculate the minimum and maximum coordinates from the marker positions
        min_x = 0
        max_x = FRAME_WIDTH
        min_y = int(max(min(markers, key=lambda x: x[1])[1] - roi_offset, 0))
        max_y = int(min(max(markers, key=lambda x: x[1])[1] + roi_offset, guard_y))
        # print("range-x : {} {}  range-y : {} {}".format(min_x, max_x, min_y, max_y))
        return min_x, min_y, max_x, max_y
    except Exception as e:
        print(e)
        return None, None, None, None


def calculate_polygon_area(points):
    """
    Calculate the area of a polygon using the Shoelace formula.

    Args:
        points (list of tuple): List of (x, y) coordinates of the polygon vertices.

    Returns:
        float: Area of the polygon.
    """
    # Add the first point to the end to close the polygon
    points.append(points[0])

    # Calculate the area using the shoelace formula
    area = 0
    for i in range(len(points) - 1):
        area += (points[i][0] * points[i + 1][1]) - (points[i + 1][0] * points[i][1])
    area = abs(area) / 2

    return area


# Function to calculate the collinearity between all the points in the line
def check_collinearity_threshold(points, threshold=COLINEARITY_THRESHOLD):
    """
    Check if three points are collinear based on a threshold value for the triangle area.

    Returns:
        bool: True if the points are collinear (triangle area is below the threshold), False otherwise.
        float: Area of the formed polygon
        :param threshold:
        :param points:
    """
    area = calculate_polygon_area(points)
    if area >= threshold:
        return False, area
    else:
        return True, area
