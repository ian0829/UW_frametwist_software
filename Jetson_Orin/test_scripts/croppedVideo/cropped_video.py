import cv2
import cv2.aruco as aruco

from crop_coords import cal_crop_coords, detect_aruco_markers


def crop_and_display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    ret, frame = cap.read()
    crop_coords = cal_crop_coords(frame)
    min_y, max_y, min_x, max_x = crop_coords

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[min_y:max_y, min_x:max_x]
        corners, ids = detect_aruco_markers(cropped_frame)
        aruco.drawDetectedMarkers(cropped_frame, corners, ids)
        cv2.imshow('Cropped Frame', cropped_frame)

        if ids is not None and len(ids) == 3:
            print(ids)
        else:
            # Update crop_coords
            crop_coords = cal_crop_coords(frame)  # Pass the current frame
            min_y, max_y, min_x, max_x = crop_coords
            cropped_frame = frame[min_y:max_y, min_x:max_x]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


video_path = "/home/frametwist/Documents/images/2024-05-09_17-53-13_original.avi"
crop_and_display_video(video_path)
