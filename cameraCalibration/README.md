#set this in aruco_detect.py  

#--- Set the camera size as the one it was calibrated with  
frame_width = 1280  
frame_height = 720  

#-- Get the camera calibration path and data  
calib_path = ""  
camera_matrix = np.loadtxt(calib_path+'cameraMatrix.txt', delimiter=',')  
camera_distortion = np.loadtxt(calib_path+'cameraDistortion.txt', delimiter=',')  
  
  
  
def draw_aruco(frame, gray):  
    ....  
    corners, ids, _ = detector.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters, cameraMatrix=camera_matrix, distCoeff=camera_distortion)  
    ....  

 
