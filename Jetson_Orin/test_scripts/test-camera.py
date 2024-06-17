from vimba import *

with Vimba.get_instance () as vimba:
    cams = vimba.get_all_cameras ()
    with cams [0] as cam:
        exposure_time = cam.ExposureTime

        time = exposure_time.get()
        inc = exposure_time.get_increment ()

        exposure_time.set(time + inc)