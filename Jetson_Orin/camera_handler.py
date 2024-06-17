import sys
from typing import Optional
from vmbpy import VmbSystem, Camera, VmbFeatureError, VmbCameraError
import vmbpy.util.log as logger


def set_nearest_value(cam: Camera, feat_name: str, feat_value: int, log: logger):
    # Helper function that tries to set a given value. If setting of the initial value failed
    # it calculates the nearest valid value and sets the result. This function is intended to
    # be used with Height and Width Features because not all Cameras allow the same values
    # for height and width.
    feat = cam.get_feature_by_name(feat_name)

    try:
        feat.set(feat_value)

    except VmbFeatureError:
        min_, max_ = feat.get_range()
        inc = feat.get_increment()

        if feat_value <= min_:
            val = min_

        elif feat_value >= max_:
            val = max_

        else:
            val = (((feat_value - min_) // inc) * inc) + min_

        try:
            feat.set(val)
            msg = ('Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
                   'Using nearest valid value \'{}\'. Note that, this causes resizing '
                   'during processing, reducing the frame rate.')
            log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value, val))

        except Exception as e:
            print(e)
            access_mode = cam.get_access_mode()
            print(f'Access mode: {access_mode}')
            msg = 'Camera {}: Failed to set value of Feature \'{}\' to \'{}\': '
            log.get_instance().info(msg.format(cam.get_id(), feat_name, feat_value))


# Function to print all available pixel formats and the current pixel format for the camera
def print_pixel_formats(cam: Camera):
    print(cam.get_pixel_formats())
    print(cam.get_pixel_format())


# Function to print the description and details of the feature type passed in the argument
def print_feature(feature):
    try:
        value = feature.get()

    except (AttributeError, VmbFeatureError):
        value = None

    print('/// Feature name   : {}'.format(feature.get_name()))
    print('/// Display name   : {}'.format(feature.get_display_name()))
    print('/// Tooltip        : {}'.format(feature.get_tooltip()))
    print('/// Description    : {}'.format(feature.get_description()))
    print('/// SFNC Namespace : {}'.format(feature.get_sfnc_namespace()))
    print('/// Unit           : {}'.format(feature.get_unit()))
    print('/// Value          : {}\n'.format(str(value))) if value is not None else None


# Function to return the camera instance for the given camera ID
def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


# Function to exit the script
def abort(reason: str, return_code: int = 1):
    print(reason + '\n')
    sys.exit(return_code)
