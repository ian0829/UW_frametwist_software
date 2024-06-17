import time

import cv2

timestamps = {}
last_access_time = None


def get_user_input(timeout=0.1):
    return cv2.waitKey(int(timeout * 10)) & 0xFF

    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        char_data = sys.stdin.readline().strip().lower()
        print(char_data)
        return char_data
    else:
        return None


def sum_timestamps() -> float:
    total_sum = sum(timestamps.values())
    return total_sum


def get_time_in_msec(start: float, end: float) -> float:
    return (end - start) * 1000


def log_timestamp(label: str):
    global last_access_time
    try:
        curr_timestamp = time.time()
        if label == 'start time':
            timestamps.clear()
            last_access_time = curr_timestamp
        timestamps[label] = get_time_in_msec(curr_timestamp, last_access_time)
        last_access_time = time.time()
    except Exception as e:
        print(e)


def subtract_starting_time(starting_time):
    global timestamps
    offset = starting_time
    # Subtract the starting time from all timestamps
    timestamps = {event: timestamp - offset for event, timestamp in timestamps.items()}
    print(timestamps)


def char_to_val(data):
    return ord(data)
    # return data.char
