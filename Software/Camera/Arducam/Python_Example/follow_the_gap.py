import cv2
import csv
import numpy as np
import ArducamDepthCamera as ac
import matplotlib.pyplot as plt

# MAX_DISTANCE value modifiable  is 2000 or 4000
MAX_DISTANCE=4000


class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


confidence_value = 30
selectRect, followRect = UserRect(), UserRect()


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_mouse(event, x, y, flags, param):
    global selectRect, followRect

    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4
        selectRect.start_y = y - 4
        selectRect.end_x = x + 4
        selectRect.end_y = y + 4
    else:
        followRect.start_x = x - 4
        followRect.start_y = y - 4
        followRect.end_x = x + 4
        followRect.end_y = y + 4


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


def usage(argv0):
    print("Usage: python " + argv0 + " [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")

def find_largest_gap(data_indices): # In the future find the 2-3 largest gaps if they exist
    """
    find_largest_gap(): Finds the largest traversable gap by counting the largest number of consecutive 
    numbers (indices) as well as their starting and ending indeces. The gaps in data_indices occurs from 
    reset_closest_points() which sets the points greater than the threshold to 0, and returns the indices 
    of the rest of the non-zero values in the array. 

    :param data_indices: a 1D array of numbers that are the indices of all the furthest data points for 
        the find the gap algorithm. 

    :return: max_length, max_start_idx, and max_end_idx. 
        The max length is the length of the widest gap that exists. 
        The max_start_idx is the starting index of the largest gap. 
        The max_end_idx isthe ending index of the largest gap. 
    """ 
    max_length = 1
    current_length = 1

    start_idx = 0
    max_start_idx = 0
    max_end_idx = 0

    for i in range(1, len(data_indices)):
        if data_indices[i] - data_indices[i - 1] == 1:  # Checking for increasing order
            current_length += 1
        else:
            if current_length > max_length: # if current len is greater than max len
                max_length = current_length # set max len to current len
                max_start_idx = start_idx 
                max_end_idx = i - 1

            current_length = 1  # Reset count
            start_idx = i # Update start index 

    if current_length > max_length: # Final check of the last value 
        max_length = current_length 
        max_start_idx = start_idx 
        max_end_idx = len(data_indices) - 1
    
    return max_length, max_start_idx, max_end_idx


def reset_closest_points(data, threshold):
    """
    reset_closest_points(): sets data great than threshold to zero and returns the indices of all 
    data points larger than zero .

    :param data: the 1D array that needs to be parsed.
    :param threshold: an experimental value that is the limit to how close an object should be distance wise 

    :return: 1D array of all the indices of all the data points larger than zero. 
    """ 
    
    data[data > threshold] = 0 # set data greater than threshold to zero 
    data_indices = np.flatnonzero(data) # find the indices of all the non zero data points in the flattened version of data
    return data_indices # returns the indices of all the data points larger than zero 


def min_data_rows(array, row_start, row_end): 
    """
    min_data_rows(): takes a 2D array and first slices it based on the necessary rows needed given by 
    row_start and row_end, and then finds the min value of all the columns for n rows between the 
    given starting and ending rows.

    :param array: the 2D array of distance data. 
    :param row_start: an experimental value of the beginning of the rows that need to be parsed. 
    :param row_end: an experimental value of the end of the rows that need to be parsed.

    :return: a 1D array of the min distance out of each col between row_start and row_end.
    """ 

    # so this function actually returns the MAX values in the column instead of the MIN, but that is because 
    # the distances far away (according to result_image2.png, the green values are further away than the red values,
    # however within the data2.csv file they are actually closer, green values: 2000s range, red values: 3000s range)
    # appear to be smaller than the vlues close by for some reason and I will need to experimentally validate 
    # why that is the case
    
    sliced_arr = array[row_start:row_end]
    col_max_val = np.max(sliced_arr, axis=0)  # axis=0 operates along columns
    return col_max_val


def move_forward(max_start_idx, max_end_idx):
    # depending on imu data, emit a sound 
    # dividing the frame into 4 quadrants. if the gaps are between any of the quadrants, relay to the speaker 
    x = 0


def main():
    print("Arducam Depth Camera Demo.")
    print("  SDK version:", ac.__version__)

    cam = ac.ArducamCamera()
    cfg_path = None
    # cfg_path = "file.cfg"

    black_color = (0, 0, 0)
    white_color = (255, 255, 255)

    ret = 0
    if cfg_path is not None:
        ret = cam.openWithFile(cfg_path, 0)
    else:
        ret = cam.open(ac.Connection.CSI, 0)
    if ret != 0:
        print("Failed to open camera. Error code:", ret)
        return

    ret = cam.start(ac.FrameType.DEPTH)
    if ret != 0:
        print("Failed to start camera. Error code:", ret)
        cam.close()
        return

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)

    r = cam.getControl(ac.Control.RANGE)

    info = cam.getCameraInfo()
    print(f"Camera resolution: {info.width}x{info.height}")

    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("preview", on_mouse)

    if info.device_type == ac.DeviceType.VGA:
        # Only VGA support confidence
        cv2.createTrackbar(
            "confidence", "preview", confidence_value, 255, on_confidence_changed
        )

    while True:
        frame = cam.requestFrame(2000)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            result_image = getPreviewRGB(result_image, confidence_buf)

            cv2.normalize(confidence_buf, confidence_buf, 1, 0, cv2.NORM_MINMAX)

            cv2.imshow("preview_confidence", confidence_buf)

            cv2.rectangle(result_image, followRect.rect, white_color, 1)
            if not selectRect.empty:
                cv2.rectangle(result_image, selectRect.rect, black_color, 2)
                print("select Rect distance:", np.mean(depth_buf[selectRect.slice]))

            cv2.imshow("preview", result_image)
            cam.releaseFrame(frame)

            ###########################################################################

            depth_array = np.array(depth_buf)
            # mean_val = np.mean(depth_array)
            # max_value = np.max(depth_array)
            # min_value = np.min(depth_array)
            # print(f"\nDepth Array: \n{depth_array}\n Max Value: {max_value}\n Min Value: {min_value}\n Mean Value: {mean_val}\n")
            # print(f"Num Rows: {depth_array.shape[0]}\n") # number of rows
            # print(f"Num Cols: {depth_array.shape[1]}\n") # number of cols

            threshold = 2999 # EXPERIMENTAL VALUE, depth values of object at closest limit to user 
            row_start = 60 # EXPERIMANTAL VALUE, depth value to first row from frame to parse
            row_end = 120 # EXPERIMENTAL VALUE, depth value of last row from frame to parse
            
            sliced_data = min_data_rows(depth_array, row_start, row_end)
            data_indices = reset_closest_points(sliced_data, threshold)
            # test_arr = [1, 2, 3, 10, 11, 12, 5, 6, 7, 8, 20, 21, 22, 23, 2, 3, 4, 30, 31, 32, 33, 34, 1, 2]
            len, start_idx, end_idx = find_largest_gap(data_indices)
            print(f"\n Largest Gap: {len}   Start Index: {start_idx}     End Index: {end_idx}\n")


            # left_frames = depth_buf[:, :info.width // 2]
            # right_frames = depth_buf[:, info.width // 2 :]
            # if np.mean(left_frames) < 1000:
            #     print("left collision")
            # if np.mean(right_frames) < 1000:
            #     print("right collision")

            ############################################################################

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cam.stop()
    cam.close()


if __name__ == "__main__":
    main()
