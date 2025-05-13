import os
import cv2
import time
import numpy as np
import RPi.GPIO as GPIO
import ArducamDepthCamera as ac

# MAX_DISTANCE value modifiable  is 2000 or 4000
MAX_DISTANCE=2000

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


confidence_value = 50
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


def main():
    print("Arducam Depth Camera Demo.")
    print("  SDK version:", ac.__version__)
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.IN, pull_up_down = GPIO.PUD_DOWN) 
    
    ir_folder = "ir_frames" 
    depth_folder = "depth_frames"

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
        
        button_state = GPIO.input(4)
        
        frame = cam.requestFrame(2000)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

            result_image = (depth_buf * (255.0 / r)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            result_image = getPreviewRGB(result_image, confidence_buf)

            cv2.normalize(confidence_buf, confidence_buf, 1, 0, cv2.NORM_MINMAX)

            cv2.rectangle(result_image, followRect.rect, white_color, 1)
            if not selectRect.empty:
                cv2.rectangle(result_image, selectRect.rect, black_color, 2)
                print("select Rect distance:", np.mean(depth_buf[selectRect.slice]))
                
            cv2.imshow("preview_confidence", confidence_buf)
            cv2.imshow("preview", result_image)
            
            cam.releaseFrame(frame)
            
            # print("Image saved as result_image.png")
            # np.save("depth_buf.npy", depth_buf)
            # print("Depth buffer saved as depth_buf.npy")

        if button_state == GPIO.HIGH:
            print("button pressed")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            ir_filename = os.path.join(ir_folder, f"ir_frame_{timestamp}.png")
            depth_filename = os.path.join(depth_folder, f"depth_frame_{timestamp}.png")
            
            cv2.imwrite(ir_filename, (confidence_buf * 255).astype(np.uint8))
            cv2.imwrite(depth_filename, result_image)
            
            time.sleep(0.2) # debouncing delay
            
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cam.stop()
    cam.close()


if __name__ == "__main__":
    main()
