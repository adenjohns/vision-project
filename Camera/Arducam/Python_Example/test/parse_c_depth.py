import numpy as np
import cv2 

width, height = 680, 480
filename = "depth_240_180_f32_1744587428857.raw"

# read the raw depth file 
with open(filename, "rb") as f: 
	depth = np.fromfile(f, dtype = np.float32).reshape((height, width))
	
# normailze for viewing 
depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX) 
depth_vis = np.uint8(depth_vis) 

# apply colormap for meter visualizaton 
colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET) 

cv2.imshow("Depth Image", colored) 
cv2.waitKey(0)
cv2.destroyAllWindows()
