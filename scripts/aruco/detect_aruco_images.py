'''
Sample Command:-
python scripts/aruco/detect_aruco_images.py --image output\aruco_test_pics\DICT_5X5_50\20221205_123438.jpg --type DICT_5X5_50

'''
import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import cv2
import sys
import pathlib as pl


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())


p_img = pl.Path(args["image"])
print(f"Loading image {p_img}")
image = cv2.imread(p_img.as_posix())
h,w,_ = image.shape
width=600
height = int(width*(h/w))
image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


# verify that the supplied ArUCo tag exists and is supported by OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("Detecting '{}' tags....".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
#
# for i in range(len(corners)):
# 	print(f"---- {i}")
# 	print(corners[i])
# 	print(ids[i])
# 	print(rejected[i])
detected_markers = aruco_display(corners, ids, rejected, image)
cv2.imshow("Image", detected_markers)

# # Uncomment to save
p_out = p_img.parent.joinpath(f"{p_img.name.split('.')[0]}_detected.png")
cv2.imwrite(p_out.as_posix(),detected_markers)

cv2.waitKey(0)