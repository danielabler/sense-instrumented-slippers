'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4

python scripts/aruco/detect_aruco_video.py --type DICT_5X5_100 --camera False --video /media/dabler/ext1TB/2023-01-12/08_calibration-pattern_video2.avi
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
import pathlib as pl

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())


if args["video"] is None:
	print("[Error] Video file location is not provided")
	sys.exit(1)

p_video = pl.Path(args["video"])
video = cv2.VideoCapture(p_video.as_posix())

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

counter = 0
while True:
	ret, frame_orig = video.read()
	counter = counter + 1
	if ret is False:
		break


	h, w, _ = frame_orig.shape

	width=1000
	height = int(width*(h/w))
	frame = cv2.resize(frame_orig, (width, height), interpolation=cv2.INTER_CUBIC)
	corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

	detected_markers = aruco_display(corners, ids, rejected, frame)

	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF

	if ids is not None:
		name = p_video.name.split('.')[0]
		p_out = p_video.parent.joinpath(f"{name}_detected").joinpath(f"{name}_frame-{counter}.png")
		p_out.parent.mkdir(exist_ok=True, parents=True)
		cv2.imwrite(p_out.as_posix(), detected_markers)
		p_out_orig = p_video.parent.joinpath(f"{name}_detected_orig_img").joinpath(f"{name}_frame-{counter}.png")
		p_out_orig.parent.mkdir(exist_ok=True, parents=True)
		cv2.imwrite(p_out_orig.as_posix(), frame_orig)
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()