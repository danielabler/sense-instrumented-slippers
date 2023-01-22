import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import pathlib as pl
import time


aruco_type = 'DICT_5X5_50'
p_img = pl.Path('C:\\Users\\daniel.abler\\Documents\\repositories\\instrumented-slippers\\output\\aruco_test_pics\\DICT_5X5_50\\20221205_123459.jpg')
aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

p_base = pl.Path('C:\\Users\daniel.abler\CloudStorage\SWITCHdrive\WORK\Projects\ACTIVE\\2022_SENSE_InstrumentedSlippers\data\\2022-12-06_marker-tests')
calibration_matrix_path = p_base.joinpath('calibration_matrix.npy')
distortion_coefficients_path = p_base.joinpath('distortion_coefficients.npy')

matrix_coefficients = np.load(calibration_matrix_path)
distortion_coefficients = np.load(distortion_coefficients_path)

img = cv2.imread(p_img.as_posix())
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
parameters = cv2.aruco.DetectorParameters_create()

corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
tag_dict = {}
if len(corners) > 0:
    for i in range(0, len(ids)):
        pose_dict = {}
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
        rotation_mat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat((rotation_mat, tvec.T))
        cam_mat, rot_mat, trans_vec, rot_mat_x, rot_mat_y, rot_mat_z, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pose_dict['tvec'] = tvec
        pose_dict['trans_vec'] = cv2.convertPointsFromHomogeneous(trans_vec.T)
        pose_dict['rvec'] = rvec
        pose_dict['rotation_mat'] = rotation_mat
        pose_dict['euler'] = euler_angles
        pose_dict['marker_points'] = markerPoints
        tag_dict[i] = pose_dict
        # Draw a square around the markers
        cv2.aruco.drawDetectedMarkers(img, corners)
        # Draw Axis
        cv2.drawFrameAxes(img, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

cv2.imshow('Estimated Pose', img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

p_out = p_img.parent.joinpath(f"{p_img.name.split('.')[0]}_pose.png")
cv2.imwrite(p_out.as_posix(), img)

for i, pose_dict in tag_dict.items():
    print(f"- {i}:  tvec:  {pose_dict['tvec']}")
    print(f"       euler:  {pose_dict['euler'].T}")
    # print(f"       {pose_dict['trans_vec']}")