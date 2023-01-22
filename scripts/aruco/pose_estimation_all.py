import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import pathlib as pl
import time
import pandas as pd

aruco_type = 'DICT_5X5_50'
p_img_path = pl.Path('C:\\Users\daniel.abler\CloudStorage\SWITCHdrive\WORK\Projects\ACTIVE\\2022_SENSE_InstrumentedSlippers\data\\2022-12-06_marker-tests\camera-videos\\normal_steps_bright_video2_detected_orig_img')
aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

p_base = pl.Path('C:\\Users\daniel.abler\CloudStorage\SWITCHdrive\WORK\Projects\ACTIVE\\2022_SENSE_InstrumentedSlippers\data\\2022-12-06_marker-tests')
calibration_matrix_path = p_base.joinpath('calibration_matrix.npy')
distortion_coefficients_path = p_base.joinpath('distortion_coefficients.npy')

matrix_coefficients = np.load(calibration_matrix_path)
distortion_coefficients = np.load(distortion_coefficients_path)

marker_size = 0.01

image_list = []
for p_img in p_img_path.glob('*'):
    img = cv2.imread(p_img.as_posix())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if len(corners) > 0:
        marker_list = []
        for i, marker_id in enumerate(ids):
            pose_dict = {}
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, matrix_coefficients, distortion_coefficients)
            rotation_mat, _ = cv2.Rodrigues(rvec)
            pose_mat = cv2.hconcat((rotation_mat, tvec.T))
            cam_mat, rot_mat, trans_vec, rot_mat_x, rot_mat_y, rot_mat_z, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            pose_dict['tvec'] = tvec
            pose_dict['trans_vec'] = cv2.convertPointsFromHomogeneous(trans_vec.T)
            pose_dict['rvec'] = rvec
            pose_dict['rotation_mat'] = rotation_mat
            pose_dict['euler'] = euler_angles
            pose_dict['marker_points'] = markerPoints
            pose_dict['marker_size'] = marker_size
            pose_dict['marker_id'] = marker_id
            marker_list.append(pd.Series(pose_dict))
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(img, corners)
            # Draw Axis
            cv2.drawFrameAxes(img, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    df_img_markers = pd.concat(marker_list, axis=1).T
    df_img_markers['filename'] = p_img.name
    image_list.append(df_img_markers)
    # cv2.imshow('Estimated Pose', img)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    p_out_dir = pl.Path(f"{p_img.parent.as_posix()}_pose")
    p_out_dir.mkdir(exist_ok=True, parents=True)
    p_out = p_out_dir.joinpath(f"{p_img.name.split('.')[0]}_pose.png")
    cv2.imwrite(p_out.as_posix(), img)
df_imgs = pd.concat(image_list)
df_imgs.to_csv(p_out_dir.joinpath(f"pose_{marker_size}.csv"))

    # for i, pose_dict in tag_dict.items():
    #     print(f"- {i}:  tvec:  {pose_dict['tvec']}")
    #     print(f"       euler:  {pose_dict['euler'].T}")
    #     # print(f"       {pose_dict['trans_vec']}")