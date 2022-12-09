'''
Sample Usage:-
python scripts/aruco/pose_estimation.py --K_Matrix output/calibration_matrix.npy --D_Coeff output/distortion_coefficients.npy --type DICT_5X5_50 --image output\aruco_test_pics\DICT_5X5_50\20221205_123438.jpg
'''


import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import pathlib as pl
import time


def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()


    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
        # cameraMatrix=matrix_coefficients,
        # distCoeff=distortion_coefficients
                                                                )

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            rotation_mat, _ = cv2.Rodrigues(rvec)
            pose_mat = cv2.hconcat((rotation_mat, tvec))
            # _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

            print("----- markers", i)
            print('rvec', rvec)
            print('tvec', tvec)
            print('marker points', markerPoints)
            print('rotation matrix', rotation_mat)
            # print('pose matrix', pose_mat)
            # print('euler angles', euler_angles)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    ap.add_argument("-i", "--image", type=str, help="path_to_image")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # video = cv2.VideoCapture(0)
    # time.sleep(2.0)
    #
    # while True:
    #     ret, frame = video.read()
    #
    #     if not ret:
    #         break
    #
    #     output = pose_esitmation(frame, aruco_dict_type, k, d)
    #
    #     cv2.imshow('Estimated Pose', output)
    #
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    #
    # video.release()
    p_img = pl.Path(args["image"])
    img = cv2.imread(p_img.as_posix())
    output = pose_esitmation(img, aruco_dict_type, k, d)
    cv2.imshow('Estimated Pose', output)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    p_out = p_img.parent.joinpath(f"{p_img.name.split('.')[0]}_pose.png")
    cv2.imwrite(p_out.as_posix(), output)

