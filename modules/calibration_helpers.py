import cv2
import logging
import numpy as np



def get_corners_ids_from_img(p_img, aruco_dict, board, criteria, min_num_corners=6):
    frame = cv2.imread(p_img.as_posix())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    corners_found, ids_found = None, None
    if len(corners) > 0:
        # SUB PIXEL DETECTION
        for corner in corners:
            cv2.cornerSubPix(gray, corner,
                             winSize=(3, 3),
                             zeroZone=(-1, -1),
                             criteria=criteria)
        res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if res2[1] is not None and res2[2] is not None and len(res2[1])>=min_num_corners:
            corners_found = res2[1]
            ids_found     = res2[2]
        else:
            logging.warning("no results")
    else:
        logging.warning("no corners found")
    return ids_found, corners_found, img_size


def get_corners_ids_from_imgs(p_imgs, aruco_dict, board, criteria, min_num_corners=6, glob_str='*.png'):
    pp_images = p_imgs.glob(glob_str)
    corners_found_list = []
    ids_found_list = []
    img_size_list = []
    for p_image in pp_images:
        logging.debug(f"-- processing image {p_image}")
        ids_found, corners_found, img_size = get_corners_ids_from_img(p_image, aruco_dict, board, criteria, min_num_corners)
        if (ids_found is not None) and (corners_found is not None):
            ids_found_list.append(ids_found)
            corners_found_list.append(corners_found)
            img_size_list.append(img_size)
    return ids_found_list, corners_found_list, img_size_list[0]


def compute_calibration(corners_found_list, ids_found_list, img_size, board, criteria):
    logging.info(f"== Computing calibration for {len(ids_found_list)} images")
    cameraMatrixInit = np.array([[1000., 0., img_size[0] / 2.],
                                 [0., 1000., img_size[1] / 2.],
                                 [0., 0., 1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=corners_found_list,
        charucoIds=ids_found_list,
        board=board,
        imageSize=img_size,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=criteria)
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
