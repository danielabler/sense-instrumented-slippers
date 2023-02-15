from config import *
from cv2 import aruco
import logging
import logging.config

logging.config.fileConfig(p_logger_config)
logger = logging.getLogger(__name__)

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

square_size = 20
marker_size = 15

p_calibration_results = p_processing_results.joinpath('calibration')
p_calibration_results.mkdir(exist_ok=True, parents=True)
p_calibration_matrix = p_calibration_results.joinpath("calibration_matrix.npy")
p_distortion_coefficients = p_calibration_results.joinpath("distortion_coefficients.npy")
p_calibration_results_dict = p_calibration_results.joinpath('calibration_results_dict.pkl')

