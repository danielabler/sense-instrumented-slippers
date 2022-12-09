Scripts for camera calibration and pose estimation using ArUco markers

Adapted from:
- https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python
- https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html

- OpenCV description of camera calibration
  https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
# Commands
- Generate calibration chessboard:python gen_pattern.py -o chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20
  - `python scripts/calibration/generate_chessboard.py -o output/chessboard.svg --rows 9 --columns 6 --type checkerboard --square_size 20`
- 
