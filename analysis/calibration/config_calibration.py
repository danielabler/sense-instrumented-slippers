from analysis.config_analysis import square_size, marker_size, p_calibration_results, aruco_dict, p_raw_data, \
    p_calibration_matrix, p_distortion_coefficients

page_size  = 'A4'
inch_to_mm = 25.4
resolution = 600 #dots/inch
n_rows      = 11 # -> height
n_columns   = 8  # -> width

page_sizes = {"A0": [840, 1188], "A1": [594, 840], "A2": [420, 594], "A3": [297, 420], "A4": [210, 297],
              "A5": [148, 210]}

grid_size_width  = square_size * n_columns
grid_size_width_inch = grid_size_width/inch_to_mm
grid_size_width_pixel= round(grid_size_width_inch * resolution)
grid_size_height = square_size * n_rows
grid_size_height_inch = round(grid_size_height/inch_to_mm)
grid_size_height_pixel = round(grid_size_height_inch * resolution)

page_width = page_sizes[page_size][0]
page_width_inch = page_width/inch_to_mm
page_width_pixel = round(page_width_inch * resolution)

page_height = page_sizes[page_size][1]
page_height_inch = page_height/inch_to_mm
page_height_pixel = round(page_height_inch * resolution)

p_calibration_grid = p_calibration_results.joinpath('charuco-calibration-grid.tiff')


