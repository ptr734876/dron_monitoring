# config.py - добавляем новые параметры
SIFT_N_FEATURES = 0
SIFT_CONTRAST_THRESHOLD = 0.01
SIFT_EDGE_THRESHOLD = 12
MATCH_RATIO_THRESHOLD = 0.7
RANSAC_THRESHOLD = 6.0
RANSAC_MAX_ITERS = 2000
MIN_MATCH_COUNT = 10
MAP_ALTITUDE = 1
MAP_GSD = None

# Новые параметры для лазерного дальномера
LASER_MAX_RANGE = 100.0  # максимальная дальность в метрах
LASER_ACCURACY = 0.1     # точность в метрах
DRONE_FOV_HORIZONTAL = 62.2  # угол обзора камеры по горизонтали
DRONE_FOV_VERTICAL = 48.8    # угол обзора камеры по вертикали
CAMERA_RESOLUTION_WIDTH = 1920
CAMERA_RESOLUTION_HEIGHT = 1080
USE_LASER_TRIANGULATION = True  # флаг режима работы