# locator.py
import cv2
import numpy as np
import config
from feature_matcher import FeatureMatcher


class DroneLocator:
    
    def __init__(self):
        self.matcher = FeatureMatcher()
        self.map_image = None
        self.map_image_gray = None
        self.map_kp = None
        self.map_des = None
        self.map_height = 0
        self.map_width = 0
    
    def load_map(self, map_path):
        self.map_image = cv2.imread(map_path)
        self.map_image_gray = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)
        if self.map_image is None:
            raise FileNotFoundError(f"Не удалось загрузить карту: {map_path}")
        self.map_height, self.map_width = self.map_image_gray.shape
        self.map_kp, self.map_des = self.matcher.detect_and_compute(self.map_image)
    
    def _find_scale_by_ncc(self, drone_gray, x_approx, y_approx):
        h_drone, w_drone = drone_gray.shape
        
        scales = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        best_scale = None
        best_score = -1
        
        crop_size = 300
        x_min = max(0, int(x_approx - crop_size // 2))
        x_max = min(self.map_width, int(x_approx + crop_size // 2))
        y_min = max(0, int(y_approx - crop_size // 2))
        y_max = min(self.map_height, int(y_approx + crop_size // 2))
        
        if x_max - x_min < 50 or y_max - y_min < 50:
            x_min = 0
            x_max = self.map_width
            y_min = 0
            y_max = self.map_height
        
        map_crop = self.map_image_gray[y_min:y_max, x_min:x_max]
        
        for scale in scales:
            new_w = int(map_crop.shape[1] * scale)
            new_h = int(map_crop.shape[0] * scale)
            
            if new_w < w_drone or new_h < h_drone:
                continue
            
            map_scaled = cv2.resize(map_crop, (new_w, new_h))
            result = cv2.matchTemplate(map_scaled, drone_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_scale = scale
        
        return best_scale, best_score
    
    def locate(self, drone_image_path):
        drone_img = cv2.imread(drone_image_path)
        if drone_img is None:
            return {'success': False, 'error': f'Не удалось загрузить: {drone_image_path}'}
        
        drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        drone_kp, drone_des = self.matcher.detect_and_compute(drone_img)
        
        if drone_kp is None or len(drone_kp) < config.MIN_MATCH_COUNT:
            return {'success': False, 'error': f'Мало точек: {len(drone_kp) if drone_kp else 0}'}
        
        matches = self.matcher.match(drone_des, self.map_des)
        
        if len(matches) < config.MIN_MATCH_COUNT:
            return {'success': False, 'error': f'Мало совпадений: {len(matches)}'}
        
        src_pts, dst_pts = self.matcher.get_matched_points(drone_kp, self.map_kp, matches)
        
        M, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            config.RANSAC_THRESHOLD,
            maxIters=config.RANSAC_MAX_ITERS
        )
        
        if M is None:
            return {'success': False, 'error': 'Не удалось вычислить гомографию'}
        
        h, w = drone_img.shape[:2]
        center = np.float32([w / 2, h / 2]).reshape(-1, 1, 2)
        center_on_map = cv2.perspectiveTransform(center, M)
        x_sift, y_sift = center_on_map[0][0]
        
        scale_ncc, conf_ncc = self._find_scale_by_ncc(drone_gray, x_sift, y_sift)
        
        if scale_ncc is None:
            scale_ncc = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
            conf_ncc = 0
        
        inliers_count = np.sum(mask) if mask is not None else 0
        conf_sift = inliers_count / len(matches) if matches else 0
        
        drone_altitude = config.MAP_ALTITUDE / scale_ncc if scale_ncc > 0 else None
        in_bounds = 0 <= x_sift < self.map_width and 0 <= y_sift < self.map_height
        
        return {
            'success': in_bounds,
            'x': float(x_sift),
            'y': float(y_sift),
            'scale': float(scale_ncc),
            'altitude': float(drone_altitude) if drone_altitude else None,
            'confidence_sift': float(conf_sift),
            'confidence_ncc': float(conf_ncc),
            'matches_count': len(matches),
            'inliers_count': int(inliers_count)
        }