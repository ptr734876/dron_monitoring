# locator.py
import cv2
import numpy as np
import math
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
        self.maps_data = []
    
    def load_map(self, map_path):
        self.map_image = cv2.imread(map_path)
        self.map_image_gray = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)
        if self.map_image is None:
            raise FileNotFoundError(f"Не удалось загрузить карту: {map_path}")
        self.map_height, self.map_width = self.map_image_gray.shape
        self.map_kp, self.map_des = self.matcher.detect_and_compute(self.map_image)
    
    def load_multiple_maps(self, map_paths):
        self.maps_data = []
        for path in map_paths:
            map_img = cv2.imread(path)
            if map_img is None:
                raise FileNotFoundError(f"Не удалось загрузить карту: {path}")
            map_gray = cv2.cvtColor(map_img, cv2.COLOR_BGR2GRAY)
            map_kp, map_des = self.matcher.detect_and_compute(map_img)
            self.maps_data.append({
                'image': map_img,
                'gray': map_gray,
                'kp': map_kp,
                'des': map_des,
                'height': map_gray.shape[0],
                'width': map_gray.shape[1],
                'path': path
            })
    
    def locate_on_multiple_maps(self, map_paths, drone_image_path):
        self.load_multiple_maps(map_paths)
        
        drone_img = cv2.imread(drone_image_path)
        if drone_img is None:
            return {'success': False, 'error': f'Не удалось загрузить: {drone_image_path}'}
        
        drone_gray = cv2.cvtColor(drone_img, cv2.COLOR_BGR2GRAY)
        drone_kp, drone_des = self.matcher.detect_and_compute(drone_img)
        
        if drone_kp is None or len(drone_kp) < config.MIN_MATCH_COUNT:
            return {'success': False, 'error': f'Мало точек: {len(drone_kp) if drone_kp else 0}'}
        
        best_result = None
        best_score = -1
        map_scores = []
        
        print("\n" + "="*60)
        print("ANALYZING MAPS FOR DRONE LOCATION")
        print("="*60)
        
        for idx, map_data in enumerate(self.maps_data):
            matches = self.matcher.match(drone_des, map_data['des'])
            
            if len(matches) < config.MIN_MATCH_COUNT:
                map_scores.append(0.0)
                print(f"Map {idx+1} ({map_data['path']}): {len(matches)} matches (BELOW MINIMUM)")
                continue
            
            src_pts, dst_pts = self.matcher.get_matched_points(drone_kp, map_data['kp'], matches)
            
            M, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                config.RANSAC_THRESHOLD,
                maxIters=config.RANSAC_MAX_ITERS
            )
            
            if M is None:
                map_scores.append(0.0)
                print(f"Map {idx+1} ({map_data['path']}): Homography failed")
                continue
            
            M = M / M[2, 2]
            h, w = drone_img.shape[:2]
            center_x, center_y = w / 2.0, h / 2.0
            
            center_pt = np.float32([center_x, center_y]).reshape(-1, 1, 2)
            center_map = cv2.perspectiveTransform(center_pt, M)[0][0]
            x, y = center_map[0], center_map[1]
            
            in_bounds = 0 <= x < map_data['width'] and 0 <= y < map_data['height']
            
            corners_drone = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)
            
            corners_map = cv2.perspectiveTransform(corners_drone, M)
            corners_map = corners_map.reshape(4, 2)
            
            proj_w = (math.sqrt((corners_map[1][0] - corners_map[0][0])**2 + (corners_map[1][1] - corners_map[0][1])**2) +
                      math.sqrt((corners_map[2][0] - corners_map[3][0])**2 + (corners_map[2][1] - corners_map[3][1])**2)) / 2.0
            
            map_diag = math.sqrt(map_data['width']**2 + map_data['height']**2)
            proj_diag = math.sqrt(proj_w**2 + (proj_w * h / w)**2)
            
            scale = proj_diag / map_diag
            
            drone_altitude = config.MAP_ALTITUDE * scale if scale and scale > 0 else None
            
            inliers_count = np.sum(mask) if mask is not None else 0
            conf_sift = inliers_count / len(matches) if matches else 0
            
            score = inliers_count * conf_sift
            
            map_scores.append(score)
            
            in_bounds_str = "IN BOUNDS" if in_bounds else "OUT OF BOUNDS"
            print(f"Map {idx+1} ({map_data['path']}): {len(matches)} matches, {inliers_count} inliers, "
                  f"confidence: {conf_sift:.3f}, score: {score:.1f}, position: {in_bounds_str}")
            
            if in_bounds and score > best_score:
                best_score = score
                best_result = {
                    'success': True,
                    'x': float(x),
                    'y': float(y),
                    'scale': float(scale),
                    'altitude': float(drone_altitude) if drone_altitude else None,
                    'confidence_sift': float(conf_sift),
                    'matches_count': len(matches),
                    'inliers_count': int(inliers_count),
                    'map_index': idx
                }
        
        print("\n" + "="*60)
        print("MATCH PROBABILITY SUMMARY")
        print("="*60)
        
        total_score = sum(map_scores)
        if total_score > 0:
            for idx, score in enumerate(map_scores):
                probability = (score / total_score) * 100
                bar_length = int(probability / 2)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"Map {idx+1}: {bar} {probability:.1f}%")
        else:
            for idx in range(len(map_scores)):
                print(f"Map {idx+1}: No valid matches found")
        
        print("="*60)
        
        if best_result is None:
            return {'success': False, 'error': 'Не удалось найти положение ни на одной карте'}
        
        print(f"\nSelected: Map {best_result['map_index']+1} "
              f"({self.maps_data[best_result['map_index']]['path']}) "
              f"with score {best_score:.1f}")
        
        self.map_image = self.maps_data[best_result['map_index']]['image']
        self.map_image_gray = self.maps_data[best_result['map_index']]['gray']
        self.map_height = self.maps_data[best_result['map_index']]['height']
        self.map_width = self.maps_data[best_result['map_index']]['width']
        
        return best_result
    
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
        
        M = M / M[2, 2]
        h, w = drone_img.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0
        
        center_pt = np.float32([center_x, center_y]).reshape(-1, 1, 2)
        center_map = cv2.perspectiveTransform(center_pt, M)[0][0]
        x, y = center_map[0], center_map[1]
        
        corners_drone = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)
        
        corners_map = cv2.perspectiveTransform(corners_drone, M)
        corners_map = corners_map.reshape(4, 2)
        
        proj_w = (math.sqrt((corners_map[1][0] - corners_map[0][0])**2 + (corners_map[1][1] - corners_map[0][1])**2) +
                  math.sqrt((corners_map[2][0] - corners_map[3][0])**2 + (corners_map[2][1] - corners_map[3][1])**2)) / 2.0
        
        map_diag = math.sqrt(self.map_width**2 + self.map_height**2)
        proj_diag = math.sqrt(proj_w**2 + (proj_w * h / w)**2)
        
        scale = proj_diag / map_diag
        
        drone_altitude = config.MAP_ALTITUDE * scale if scale and scale > 0 else None
        
        inliers_count = np.sum(mask) if mask is not None else 0
        conf_sift = inliers_count / len(matches) if matches else 0
        
        in_bounds = 0 <= x < self.map_width and 0 <= y < self.map_height
        
        return {
            'success': in_bounds,
            'x': float(x),
            'y': float(y),
            'scale': float(scale),
            'altitude': float(drone_altitude) if drone_altitude else None,
            'confidence_sift': float(conf_sift),
            'matches_count': len(matches),
            'inliers_count': int(inliers_count)
        }