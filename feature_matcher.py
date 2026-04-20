# feature_matcher.py
import cv2
import numpy as np
import config


class FeatureMatcher:
    
    def __init__(self):
        self.detector = cv2.SIFT_create(
            nfeatures=config.SIFT_N_FEATURES,
            contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=config.SIFT_EDGE_THRESHOLD
        )
        
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_and_compute(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.detector.detectAndCompute(image, None)
    
    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []
        if len(des1) < 2 or len(des2) < 2:
            return []
        
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < config.MATCH_RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def get_matched_points(self, kp1, kp2, matches):
        if not matches:
            return None, None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        return src_pts, dst_pts