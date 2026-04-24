# set_target.py
import cv2
import numpy as np

class TargetSetter:
    
    def __init__(self):
        self.target_x = None
        self.target_y = None
        self.map_image = None
        self.map_path = None
    
    def set_target_from_click(self, map_path):
        self.map_path = map_path
        self.map_image = cv2.imread(map_path)
        
        if self.map_image is None:
            raise FileNotFoundError(f"Не удалось загрузить карту: {map_path}")
        
        h, w = self.map_image.shape[:2]
        
        if w > 900:
            scale = 900 / w
            self.display_image = cv2.resize(self.map_image, (int(w * scale), int(h * scale)))
            self.scale_factor = scale
        else:
            self.display_image = self.map_image.copy()
            self.scale_factor = 1.0
        
        self.click_x = None
        self.click_y = None
        
        cv2.namedWindow("Click to set target")
        cv2.setMouseCallback("Click to set target", self._mouse_callback)
        
        while True:
            display_copy = self.display_image.copy()
            
            if self.click_x is not None and self.click_y is not None:
                cv2.drawMarker(display_copy, (self.click_x, self.click_y), 
                              (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            
            cv2.imshow("Click to set target", display_copy)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.click_x is not None:
                self.target_x = self.click_x / self.scale_factor
                self.target_y = self.click_y / self.scale_factor
                break
            elif key == 27:
                break
        
        cv2.destroyAllWindows()
        
        return self.target_x, self.target_y
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_x = x
            self.click_y = y
    
    def get_target(self):
        return self.target_x, self.target_y
    
    def save_target(self, path="target.txt"):
        if self.target_x is not None and self.target_y is not None:
            with open(path, 'w') as f:
                f.write(f"{self.target_x},{self.target_y}")
    
    def load_target(self, path="target.txt"):
        try:
            with open(path, 'r') as f:
                data = f.read().strip().split(',')
                self.target_x = float(data[0])
                self.target_y = float(data[1])
                return self.target_x, self.target_y
        except:
            return None, None