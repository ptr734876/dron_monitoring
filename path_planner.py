# path_planner.py
import numpy as np
import math


class PathPlanner:
    
    def __init__(self, update_distance=5.0):
        self.update_distance = update_distance
        self.current_x = None
        self.current_y = None
        self.target_x = None
        self.target_y = None
        self.waypoints = []
        self.last_update_x = None
        self.last_update_y = None
    
    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
    
    def set_current_position(self, x, y):
        self.current_x = x
        self.current_y = y
    
    def distance_to_target(self):
        if self.current_x is None or self.target_x is None:
            return float('inf')
        return math.sqrt((self.target_x - self.current_x)**2 + (self.target_y - self.current_y)**2)
    
    def angle_to_target(self):
        if self.current_x is None or self.target_x is None:
            return 0.0
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        return math.degrees(math.atan2(dy, dx))
    
    def need_replan(self, current_x, current_y):
        if self.last_update_x is None or self.last_update_y is None:
            return True
        
        dist = math.sqrt((current_x - self.last_update_x)**2 + (current_y - self.last_update_y)**2)
        return dist >= self.update_distance
    
    def replan(self, current_x, current_y):
        self.last_update_x = current_x
        self.last_update_y = current_y
        self.current_x = current_x
        self.current_y = current_y
        
        if self.target_x is None:
            return None
        
        angle = self.angle_to_target()
        distance = self.distance_to_target()
        
        return {
            'target_x': self.target_x,
            'target_y': self.target_y,
            'current_x': current_x,
            'current_y': current_y,
            'angle': angle,
            'distance': distance,
            'command': self._get_flight_command(angle, distance)
        }
    
    def _get_flight_command(self, angle, distance):
        if distance < 5.0:
            return "HOVER"
        
        if -22.5 <= angle <= 22.5:
            return "MOVE_RIGHT"
        elif 22.5 < angle <= 67.5:
            return "MOVE_UP_RIGHT"
        elif 67.5 < angle <= 112.5:
            return "MOVE_UP"
        elif 112.5 < angle <= 157.5:
            return "MOVE_UP_LEFT"
        elif angle > 157.5 or angle <= -157.5:
            return "MOVE_LEFT"
        elif -157.5 < angle <= -112.5:
            return "MOVE_DOWN_LEFT"
        elif -112.5 < angle <= -67.5:
            return "MOVE_DOWN"
        elif -67.5 < angle <= -22.5:
            return "MOVE_DOWN_RIGHT"
        
        return "MOVE_FORWARD"