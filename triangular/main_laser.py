# main_laser.py - пример использования
from laser_locator import LaserTriangulationLocator
import cv2
import numpy as np
import config

class DroneNavigationSystem:
    def __init__(self):
        self.locator = LaserTriangulationLocator()
        self.target_position = None
        self.target_locked = False
        
        # Инициализация камеры и лазера (заглушки для реального железа)
        self.camera = None
        self.laser = None
        
    def initialize_sensors(self):
        """Инициализация сенсоров дрона"""
        # В реальности здесь будет подключение к камере и лазеру
        self.camera = cv2.VideoCapture(0)  # Пример
        self.laser = LaserRangefinder()  # Класс для работы с лазером
        
        return True
    
    def update_position(self, x, y, altitude, yaw):
        """Обновление позиции дрона от GPS/IMU"""
        self.locator.set_drone_state(x, y, altitude, yaw)
    
    def scan_for_target(self):
        """Сканирование и обнаружение цели"""
        if self.camera is None:
            return False
        
        ret, frame = self.camera.read()
        if not ret:
            return False
        
        # Ищем цель на изображении
        px, py, detected = self.locator.detect_target_in_frame(frame)
        
        if detected:
            # Получаем расстояние с лазера
            laser_distance = self.laser.get_distance()
            
            if laser_distance is not None:
                # Вычисляем позицию цели
                tx, ty, confidence = self.locator.calculate_target_position(
                    laser_distance, px, py
                )
                
                if tx is not None and confidence > 0.5:
                    self.target_position = (
                        self.locator.get_filtered_position()
                    )
                    self.target_locked = True
                    return True
        
        return False
    
    def navigate_to_target(self):
        """Навигация к цели"""
        if not self.target_locked:
            return None
        
        target_x, target_y, conf = self.target_position
        
        # Здесь логика управления дроном
        current_x, current_y, current_alt = self.locator.drone_position
        
        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.degrees(np.arctan2(dx, dy))
        
        return {
            'target_x': target_x,
            'target_y': target_y,
            'distance': distance,
            'angle': angle,
            'confidence': conf,
            'command': self._get_flight_command(angle, distance)
        }
    
    def _get_flight_command(self, angle, distance):
        """Генерация команды управления"""
        if distance < 0.5:  # Прибыли
            return "LAND"
        
        # Определение направления
        if -22.5 <= angle <= 22.5:
            return "FORWARD"
        elif 22.5 < angle <= 67.5:
            return "FORWARD_RIGHT"
        elif 67.5 < angle <= 112.5:
            return "RIGHT"
        # ... и так далее
        
        return "FORWARD"

# Заглушка для лазерного дальномера
class LaserRangefinder:
    def __init__(self):
        self.last_distance = 10.0
    
    def get_distance(self):
        """Получение расстояния от лазерного дальномера"""
        # В реальности здесь будет чтение с сенсора
        return self.last_distance