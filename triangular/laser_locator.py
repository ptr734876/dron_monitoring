# laser_locator.py - новый модуль для локации через триангуляцию
import numpy as np
import math
import cv2
import config

class LaserTriangulationLocator:
    """
    Система локации на основе лазерного дальномера и камеры
    """
    
    def __init__(self):
        self.drone_position = None  # (x, y, altitude)
        self.drone_yaw = 0.0  # угол рыскания дрона
        
        # Калибровочные параметры камеры
        self.fov_h = config.DRONE_FOV_HORIZONTAL
        self.fov_v = config.DRONE_FOV_VERTICAL
        self.res_w = config.CAMERA_RESOLUTION_WIDTH
        self.res_h = config.CAMERA_RESOLUTION_HEIGHT
        
        # Параметры лазерного дальномера
        self.max_range = config.LASER_MAX_RANGE
        
        # Хранилище измерений для усреднения
        self.measurements_history = []
        self.max_history = 10
    
    def set_drone_state(self, x, y, altitude, yaw=0.0):
        """
        Установка текущего состояния дрона
        x, y - координаты в локальной системе (например, от takeoff)
        altitude - высота в метрах
        yaw - угол рыскания в градусах (0 - на север, 90 - на восток)
        """
        self.drone_position = (x, y, altitude)
        self.drone_yaw = yaw
    
    def calculate_target_position(self, laser_distance, pixel_x, pixel_y):
        """
        Расчёт позиции цели по данным лазерного дальномера и камеры
        
        Parameters:
        - laser_distance: расстояние до цели в метрах
        - pixel_x, pixel_y: координаты цели на изображении (центр = 0,0)
        
        Returns:
        - target_x, target_y: координаты цели в локальной системе
        - confidence: уверенность в измерении (0-1)
        """
        if self.drone_position is None:
            return None, None, 0.0
        
        drone_x, drone_y, drone_alt = self.drone_position
        
        # Проверка валидности измерения
        if laser_distance <= 0 or laser_distance > self.max_range:
            return None, None, 0.0
        
        # 1. Находим углы отклонения от оптической оси камеры
        angle_x = (pixel_x / self.res_w) * self.fov_h  # горизонтальный угол
        angle_y = -(pixel_y / self.res_h) * self.fov_v  # вертикальный угол (инвертируем)
        
        # 2. Корректируем с учётом рыскания дрона
        total_angle_x = angle_x + self.drone_yaw
        
        # 3. Находим горизонтальную проекцию расстояния
        horizontal_distance = math.sqrt(
            laser_distance**2 - drone_alt**2
        ) if laser_distance > drone_alt else 0
        
        # 4. Вычисляем смещение цели относительно дрона
        # Предполагаем, что лазер направлен в ту же точку, что и центр камеры
        target_offset_x = horizontal_distance * math.sin(math.radians(total_angle_x))
        target_offset_y = horizontal_distance * math.cos(math.radians(total_angle_x))
        
        # 5. Абсолютные координаты цели
        target_x = drone_x + target_offset_x
        target_y = drone_y + target_offset_y
        
        # 6. Оценка уверенности
        confidence = self._calculate_confidence(laser_distance, angle_x, angle_y)
        
        # Сохраняем измерение
        self._add_measurement(target_x, target_y, confidence)
        
        return target_x, target_y, confidence
    
    def get_filtered_position(self):
        """
        Возвращает отфильтрованную позицию на основе истории измерений
        Использует взвешенное скользящее среднее
        """
        if not self.measurements_history:
            return None, None, 0.0
        
        # Взвешенное среднее (последние измерения имеют больший вес)
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        weighted_conf = 0
        
        for i, (x, y, conf) in enumerate(self.measurements_history):
            weight = (i + 1) / len(self.measurements_history)  # линейный вес
            weighted_x += x * weight * conf
            weighted_y += y * weight * conf
            weighted_conf += conf * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_x = weighted_x / total_weight
            avg_y = weighted_y / total_weight
            avg_conf = weighted_conf / len(self.measurements_history)
            return avg_x, avg_y, avg_conf
        
        return None, None, 0.0
    
    def _calculate_confidence(self, distance, angle_x, angle_y):
        """
        Оценка уверенности измерения на основе:
        - расстояния (ближе = лучше)
        - угла отклонения от центра (меньше = лучше)
        """
        # Уверенность падает с расстоянием
        distance_factor = max(0, 1 - distance / self.max_range)
        
        # Уверенность падает при больших углах (на краях кадра)
        max_angle = max(self.fov_h, self.fov_v) / 2
        angle_factor_x = 1 - abs(angle_x) / max_angle
        angle_factor_y = 1 - abs(angle_y) / max_angle
        angle_factor = min(angle_factor_x, angle_factor_y)
        
        # Комбинированная уверенность
        confidence = 0.5 * distance_factor + 0.5 * angle_factor
        return max(0, min(1, confidence))
    
    def _add_measurement(self, x, y, confidence):
        """Добавление измерения в историю"""
        self.measurements_history.append((x, y, confidence))
        if len(self.measurements_history) > self.max_history:
            self.measurements_history.pop(0)
    
    def detect_target_in_frame(self, frame):
        """
        Простое обнаружение цели на изображении
        Возвращает координаты центра цели в пикселях
        """
        # Здесь можно использовать детектор объектов, YOLO, или цветовой фильтр
        # Для примера - простой детектор красного цвета
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Диапазон красного цвета
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Находим контуры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Берём самый большой контур
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Центрируем координаты
                px = cx - self.res_w // 2
                py = cy - self.res_h // 2
                
                return px, py, True
        
        return 0, 0, False
    
    def triangulate_with_multiple_measurements(self, measurements):
        """
        Триангуляция по нескольким измерениям с разных позиций
        
        measurements: список кортежей (drone_x, drone_y, drone_alt, laser_dist, pixel_x, pixel_y)
        """
        if len(measurements) < 2:
            return None, None
        
        # Сохраняем состояние
        original_state = self.drone_position, self.drone_yaw
        
        positions = []
        for drone_x, drone_y, drone_alt, laser_dist, pixel_x, pixel_y in measurements:
            self.set_drone_state(drone_x, drone_y, drone_alt)
            tx, ty, conf = self.calculate_target_position(laser_dist, pixel_x, pixel_y)
            if tx is not None:
                positions.append((tx, ty, conf))
        
        # Восстанавливаем состояние
        self.drone_position, self.drone_yaw = original_state
        
        if positions:
            # Простое усреднение
            avg_x = np.mean([p[0] for p in positions])
            avg_y = np.mean([p[1] for p in positions])
            return avg_x, avg_y
        
        return None, None