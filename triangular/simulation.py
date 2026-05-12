# visual_simulation.py
import cv2
import numpy as np
import math
import time
import config

class VisualLaserSimulation:
    def __init__(self, map_path, drone_image_path):
        self.map_img = cv2.imread(map_path)
        self.drone_img = cv2.imread(drone_image_path)
        
        if self.map_img is None:
            raise FileNotFoundError(f"Карта не найдена: {map_path}")
        if self.drone_img is None:
            raise FileNotFoundError(f"Фото дрона не найдено: {drone_image_path}")
        
        self.map_h, self.map_w = self.map_img.shape[:2]
        self.drone_h, self.drone_w = self.drone_img.shape[:2]
        
        self.drone_x = self.map_w // 2
        self.drone_y = self.map_h // 2
        self.altitude = 50.0
        self.yaw = 0.0
        
        self.targets = []
        self.trajectory = []
        self.dragging = False
        self.show_fov = True
        self.paused = False
        
    def create_window(self):
        self.window = "Laser Triangulation Demo"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1200, 800)
        cv2.setMouseCallback(self.window, self._mouse)
        
        print("\n" + "="*60)
        print("  УПРАВЛЕНИЕ СИМУЛЯЦИЕЙ")
        print("="*60)
        print("  МЫШЬ:")
        print("    ЛКМ + тянуть  — перемещать дрон")
        print("    ПКМ          — поставить цель")
        print("    Колёсико     — изменить высоту")
        print("  КЛАВИШИ:")
        print("    W/A/S/D      — движение дрона")
        print("    Q/E          — поворот дрона")
        print("    F            — поле зрения вкл/выкл")
        print("    R            — авто-движение к цели")
        print("    C            — очистить всё")
        print("    ESC          — выход")
        print("="*60 + "\n")
        
    def _mouse(self, event, x, y, flags, param):
        if y >= self.map_h:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            dist = math.hypot(x - self.drone_x, y - self.drone_y)
            if dist < 30:
                self.dragging = True
                
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drone_x = np.clip(x, 0, self.map_w)
            self.drone_y = np.clip(y, 0, self.map_h)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._add_target(x, y)
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.altitude = np.clip(self.altitude + flags * 5, 10, 200)
            
    def _add_target(self, x, y):
        color = (
            np.random.randint(150, 255),
            np.random.randint(50, 200),
            np.random.randint(50, 200)
        )
        self.targets.append({
            'x': x, 'y': y, 'color': color,
            'id': len(self.targets),
            'est_x': None, 'est_y': None,
            'error': None, 'visible': False
        })
        print(f"🎯 Цель #{len(self.targets)-1}: ({x:.0f}, {y:.0f})")
        
    def _measure(self, tx, ty):
        """Симуляция измерения лазерного дальномера"""
        dx = tx - self.drone_x
        dy = ty - self.drone_y
        ground_dist = math.hypot(dx, dy)
        
        # Угол относительно курса дрона
        angle = math.degrees(math.atan2(dx, dy)) - self.yaw
        angle = ((angle + 180) % 360) - 180
        
        # Вне поля зрения?
        if abs(angle) > config.DRONE_FOV_HORIZONTAL / 2:
            return None
        
        # Наклонная дальность с шумом
        slant = math.hypot(ground_dist, self.altitude)
        slant += np.random.normal(0, slant * 0.02)
        
        # Пиксельные координаты на матрице камеры
        px = (angle / (config.DRONE_FOV_HORIZONTAL / 2)) * (self.drone_w / 2)
        
        v_angle = math.degrees(math.atan2(self.altitude, ground_dist))
        py = (1 - v_angle / (config.DRONE_FOV_VERTICAL / 2)) * (self.drone_h / 2)
        
        return slant, px, py
        
    def _triangulate(self, slant, px, py):
        """Триангуляция: вычисление позиции цели"""
        angle_h = math.radians((px / (self.drone_w / 2)) * (config.DRONE_FOV_HORIZONTAL / 2))
        total_angle = angle_h + math.radians(self.yaw)
        
        horiz = math.sqrt(max(0, slant**2 - self.altitude**2))
        
        est_x = self.drone_x + horiz * math.sin(total_angle)
        est_y = self.drone_y + horiz * math.cos(total_angle)
        
        return est_x, est_y
        
    def _process_targets(self):
        """Обработка всех целей"""
        for t in self.targets:
            result = self._measure(t['x'], t['y'])
            
            if result is None:
                t['visible'] = False
                continue
                
            slant, px, py = result
            t['est_x'], t['est_y'] = self._triangulate(slant, px, py)
            t['error'] = math.hypot(t['est_x'] - t['x'], t['est_y'] - t['y'])
            t['visible'] = True
            t['slant'] = slant
            t['px'], t['py'] = px, py
            
    def _draw_fov(self, img):
        """Отрисовка поля зрения"""
        fov_half = math.radians(config.DRONE_FOV_HORIZONTAL / 2)
        length = self.altitude * 3
        
        a1 = math.radians(self.yaw) - fov_half
        a2 = math.radians(self.yaw) + fov_half
        
        pts = np.array([
            [int(self.drone_x), int(self.drone_y)],
            [int(self.drone_x + length * math.sin(a1)),
             int(self.drone_y + length * math.cos(a1))],
            [int(self.drone_x + length * math.sin(a2)),
             int(self.drone_y + length * math.cos(a2))]
        ], np.int32)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (255, 200, 100))
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [pts], True, (255, 180, 50), 2)
        
    def _draw_targets(self, img):
        """Отрисовка целей и измерений"""
        for t in self.targets:
            # Цель
            cv2.circle(img, (int(t['x']), int(t['y'])), 10, t['color'], -1)
            cv2.circle(img, (int(t['x']), int(t['y'])), 12, (255, 255, 255), 2)
            
            # Подпись
            label = f"T{t['id']}"
            cv2.putText(img, label, (int(t['x']) + 15, int(t['y']) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, t['color'], 2)
            
            # Не видна — серая
            if not t['visible']:
                cv2.circle(img, (int(t['x']), int(t['y'])), 12, (100, 100, 100), 2)
                cv2.putText(img, "(no LOS)", (int(t['x']) + 15, int(t['y']) + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                continue
                
            # Оценка позиции (жёлтый ромб)
            est_x, est_y = int(t['est_x']), int(t['est_y'])
            size = 8
            pts = np.array([
                [est_x, est_y - size],
                [est_x + size, est_y],
                [est_x, est_y + size],
                [est_x - size, est_y]
            ], np.int32)
            cv2.fillPoly(img, [pts], (0, 255, 255))
            cv2.polylines(img, [pts], True, (255, 255, 255), 1)
            
            # Линия от дрона (лазерный луч)
            cv2.line(img, (int(self.drone_x), int(self.drone_y)),
                    (int(t['x']), int(t['y'])), (0, 255, 100), 1)
            
            # Линия ошибки
            cv2.line(img, (int(t['x']), int(t['y'])),
                    (est_x, est_y), (0, 0, 255), 2)
            
            # Величина ошибки
            mid_x = (t['x'] + est_x) / 2
            mid_y = (t['y'] + est_y) / 2
            cv2.putText(img, f"{t['error']:.1f}m",
                       (int(mid_x) + 5, int(mid_y)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                       
    def _draw_drone(self, img):
        """Отрисовка дрона"""
        # Тень
        cv2.circle(img, (int(self.drone_x), int(self.drone_y)), 18, (0, 0, 0), -1)
        cv2.circle(img, (int(self.drone_x), int(self.drone_y)), 16, (255, 120, 0), -1)
        cv2.circle(img, (int(self.drone_x), int(self.drone_y)), 16, (255, 255, 255), 2)
        
        # Направление
        a = math.radians(self.yaw)
        ex = int(self.drone_x + 25 * math.sin(a))
        ey = int(self.drone_y + 25 * math.cos(a))
        cv2.arrowedLine(img, (int(self.drone_x), int(self.drone_y)),
                       (ex, ey), (0, 0, 255), 3, tipLength=0.4)
        
    def _draw_info_panel(self, img):
        """Информационная панель"""
        h = 120
        panel = np.zeros((h, self.map_w, 3), dtype=np.uint8)
        
        # Заголовок
        cv2.putText(panel, "LASER TRIANGULATION", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
        
        # Данные дрона
        info = f"POS: ({self.drone_x:.0f}, {self.drone_y:.0f})  |  ALT: {self.altitude:.0f}m  |  HDG: {self.yaw:.0f}°"
        cv2.putText(panel, info, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Статистика
        visible = sum(1 for t in self.targets if t['visible'])
        total = len(self.targets)
        cv2.putText(panel, f"TARGETS: {visible}/{total} visible", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if total > 0:
            errors = [t['error'] for t in self.targets if t['visible']]
            if errors:
                avg = np.mean(errors)
                cv2.putText(panel, f"AVG ERROR: {avg:.1f}m", (300, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        
        # Индикатор масштаба
        cv2.putText(panel, f"SCALE: 1px ≈ {self.altitude/20:.1f}m", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Легенда
        legend_x = self.map_w - 300
        cv2.circle(panel, (legend_x, 25), 5, (0, 255, 0), -1)
        cv2.putText(panel, "Target", (legend_x + 15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.drawMarker(panel, (legend_x, 55), (0, 255, 255), cv2.MARKER_DIAMOND, 10, 1)
        cv2.putText(panel, "Estimated", (legend_x + 15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.line(panel, (legend_x - 5, 85), (legend_x + 15, 85), (0, 0, 255), 2)
        cv2.putText(panel, "Error", (legend_x + 25, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return np.vstack([img, panel])
        
    def run(self):
        self.create_window()
        
        # Пресеты целей
        self._add_target(int(self.map_w * 0.7), int(self.map_h * 0.3))
        self._add_target(int(self.map_w * 0.25), int(self.map_h * 0.7))
        self._add_target(int(self.map_w * 0.6), int(self.map_h * 0.8))
        
        clock = cv2.getTickFrequency()
        last = cv2.getTickCount()
        auto = False
        
        print("🚁 СИМУЛЯЦИЯ ЗАПУЩЕНА\n")
        
        while True:
            dt = (cv2.getTickCount() - last) / clock
            last = cv2.getTickCount()
            
            # Обработка целей
            self._process_targets()
            
            # Авто-движение к ближайшей видимой цели
            if auto:
                visible = [t for t in self.targets if t['visible']]
                if visible:
                    target = min(visible, key=lambda t: math.hypot(
                        t['x'] - self.drone_x, t['y'] - self.drone_y))
                    dx = target['x'] - self.drone_x
                    dy = target['y'] - self.drone_y
                    dist = math.hypot(dx, dy)
                    
                    if dist > 10:
                        step = min(dist, 100 * dt)
                        self.drone_x += (dx / dist) * step
                        self.drone_y += (dy / dist) * step
                        self.trajectory.append((self.drone_x, self.drone_y))
                    else:
                        print(f"✅ Достигнута цель #{target['id']}")
                        self.targets.remove(target)
            
            # Отрисовка
            display = self.map_img.copy()
            self._draw_fov(display)
            
            # Траектория
            for i in range(1, len(self.trajectory)):
                cv2.line(display,
                        (int(self.trajectory[i-1][0]), int(self.trajectory[i-1][1])),
                        (int(self.trajectory[i][0]), int(self.trajectory[i][1])),
                        (255, 180, 100), 2)
            
            self._draw_targets(display)
            self._draw_drone(display)
            display = self._draw_info_panel(display)
            
            # FPS
            fps = 1.0 / max(dt, 0.001)
            cv2.putText(display, f"FPS: {fps:.0f}", (self.map_w - 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            cv2.imshow(self.window, display)
            
            # Клавиши
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('f'):
                self.show_fov = not self.show_fov
            elif key == ord('r'):
                auto = not auto
                print(f"⏩ Авто-движение: {'ON' if auto else 'OFF'}")
            elif key == ord('c'):
                self.targets.clear()
                self.trajectory.clear()
                print("🗑️  Очищено")
            elif key == ord('w'):
                self.drone_y -= 5
            elif key == ord('s'):
                self.drone_y += 5
            elif key == ord('a'):
                self.drone_x -= 5
            elif key == ord('d'):
                self.drone_x += 5
            elif key == ord('q'):
                self.yaw = (self.yaw - 3) % 360
            elif key == ord('e'):
                self.yaw = (self.yaw + 3) % 360
                
        cv2.destroyAllWindows()
        print("\n👋 Симуляция завершена")


def main():
    sim = VisualLaserSimulation("./photos/map.png", "./photos/dron.png")
    sim.run()


if __name__ == "__main__":
    main()