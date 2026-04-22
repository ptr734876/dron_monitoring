# main_navigation.py
from locator import DroneLocator
from set_target import TargetSetter
import cv2
import math
import os
import sys
import config


def distance_pixels(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def main():
    map_num = ""
    dron_num = ""
    
    if len(sys.argv) > 1:
        if sys.argv[1] != "-1":
            map_num = sys.argv[1]
    if len(sys.argv) > 2:
        if sys.argv[2] != "-1":
            dron_num = sys.argv[2]
    
    map_path = f"./photos/map{map_num}.png"
    dron_path = f"./photos/dron{dron_num}.png"
    
    locator = DroneLocator()
    locator.load_map(map_path)
    
    result = locator.locate(dron_path)
    
    if not result['success']:
        print(f"Location failed: {result['error']}")
        return
    
    current_x = result['x']
    current_y = result['y']
    altitude = result['altitude']
    scale = result['scale']
    
    target_setter = TargetSetter()
    
    print("Set target on map (click + Enter)")
    target_x, target_y = target_setter.set_target_from_click(map_path)
    if target_x is not None:
        target_setter.save_target("target.txt")
        print(f"Target set: {target_x:.1f}, {target_y:.1f}")
    else:
        print("Target not set")
        return
    
    dist_pixels = distance_pixels(current_x, current_y, target_x, target_y)
    
    print(f"\nDrone position: ({current_x:.1f}, {current_y:.1f})")
    print(f"Target position: ({target_x:.1f}, {target_y:.1f})")
    print(f"Map altitude: {config.MAP_ALTITUDE}m")
    print(f"Drone altitude: {altitude:.2f}m")
    print(f"Scale factor: {scale:.3f}")
    
    if hasattr(config, 'MAP_GSD') and config.MAP_GSD is not None:
        dist_meters = dist_pixels * config.MAP_GSD
        print(f"Distance: {dist_meters:.3f}m")
    else:
        print(f"Distance in pixels: {dist_pixels:.1f}")
    
    display = locator.map_image.copy()
    
    cv2.line(display, (int(current_x), int(current_y)), 
            (int(target_x), int(target_y)), (0, 0, 255), 2)
    
    cv2.circle(display, (int(current_x), int(current_y)), 8, (255, 0, 0), -1)
    cv2.circle(display, (int(current_x), int(current_y)), 14, (255, 0, 0), 2)
    
    cv2.circle(display, (int(target_x), int(target_y)), 8, (0, 255, 0), -1)
    cv2.circle(display, (int(target_x), int(target_y)), 14, (0, 255, 0), 2)
    
    h, w = display.shape[:2]
    if w > 1200:
        s = 1200 / w
        display = cv2.resize(display, (int(w * s), int(h * s)))
    
    cv2.imshow("Route", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("route.png", display)
    print("\nRoute saved to route.png")


if __name__ == "__main__":
    main()