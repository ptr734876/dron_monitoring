# main.py
import cv2
from locator import DroneLocator


def draw_result(map_image, x, y, altitude, scale):
    display = map_image.copy()
    h, w = display.shape[:2]
    
    cross_size = max(20, min(w, h) // 40)
    thickness = max(2, min(w, h) // 200)
    color = (0, 0, 255)
    
    cv2.line(display, (int(x - cross_size), int(y)), (int(x + cross_size), int(y)), color, thickness)
    cv2.line(display, (int(x), int(y - cross_size)), (int(x), int(y + cross_size)), color, thickness)
    cv2.circle(display, (int(x), int(y)), cross_size, color, thickness)
    
    text = f"H: {altitude:.3f} m | Scale: {scale:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
    cv2.rectangle(display, (int(x) + 15, int(y) - th - 10), 
                 (int(x) + 25 + tw, int(y) - 5), (255, 255, 255), -1)
    cv2.putText(display, text, (int(x) + 20, int(y) - 10), font, 0.6, (0, 0, 0), 1)
    
    if w > 1200:
        s = 1200 / w
        display = cv2.resize(display, (int(w * s), int(h * s)))
    
    cv2.imshow("Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("result.png", display)


def main():
    locator = DroneLocator()
    locator.load_map("dron_lab/map.png")
    result = locator.locate("dron_lab/dron1.png")
    
    if result['success']:
        print(f"X: {result['x']:.1f}, Y: {result['y']:.1f}")
        print(f"Altitude: {result['altitude']:.3f} m")
        print(f"Scale: {result['scale']:.3f}")
        print(f"SIFT conf: {result['confidence_sift']:.3f}")
        print(f"NCC conf: {result['confidence_ncc']:.3f}")
        print(f"Matches: {result['matches_count']} ({result['inliers_count']} inliers)")
        draw_result(locator.map_image, result['x'], result['y'], result['altitude'], result['scale'])
    else:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()