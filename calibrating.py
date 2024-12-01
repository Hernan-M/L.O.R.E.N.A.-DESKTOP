import cv2


def calibrate():

    def on_mouse_event(event, x, y, flags, param):
        nonlocal click_count, current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_count[current_point] += 1
            if click_count[current_point] == 1:
                colors[current_point] = (0, 255, 255)
            elif click_count[current_point] == 2:
                colors[current_point] = (0, 255, 0)             
                current_point += 1

    

    points = [(x, y) for x in [0, screen_w // 2, screen_w] for y in [0, screen_h // 2, screen_h]]
    colors = [(255, 255, 255) for _ in range(len(points))]
    click_count = [0] * len(points)
    current_point = 0
 
    
    cv2.namedWindow('Calibração', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Calibração', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Calibração', on_mouse_event)

    while current_point < len(points):
        img = np.zeros((screen_h, screen_w, 3), np.uint8)
        draw_calibration_points(img, points, colors, current_point)
        cv2.imshow('Calibração', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
