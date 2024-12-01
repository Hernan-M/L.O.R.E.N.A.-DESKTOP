import cv2


def zoom_at(image, coord=None, zoom_type=None):
    global gb_zoom
    if zoom_type == 'transition' and gb_zoom < 5:
        gb_zoom += 0.1
    if gb_zoom != 5 and zoom_type is None:
        gb_zoom -= 0.1
    zoom = gb_zoom
    cy, cx = [i / 2 for i in image.shape[:-1]] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def draw_calibration_points(img, points, colors, current_point):
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 20, colors[i], -1)
        if current_point < len(points):
            cv2.circle(img, points[current_point], 20, (255, 0, 0), 2)

def get_point(x, y):
    list_points = []