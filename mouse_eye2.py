import math
import cv2
import mediapipe as mp
from vidstab import VidStab
from pyvirtualcam import PixelFormat
import pyvirtualcam
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Índices dos olhos
RIGHT_EYE = [362, 263, 374, 386, 373]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Índices das íris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

# Variáveis globais
gb_zoom = 5
eye = None

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

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


def play():
    mouse_tracking();


def draw_calibration_points(img, points, colors, current_point):
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 20, colors[i], -1)
        if current_point < len(points):
            cv2.circle(img, points[current_point], 20, (255, 0, 0), 2)

def get_point(x, y):
    list_points = []




def frame_manipulate(img):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 1)
        results = face_detection.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        coordinates = None
        zoom_transition = None
        if results.detections:
            for detection in results.detections:
                height, width, _ = img.shape
                left_eye = detection.location_data.relative_keypoints[1]
                right_ear = detection.location_data.relative_keypoints[4]
                left_ear = detection.location_data.relative_keypoints[5]

                right_ear_x = int(right_ear.x * width)
                left_ear_x = int(left_ear.x * width)

                center_x = int(left_eye.x * width)
                center_y = int(left_eye.y * height)
                
                # coordinates = [center_x, center_y]
                # if (left_ear_x - right_ear_x) < 120:
                #     zoom_transition = 'transition'
        # img = zoom_at(img, coord=coordinates, zoom_type=zoom_transition)
        global eye
        eye = img
    return eye


def mouse_tracking():

    global frame_h, frame_w, percent_iris_x, percent_iris_y

    #debug
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_h)
    cap.set(cv2.CAP_PROP_FPS, 120)

    while True:
        success, img = cap.read()
        if not success:
            continue
        img = frame_manipulate(img)
        output = face_mesh.process(img)
        landmark_points = output.multi_face_landmarks
        img_h, img_w = img.shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        frame_h, frame_w, _ = img.shape

        if landmark_points:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in landmark_points[0].landmark])
            
            #debug
            cv2.polylines(img, [mesh_points[LEFT_EYE]], True, (0, 255, 0), 1, cv2.LINE_AA)

            landmarks = landmark_points[0].landmark
            right_iris_landmark = [landmarks[RIGHT_IRIS[0]], landmarks[RIGHT_IRIS[2]]]

            absolute_iris_x, absolute_iris_y, absolute_iris_z = ((right_iris_landmark[0].x + right_iris_landmark[1].x) / 2, (right_iris_landmark[0].y + right_iris_landmark[1].y) / 2, (right_iris_landmark[0].z + right_iris_landmark[1].z) / 2)
            
            

            #esse aq so mostra msm
            iris_center_x = int(absolute_iris_x * frame_w)
            iris_center_y = int(absolute_iris_y * frame_h)
            #debug
            cv2.circle(img, (iris_center_x, iris_center_y), 3, (0, 0, 255))
            
            #tamain dos zoi
            eye_width = (mesh_points[263][0] - mesh_points[362][0]) 
            eye_height = (mesh_points[374][1] - mesh_points[386][1])                                 

            #valor do olho 100% atualizdo eh ruim de aturar
            calc_centr_x = ((mesh_points[474][0] + mesh_points[476][0])/2) - mesh_points[362][0]   
            calc_centr_y = mesh_points[474][1] - mesh_points[386][1] 
            # ratio_x =  screen_w / relative_iris_x
            # ratio_y = screen_h / relative_iris_y

            # olho_x = (absolute_iris_x  * ratio_x) 
            # olho_y = (absolute_iris_y * ratio_y)

            percent_iris_x = ((calc_centr_x * 100) / eye_width) 
            percent_iris_y = ((calc_centr_y * 100) / eye_height)
            # cv2.putText(img, f'x: {calc_centr_x}, y: {calc_centr_y} -> iris' ,(10, 100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255 )
            # cv2.putText(img, f'x: {eye_width}, y: {eye_height} -> eyes size' ,(10, 150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255 )
            cv2.putText(img, f'x: {percent_iris_x}%, y: {percent_iris_y}% -> iris percent' ,(10, 200),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255 )


            if absolute_iris_z > 0.0001:
                cv2.putText(img, 'ta longe fi de rapariga' ,(500, 500),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255 )

            if( percent_iris_x < 100 and  percent_iris_y < 100): 
                point_x = convert_percent(percent_iris_x, screen_w, 3)
                point_y = convert_percent(percent_iris_y, screen_h, 5)
                cv2.putText(img, f'x: {point_x}, y: a -> pos iris' ,(10, 150),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255 )
                cv2.circle(img, (point_x, point_y), 20, (0, 255, 0), 10)
                
            # else : 
            #     percent = ((relative_iris_x * 100) / eye_width)

            # screen_pos_x = (percent_iris_x * screen_w)/100
            # screen_pos_y = (percent_iris_y * screen_h)/100
            # cv2.putText(img, f'{calc_centr_x}, {calc_centr_y}',(10, 100),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            # cv2.putText(img, f'extremidade: {eye_width}, extremidade da iris: {right_iris_landmark[1].x}',(10, 400),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            #cv2.putText(img,str((landmarks[RIGHT_EYE[2]].y ))+  ' eye height',(10,120),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            # cv2.putText(img,'(absolute, relative) : '+ '(' + str(olho_x) + ',' + str(relative_iris_x) + ')',(10,140),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            # cv2.putText(img,'(absolute, relative) : '+ '(' + str(olho_y) + ',' + str(relative_iris_y) + ')',(10,160),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            
            
            #  cv2.putText(img,iris_center_y,(10,200),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
            for index in RIGHT_EYE:
                cv2.circle(img, mesh_points[index], 3, (0, 255, 255))
                cv2.putText(img, f'{index}' , mesh_points[index] ,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
           
           
            #debug
            for index in RIGHT_IRIS:
                cv2.circle(img, mesh_points[index], 3, (0, 255, 255))
                # cv2.putText(img, f'{index}' , mesh_points[index] ,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)

            #debug
            left_eye_landmarks = [landmarks[145], landmarks[159]]
            for landmark in left_eye_landmarks:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(img, (x, y), 3, (0, 0, 255))

            #pyautogui.moveTo(screen_pos_x, screen_pos_y)
        
        cv2.imshow('Mouse eye', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



def convert_percent(percent, screen, sensibility): 
        distance_from_50 = (percent - 50) / 50
        base_value = (percent / 100 * screen)
        k = sensibility
        converted_value = base_value * (1 + k * distance_from_50)

        #suavização (falta [smooth_value = 5] <- no param da função) 
            # filtered_values = np.convolve(converted_value, np.ones(smooth_value) / smooth_value, mode='valid')
            # final_values = np.clip(filtered_values, 0, screen)
            # return final_values
            
        if (0 >= converted_value):
             return 0
        elif (converted_value > screen): 
             return screen
        else: 
            return int(converted_value)


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

    

    cv2.destroyAllWindows()



# Função principal para criar a interface gráfica
def main():
    root = tk.Tk()
    root.title("Mouse eye")

    label = tk.Label(root, text="Escolha uma opção:", font=("Helvetica", 16))
    label.pack(pady=20)

    play_button = tk.Button(root, text="Play", command=play, font=("Helvetica", 14))
    play_button.pack(pady=10)

    calibrate_button = tk.Button(root, text="Calibrate", command=calibrate, font=("Helvetica", 14))
    calibrate_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()