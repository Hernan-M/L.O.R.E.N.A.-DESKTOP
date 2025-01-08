import cv2
from collections import defaultdict
from gaze_tracking import GazeTracking
import pyautogui
import numpy as np
import tkinter as tk
from tkinter import messagebox

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

point_pos = []
screen_w, screen_h = pyautogui.size()

def play():
    mouse_tracking();

def mouse_tracking():
   while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    # text = ""

    # if gaze.is_blinking():
    #     text = "Blinking"
    # elif gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    ratio_y = gaze.vertical_ratio()
    ratio_x = gaze.horizontal_ratio()
    cv2.putText(frame, "y:  " + str(ratio_y), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "x: " + str(ratio_x), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
    webcam.release()
    cv2.destroyAllWindows()


def calibrate():
    
    points = [(x, y) for x in [0, screen_w // 2, screen_w] for y in [0, screen_h // 2, screen_h]]
    colors = [(255, 255, 255) for _ in range(len(points))]
    click_data = {i: [] for i in range(len(points))}
    click_count = [0] * len(points)
    current_point = 0

    def draw_calibration_points(img, points, colors, current_point):
        """
        Desenha os pontos de calibração na tela.
        """
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 20, colors[i], -1)
        if current_point < len(points):
            cv2.circle(img, points[current_point], 20, (255, 0, 0), 2)

    def on_mouse_event(event, x, y, flags, param):
        nonlocal current_point
        if event == cv2.EVENT_LBUTTONDOWN:  # Clique esquerdo do mouse
            click_data[current_point].append((x, y))  # Salvar coordenadas do clique
            num_clicks = len(click_data[current_point])
            
            if num_clicks == 1:
                colors[current_point] = (0, 255, 255)  # Amarelo após o 1º clique
            elif num_clicks == 2:
                colors[current_point] = (0, 255, 0)  # Verde após o 2º clique
            elif num_clicks == 3:
                current_point += 1  # Avança para o próximo ponto

    # Configurar a janela de calibração
    cv2.namedWindow('Calibração', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Calibração', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Calibração', on_mouse_event)

    try:
        while current_point < len(points):
            img = np.zeros((screen_h, screen_w, 3), np.uint8)
            draw_calibration_points(img, points, colors, current_point)
            cv2.imshow('Calibração', img)
            if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'Esc' para sair
                break
    finally:
        # Certificar-se de liberar a janela ao final
        cv2.destroyAllWindows()

    # Calcular a média dos cliques para cada ponto
    avg_click_data = {}
    for i, clicks in click_data.items():
        if len(clicks) == 3:  # Apenas processar pontos com 3 cliques
            avg_x = sum(x for x, y in clicks) / 3
            avg_y = sum(y for x, y in clicks) / 3
            avg_click_data[i] = (avg_x, avg_y)

    print(avg_click_data)
    return avg_click_data



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