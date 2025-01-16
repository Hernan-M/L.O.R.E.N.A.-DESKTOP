import cv2
from gaze_tracking import GazeTracking
import pyautogui 
import numpy as np
import tkinter as tk
from tkinter import Tk, PhotoImage
from tkinter.ttk import Frame, Label, Button, Style
import threading
import time
import ctypes
from ui import menu
import json
import os
from threading import Lock
from statistics import median
from collections import deque
import math


frame_lock = Lock()
root = Tk()


gaze = GazeTracking()

webcam = cv2.VideoCapture(1)

alert_asset = cv2.imread("ui/assets/imgs/alert.png")
calibration_path = "settings/calibration"
os.makedirs(calibration_path, exist_ok=True)

screen_w, screen_h = pyautogui.size()

ratio_x, ratio_y = None, None
frame, ret = None, False
avg_click_data = {}

# flags states
is_on_calibrate = False
tracking_thread_running = False
loaded_calibration_file = False
its_recognizing = False
tracking_thread = None

# Calibration
x_axis_limits, y_axis_limits, horizontal_scale, vertical_scale = None, None, None, None

def play():
    global tracking_thread_running
    if avg_click_data:
        try:
            alert('Para sair do rastreio ocular aperte ESC')
            start_mouse_tracking()
            # face_detect()
        except Exception as e:
            error(f"Erro de rastreio: {e}")
            cv2.destroyAllWindows()
            main();
    else:
        alert("Calibre o sistema ao menos uma vez antes de iniciar o rastreio.")


def mouse_tracking():
    global ratio_x, ratio_y, frame, ret, is_on_calibrate, tracking_thread_running, avg_click_data
    
    last_three_x = deque(maxlen=3)
    last_three_y = deque(maxlen=3)

    try:
        timer = 0   
        counter = 0
        detection_started = False
        while tracking_thread_running:
            
            ret, frame = webcam.read()
            if not ret or frame is None:
                error("Falha ao capturar imagem da webcam")
                break

            frame = cv2.flip(frame, 1)  
            gaze.refresh(frame)
            ratio_x = gaze.horizontal_ratio()
            ratio_y = gaze.vertical_ratio()
            if ratio_x is not None and ratio_y is not None:
                ratio_x = ratio_x + 0.010000000 
                ratio_y = ratio_y + 0.010000000 
            click = gaze.is_left_blinking()

            if not is_on_calibrate and click:
                img = np.ones((screen_h, screen_w, 3), np.uint8) * 255
                cv2.namedWindow('Rastreio em execucao', cv2.WINDOW_NORMAL)
                cv2.putText(img, "Rastreio em execucao, pressione ESC para finalizar", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 1.5, (147, 58, 31), 2)
                cv2.setWindowProperty('Rastreio em execucao', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Rastreio em execucao', img)
                pyautogui.click()
                click = False
                continue

            elif not is_on_calibrate and ratio_x is not None and ratio_y is not None:
                coord = calculate_absolute()
                last_three_x.append(coord[0])
                last_three_y.append(coord[1])
                if len(last_three_x) == 3:
                    coordinates = (median(last_three_x), median(last_three_y)) 
                    pyautogui.moveTo(coordinates)

            elif not is_on_calibrate and (avg_click_data and ratio_x is None):
                # face_detect()
                if not detection_started:
                    detection_started = True
                    start_time = time.time()
            
                elapsed_time = time.time() - start_time
                if elapsed_time >= 2: 
                    alert('Tente ficar próximo à câmera e verifique a iluminação do ambiente')
                    
                detection_started = False
            # cv2.putText(frame, "px: " + str(coord), (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
            # cv2.imshow("Demo", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'ESC' para sair
                break

            time.sleep(0.05)
        cv2.destroyAllWindows()
    except Exception as e:
        error(f"Erro de rastreio: {e}")
    finally:
        cv2.destroyAllWindows()


def start_mouse_tracking():
    global tracking_thread, tracking_thread_running
    with frame_lock:
        if not tracking_thread_running:
            tracking_thread_running = True
            tracking_thread = threading.Thread(target=mouse_tracking, daemon=True)
            tracking_thread.start()

def face_detect():
    global ratio_x, ratio_y, frame, ret

    start_time = None  
    detection_started = False
    
    try:
        while True:
            ret, frame = webcam.read()
            if frame is None or not ret:
                alert("Falha ao capturar imagem da webcam")
                break
            
            debug_frame = frame.copy()
            cam_w = webcam.get(cv2.CAP_PROP_FRAME_WIDTH)

            if ratio_x and ratio_y:
                if not detection_started:
                    detection_started = True
                    start_time = time.time()

                elapsed_time = time.time() - start_time
                text= f"Mantenha sua posicao por {5 - int(elapsed_time)} {'segundos!' if 2 - int(elapsed_time) > 1 else 'segundo!'}"
                color_txt = (0, 255, 0)

                if elapsed_time >= 2: 
                    alert("Detecção concluida com sucesso!")
                    break

            else:
                text = "Alinhe seu rosto onde esta indicado!"
                color_txt = (0, 220, 200)
                detection_started = False

            cv2.putText(debug_frame, text, (getCenterText(text, cam_w, 0.7, 1), 80),cv2.FONT_HERSHEY_TRIPLEX,0.7, color_txt ,1,cv2.LINE_AA)
            frame_mesc = cv2.addWeighted(alert_asset, 0.5, debug_frame, 0.7, 0)    
            cv2.imshow("Deteccao", frame_mesc)

            cv2.moveWindow('Deteccao', int((screen_w - cam_w) / 2), 0)

            # Permitir sair manualmente pressionando 'ESC'
            if cv2.waitKey(1) & 0xFF == 27:
                alert("Detecção interrompida pelo usuário.")
                break
            
            time.sleep(0.1)

    finally:
        cv2.destroyWindow("Deteccao")
        return detection_started

def getCenterText(text, window_width = screen_w, size = 1, thickness = 2):
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, size, thickness)
        text_center = (window_width - text_w) // 2
        return int(text_center)

def calibrate():
    global is_on_calibrate, tracking_thread, tracking_thread_running, root
    is_on_calibrate = True
    alert("Olhe para os pontos indicados em azul e clique com o botão esquerdo do mouse até os pontos ficarem verdes para calibrar o sistema")
    start_mouse_tracking()
    points = [(x, y) for x in [55, screen_w // 2, screen_w - 55] for y in [55, screen_h // 2, screen_h - 55]]
    colors = [(100, 100, 0) for _ in range(len(points))]
    click_data = {i: [] for i in range(len(points))}
    current_point = 0

    def draw_calibration_points(img, points, colors, current_point):
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 50, colors[i], -1)
        if current_point < len(points):
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 3 - int(elapsed_time))
            cv2.circle(img, points[current_point], 50, (200, 100, 0), 10)
            if remaining_time > 0:
                cv2.putText(img, f"{remaining_time}", (points[current_point][0] - 10, points[current_point][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
            else:
                cv2.putText(img, "Pronto", (points[current_point][0] - 32, points[current_point][1] + 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (222, 222, 222), 2)

    def on_mouse_event(event, x = 0, y = 0, flags = '', param = ''):
        nonlocal current_point, start_time
        if event == cv2.EVENT_LBUTTONDOWN and not its_recognizing:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 3 - int(elapsed_time))
            if remaining_time == 0:  # Permitir clique somente após o tempo acabar
                if ratio_x is not None and ratio_y is not None:
                    click_data[current_point].append((ratio_x, ratio_y))
                    num_clicks = len(click_data[current_point])
                    if num_clicks == 1:
                        colors[current_point] = (0, 180, 200)
                    elif num_clicks == 2:
                        colors[current_point] = (100, 200, 0)
                    elif num_clicks == 3:
                        current_point += 1
                        if current_point < len(points):
                            start_time = time.time()
                else:
                    error_face()

    
        
    cv2.namedWindow('Calibracao', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Calibracao', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Calibracao', on_mouse_event)

    try:
        while current_point < len(points):
            if cv2.waitKey(1) & 0xFF == 27:
                tracking_thread_running = False
                is_on_calibrate = False
                cv2.destroyAllWindows()
                break

            img = np.ones((screen_h, screen_w, 3), np.uint8) * 255
            if current_point < len(points):
                start_time = time.time() if 'start_time' not in locals() else start_time
            draw_calibration_points(img, points, colors, current_point)
            text_info = "Olhe para o ponto indicado, e clique com ENTER apos a contagem para registrar o rastreio!"
            if ratio_x is None:
                text= "Erro na identificacao, alinhe seu rosto com a camera"
                color_txt = (0, 0, 200)
            else:
                text= "Rosto identificado, pronto para calibragem!"
                color_txt = (100, 200, 0)
            cv2.putText(img, text_info, (getCenterText(text_info), int(screen_h/4)), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, text, (getCenterText(text), int(screen_h/1.5)), cv2.FONT_HERSHEY_TRIPLEX, 1, color_txt, 2)    
            cv2.imshow('Calibracao', img)
            #permite usar tecla enter para calibrar
            if cv2.waitKey(1) & 0xFF == 13:
                on_mouse_event(cv2.EVENT_LBUTTONDOWN)
            
            time.sleep(0.1)
    finally:
        if is_on_calibrate:
            cv2.destroyAllWindows()

    avg_click_data = {}
    for i, clicks in click_data.items():
        if len(clicks) == 3:
            avg_x = sum(x for x, y in clicks) / 3
            avg_y = sum(y for x, y in clicks) / 3
            avg_click_data[i] = (avg_x, avg_y)

    if len(avg_click_data) > 7:
        print(avg_click_data)
        save_calibration_data(avg_click_data)
    else:
        alert('Erro ao calibrar!')

    with frame_lock:
        tracking_thread_running = False
        if tracking_thread:
            tracking_thread.join()
    cv2.destroyAllWindows()

def save_calibration_data(data):
    global avg_click_data, loaded_calibration_file
    if data:
        try:
            name = os.path.join(calibration_path, "calibration_data.txt")
            with open(name, "w") as archive:
                json.dump(data, archive, indent=4)
            avg_click_data = data
            # breakpoint()
            loaded_calibration_file = False
            calculate_width_ratio(avg_click_data)
            alert("Calibração salva com sucesso!")
        except Exception as e:
            error(f"Erro ao salvar dados de calibração: {e}")

def load_calibration_data():
    global loaded_calibration_file, avg_click_data
    try:
        name = os.path.join(calibration_path, "calibration_data.txt")
        with open(name, "r") as archive:
            loaded_calibration_file = True
            avg_click_data = json.load(archive)
            
            calculate_width_ratio(avg_click_data)
            return avg_click_data
    except Exception as e:
        loaded_calibration_file = False
        avg_click_data = {}
        return avg_click_data

def error(message):
    ctypes.windll.user32.MessageBoxW(
        0,
        message,
        "ERRO",
        0x10
    )
    return

def error_face():
    global its_recognizing
    its_recognizing = True
    error("Rastreio não identificado, alinhe seu rosto com a câmera e tente novamente.")
    face_detect()
    its_recognizing = False
    # breakpoint()

def alert(message):
    ctypes.windll.user32.MessageBoxW(
        0, message, "Aviso", 0x30
    )
    return

def calculate_width_ratio(avg_click_data):

    global x_axis_limits, y_axis_limits, horizontal_scale, vertical_scale

    def robust_mean(value1, value2, value3):
        values = [value1, value2, value3]
        mean = sum(values) / 3
        std_dev = (sum((x - mean) ** 2 for x in values) / 3) ** 0.5       
        # Filtrar valores fora de 1 desvio padrão
        filtered_values = [x for x in values if abs(x - mean) <= std_dev]
        # breakpoint()
        return sum(filtered_values) / len(filtered_values) if filtered_values else mean
    
    def get(data, key):
        # Convertendo a chave para string caso venha do json
        key = str(key) if loaded_calibration_file else int(key)
        return data.get(key)

    # Encontrar os quatro pontos mais próximos na matriz de calibração
    col_left = robust_mean(get(avg_click_data, 0)[0], get(avg_click_data, 1)[0], get(avg_click_data, 2)[0])
    col_right = robust_mean(get(avg_click_data, 6)[0], get(avg_click_data, 7)[0], get(avg_click_data, 8)[0])
    row_top = robust_mean(get(avg_click_data, 0)[1], get(avg_click_data, 3)[1], get(avg_click_data, 6)[1])
    row_bottom = robust_mean(get(avg_click_data, 2)[1], get(avg_click_data, 5)[1], get(avg_click_data, 8)[1])
    
    # breakpoint()

    # Obter os valores de calibração para os pontos
    # top_left = avg_click_data[0]
    # top_right = avg_click_data[6]
    # bottom_left = avg_click_data[2]
    # bottom_right = avg_click_data[8]

    relative_total_x = col_right - col_left
    relative_total_y = row_bottom - row_top
    # breakpoint()
    if relative_total_x == 0 or relative_total_y == 0:
        raise ValueError("Intervalo entre colunas ou linhas não pode ser zero.")

    # Calcular escala
    scale_x = screen_w / relative_total_x
    scale_y = screen_h / relative_total_y

    x_axis_limits = (col_left, col_right)
    y_axis_limits = (row_top, row_bottom)
    horizontal_scale = scale_x
    vertical_scale = scale_y
    return x_axis_limits, y_axis_limits, horizontal_scale, vertical_scale



def calculate_absolute():
    global x_axis_limits, y_axis_limits, horizontal_scale, vertical_scale, ratio_x, ratio_y
     # Garantindo o range das laterais da tela
    ratio_x_normalized = max(min(ratio_x, x_axis_limits[1]), x_axis_limits[0])
    ratio_y_normalized = max(min(ratio_y, y_axis_limits[1]), y_axis_limits[0])
    
    # Calcular os valores em pixels maiores que 0
    value_px_x = max((ratio_x_normalized - x_axis_limits[0]) * horizontal_scale, 10)
    value_px_y = max((ratio_y_normalized - y_axis_limits[0]) * vertical_scale, 10) 
    value_px_x = (screen_w - 40) if value_px_x >= screen_w else value_px_x
    value_px_y = (screen_h - 20) if value_px_y >= screen_h else value_px_y
    print(math.floor(value_px_x), math.floor(value_px_y))
    return int(value_px_x), int(value_px_y)


def main(): 
    global avg_click_data, tracking_thread_running, root, loaded_calibration_file
    
    # Configuração da janela principal
    root.title('L.O.R.E.N.A.')
    if loaded_calibration_file == False: 
        load_calibration_data()
    
    # Calculando a posição para centralizar a janela
    pos_x = screen_w  // 4
    pos_y = screen_h  // 4

    root.geometry(f'{screen_w // 2 }x{screen_h // 2}+{pos_x}+{pos_y}')

    root.iconphoto(False, PhotoImage(file='ui/assets/imgs/logo.png'))
    root.configure(bg='#0b1f30')  # Alterar a cor de fundo para #0b1f30

    configure_style()      

    # Frame principal
    borda = Frame(root, style='Borda.TFrame')
    borda.pack(fill='x', expand=True)

    # Topo com logo
    topo = Frame(borda)
    topo.pack(pady=20)

    # Centro com título
    centro = Frame(borda)
    centro.pack()

    Label(centro, text="Bem-vindo(a) ao LORENA,", font=('Montserrat', 14), background="#0b1f30", foreground="#fff").pack(pady=0)
    Label(centro, text="seu assistente virtual por eyeTracking", font=('Montserrat', 14), background="#0b1f30", foreground="#fff").pack(pady=0)
    logo = PhotoImage(file=r'ui/assets/imgs/logo.png')
    logo = logo.subsample(5, 5)
    lb_logo = Label(topo, image=logo, border=0, background="#0b1f30")
    lb_logo.image = logo
    lb_logo.pack()

    Label(centro, text="Escolha uma opção:", font=('Montserrat', 14), background="#0b1f30", foreground="#fff").pack(pady=10)

    # Base com botões
    base = Frame(borda)
    base.pack()

    btn_play = Button(base, text='Play', command=play, cursor='hand1', takefocus=True)
    btn_play.grid(row=0, column=0, padx=10, pady=10)

    btn_calibrate = Button(base, text='Calibrar', command=calibrate, cursor='hand1', takefocus=True)
    btn_calibrate.grid(row=0, column=1, padx=10, pady=10)

    # Configura o foco inicial no botão "Play"
    # root.after(100, lambda: btn_play.focus_set())

    # Vincula Enter ao botão em foco
    root.bind('<Return>', lambda event: root.focus_get().invoke())

    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
    root.mainloop()

def on_close(root):
    global tracking_thread_running
    tracking_thread_running = False
    if tracking_thread:
        tracking_thread.join()
    webcam.release()
    cv2.destroyAllWindows()
    root.destroy()

def configure_style():
    # Estilos
    style = Style()
    style.theme_use('clam')
    style.configure('TFrame', background='#0b1f30')
    style.configure('Borda.TFrame', width=500, justify='row')
    style.configure('TLabel', justify='center', font=('Montserrat', 10), background='#0b1f30', foreground='#fff')
    style.configure('TButton', padding=(60, 15), font=('Montserrat', 12, 'bold'), foreground='#fff', background='#20bcbb', relief='flat')
    style.configure('TSeparator', background='#0b1f30')

    # Map de interações dos botões
    style.map('TButton',
              foreground=[('pressed', '#e25ca5'), ('active', '#fff')],
              background=[('pressed', '!focus', '#3f8efc'), ('active', '#025b5a')])

    style.configure('TLabel', font=('Montserrat', 14), background='#0b1f30', foreground='#fff')
    style.configure('TButton', font=('Montserrat', 12, 'bold'))


if __name__ == "__main__":
   main()



