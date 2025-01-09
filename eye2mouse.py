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


frame_lock = Lock()

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

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
tracking_thread = None

def play():
    global tracking_thread_running
    if avg_click_data:
        try:
            start_mouse_tracking()
        except Exception as e:
            error(f"Erro de rastreio: {e}")
    else:
        alert("Calibre o sistema ao menos uma vez antes de iniciar o rastreio.")

def mouse_tracking():
    global ratio_x, ratio_y, frame, ret, is_on_calibrate
    try:
       while True:
            with frame_lock:
                if not tracking_thread_running:  # Verifica se a thread deve parar
                    break
            ret, frame = webcam.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                error("Falha ao capturar imagem da webcam")
                break

            gaze.refresh(frame)
            ratio_x = gaze.horizontal_ratio()
            ratio_y = gaze.vertical_ratio() 
            print('cu')
            if not is_on_calibrate and ratio_x and ratio_y:
                face_detect()
                img = np.ones((screen_h, screen_w, 3), np.uint8) * 255
                cv2.namedWindow('pleyou', cv2.WINDOW_NORMAL)
                cv2.setWindowProperty('pleyou', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('pleyou', img)
                coord = calculate_width_ratio()
                cv2.circle(img, coord, 3, (0, 0, 255))
                
            elif avg_click_data and ratio_x is None:
                error_face()

            time.sleep(0.2)

            # if cv2.waitKey(1) & 0xFF == 27:
            #     alert("Detecção interrompida pelo usuário.")
            #     break
    except Exception as e:
        breakpoint()
        error(f"Erro de rastreio: {e}")

def start_mouse_tracking():
    global tracking_thread, tracking_thread_running
    with frame_lock:
        print(tracking_thread_running)
        if not tracking_thread_running:
            print('aisadasdas')
            tracking_thread_running = True
            tracking_thread = threading.Thread(target=mouse_tracking, daemon=True)
            tracking_thread.start()

def face_detect():
    global ratio_x, ratio_y

    start_time = None  
    detection_started = False

    while True:
        frame, ret = webcam.read()
        if frame is None or not ret:
            alert("Falha ao capturar imagem da webcam")
            break
        
        debug_frame = frame.copy()
        middle_cam = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)

        if ratio_x and ratio_y:
            if not detection_started:
                detection_started = True
                start_time = time.time()

            elapsed_time = time.time() - start_time

            cv2.putText(
                debug_frame,
                f"Mantenha sua posição por {5 - int(elapsed_time)} segundos!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            if elapsed_time >= 5: 
                alert("Detecção concluída com sucesso!")
                break

        else:
            cv2.putText(
                debug_frame,
                "Alinhe seu rosto de acordo com a marcação!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            detection_started = False

        frame_mesc = cv2.addWeighted(alert_asset, 0.5, debug_frame, 0.7, 0)    
        cv2.imshow("Deteccao", frame_mesc)

        cv2.moveWindow('Deteccao', int(screen_w / 2 - middle_cam), 0)

        # Permitir sair manualmente pressionando 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            alert("Detecção interrompida pelo usuário.")
            break

    cv2.destroyWindow('Deteccao')

def calibrate():
    global is_on_calibrate, tracking_thread, tracking_thread_running
    is_on_calibrate = True
    alert("Olhe para os pontos indicados em azul e clique com o botão esquerdo do mouse até os pontos ficarem verdes para calibrar o sistema")
    start_mouse_tracking()
    points = [(x, y) for x in [0, screen_w // 2, screen_w] for y in [0, screen_h // 2, screen_h]]
    colors = [(100, 100, 0) for _ in range(len(points))]
    click_data = {i: [] for i in range(len(points))}
    current_point = 0

    def draw_calibration_points(img, points, colors, current_point):
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 50, colors[i], -1)
        if current_point < len(points):
            cv2.circle(img, points[current_point], 50, (200, 100, 0), 10)

    def on_mouse_event(event, x, y, flags, param):
        nonlocal current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            if ratio_x is not None and ratio_y is not None:
                try : 
                    click_data[current_point].append((ratio_x, ratio_y))
                except: 
                    error_face()
                num_clicks = len(click_data[current_point])
                if num_clicks == 1:
                    colors[current_point] = (0, 255, 255)
                elif num_clicks == 2:
                    colors[current_point] = (0, 255, 0)
                elif num_clicks == 3:
                    current_point += 1
            else:
                error_face()

    cv2.namedWindow('Calibracao', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Calibracao', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Calibracao', on_mouse_event)

    try:
        while current_point < len(points):
            img = np.ones((screen_h, screen_w, 3), np.uint8) * 255
            draw_calibration_points(img, points, colors, current_point)
            cv2.imshow('Calibracao', img)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow('Calibracao')
                break
    finally:
        cv2.destroyAllWindows()

    global avg_click_data
    avg_click_data = {}
    for i, clicks in click_data.items():
        if len(clicks) == 3:
            avg_x = sum(x for x, y in clicks) / 3
            avg_y = sum(y for x, y in clicks) / 3
            avg_click_data[i] = (avg_x, avg_y)

    with frame_lock:
        tracking_thread_running = False
    tracking_thread.join()
    save_calibration_data()
    is_on_calibrate = False
    return avg_click_data

def save_calibration_data():
    if avg_click_data:
        try:
            name = os.path.join(calibration_path, "calibration_data.txt")
            with open(name, "w") as archive:
                json.dump(avg_click_data, archive, indent=4)
            alert("Calibração salva com sucesso!")
        except Exception as e:
            error(f"Erro ao salvar dados de calibração: {e}")

def load_calibration_data():
    try:
        name = os.path.join(calibration_path, "calibration_data.txt")
        with open(name, "r") as archive:
            return json.load(archive)
    except Exception as e:
        return {}

def error(message):
    ctypes.windll.user32.MessageBoxW(
        0,
        message,
        "ERRO",
        0x10
    )
    return

def error_face():
    error("Rastreio não identificado, alinhe seu rosto com a câmera e tente novamente.")
    print('aiiuaa')
    face_detect()

def alert(message):
    ctypes.windll.user32.MessageBoxW(
        0, message, "Aviso", 0x30
    )
    return

def calculate_width_ratio():

    def robust_mean(value1, value2, value3):
        values = [value1, value2, value3]
        mean = sum(values) / 3
        std_dev = (sum((x - mean) ** 2 for x in values) / 3) ** 0.5       
        # Filtrar valores fora de 1 desvio padrão
        filtered_values = [x for x in values if abs(x - mean) <= std_dev]
        return sum(filtered_values) / len(filtered_values) if filtered_values else mean

    # Encontrar os quatro pontos mais próximos na matriz de calibração
    col_left = robust_mean(avg_click_data[0][0], avg_click_data[1][0], avg_click_data[2][0])
    col_right = robust_mean(avg_click_data[6][0], avg_click_data[7][0], avg_click_data[8][0])
    row_top = robust_mean(avg_click_data[0][1], avg_click_data[3][1], avg_click_data[6][1])
    row_bottom = robust_mean(avg_click_data[2][1], avg_click_data[5][1], avg_click_data[8][1])
    
    # Obter os valores de calibração para os pontos
    top_left = avg_click_data[0]
    top_right = avg_click_data[6]
    bottom_left = avg_click_data[2]
    bottom_right = avg_click_data[8]

    relative_total_x = col_right - col_left
    relative_total_y = row_bottom - row_top
    # breakpoint()
    if relative_total_x == 0 or relative_total_y == 0:
        raise ValueError("Intervalo entre colunas ou linhas não pode ser zero.")

    # Calcular escala
    scale_x = screen_w / relative_total_x
    scale_y = screen_h / relative_total_y

    # Garantindo o range das laterais da tela
    ratio_x_normalized = max(min(ratio_x, col_right), col_left)
    ratio_y_normalized = max(min(ratio_y, row_bottom), row_top)
    
    # Calcular os valores em pixels maiores que 0
    value_px_x = max((ratio_x_normalized - col_left) * scale_x, 10)
    value_px_y = max((ratio_y_normalized - row_top) * scale_y, 10) 
    value_px_x = screen_w - 10 if value_px_x > screen_w else value_px_x
    value_px_y = screen_h - 10 if value_px_y > screen_h else value_px_y
    print(value_px_x, value_px_y)
    return int(value_px_x), int(value_px_y)





def main(): 
        global avg_click_data
        avg_click_data = load_calibration_data()
        # Configuração da janela principal
        root = Tk()
        root.title('L.O.R.E.N.A.')

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        
        # Calculando a posição para centralizar a janela
        pos_x = (screen_w - 550) // 2
        pos_y = (screen_h - 550) // 2

        root.geometry(f'{550}x{550}+{pos_x}+{pos_y}')

        root.iconphoto(False, PhotoImage(file='ui/assets/imgs/logo.png'))
        root.configure(bg='#fff')  

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

        Label(centro, text="Bem-vindo(a) ao LORENA,", font=('Montserrat', 16), background="#fff").pack(pady=0)
        Label(centro, text="seu assistente virtual por eyeTracking", font=('Montserrat', 16), background="#fff").pack(pady=0)
        logo = PhotoImage(file=r'ui/assets/imgs/logo.png')
        logo = logo.subsample(5, 5)
        lb_logo = Label(topo, image=logo)
        lb_logo.image = logo
        lb_logo.pack()

        Label(centro, text="Escolha uma opção:", font=('Montserrat', 16), background="#fff").pack(pady=10)

        # Base com botões
        base = Frame(borda)
        base.pack()

        Button(base, text='Play', command=play, cursor='hand1').pack(fill='x', pady=(0, 10))
        Button(base, text='Calibrar', command=calibrate, cursor='hand1').pack(fill='x', pady=(0, 10))

        root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
        root.mainloop()

def on_close(root):
    webcam.release()
    cv2.destroyAllWindows()
    root.destroy()



def configure_style():
            # Estilos
            style = Style()
            style.theme_use('default')
            style.configure('TFrame', background='#fff')
            style.configure('Borda.TFrame', width=300)
            style.configure('TLabel', justify='center', font=('Montserrat', 10), background='#fff', foreground='#808080')
            style.configure('TButton', padding=(60, 7), font=('Montserrat', 12, 'bold'), foreground='#fff', background='#20bcbb', relief='flat')
            style.configure('TSeparator', background='#bafafa')

            # Map de interações dos botões
            style.map('TButton',
                    foreground=[('pressed', '#e25ca5'), ('active', '#fff')],
                    background=[('pressed', '!focus', '#3f8efc'), ('active', '#025b5a')])

            style.configure('TLabel', font=('Montserrat', 14), background='#fff')
            style.configure('TButton', font=('Montserrat', 12, 'bold'))


if __name__ == "__main__":
   main()
