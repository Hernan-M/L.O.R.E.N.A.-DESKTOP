from tkinter import Tk, PhotoImage
from tkinter.ttk import Frame, Label, Button, Style


class Menu:
    
    @staticmethod
    def calibrate():
        print("calibrou")

    @staticmethod
    def play():
        print("playou")

    def __init__(self):
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

        # Definição de estilo
        self._configure_style()

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

        Button(base, text='Play', command=self.play, cursor='hand1').pack(fill='x', pady=(0, 10))
        Button(base, text='Calibrar', command=self.calibrate, cursor='hand1').pack(fill='x', pady=(0, 10))

        # Executa a interface
        root.mainloop()

    def _configure_style(self):
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


# Criação e execução do menu
if __name__ == "__main__":
    menu = Menu()
