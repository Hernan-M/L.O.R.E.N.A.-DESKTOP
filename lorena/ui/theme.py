"""Identidade visual refinada para interface desktop moderna no Windows."""
from tkinter.ttk import Style

BG = "#0b1f30"
SURFACE = "#112638"
SURFACE_BORDER = "#1f4463"
FG = "#ffffff"
MUTED = "#8fa6be"
ACCENT = "#20bcbb"
ACCENT_ACTIVE = "#025b5a"
ACCENT_PRESSED = "#3f8efc"
PINK = "#e25ca5"
OK = "#3ccf7a"
WARN = "#f2c94c"
ERROR = "#eb5757"
FONT = "Segoe UI"


def apply(style: Style | None = None) -> Style:
    style = style or Style()
    style.theme_use("clam")

    # Containers
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=SURFACE)

    # Rótulos
    style.configure("TLabel", font=(FONT, 11), background=BG, foreground=FG, justify="center")
    style.configure("Header.TLabel", font=(FONT, 18, "bold"), background=BG, foreground=FG)
    style.configure("Subtitle.TLabel", font=(FONT, 10), background=BG, foreground=MUTED)
    style.configure("Card.TLabel", font=(FONT, 10), background=SURFACE, foreground=FG)
    style.configure("CardHeader.TLabel", font=(FONT, 11, "bold"), background=SURFACE, foreground=ACCENT)
    style.configure("Small.TLabel", font=(FONT, 9), background=BG, foreground=MUTED)
    style.configure("StatusOk.TLabel", font=(FONT, 9, "bold"), background=SURFACE, foreground=OK)
    style.configure("StatusWarn.TLabel", font=(FONT, 9, "bold"), background=SURFACE, foreground=WARN)
    style.configure("Hint.TLabel", font=(FONT, 9), background=BG, foreground="#6b8299")

    # Botão padrão / Primário
    style.configure(
        "TButton",
        padding=(30, 12),
        font=(FONT, 11, "bold"),
        foreground=FG,
        background=ACCENT,
        relief="flat",
    )
    style.map(
        "TButton",
        foreground=[("pressed", PINK), ("active", FG), ("disabled", "#6c8194")],
        background=[("pressed", "!focus", ACCENT_PRESSED), ("active", ACCENT_ACTIVE), ("disabled", "#0f2334")],
    )

    # Botão Secundário
    style.configure(
        "Secondary.TButton",
        padding=(30, 12),
        font=(FONT, 11, "bold"),
        foreground="#dcebf7",
        background="#183b54",
        relief="flat",
    )
    style.map(
        "Secondary.TButton",
        foreground=[("pressed", FG), ("active", FG), ("disabled", "#6c8194")],
        background=[("pressed", "!focus", "#1f4a6b"), ("active", "#225175"), ("disabled", "#0f2334")],
    )

    # Seletor Combobox
    style.configure(
        "TCombobox",
        fieldbackground="#132f4c",
        background=ACCENT,
        foreground=FG,
        arrowcolor=FG,
        font=(FONT, 10),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#132f4c"), ("disabled", "#081622")],
        foreground=[("disabled", "#5a6e85"), ("readonly", FG)],
        selectbackground=[("readonly", ACCENT_ACTIVE)],
        selectforeground=[("readonly", FG)],
    )

    return style
