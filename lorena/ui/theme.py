"""Identidade visual (antes duplicada entre eye2mouse.py e ui/menu.py)."""
from tkinter.ttk import Style

BG = "#0b1f30"
FG = "#ffffff"
ACCENT = "#20bcbb"
ACCENT_ACTIVE = "#025b5a"
ACCENT_PRESSED = "#3f8efc"
PINK = "#e25ca5"
OK = "#3ccf7a"
WARN = "#f2c94c"
ERROR = "#eb5757"
FONT = "Montserrat"


def apply(style: Style | None = None) -> Style:
    style = style or Style()
    style.theme_use("clam")
    style.configure("TFrame", background=BG)
    style.configure("TLabel", font=(FONT, 14), background=BG, foreground=FG, justify="center")
    style.configure("Small.TLabel", font=(FONT, 10), background=BG, foreground="#9fb3c8")
    style.configure("TButton", padding=(60, 15), font=(FONT, 12, "bold"), foreground=FG, background=ACCENT, relief="flat")
    style.map(
        "TButton",
        foreground=[("pressed", PINK), ("active", FG)],
        background=[("pressed", "!focus", ACCENT_PRESSED), ("active", ACCENT_ACTIVE)],
    )
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
