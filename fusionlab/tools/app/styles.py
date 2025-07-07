# Fusionlab-learn palette 
PRIMARY   = "#2E3191"   
SECONDARY = "#F28620"   
BG_LIGHT  = "#fafafa"
FG_DARK   = "#1e1e1e"
PRIMARY_T75    = "rgba(46,49,145,0.75)"      # 75 % alpha
SECONDARY_T70  = "rgba(242,134,32,0.70)"     # 70 % alpha
# ------------------------------------------------------------------ #
#  Inference-mode toggle
# ------------------------------------------------------------------ #
INFERENCE_ON  = PRIMARY
INFERENCE_OFF = "#dadada"        

# --- Color Palette Definition ---
# Using a central palette makes themes easier to manage.
# Inspired by common UI color systems.
PALETTE = {
    # Primary Brand Colors
    "primary": "#2E3191",    # Deep Blue
    "primary_hover": "#4338ca",
    "secondary": "#F28620",  # Orange
    
    # Dark Theme Colors
    "dark_bg": "#1e293b",       # slate-800
    "dark_card_bg": "#334155",  # slate-700
    "dark_input_bg": "#0f172a",  # slate-900
    "dark_border": "#475569",     # slate-600
    "dark_text": "#cbd5e1",      # slate-300
    "dark_text_title": "#ffffff",
    "dark_text_muted": "#94a3b8",   # slate-400
    "dark_reset_bg": "#475569",  # slate-600

    # Light Theme Colors
    "light_bg": "#f8fafc",      # slate-50
    "light_card_bg": "#ffffff",
    "light_input_bg": "#f1f5f9", # slate-100
    "light_border": "#cbd5e1",    # slate-300
    "light_text": "#0f172a",      # slate-900
    "light_text_title": "#2E3191", # Primary color for titles
    "light_text_muted": "#64748b",  # slate-500
    "light_reset_bg": "#e2e8f0", # slate-200
}

FLAB_STYLE_SHEET = f"""
QMainWindow {{
    background: {BG_LIGHT};
    color: {FG_DARK};
    font-family: 'Helvetica Neue', sans-serif;
}}

QFrame#card {{
    background: white;
    border: 2px solid {PRIMARY};
    border-radius: 12px;
}}

QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PRIMARY};
}}

QPushButton {{
    background: {PRIMARY};
    color: white;
    border-radius: 6px;
    padding: 6px 12px;
}}
QPushButton:hover {{
    background: {SECONDARY};
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #f0f3ff;
    border: 1px solid {PRIMARY};
    border-radius: 4px;
    padding: 4px;
}}

QTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
}}

QPushButton#reset, QPushButton#stop {{
    background: #dadada;
    color: #333;
}}
QPushButton#reset:hover:enabled,
QPushButton#stop:hover:enabled {{
    background: {SECONDARY};
    color: white;
}}

/* Disabled state */
QPushButton#reset:disabled,
QPushButton#stop:disabled {{
    background: #dadada;
    color: #333;
}}

/* Enabled (normal) state */
QPushButton#stop:enabled {{
    background: {PRIMARY};
    color: white;
}}


QToolTip {{
    /* translucent orange bubble, white text, subtle outline */
    background: {SECONDARY_T70};
    color: white;
    border: 1px solid {SECONDARY};
    border-radius: 4px;
    padding: 4px 6px;
}}

QPushButton#inference {{
    background: {PRIMARY};      /* overwritten at runtime */
    color: white;
    border-radius: 6px;
    padding: 6px 14px;   /* a tad wider than Stop / Reset */
}} 

QPushButton#inference:disabled {{
    background: {INFERENCE_OFF};      /* grey when no manifest yet      */
    color: #666;
}}

QPushButton#tune {{
    background: {PRIMARY};     
    color: white;
    border-radius: 6px;
    padding: 6px 14px;   
}} 

QPushButton#tune:disabled {{
    background: {INFERENCE_OFF};    
    color: #666;
}}


QFrame#card[inferenceMode="true"] {{
    border: 2px solid #2E3191; /* Primary blue color */
}}

/* --- QMessageBox Styling (light theme) --- */
QMessageBox {{
    background-color: {PALETTE['light_bg']};
}}
QMessageBox QLabel {{
    color: {PALETTE['light_text']}; 
    font-size: 14px;
}}
QMessageBox QPushButton {{
    background-color: {PALETTE['primary']};
    color: white;
    border-radius: 4px;
    padding: 8px 20px;
    min-width: 80px;
}}
QMessageBox QPushButton:hover {{
    background-color: {PALETTE['primary_hover']};
}}
QMessageBox QPushButton:pressed {{
    background-color: {PALETTE['secondary']};     
    color: white;
}}

"""
#

DARK_THEME_STYLESHEET = f"""
/* Main Window and General Widgets */
QMainWindow, QWidget {{
    background-color: {PALETTE['dark_bg']};
    color: {PALETTE['dark_text']};
    font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
}}
/* Main Title Label */
QLabel#title {{
    font-size: 28px;
    font-weight: bold;
    color: {PALETTE['dark_text_title']};
    padding: 10px;
}}
/* General Description Label */
QLabel#description {{
    font-size: 14px;
    color: {PALETTE['dark_text_muted']};
}}
/* Card Frame Styling */
QFrame#card {{
    background-color: {PALETTE['dark_card_bg']};
    border: 1px solid {PALETTE['dark_border']};
    border-radius: 12px;
}}
/* Card Title Label */
QLabel#cardTitle {{
    font-size: 18px;
    font-weight: 600;
    color: {PALETTE['dark_text_title']};
    padding-bottom: 5px;
}}
/* Card Description Label */
QLabel#cardDescription {{
    font-size: 13px;
    color: {PALETTE['dark_text_muted']};
}}
/* Main Action Buttons (Run, Select File) */
QPushButton {{
    background-color: {PALETTE['primary']};
    color: white;
    border: none;
    padding: 10px 15px;
    border-radius: 6px;
    font-weight: bold;
    outline: none; /* Remove outline on focus */
}}
QPushButton:hover {{
    background-color: #4338ca; /* Lighter shade of primary for hover */
}}
QPushButton:disabled {{
    background-color: #334155;
    color: {PALETTE['dark_text_muted']};
}}
/* Special Buttons (Reset, Stop) */
QPushButton#resetButton, QPushButton#stopButton {{
    background-color: {PALETTE['dark_reset_bg']};
    color: {PALETTE['dark_text']};
}}
QPushButton#resetButton:hover, QPushButton#stopButton:hover {{
    background-color: {PALETTE['dark_border']};
}}
/* Input Fields (QLineEdit, QSpinBox, etc.) */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {PALETTE['dark_input_bg']};
    border: 1px solid {PALETTE['dark_border']};
    padding: 8px;
    border-radius: 6px;
    color: white;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {PALETTE['primary']}; /* Highlight on focus */
}}
/* Log Output Area */
QTextEdit {{
    background-color: #020617; /* Near black for contrast */
    color: #e2e8f0;
    border: 1px solid {PALETTE['dark_border']};
    font-family: "Consolas", "Courier New", monospace;
    font-size: 13px;
}}
/* Horizontal Separator Line */
QFrame#hLine {{
    border: none;
    border-top: 1px solid {PALETTE['dark_border']};
}}

QFrame#card[inferenceMode="true"] {{
    border: 2px solid #F28620; /* Secondary orange color */
}}

/* --- QMessageBox Styling (dark theme) --- */
QMessageBox {{
    background-color: {PALETTE['dark_card_bg']};  /* e.g. #334155 */
}}

/* Main message text */
QMessageBox QLabel {{
    color: {PALETTE['dark_text']};   /* e.g. #cbd5e1 */
    font-size: 14px;
}}

/* Buttons (Yes / No / OK / Cancel) */
QMessageBox QPushButton {{
    background-color: {PALETTE['primary']};      /* keep your brand blue */
    color: white;
    border-radius: 4px;
    padding: 8px 20px;
    min-width: 80px;
}}
QMessageBox QPushButton:hover {{
    background-color: {PALETTE['primary_hover']}; /* lighter on hover */
}}
QMessageBox QPushButton:pressed {{
    background-color: {PALETTE['secondary']};     /* or primary again */
    color: white;
}}

"""

TAB_STYLES = f"""
QTabBar::tab           {{             
    background : #F9F7F5;
    color      : black;
    padding    : 6px 14px;             
    border-top-left-radius  : 4px;
    border-top-right-radius : 4px;
}}
QTabBar::tab:selected {{              
    background : {SECONDARY};
    color      : white;
}}
QTabBar::tab:hover    {{             
    background : {SECONDARY};
    color      : white;
}}
"""

LOG_STYLES = """
/* apply this to your QTextEdit via setObjectName("logWidget") */
QTextEdit#logWidget {
    background-color: #1e1e1e;
    color:           #e2e8f0;
    font-family:     Consolas, "Courier New", monospace;
    font-size:       12px;
    border:          1px solid rgba(0,0,0,0.25);
    padding:         6px;
}
"""

_LOG_STYLES = f"""
/* ─────────────────────  Frame & shadow ─────────────────────────────── */
QDockWidget#logDock {{
    border        : 1px solid rgba(0,0,0,0.25);   /* light outline      */
    border-radius : 4px;
    background    : #ffffff;
}}

QDockWidget#logDock::separator {{
    width  : 0px;   /* hide default splitter handle (optional)          */
    height : 0px;
}}

/* ─────────────────────  Title-bar (normal + hover) ─────────────────── */
QDockWidget#logDock::title {{
    background    : {PRIMARY};
    color         : white;
    padding       : 4px 10px;
    font-weight   : 600;
    border-top-left-radius  : 4px;
    border-top-right-radius : 4px;
}}

QDockWidget#logDock::title:hover {{
    background : {SECONDARY};         /* on-hover accent                 */
}}

/* ─────────────────────  The QTextEdit inside ───────────────────────── */
QDockWidget#logDock QTextEdit {{
    border        : none;
    background    : #fafafa;
    font-family   : Consolas, monospace;
    font-size     : 11px;
    padding       : 6px;
}}
"""

# ‣ colour logic: PRIMARY when *not* selected, SECONDARY when selected
TUNER_STYLES = f"""
QPushButton {{
    background      : {PRIMARY};
    color           : white;
    padding         : 6px 18px;     /* top/bottom | left/right */
    border          : none;
    border-radius   : 4px;
}}

QPushButton:checked {{
    background : {SECONDARY};
}}
"""

ERROR_STYLES=f"""
/* --- Error Dialog Styling ----- */
QDialog#errorDialog {{
    background: {BG_LIGHT};       /* same as main window */
    border: 2px solid {PRIMARY};  /* primary brand color */
    border-radius: 8px;
    padding: 12px;
    min-width: 600px;             /* enforce a reasonable width */
    max-width: 800px;
}}

QDialog#errorDialog QLabel {{
    color: {FG_DARK};
    font-size: 14px;
}}

QDialog#errorDialog QTextEdit {{
    background: #f6f6f6;
    border: 1px solid #cccccc;
    font-family: Consolas, monospace;
}}

QDialog#errorDialog QPushButton {{
    background: {PRIMARY};
    color: white;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}}

QDialog#errorDialog QPushButton:hover:enabled {{
    background: {SECONDARY};
}}

QDialog#errorDialog QPushButton:disabled {{
    background: {INFERENCE_OFF};
    color: #666;
}}

"""


TUNER_DIALOG_STYLES = f"""
QMessageBox {{
    background-color: {BG_LIGHT};        /* light backdrop */
    border-radius: 8px;
    padding: 12px;
}}
QMessageBox QLabel {{
    color: {FG_DARK};
    font-size: 13px;                     /* slightly smaller text */
    qproperty-alignment: AlignLeft;
}}
QMessageBox QLabel#qt_msgbox_label {{
    font-weight: 600;
    font-size: 14px;                     /* title text a touch bigger */
}}
QMessageBox QPushButton {{
    background-color: {PRIMARY};
    color: white;
    border-radius: 4px;
    padding: 4px 12px;
    min-width: 70px;
}}
QMessageBox QPushButton:hover:enabled {{
    background-color: {SECONDARY};
}}
QMessageBox QPushButton:disabled {{
    background-color: {INFERENCE_OFF};
    color: #888;
}}
"""
