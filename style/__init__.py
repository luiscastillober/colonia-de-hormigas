from .headers import get_header_styles
from .cards import get_card_styles
from .buttons import get_button_styles
from .layout import get_layout_styles

def load_all_styles():
    """Cargar todos los estilos"""
    return (
        get_header_styles() +
        get_card_styles() +
        get_button_styles() +
        get_layout_styles()
    )

def load_basic_styles():
    """Cargar estilos básicos (sin botones)"""
    return get_header_styles() + get_card_styles()

def load_minimal_styles():
    """Cargar estilos mínimos"""
    return get_header_styles()