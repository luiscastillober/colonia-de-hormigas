"""
Validadores de datos para la aplicación web
"""

import re
from typing import Tuple, Optional

def validate_place_name(place_name: str) -> Tuple[bool, Optional[str]]:
    """Validar nombre de lugar para OSM"""
    if not place_name or not place_name.strip():
        return False, "El nombre del lugar no puede estar vacío"

    if len(place_name.strip()) < 2:
        return False, "El nombre del lugar es demasiado corto"

    return True, None

def validate_iterations(iterations: int) -> Tuple[bool, Optional[str]]:
    """Validar número de iteraciones"""
    if iterations < 1:
        return False, "Las iteraciones deben ser al menos 1"
    if iterations > 1000:
        return False, "Las iteraciones no pueden exceder 1000"

    return True, None