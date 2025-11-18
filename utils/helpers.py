"""
Funciones auxiliares para la aplicación web
"""

import random
from typing import List, Tuple, Optional
import streamlit as st

def validate_coordinates(lat: str, lon: str) -> Tuple[bool, Optional[float], Optional[float]]:
    """Validar coordenadas geográficas"""
    try:
        lat_num = float(lat)
        lon_num = float(lon)

        if -90 <= lat_num <= 90 and -180 <= lon_num <= 180:
            return True, lat_num, lon_num
        else:
            return False, None, None

    except (ValueError, TypeError):
        return False, None, None

def format_time(seconds: float) -> str:
    """Formatear tiempo en segundos a string legible"""
    if seconds < 60:
        return f"{seconds:.1f} segundos"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutos"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} horas"

def calculate_success_rate(optimizer) -> float:
    """Calcular tasa de éxito de la simulación"""
    if not optimizer or not optimizer.vehicles:
        return 0.0

    arrived = sum(1 for v in optimizer.vehicles if v.has_arrived())
    return (arrived / len(optimizer.vehicles)) * 100

def get_simulation_stats(optimizer) -> dict:
    """Obtener estadísticas de la simulación"""
    if not optimizer:
        return {}

    stats = {
        'total_vehicles': len(optimizer.vehicles),
        'arrived_vehicles': sum(1 for v in optimizer.vehicles if v.has_arrived()),
        'success_rate': calculate_success_rate(optimizer),
        'total_iterations': optimizer.iteration,
        'average_travel_time': 0.0
    }

    # Calcular tiempo promedio de viaje
    arrived_times = [v.get_path_travel_time() for v in optimizer.vehicles if v.has_arrived()]
    if arrived_times:
        stats['average_travel_time'] = sum(arrived_times) / len(arrived_times)

    return stats