"""
Módulo Core - Sistema de Optimización de Tráfico ACO
"""

# Importar las clases principales para acceso directo
from .city_map import EnhancedCityMap
from .aco_algorithm import (
    ACOTrafficOptimizer,
    ACOVehicle,
    VehicleState,
    ACO_PARAMS
)

# Versión del módulo
__version__ = "1.0.0"
__author__ = "Tu Nombre"
__description__ = "Sistema de optimización de tráfico usando algoritmo ACO"

# Lista de lo que se exporta
__all__ = [
    'EnhancedCityMap',
    'ACOTrafficOptimizer',
    'ACOVehicle',
    'VehicleState',
    'ACO_PARAMS'
]

# Mensaje de inicialización
print(f"✅ Módulo Core ACO v{__version__} cargado correctamente")
print(f"📦 Clases disponibles: {', '.join(__all__)}")