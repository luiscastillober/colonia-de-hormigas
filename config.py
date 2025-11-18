"""
Configuración global de la aplicación
"""

# Configuración de OSMnx
OSMNX_CONFIG = {
    'timeout': 300,
    'use_cache': True,
    'log_console': True
}

# Parámetros del algoritmo ACO
ACO_PARAMS = {
    'alpha': 1.2,           # Influencia de feromonas
    'beta': 3.0,            # Influencia de heurística
    'evaporation_rate': 0.90,
    'Q': 200.0,             # Constante de deposición
    'min_pheromone': 0.5,
    'max_pheromone': 100.0,
    'max_steps': 200,       # Máximo de pasos por vehículo
    'max_stuck_count': 5    # Intentos antes de cambiar estrategia
}

# Configuración de la UI
UI_CONFIG = {
    'window_size': '1200x900',
    'map_figsize': (15, 15),
    'colors': ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta']
}

# Mensajes del sistema
MESSAGES = {
    'map_loaded': "✅ Mapa cargado: {} intersecciones, {} segmentos",
    'simulation_start': "🔄 Iniciando simulación con {} iteraciones...",
    'vehicle_added': "🚗 Vehículo {} añadido: {} → {}",
    'road_blocked': "🚧 Calle bloqueada: {} → {}",
    'traffic_added': "🚦 Tráfico añadido: {} → {}"
}