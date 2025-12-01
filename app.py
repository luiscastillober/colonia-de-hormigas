"""
Sistema de Optimización de Tráfico ACO - Versión FOLIUM + STREAMLIT
Aplicación principal con mapas interactivos Folium/Leaflet
"""

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

# Importar módulos del sistema
from core.city_map import EnhancedCityMap
from core.aco_algorithm import ACOTrafficOptimizer, ACO_PARAMS

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="🚗 ACO Traffic Optimizer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS PERSONALIZADO
# ============================================

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ESTADO DE LA APLICACIÓN
# ============================================

if 'city_map' not in st.session_state:
    st.session_state.city_map = None
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = None
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = []
if 'selected_start' not in st.session_state:
    st.session_state.selected_start = None
if 'selected_end' not in st.session_state:
    st.session_state.selected_end = None
if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
if 'blocked_roads' not in st.session_state:
    st.session_state.blocked_roads = []
if 'traffic_areas' not in st.session_state:
    st.session_state.traffic_areas = []

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def create_base_map(city_map):
    """Crear mapa base de Folium con el grafo de la ciudad"""
    if not city_map or not city_map.intersections:
        return None
    
    # Calcular centro del mapa
    coords = [node['coords'] for node in city_map.intersections.values()]
    lons, lats = zip(*coords)
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Calcular bounds para ajustar el zoom
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Crear mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap',
        control_scale=True,
        prefer_canvas=True  # Mejor rendimiento
    )
    
    # Ajustar bounds
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    
    # Añadir capas de tiles alternativos
    folium.TileLayer('cartodbdark_matter', name='🌙 Dark Mode').add_to(m)
    folium.TileLayer('cartodbpositron', name='☀️ Light Mode').add_to(m)
    
    return m

def add_roads_to_map(m, city_map, max_roads=1000):
    """Añadir calles al mapa (limitado para rendimiento)"""
    if not city_map:
        return m
    
    # Grupo de calles normales
    streets_group = folium.FeatureGroup(name='🛣️ Calles', show=True)
    
    # Filtrar solo calles importantes si hay muchas
    roads_to_draw = []
    
    for (start, end), road_data in city_map.roads.items():
        # Priorizar calles importantes
        importance = road_data.get('importance', 0)
        if importance > 0 or len(roads_to_draw) < max_roads:
            roads_to_draw.append(((start, end), road_data))
    
    # Limitar cantidad para rendimiento
    roads_to_draw = roads_to_draw[:max_roads]
    
    # Dibujar calles
    drawn_roads = []
    for (start, end), road_data in roads_to_draw:
        # Verificar si ya dibujamos esta calle
        road_drawn = False
        for drawn in drawn_roads:
            if (drawn[0] == start and drawn[1] == end) or (drawn[0] == end and drawn[1] == start):
                road_drawn = True
                break
        
        if not road_drawn:
            if start in city_map.intersections and end in city_map.intersections:
                x0, y0 = city_map.intersections[start]['coords']
                x1, y1 = city_map.intersections[end]['coords']
                
                # Color según tipo de calle
                color = 'gray'
                weight = 2
                opacity = 0.3
                
                if road_data.get('is_major_road', False):
                    color = '#4A90E2'
                    weight = 3
                    opacity = 0.7
                
                # Obtener nombre de forma segura
                road_name = road_data.get('name', 'Sin nombre')
                if isinstance(road_name, list):
                    road_name = ', '.join(str(n) for n in road_name)
                
                # Crear línea (sin popup para mejor rendimiento)
                folium.PolyLine(
                    locations=[[y0, x0], [y1, x1]],
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    tooltip=str(road_name) if road_data.get('is_major_road', False) else None
                ).add_to(streets_group)
                
                drawn_roads.append((start, end))
    
    streets_group.add_to(m)
    return m

def add_blocked_roads_to_map(m, city_map):
    """Añadir calles bloqueadas al mapa"""
    if not city_map or not city_map.blocked_roads:
        return m
    
    blocked_group = folium.FeatureGroup(name='🚧 Calles Bloqueadas', show=True)
    
    for (start, end) in city_map.blocked_roads:
        if start in city_map.intersections and end in city_map.intersections:
            x0, y0 = city_map.intersections[start]['coords']
            x1, y1 = city_map.intersections[end]['coords']
            
            folium.PolyLine(
                locations=[[y0, x0], [y1, x1]],
                color='red',
                weight=5,
                opacity=0.8,
                dash_array='10, 5',
                popup=f"🚧 Calle Bloqueada<br>Nodos: {start} → {end}",
                tooltip="🚧 Bloqueada"
            ).add_to(blocked_group)
    
    blocked_group.add_to(m)
    return m

def add_traffic_areas_to_map(m, city_map):
    """Añadir áreas con tráfico al mapa"""
    if not city_map or not city_map.high_traffic_roads:
        return m
    
    traffic_group = folium.FeatureGroup(name='🚦 Tráfico Alto', show=True)
    
    for (start, end) in city_map.high_traffic_roads:
        if start in city_map.intersections and end in city_map.intersections:
            x0, y0 = city_map.intersections[start]['coords']
            x1, y1 = city_map.intersections[end]['coords']
            
            folium.PolyLine(
                locations=[[y0, x0], [y1, x1]],
                color='orange',
                weight=4,
                opacity=0.7,
                popup=f"🚦 Tráfico Alto<br>Nodos: {start} → {end}",
                tooltip="🚦 Congestión"
            ).add_to(traffic_group)
    
    traffic_group.add_to(m)
    return m

def add_nodes_to_map(m, city_map, highlight_nodes=None, show_all=False):
    """Añadir nodos importantes al mapa (optimizado)"""
    if not city_map:
        return m
    
    nodes_group = folium.FeatureGroup(name='📍 Intersecciones', show=False)
    
    # Solo mostrar nodos importantes o destacados para mejor rendimiento
    for node_id, node_data in city_map.intersections.items():
        x, y = node_data['coords']
        
        # Determinar si debemos mostrar este nodo
        is_important = node_data.get('is_important', False)
        is_highlighted = highlight_nodes and node_id in highlight_nodes
        
        # Solo mostrar nodos importantes o destacados (a menos que show_all=True)
        if not show_all and not is_important and not is_highlighted:
            continue
        
        # Color y tamaño según importancia
        if is_highlighted:
            color = 'red'
            radius = 8
            fill_opacity = 1.0
        elif is_important:
            color = 'blue'
            radius = 4
            fill_opacity = 0.6
        else:
            color = 'lightblue'
            radius = 2
            fill_opacity = 0.3
        
        folium.CircleMarker(
            location=[y, x],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            popup=f"<b>Nodo {node_id}</b><br>"
                  f"Conexiones: {node_data.get('degree', 0)}",
            tooltip=f"Nodo {node_id}"
        ).add_to(nodes_group)
    
    nodes_group.add_to(m)
    return m

def add_vehicle_routes_to_map(m, city_map, vehicles):
    """Añadir rutas de vehículos al mapa"""
    if not city_map or not vehicles:
        return m
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8B88B', '#ABEBC6']
    
    for idx, vehicle in enumerate(vehicles):
        if len(vehicle.path) < 2:
            continue
        
        color = colors[idx % len(colors)]
        
        # Crear grupo para esta ruta
        route_group = folium.FeatureGroup(
            name=f'🚗 Vehículo {idx + 1}',
            show=True
        )
        
        # Obtener coordenadas del path
        path_coords = []
        for node_id in vehicle.path:
            if node_id in city_map.intersections:
                x, y = city_map.intersections[node_id]['coords']
                path_coords.append([y, x])
        
        if len(path_coords) < 2:
            continue
        
        # Dibujar la ruta
        folium.PolyLine(
            locations=path_coords,
            color=color,
            weight=4,
            opacity=0.8,
            popup=f"<b>Vehículo {idx + 1}</b><br>"
                  f"Inicio: {vehicle.start}<br>"
                  f"Destino: {vehicle.end}<br>"
                  f"Nodos visitados: {len(vehicle.path)}<br>"
                  f"Tiempo: {vehicle.travel_time:.2f}<br>"
                  f"Estado: {'✅ Llegó' if vehicle.arrived else '🔄 En ruta'}",
            tooltip=f"Vehículo {idx + 1}"
        ).add_to(route_group)
        
        # Añadir marcadores de inicio y fin
        if path_coords:
            # Inicio
            folium.Marker(
                location=path_coords[0],
                popup=f"🏁 Inicio V{idx + 1}",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(route_group)
            
            # Fin
            folium.Marker(
                location=path_coords[-1],
                popup=f"🎯 {'Llegó' if vehicle.arrived else 'Destino'} V{idx + 1}",
                icon=folium.Icon(
                    color='red' if vehicle.arrived else 'orange',
                    icon='flag-checkered' if vehicle.arrived else 'flag',
                    prefix='fa'
                )
            ).add_to(route_group)
        
        route_group.add_to(m)
    
    return m

def create_full_map():
    """Crear mapa completo con todas las capas"""
    city_map = st.session_state.city_map
    
    if not city_map:
        return None
    
    # Obtener nivel de detalle
    max_roads = st.session_state.get('max_roads', 1000)
    
    # Crear mapa base
    m = create_base_map(city_map)
    
    # Añadir capas con límite de rendimiento
    m = add_roads_to_map(m, city_map, max_roads=max_roads if max_roads else 999999)
    m = add_blocked_roads_to_map(m, city_map)
    m = add_traffic_areas_to_map(m, city_map)
    
    # Añadir nodos destacados
    highlight = []
    if st.session_state.selected_start:
        highlight.append(st.session_state.selected_start)
    if st.session_state.selected_end:
        highlight.append(st.session_state.selected_end)
    
    m = add_nodes_to_map(m, city_map, highlight)
    
    # Añadir rutas de vehículos si hay simulación
    if st.session_state.optimizer and st.session_state.simulation_done:
        m = add_vehicle_routes_to_map(m, city_map, st.session_state.optimizer.vehicles)
    
    # Añadir control de capas
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Añadir plugin de pantalla completa
    plugins.Fullscreen(
        position='topright',
        title='Pantalla completa',
        title_cancel='Salir de pantalla completa'
    ).add_to(m)
    
    # Añadir minimap (opcional, puede afectar rendimiento)
    # plugins.MiniMap(toggle_display=True).add_to(m)
    
    return m

# ============================================
# INTERFAZ DE USUARIO
# ============================================

# Header
st.title("🚗 Optimizador de Tráfico con ACO")
st.markdown("**Sistema de Optimización de Rutas usando Algoritmo de Colonia de Hormigas**")
st.markdown("---")

# Sidebar - Controles
with st.sidebar:
    st.header("🎛️ Panel de Control")
    
    # Tab 1: Cargar Mapa
    with st.expander("🗺️ **1. Cargar Mapa**", expanded=True):
        load_method = st.radio(
            "Método de carga:",
            ["Por Nombre", "Por Coordenadas"],
            horizontal=True
        )
        
        # Selector de nivel de detalle
        detail_level = st.select_slider(
            "Nivel de detalle del mapa:",
            options=["Bajo (500)", "Medio (1000)", "Alto (2000)", "Completo"],
            value="Medio (1000)",
            help="Más detalle = más lento"
        )
        
        # Guardar nivel de detalle en session_state
        if detail_level == "Bajo (500)":
            st.session_state.max_roads = 500
        elif detail_level == "Medio (1000)":
            st.session_state.max_roads = 1000
        elif detail_level == "Alto (2000)":
            st.session_state.max_roads = 2000
        else:
            st.session_state.max_roads = None  # Mostrar todo
        
        if load_method == "Por Nombre":
            place_name = st.text_input(
                "Nombre del lugar:",
                value="Trujillo, Peru",
                help="Ejemplo: Trujillo, Peru"
            )
            
            if st.button("🔍 Cargar Mapa", key="load_place"):
                with st.spinner("Cargando mapa..."):
                    try:
                        city_map = EnhancedCityMap()
                        city_map.load_city_from_osm(place_name)
                        st.session_state.city_map = city_map
                        st.success(f"✅ Mapa cargado: {len(city_map.intersections)} intersecciones")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("Latitud:", value=-8.109, format="%.6f")
            with col2:
                lon = st.number_input("Longitud:", value=-79.022, format="%.6f")
            
            dist = st.slider("Radio (metros):", 500, 3000, 1000, 100)
            
            if st.button("🔍 Cargar Mapa", key="load_coords"):
                with st.spinner("Cargando mapa..."):
                    try:
                        city_map = EnhancedCityMap()
                        city_map.load_city_from_point(lat, lon, dist)
                        st.session_state.city_map = city_map
                        st.success(f"✅ Mapa cargado: {len(city_map.intersections)} intersecciones")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # Tab 2: Configurar Vehículos
    with st.expander("🚗 **2. Configurar Vehículos**", expanded=False):
        if not st.session_state.city_map:
            st.warning("⚠️ Carga un mapa primero")
        else:
            st.info("💡 Ingresa los IDs de nodos manualmente")
            
            col1, col2 = st.columns(2)
            with col1:
                start_node = st.number_input(
                    "Nodo Inicio:",
                    min_value=0,
                    max_value=len(st.session_state.city_map.intersections) - 1,
                    value=0,
                    key="start_input"
                )
            with col2:
                end_node = st.number_input(
                    "Nodo Destino:",
                    min_value=0,
                    max_value=len(st.session_state.city_map.intersections) - 1,
                    value=min(10, len(st.session_state.city_map.intersections) - 1),
                    key="end_input"
                )
            
            if st.button("➕ Añadir Vehículo", key="add_vehicle"):
                st.session_state.vehicles.append({
                    'start': start_node,
                    'end': end_node
                })
                st.success(f"✅ Vehículo añadido: {start_node} → {end_node}")
            
            # Mostrar vehículos añadidos
            if st.session_state.vehicles:
                st.markdown("**Vehículos configurados:**")
                for i, v in enumerate(st.session_state.vehicles):
                    st.text(f"🚗 V{i+1}: {v['start']} → {v['end']}")
                
                if st.button("🗑️ Limpiar Vehículos"):
                    st.session_state.vehicles = []
                    st.rerun()
    
    # Tab 3: Obstáculos
    with st.expander("🚧 **3. Obstáculos**", expanded=False):
        if not st.session_state.city_map:
            st.warning("⚠️ Carga un mapa primero")
        else:
            st.subheader("Bloquear Calle")
            col1, col2 = st.columns(2)
            with col1:
                block_start = st.number_input("Desde:", 0, key="block_start")
            with col2:
                block_end = st.number_input("Hasta:", 1, key="block_end")
            
            if st.button("🚧 Bloquear"):
                st.session_state.city_map.block_road_between_nodes(block_start, block_end)
                st.session_state.blocked_roads.append((block_start, block_end))
                st.success(f"Bloqueada: {block_start} ↔ {block_end}")
            
            st.markdown("---")
            
            st.subheader("Añadir Tráfico")
            traffic_center = st.number_input("Nodo Central:", 0, key="traffic_center")
            traffic_radius = st.slider("Radio:", 1, 5, 2)
            traffic_factor = st.slider("Factor:", 1.0, 5.0, 2.5, 0.5)
            
            if st.button("🚦 Añadir Tráfico"):
                st.session_state.city_map.add_traffic_to_area(
                    traffic_center,
                    traffic_radius,
                    traffic_factor
                )
                st.session_state.traffic_areas.append(traffic_center)
                st.success(f"Tráfico añadido en nodo {traffic_center}")
    
    # Tab 4: Simulación
    with st.expander("▶️ **4. Ejecutar Simulación**", expanded=False):
        if not st.session_state.city_map:
            st.warning("⚠️ Carga un mapa primero")
        elif not st.session_state.vehicles:
            st.warning("⚠️ Añade vehículos primero")
        else:
            st.subheader("Parámetros")
            
            iterations = st.slider(
                "Iteraciones máximas:",
                10, 500, 100, 10
            )
            
            with st.expander("⚙️ Parámetros Avanzados"):
                alpha = st.slider("Alpha (feromonas):", 0.1, 3.0, float(ACO_PARAMS['alpha']), 0.1)
                beta = st.slider("Beta (heurística):", 0.1, 5.0, float(ACO_PARAMS['beta']), 0.1)
                evap = st.slider("Evaporación:", 0.1, 0.9, float(ACO_PARAMS['evaporation_rate']), 0.05)
                
                ACO_PARAMS['alpha'] = alpha
                ACO_PARAMS['beta'] = beta
                ACO_PARAMS['evaporation_rate'] = evap
            
            if st.button("▶️ **EJECUTAR SIMULACIÓN**", type="primary", key="run_sim"):
                # Crear optimizador
                optimizer = ACOTrafficOptimizer(st.session_state.city_map)
                
                # Añadir vehículos
                for v in st.session_state.vehicles:
                    optimizer.add_vehicle(v['start'], v['end'])
                
                st.session_state.optimizer = optimizer
                
                # Ejecutar simulación
                result = optimizer.run_until_all_arrive(max_iterations=iterations)
                
                st.session_state.simulation_done = True
                
                # Mostrar resultados
                if result['success']:
                    st.success(f"✅ Simulación exitosa en {result['iterations']} iteraciones")
                else:
                    st.warning(f"⏰ Máximo de iteraciones alcanzado")
                
                st.metric("Vehículos que llegaron", 
                         f"{result['final_stats']['arrived_count']}/{result['final_stats']['total_vehicles']}")
                st.metric("Tasa de éxito", 
                         f"{result['final_stats']['success_rate']:.1f}%")

# Área principal - Mapa
st.header("🗺️ Visualización del Sistema")

if st.session_state.city_map:
    # Crear y mostrar el mapa
    try:
        with st.spinner("🗺️ Generando mapa interactivo..."):
            map_obj = create_full_map()
            
            if map_obj:
                st.success(f"✅ Mapa listo: {len(st.session_state.city_map.intersections)} intersecciones")
                
                # Mostrar el mapa con Folium
                map_data = st_folium(
                    map_obj,
                    width=1400,
                    height=700,
                    returned_objects=["last_clicked"],
                    key="main_map"
                )
            else:
                st.error("❌ No se pudo crear el mapa")
    except Exception as e:
        st.error(f"❌ Error al generar el mapa: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # Estadísticas si hay simulación
    if st.session_state.simulation_done and st.session_state.optimizer:
        st.markdown("---")
        st.header("📊 Resultados de la Simulación")
        
        col1, col2, col3, col4 = st.columns(4)
        
        optimizer = st.session_state.optimizer
        vehicles = optimizer.vehicles
        arrived = [v for v in vehicles if v.has_arrived()]
        
        with col1:
            st.metric("Total Vehículos", len(vehicles))
        with col2:
            st.metric("Llegaron", len(arrived), 
                     delta=f"{len(arrived)/len(vehicles)*100:.1f}%")
        with col3:
            avg_time = np.mean([v.travel_time for v in arrived]) if arrived else 0
            st.metric("Tiempo Promedio", f"{avg_time:.2f}")
        with col4:
            st.metric("Iteraciones", optimizer.iteration)
        
        # Tabla de vehículos
        st.subheader("📋 Detalle de Vehículos")
        vehicle_data = []
        for i, v in enumerate(vehicles):
            vehicle_data.append({
                "Vehículo": f"V{i+1}",
                "Inicio": v.start,
                "Destino": v.end,
                "Estado": "✅ Llegó" if v.arrived else "❌ No llegó",
                "Nodos": len(v.path),
                "Tiempo": f"{v.travel_time:.2f}"
            })
        
        df = pd.DataFrame(vehicle_data)
        st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Carga un mapa desde el panel izquierdo para comenzar")
    
    # Mostrar ejemplo
    st.markdown("""
    ### 📖 Guía Rápida
    
    1. **🗺️ Cargar Mapa**: Elige un lugar por nombre o coordenadas
    2. **🚗 Añadir Vehículos**: Define rutas de inicio a destino
    3. **🚧 Obstáculos (Opcional)**: Bloquea calles o añade tráfico
    4. **▶️ Simular**: Ejecuta el algoritmo ACO
    5. **📊 Analizar**: Visualiza rutas y estadísticas
    
    ### 🎯 Ejemplos de Lugares
    - Trujillo, Peru
    - Lima, Peru  
    - Arequipa, Peru
    - San Isidro, Lima, Peru
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    🐜 <b>Sistema ACO - Optimización de Tráfico</b> | 
    Powered by Folium + Streamlit | 
    Versión 2.0
</div>
""", unsafe_allow_html=True)