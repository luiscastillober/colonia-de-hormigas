import random
from typing import List, Dict, Tuple, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import os
import numpy as np

try:
    import osmnx as ox
    import networkx as nx

    OSMNX_AVAILABLE = True
except ImportError:
    OSMNX_AVAILABLE = False


class EnhancedCityMap:
    """Versión MEJORADA del mapa con mejores visualizaciones y controles"""

    def __init__(self):
        self.intersections = {}
        self.roads = {}
        self.one_way_streets = set()
        self.blocked_roads = set()
        self.high_traffic_roads = set()  # Nuevo: calles con alto tráfico
        self.graph = None
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.selected_nodes = set()  # Para selección visual
        self.hover_node = None  # Nodo bajo el mouse

    def load_city_from_osm(self, place_name: str, network_type: str = 'drive'):
        """Cargar mapa con mejor procesamiento"""
        if not OSMNX_AVAILABLE:
            raise ImportError("OSMnx no está instalado")

        try:
            # Método más robusto para cargar
            try:
                self.graph = ox.graph_from_place(
                    place_name,
                    network_type=network_type,
                    simplify=True,
                    truncate_by_edge=True
                )
            except Exception as e:
                print(f"Intento 1 falló: {e}, intentando método alternativo...")
                self.graph = ox.graph_from_address(
                    place_name,
                    network_type=network_type,
                    dist=2000,
                    simplify=True
                )

            self._process_enhanced_graph()
            return self

        except Exception as e:
            raise Exception(f"No se pudo cargar el mapa: {e}")

    def load_city_from_point(self, lat: float, lon: float, dist: int = 1000):
        """Cargar mapa desde coordenadas específicas - MÉTODO NUEVO"""
        if not OSMNX_AVAILABLE:
            raise ImportError("OSMnx no está disponible")

        try:
            print(f"📍 Cargando mapa alrededor de ({lat}, {lon})...")

            self.graph = ox.graph_from_point(
                (lat, lon),
                dist=dist,
                network_type='drive',
                simplify=True
            )
            self._process_enhanced_graph()
            return self
        except Exception as e:
            raise Exception(f"No se pudo cargar el mapa desde coordenadas: {str(e)}")

    def _process_enhanced_graph(self):
        """Procesamiento MEJORADO del grafo"""
        if self.graph is None:
            return

        # Limpiar datos
        self.intersections.clear()
        self.roads.clear()
        self.one_way_streets.clear()
        self.node_mapping.clear()
        self.reverse_mapping.clear()

        # Mapear nodos con MÁS INFORMACIÓN
        osm_nodes = list(self.graph.nodes())
        for idx, osm_id in enumerate(osm_nodes):
            self.node_mapping[osm_id] = idx
            self.reverse_mapping[idx] = osm_id
            node_data = self.graph.nodes[osm_id]
            x, y = node_data['x'], node_data['y']

            # Añadir más datos a las intersecciones
            self.intersections[idx] = {
                'coords': (x, y),
                'osm_id': osm_id,
                'degree': self.graph.degree(osm_id),  # Número de conexiones
                'is_important': self.graph.degree(osm_id) > 2  # Nodo importante
            }

        # Procesar calles con MÁS METADATOS
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            start_id = self.node_mapping[u]
            end_id = self.node_mapping[v]

            length = data.get('length', 50.0)
            oneway = data.get('oneway', False)
            highway_type = data.get('highway', 'unclassified')
            name = data.get('name', 'Sin nombre')

            # Calcular importancia de la calle
            importance = self._calculate_road_importance(highway_type, length)

            road_data = {
                'distance': length / 1000,  # km
                'pheromone': 1.0,
                'traffic': 1.0,
                'base_travel_time': length / 1000,
                'osm_data': data,
                'highway_type': highway_type,
                'name': name,
                'importance': importance,
                'is_major_road': importance > 2
            }

            self.roads[(start_id, end_id)] = road_data

            if oneway:
                self.one_way_streets.add((start_id, end_id))
            else:
                # Crear calle bidireccional
                self.roads[(end_id, start_id)] = road_data

    def _calculate_road_importance(self, highway_type: str, length: float) -> int:
        """Calcular importancia de la calle para visualización"""
        importance_map = {
            'motorway': 5, 'motorway_link': 4,
            'trunk': 4, 'trunk_link': 3,
            'primary': 3, 'primary_link': 3,
            'secondary': 2, 'secondary_link': 2,
            'tertiary': 1, 'tertiary_link': 1,
            'residential': 0, 'unclassified': 0
        }
        return importance_map.get(highway_type, 0)

    def block_road_between_nodes(self, node1: int, node2: int):
        """Bloquear calle entre dos nodos (ambas direcciones)"""
        self.blocked_roads.add((node1, node2))
        self.blocked_roads.add((node2, node1))

    def unblock_road_between_nodes(self, node1: int, node2: int):
        """Desbloquear calle entre dos nodos"""
        self.blocked_roads.discard((node1, node2))
        self.blocked_roads.discard((node2, node1))

    def add_traffic_to_area(self, center_node: int, radius: int = 3, traffic_factor: float = 3.0):
        """Añadir tráfico a un ÁREA completa, no solo una calle"""
        nodes_in_area = self._get_nodes_in_radius(center_node, radius)

        for node in nodes_in_area:
            neighbors = self.get_neighbors(node)
            for neighbor in neighbors:
                if (node, neighbor) in self.roads:
                    self.roads[(node, neighbor)]['traffic'] = traffic_factor
                    self.high_traffic_roads.add((node, neighbor))

    def _get_nodes_in_radius(self, center_node: int, radius: int) -> Set[int]:
        """Obtener todos los nodos dentro de un radio"""
        if center_node not in self.intersections:
            return set()

        visited = set()
        queue = [(center_node, 0)]

        while queue:
            current_node, current_radius = queue.pop(0)
            if current_node in visited or current_radius > radius:
                continue

            visited.add(current_node)

            # Añadir vecinos
            neighbors = self.get_neighbors(current_node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    queue.append((neighbor, current_radius + 1))

        return visited

    def find_closest_node(self, lat: float, lon: float) -> Optional[int]:
        """Encontrar el nodo más cercano a unas coordenadas"""
        if not self.intersections:
            return None

        min_distance = float('inf')
        closest_node = None

        for node_id, node_data in self.intersections.items():
            node_lon, node_lat = node_data['coords']
            distance = self._haversine_distance(lon, lat, node_lon, node_lat)

            if distance < min_distance:
                min_distance = distance
                closest_node = node_id

        return closest_node

    def _haversine_distance(self, lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calcular distancia haversine entre dos puntos"""
        from math import radians, sin, cos, sqrt, atan2  # IMPORTS AÑADIDOS
        R = 6371000  # Radio de la Tierra en metros

        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad

        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def get_road_between_nodes(self, node1: int, node2: int) -> Optional[Dict]:
        """Obtener datos de la calle entre dos nodos"""
        return self.roads.get((node1, node2)) or self.roads.get((node2, node1))

    def get_node_info(self, node_id: int) -> Dict:
        """Obtener información completa de un nodo"""
        if node_id not in self.intersections:
            return {}

        node_data = self.intersections[node_id]
        neighbors = self.get_neighbors(node_id)
        roads_info = []

        for neighbor in neighbors:
            road = self.get_road_between_nodes(node_id, neighbor)
            if road:
                roads_info.append({
                    'to_node': neighbor,
                    'name': road.get('name', 'Sin nombre'),
                    'type': road.get('highway_type', 'unknown'),
                    'distance': road.get('distance', 0),
                    'traffic': road.get('traffic', 1.0)
                })

        return {
            'node_id': node_id,
            'coords': node_data['coords'],
            'degree': node_data['degree'],
            'is_important': node_data['is_important'],
            'connected_roads': roads_info
        }

    def visualize_enhanced_map(self,
                               vehicles: List = None,
                               title: str = "Mapa de Tráfico Mejorado",
                               show_ids: bool = True,
                               highlight_nodes: List[int] = None,
                               show_road_names: bool = False,
                               figsize: Tuple[int, int] = (14, 12)):
        """Visualización MEJORADA del mapa"""
        if not self.graph:
            return None

        try:
            fig, ax = plt.subplots(figsize=figsize)

            # Configurar estilo profesional
            plt.style.use('default')
            ax.set_facecolor('#f8f9fa')

            # 1. DIBUJAR CALLES POR IMPORTANCIA
            self._draw_streets_by_importance(ax)

            # 2. DIBUJAR CALLES BLOQUEADAS
            self._draw_blocked_streets(ax)

            # 3. DIBUJAR CALLES CON TRÁFICO
            self._draw_traffic_streets(ax)

            # 4. DIBUJAR NODOS MEJORADOS
            self._draw_enhanced_nodes(ax, show_ids, highlight_nodes)

            # 5. DIBUJAR RUTAS DE VEHÍCULOS
            if vehicles:
                self._draw_vehicle_routes(ax, vehicles)

            # 6. DIBUJAR NOMBRES DE CALLES (opcional)
            if show_road_names:
                self._draw_road_names(ax)

            # Configurar título y leyenda
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            self._create_legend(ax)

            # Mejorar aspecto general
            ax.tick_params(axis='both', which='major', labelsize=10)
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Error en visualización mejorada: {e}")
            return None

    def _draw_streets_by_importance(self, ax):
        """Dibujar calles clasificadas por importancia"""
        if not self.graph:
            return

        # Dibujar todas las calles base en gris claro
        ox.plot_graph(self.graph, ax=ax, node_size=0, edge_linewidth=0.8,
                      edge_color='lightgray', show=False, close=False)

    def _get_highway_color(self, highway_type: str) -> str:
        """Obtener color según tipo de calle"""
        color_map = {
            'motorway': '#FF6B6B',
            'trunk': '#FFA726',
            'primary': '#4ECDC4',
            'secondary': '#45B7D1',
            'tertiary': '#96CEB4'
        }
        return color_map.get(highway_type, '#CCCCCC')

    def _draw_blocked_streets(self, ax):
        """Dibujar calles bloqueadas"""
        for (start, end) in self.blocked_roads:
            if start in self.reverse_mapping and end in self.reverse_mapping:
                try:
                    start_osm = self.reverse_mapping[start]
                    end_osm = self.reverse_mapping[end]
                    x1, y1 = self.graph.nodes[start_osm]['x'], self.graph.nodes[start_osm]['y']
                    x2, y2 = self.graph.nodes[end_osm]['x'], self.graph.nodes[end_osm]['y']

                    # Línea roja gruesa con patrón
                    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=4, alpha=0.7,
                            dash_capstyle='round', zorder=3,
                            label='Bloqueada' if start == list(self.blocked_roads)[0][0] else "")
                except KeyError:
                    continue

    def _draw_traffic_streets(self, ax):
        """Dibujar calles con tráfico"""
        for (start, end) in self.high_traffic_roads:
            if start in self.reverse_mapping and end in self.reverse_mapping:
                try:
                    start_osm = self.reverse_mapping[start]
                    end_osm = self.reverse_mapping[end]
                    x1, y1 = self.graph.nodes[start_osm]['x'], self.graph.nodes[start_osm]['y']
                    x2, y2 = self.graph.nodes[end_osm]['x'], self.graph.nodes[end_osm]['y']

                    # Línea naranja para tráfico
                    ax.plot([x1, x2], [y1, y2], color='orange', linewidth=3,
                            alpha=0.6, zorder=2,
                            label='Tráfico Alto' if start == list(self.high_traffic_roads)[0][0] else "")
                except KeyError:
                    continue

    def _draw_enhanced_nodes(self, ax, show_ids: bool, highlight_nodes: List[int]):
        """Dibujar nodos con visualización mejorada"""
        if not self.intersections:
            return

        # Dibujar nodos normales
        normal_nodes = []
        important_nodes = []
        highlighted_nodes = []

        for node_id, node_data in self.intersections.items():
            x, y = node_data['coords']

            if highlight_nodes and node_id in highlight_nodes:
                highlighted_nodes.append((x, y, node_id))
            elif node_data['is_important']:
                important_nodes.append((x, y, node_id))
            else:
                normal_nodes.append((x, y, node_id))

        # Dibujar nodos normales (pequeños, grises)
        if normal_nodes:
            xs, ys, _ = zip(*normal_nodes)

        # Dibujar nodos importantes (medianos, azules)
        if important_nodes:
            xs, ys, _ = zip(*important_nodes)

        # Dibujar nodos destacados (grandes, verdes)
        if highlighted_nodes:
            xs, ys, node_ids = zip(*highlighted_nodes)

            # Mostrar IDs de nodos destacados SIEMPRE
            for x, y, node_id in highlighted_nodes:
                ax.annotate(str(node_id), (x, y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lime", alpha=0.8),
                            zorder=7)

        # Mostrar IDs de nodos importantes si está activado
        if show_ids and important_nodes:
            for x, y, node_id in important_nodes[:50]:  # Limitar para rendimiento
                ax.annotate(str(node_id), (x, y), xytext=(2, 2),
                            textcoords='offset points', fontsize=6, alpha=0.7,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5),
                            zorder=5)

    def _draw_vehicle_routes(self, ax, vehicles: List):
        """Dibujar rutas de vehículos mejoradas"""
        colors = plt.cm.Set3(np.linspace(0, 1, len(vehicles)))

        for idx, vehicle in enumerate(vehicles):
            color = colors[idx]

            if len(vehicle.path) > 1:
                path_coords = []
                for node_id in vehicle.path:
                    if node_id in self.reverse_mapping:
                        try:
                            osm_id = self.reverse_mapping[node_id]
                            x = self.graph.nodes[osm_id]['x']
                            y = self.graph.nodes[osm_id]['y']
                            path_coords.append((x, y))
                        except KeyError:
                            continue

                if len(path_coords) > 1:
                    xs, ys = zip(*path_coords)

                    # Línea principal
                    ax.plot(xs, ys, color=color, linewidth=3, alpha=0.8,
                            label=f'Vehículo {idx + 1}', zorder=4)

                    # Puntos en cada nodo de la ruta
                    ax.scatter(xs, ys, color=color, s=30, alpha=0.9, zorder=5)

                    # Flecha indicando dirección
                    if len(xs) >= 2:
                        mid_idx = len(xs) // 2
                        dx = xs[mid_idx + 1] - xs[mid_idx]
                        dy = ys[mid_idx + 1] - ys[mid_idx]
                        ax.arrow(xs[mid_idx], ys[mid_idx], dx * 0.3, dy * 0.3,
                                 head_width=0.0002, head_length=0.0003,
                                 fc=color, ec=color, alpha=0.8, zorder=6)

    def _draw_road_names(self, ax):
        """Dibujar nombres de calles importantes"""
        drawn_names = set()

        for (start, end), road_data in self.roads.items():
            if road_data.get('is_major_road', False) and road_data.get('name') and road_data['name'] != 'Sin nombre':
                if start in self.reverse_mapping and end in self.reverse_mapping:
                    try:
                        start_osm = self.reverse_mapping[start]
                        end_osm = self.reverse_mapping[end]
                        x1, y1 = self.graph.nodes[start_osm]['x'], self.graph.nodes[start_osm]['y']
                        x2, y2 = self.graph.nodes[end_osm]['x'], self.graph.nodes[end_osm]['y']

                        # Posición media para el texto
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

                        name = road_data['name']
                        if name not in drawn_names:
                            ax.text(mid_x, mid_y, name, fontsize=7, alpha=0.7,
                                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8),
                                    ha='center', va='center', rotation=45, zorder=3)
                            drawn_names.add(name)
                    except KeyError:
                        continue

    def _create_legend(self, ax):
        """Crear leyenda mejorada"""
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=4, label='Calle Bloqueada'),
            Line2D([0], [0], color='orange', linewidth=3, label='Tráfico Alto'),
        ]

        ax.legend(handles=legend_elements, loc='upper right',
                  framealpha=0.9, fancybox=True, shadow=True)

    # ================= MÉTODOS DE COMPATIBILIDAD =================

    def block_road(self, start: int, end: int):
        """Mantener compatibilidad con código existente"""
        self.blocked_roads.add((start, end))

    def add_traffic_to_road(self, start: int, end: int, traffic_factor: float = 2.0):
        """Mantener compatibilidad con código existente"""
        if (start, end) in self.roads:
            self.roads[(start, end)]['traffic'] = traffic_factor
            self.high_traffic_roads.add((start, end))

    def visualize_on_map(self, vehicles=None, title="Mapa de Tráfico", show_ids=False):
        """Mantener compatibilidad con llamadas existentes"""
        return self.visualize_enhanced_map(vehicles, title, show_ids)

    # ================= MÉTODOS BÁSICOS =================

    def get_neighbors(self, intersection: int) -> List[int]:
        neighbors = []
        for (start, end) in self.roads:
            if start == intersection and (start, end) not in self.blocked_roads:
                neighbors.append(end)
        return list(set(neighbors))

    def get_random_intersections(self, count: int = 2) -> List[int]:
        if not self.intersections:
            return []
        return random.sample(list(self.intersections.keys()), min(count, len(self.intersections)))

    def get_road_travel_time(self, start: int, end: int, current_time: float = 0) -> float:
        if (start, end) not in self.roads:
            return float('inf')
        if (start, end) in self.blocked_roads:
            return float('inf')

        road_data = self.roads[(start, end)]
        base_time = road_data['base_travel_time']
        traffic_factor = road_data['traffic']

        return base_time * traffic_factor

    def visualize_interactive_map(self,
                                  vehicles: List = None,
                                  title: str = "Mapa Interactivo - Selecciona Nodos",
                                  selected_nodes: Dict[str, int] = None,
                                  show_road_names: bool = False,
                                  figsize: Tuple[int, int] = (14, 12)):
        """Visualización INTERACTIVA para seleccionar nodos del mapa"""
        if not self.graph:
            return None

        try:
            fig, ax = plt.subplots(figsize=figsize)
            plt.style.use('default')
            ax.set_facecolor('#f8f9fa')

            # 1. DIBUJAR CALLES TODAS EN GRIS
            ox.plot_graph(self.graph, ax=ax, node_size=0, edge_linewidth=1.0,
                          edge_color='lightgray', show=False, close=False)

            # 2. DIBUJAR CALLES BLOQUEADAS
            self._draw_blocked_streets(ax)

            # 3. DIBUJAR CALLES CON TRÁFICO
            self._draw_traffic_streets(ax)

            # 4. DIBUJAR RUTAS DE VEHÍCULOS
            if vehicles:
                self._draw_vehicle_routes(ax, vehicles)

            # 5. DIBUJAR NODOS SELECCIONADOS
            if selected_nodes:
                self._draw_selected_nodes(ax, selected_nodes)

            # 6. DIBUJAR TODOS LOS NODOS COMO PUNTOS PEQUEÑOS PARA SELECCIÓN
            self._draw_all_nodes_for_selection(ax)

            # 7. DIBUJAR NOMBRES DE CALLES (opcional)
            if show_road_names:
                self._draw_road_names(ax)

            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            self._create_interactive_legend(ax, selected_nodes)
            ax.tick_params(axis='both', which='major', labelsize=10)
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Error en visualización interactiva: {e}")
            return None

    def _draw_selected_nodes(self, ax, selected_nodes: Dict[str, int]):
        """Dibujar nodos seleccionados con colores específicos"""
        node_colors = {
            'start': 'lime',
            'end': 'red',
            'blocked': 'darkred',
            'traffic': 'orange'
        }

        node_labels = {
            'start': 'Inicio',
            'end': 'Destino',
            'blocked': 'Bloqueado',
            'traffic': 'Tráfico'
        }

        for node_type, node_id in selected_nodes.items():
            if node_id in self.intersections:
                node_data = self.intersections[node_id]
                x, y = node_data['coords']
                color = node_colors.get(node_type, 'blue')
                label = node_labels.get(node_type, 'Seleccionado')

                ax.scatter(x, y, color=color, s=150, alpha=1.0, zorder=20,
                           edgecolors='black', linewidth=2.5, marker='o' if node_type in ['start', 'end'] else 's')

                # Etiqueta del nodo
                ax.annotate(f"{label}\nID: {node_id}", (x, y), xytext=(10, 10),
                            textcoords='offset points', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                            zorder=21)

    def _draw_all_nodes_for_selection(self, ax):
        """Dibujar todos los nodos como puntos pequeños para selección"""
        all_coords = []
        all_ids = []

        for node_id, node_data in self.intersections.items():
            x, y = node_data['coords']
            all_coords.append((x, y))
            all_ids.append(node_id)

        if all_coords:
            xs, ys = zip(*all_coords)
            # Puntos muy pequeños y transparentes, pero clickeables
            ax.scatter(xs, ys, color='blue', s=8, alpha=0.1, zorder=5)

    def _create_interactive_legend(self, ax, selected_nodes: Dict[str, int]):
        """Crear leyenda para mapa interactivo"""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        legend_elements = [
            Line2D([0], [0], color='red', linewidth=4, label='Calle Bloqueada'),
            Line2D([0], [0], color='orange', linewidth=3, label='Tráfico Alto'),
            Line2D([0], [0], color='lime', marker='o', markersize=10, linewidth=0, label='Inicio'),
            Line2D([0], [0], color='red', marker='o', markersize=10, linewidth=0, label='Destino'),
            Line2D([0], [0], color='darkred', marker='s', markersize=8, linewidth=0, label='Nodo Bloqueado'),
            Line2D([0], [0], color='orange', marker='s', markersize=8, linewidth=0, label='Nodo Tráfico'),
        ]

        ax.legend(handles=legend_elements, loc='upper right',
                  framealpha=0.95, fancybox=True, shadow=True, fontsize=9)

    def find_node_by_coords(self, x: float, y: float, tolerance: float = 0.001) -> Optional[int]:
        """Encontrar nodo más cercano a coordenadas del click"""
        if not self.intersections:
            return None

        min_distance = float('inf')
        closest_node = None

        for node_id, node_data in self.intersections.items():
            node_x, node_y = node_data['coords']
            distance = ((node_x - x) ** 2 + (node_y - y) ** 2) ** 0.5

            if distance < min_distance and distance < tolerance:
                min_distance = distance
                closest_node = node_id

        return closest_node