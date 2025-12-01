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
    """Versión MEJORADA del mapa con corrección de errores de hashable"""

    def __init__(self):
        self.intersections = {}
        self.roads = {}
        self.one_way_streets = set()
        self.blocked_roads = set()
        self.high_traffic_roads = set()
        self.graph = None
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.selected_nodes = set()
        self.hover_node = None

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
        """Cargar mapa desde coordenadas específicas"""
        if not OSMNX_AVAILABLE:
            raise ImportError("OSMnx no está disponible")

        try:
            print(f"🔍 Cargando mapa alrededor de ({lat}, {lon})...")

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
        """Procesamiento MEJORADO del grafo con corrección de tipos"""
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
                'degree': self.graph.degree(osm_id),
                'is_important': self.graph.degree(osm_id) > 2
            }

        # Procesar calles con MÁS METADATOS y CORRECCIÓN DE TIPOS
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            start_id = self.node_mapping[u]
            end_id = self.node_mapping[v]

            length = data.get('length', 50.0)
            oneway = data.get('oneway', False)
            
            # ✅ CORRECCIÓN: Manejar highway_type que puede ser lista
            highway_type = data.get('highway', 'unclassified')
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else 'unclassified'
            
            # ✅ CORRECCIÓN: Manejar name que puede ser lista
            name = data.get('name', 'Sin nombre')
            if isinstance(name, list):
                name = ', '.join(str(n) for n in name) if name else 'Sin nombre'
            elif not isinstance(name, str):
                name = str(name) if name else 'Sin nombre'

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
                self.roads[(end_id, start_id)] = road_data.copy()

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
        from math import radians, sin, cos, sqrt, atan2
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

    # MÉTODOS DE COMPATIBILIDAD
    def block_road(self, start: int, end: int):
        """Mantener compatibilidad con código existente"""
        self.blocked_roads.add((start, end))

    def add_traffic_to_road(self, start: int, end: int, traffic_factor: float = 2.0):
        """Mantener compatibilidad con código existente"""
        if (start, end) in self.roads:
            self.roads[(start, end)]['traffic'] = traffic_factor
            self.high_traffic_roads.add((start, end))