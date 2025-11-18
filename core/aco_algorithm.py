import time
import random
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
from enum import Enum

# PARÁMETROS GLOBALES DEL ALGORITMO ACO
ACO_PARAMS = {
    'max_steps': 1000,
    'max_stuck_count': 20,
    'alpha': 1.0,  # Peso de las feromonas
    'beta': 2.0,  # Peso de la heurística
    'evaporation_rate': 0.5,
    'Q': 100.0,  # Constante de depósito de feromonas
    'min_pheromone': 0.1,
    'max_pheromone': 100.0,
    'ant_count': 10
}


class VehicleState(Enum):
    EXPLORING = "exploring"
    FOLLOWING_PHEROMONE = "following_pheromone"
    BACKTRACKING = "backtracking"
    ARRIVED = "arrived"
    STUCK = "stuck"


class ACOVehicle:
    """Versión MEJORADA del vehículo con inteligencia avanzada"""

    def __init__(self, vehicle_id: int, start: int, end: int, city_map):
        self.id = vehicle_id
        self.start = start
        self.end = end
        self.current = start
        self.city_map = city_map
        self.path = [start]
        self.travel_time = 0.0
        self.visited = set([start])
        self.arrived = False
        self.stuck_count = 0
        self.state = VehicleState.EXPLORING
        self.memory = []  # Memoria de decisiones recientes
        self.exploration_factor = 1.0  # Factor de exploración dinámico

        # Parámetros desde config
        self.max_steps = ACO_PARAMS['max_steps']
        self.max_stuck_count = ACO_PARAMS['max_stuck_count']
        self.exploration_decay = 0.95  # Decaimiento de exploración

    def visualize_interactive_map(self,
                                  vehicles: List = None,
                                  title: str = "Mapa Interactivo - Haz click en los nodos",
                                  selected_nodes: Dict[str, int] = None,
                                  show_road_names: bool = False,
                                  figsize: Tuple[int, int] = (14, 12),
                                  on_node_click: callable = None):
        """Visualización INTERACTIVA con detección de clicks en nodos"""
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

            # 5. DIBUJAR TODOS LOS NODOS COMO PUNTOS CLICKEABLES
            clickable_nodes = self._draw_clickable_nodes(ax, selected_nodes)

            # 6. DIBUJAR NOMBRES DE CALLES (opcional)
            if show_road_names:
                self._draw_road_names(ax)

            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            self._create_interactive_legend(ax, selected_nodes)
            ax.tick_params(axis='both', which='major', labelsize=10)

            # Configurar el mapa para interactividad
            if on_node_click and clickable_nodes:
                self._setup_click_detection(fig, ax, clickable_nodes, on_node_click)

            plt.tight_layout()
            return fig

        except Exception as e:
            print(f"Error en visualización interactiva: {e}")
            return None

    def _draw_clickable_nodes(self, ax, selected_nodes: Dict[str, int]):
        """Dibujar todos los nodos como puntos clickeables"""
        all_nodes = []
        node_coords = []
        node_ids = []

        for node_id, node_data in self.intersections.items():
            x, y = node_data['coords']
            all_nodes.append((x, y, node_id))
            node_coords.append((x, y))
            node_ids.append(node_id)

        if node_coords:
            xs, ys = zip(*node_coords)
            # Puntos semi-transparentes pero clickeables
            scatter = ax.scatter(xs, ys, color='blue', s=50, alpha=0.3, zorder=10,
                                 picker=True, pickradius=5)  # picker habilita la detección de clicks

            # Dibujar nodos seleccionados encima
            if selected_nodes:
                self._draw_selected_nodes(ax, selected_nodes)

            return list(zip(node_ids, xs, ys))
        return []

    def _draw_selected_nodes(self, ax, selected_nodes: Dict[str, int]):
        """Dibujar nodos seleccionados con colores específicos"""
        node_colors = {
            'start': 'lime',
            'end': 'red',
            'blocked': 'darkred',
            'traffic': 'orange',
            'current_start': 'lightgreen',
            'current_end': 'lightcoral'
        }

        node_labels = {
            'start': 'Inicio',
            'end': 'Destino',
            'blocked': 'Bloqueado',
            'traffic': 'Tráfico',
            'current_start': 'Inicio (seleccionado)',
            'current_end': 'Destino (seleccionado)'
        }

        for node_type, node_id in selected_nodes.items():
            if node_id in self.intersections:
                node_data = self.intersections[node_id]
                x, y = node_data['coords']
                color = node_colors.get(node_type, 'blue')
                label = node_labels.get(node_type, 'Seleccionado')

                # Tamaño diferente según el tipo
                size = 120 if node_type in ['start', 'end'] else 100

                ax.scatter(x, y, color=color, s=size, alpha=1.0, zorder=20,
                           edgecolors='black', linewidth=2.5,
                           marker='o' if 'start' in node_type else 's')

                # Etiqueta del nodo
                ax.annotate(f"{label}\nID: {node_id}", (x, y), xytext=(10, 10),
                            textcoords='offset points', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                            zorder=21)

    def _setup_click_detection(self, fig, ax, clickable_nodes, on_node_click):
        """Configurar detección de clicks en los nodos"""

        def on_pick(event):
            if event.artist != ax.collections[0]:  # Solo los puntos de nodos
                return

            # Obtener el índice del punto clickeado
            ind = event.ind[0]
            if ind < len(clickable_nodes):
                node_id, x, y = clickable_nodes[ind]
                on_node_click(node_id, x, y)

        # Conectar el evento de pick
        fig.canvas.mpl_connect('pick_event', on_pick)

    def get_node_at_coordinates(self, x: float, y: float, tolerance: float = 0.001) -> Optional[int]:
        """Encontrar nodo más cercano a coordenadas del click"""
        if not self.intersections:
            return None

        min_distance = float('inf')
        closest_node = None

        for node_id, node_data in self.intersections.items():
            node_x, node_y = node_data['coords']
            distance = ((node_x - x) ** 2 + (node_y - y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_node = node_id

        # Solo retornar si está dentro de la tolerancia
        return closest_node if min_distance < tolerance else None

    def _escape_strategy(self) -> int:
        """Estrategia de escape MEJORADA para bucles"""
        neighbors = self.city_map.get_neighbors(self.current)
        if not neighbors:
            return self.current

        # BUSCAR NODO QUE NO ESTÉ EN EL HISTORIAL RECIENTE
        recent_path = set(self.path[-20:])  # Últimos 20 nodos visitados

        # Priorizar nodos no visitados recientemente
        best_neighbor = None
        best_score = -float('inf')

        for neighbor in neighbors:
            # Penalizar nodos visitados recientemente
            recent_penalty = 10.0 if neighbor in recent_path else 0.0

            # Score basado en heurística y novedad
            heuristic_score = self._calculate_detailed_heuristic(neighbor)
            novelty_bonus = 5.0 if neighbor not in self.visited else 0.0
            road_quality = self.city_map.roads.get((self.current, neighbor), {}).get('pheromone', 0.1)

            total_score = heuristic_score + novelty_bonus + road_quality - recent_penalty

            if total_score > best_score:
                best_score = total_score
                best_neighbor = neighbor

        # RESETEAR COMPLETAMENTE si encontramos un buen candidato
        if best_neighbor and best_neighbor not in recent_path:
            self.visited.clear()
            self.visited.add(self.current)
            self.visited.add(best_neighbor)
            self.stuck_count = 0
            self.state = VehicleState.EXPLORING
            self.exploration_factor = 1.0  # Resetear exploración

        return best_neighbor if best_neighbor else random.choice(neighbors)

    def move_to(self, intersection: int):
        """Movimiento MEJORADO con prevención de bucles"""
        if self.current == intersection:
            return

        travel_time = self._get_road_travel_time(self.current, intersection, self.travel_time)

        # PREVENIR travel_time INFINITO
        if travel_time == float('inf'):
            travel_time = 100.0  # Valor grande pero finito

        self.travel_time += travel_time
        self.current = intersection
        self.path.append(intersection)
        self.visited.add(intersection)

        # DETECCIÓN MEJORADA DE BUCLE
        if len(self.path) > 10:
            recent_nodes = self.path[-10:]
            if len(set(recent_nodes)) < len(recent_nodes) * 0.4:  # Más del 60% de repetición
                self.stuck_count += 2
            elif self._is_in_local_loop(intersection):
                self.stuck_count += 1
            else:
                self.stuck_count = max(0, self.stuck_count - 0.5)

        if self.current == self.end:
            self.arrived = True
            self.state = VehicleState.ARRIVED
            self._reinforce_path()

        # FORZAR ESCAPE si está muy atascado
        if self.stuck_count > self.max_stuck_count:
            self.state = VehicleState.STUCK

    def _update_vehicle_state(self):
        """Actualizar estado del vehículo basado en condiciones"""
        if len(self.path) > self.max_steps * 0.8:
            self.state = VehicleState.STUCK
        elif self.stuck_count > self.max_stuck_count // 2:
            self.state = VehicleState.BACKTRACKING
        elif len(self.path) > 10 and self._is_making_progress():
            self.state = VehicleState.FOLLOWING_PHEROMONE
        else:
            self.state = VehicleState.EXPLORING

    def _is_making_progress(self) -> bool:
        """Verificar si el vehículo está avanzando hacia el destino"""
        if len(self.path) < 5:
            return True

        # Calcular progreso basado en distancia al destino
        current_heuristic = self._calculate_detailed_heuristic(self.current)
        prev_heuristic = self._calculate_detailed_heuristic(self.path[-5])

        return current_heuristic > prev_heuristic * 0.9  # 10% de mejora

    def _exploration_strategy(self, alpha: float, beta: float) -> int:
        """Estrategia de exploración balanceada"""
        neighbors = self.city_map.get_neighbors(self.current)
        valid_neighbors = [n for n in neighbors if n not in self.visited]

        if not valid_neighbors:
            return self._handle_no_valid_neighbors()

        # Aplicar factor de exploración
        exploration_bonus = self.exploration_factor

        probabilities = []
        for neighbor in valid_neighbors:
            base_prob = self._calculate_move_probability(neighbor, alpha, beta)
            # Añadir bonus de exploración para nodos menos visitados
            exploration_prob = base_prob * (1 + exploration_bonus * random.random())
            probabilities.append(exploration_prob)

        total = sum(probabilities)
        if total == 0:
            chosen = random.choice(valid_neighbors)
        else:
            probabilities = [p / total for p in probabilities]
            chosen = random.choices(valid_neighbors, weights=probabilities)[0]

        # Actualizar memoria y factor de exploración
        self.memory.append(chosen)
        if len(self.memory) > 10:
            self.memory.pop(0)

        self.exploration_factor *= self.exploration_decay

        return chosen

    def _pheromone_follow_strategy(self, alpha: float, beta: float) -> int:
        """Estrategia de seguir feromonas intensamente"""
        neighbors = self.city_map.get_neighbors(self.current)
        valid_neighbors = [n for n in neighbors if n not in self.visited]

        if not valid_neighbors:
            return self._handle_no_valid_neighbors()

        # Solo considerar los mejores candidatos basados en feromonas
        candidate_scores = []
        for neighbor in valid_neighbors:
            road_key = (self.current, neighbor)
            if road_key in self.city_map.roads:
                pheromone = self.city_map.roads[road_key]['pheromone']
                heuristic = self._calculate_detailed_heuristic(neighbor)
                score = (pheromone ** alpha) * (heuristic ** beta)
                candidate_scores.append((neighbor, score))

        if not candidate_scores:
            return random.choice(valid_neighbors)

        # Elegir entre los top 3 candidatos
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidate_scores[:min(3, len(candidate_scores))]

        if random.random() < 0.8:  # 80% de probabilidad de elegir el mejor
            return top_candidates[0][0]
        else:
            return random.choice([c[0] for c in top_candidates])

    def _backtrack_strategy(self) -> int:
        """Estrategia de backtracking inteligente"""
        if len(self.path) <= 1:
            return self.start

        # Retroceder varios pasos, no solo uno
        backtrack_steps = min(3, len(self.path) - 1)
        backtrack_node = self.path[-backtrack_steps]

        # Limpiar visited de los nodos backtracked
        for node in self.path[-backtrack_steps:]:
            self.visited.discard(node)

        self.stuck_count = 0
        self.state = VehicleState.EXPLORING
        self.exploration_factor = 1.0  # Resetear exploración

        return backtrack_node

    def _escape_strategy(self) -> int:
        """Estrategia de escape cuando está realmente atrapado"""
        neighbors = self.city_map.get_neighbors(self.current)
        if not neighbors:
            return self.current

        # Buscar nodo que lleve en dirección general al destino
        best_neighbor = None
        best_score = -float('inf')

        for neighbor in neighbors:
            # Score basado en combinación de heurística y novedad
            heuristic_score = self._calculate_detailed_heuristic(neighbor)
            novelty_bonus = 2.0 if neighbor not in self.visited else 0.0
            road_quality = self.city_map.roads.get((self.current, neighbor), {}).get('pheromone', 0.1)

            total_score = heuristic_score + novelty_bonus + road_quality

            if total_score > best_score:
                best_score = total_score
                best_neighbor = neighbor

        # Resetear estado después de escape
        if best_neighbor:
            self.visited.clear()
            self.visited.add(self.current)
            self.visited.add(best_neighbor)
            self.stuck_count = 0
            self.state = VehicleState.EXPLORING

        return best_neighbor if best_neighbor else random.choice(neighbors)

    def _calculate_move_probability(self, neighbor: int, alpha: float, beta: float) -> float:
        """Cálculo MEJORADO de probabilidad"""
        road_key = (self.current, neighbor)

        if road_key not in self.city_map.roads:
            return 0.0

        pheromone = self.city_map.roads[road_key]['pheromone']
        travel_time = self._get_road_travel_time(self.current, neighbor, self.travel_time)
        heuristic = self._calculate_detailed_heuristic(neighbor)

        if travel_time <= 0:
            travel_time = 0.1

        # Evitar división por cero y añadir smoothing
        probability = (pheromone ** alpha) * (heuristic ** beta) / (travel_time + 0.1)

        return max(probability, 1e-6)  # Mínimo valor para evitar ceros

    def _calculate_detailed_heuristic(self, neighbor: int) -> float:
        """Heurística MEJORADA con múltiples factores"""
        try:
            if neighbor == self.end:
                return 1000.0

            # Factor 1: Distancia en línea recta (fallback)
            simple_heuristic = 1.0 / (abs(neighbor - self.end) + 1)

            # Factor 2: Distancia de red real (si está disponible)
            network_heuristic = 0.0
            if (hasattr(self.city_map, 'graph') and self.city_map.graph and
                    neighbor in self.city_map.reverse_mapping and
                    self.end in self.city_map.reverse_mapping):

                try:
                    neighbor_osm = self.city_map.reverse_mapping[neighbor]
                    end_osm = self.city_map.reverse_mapping[self.end]

                    shortest_path = nx.shortest_path_length(
                        self.city_map.graph,
                        neighbor_osm,
                        end_osm,
                        weight='length'
                    )
                    network_heuristic = 100.0 / (shortest_path + 1)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    network_heuristic = simple_heuristic * 10
            else:
                network_heuristic = simple_heuristic * 10

            # Factor 3: Calidad de la carretera
            road_quality = 1.0
            if (self.current, neighbor) in self.city_map.roads:
                road_data = self.city_map.roads[(self.current, neighbor)]
                if road_data.get('is_major_road', False):
                    road_quality = 1.5
                elif road_data.get('traffic', 1.0) > 2.0:
                    road_quality = 0.5

            # Combinar factores
            final_heuristic = network_heuristic * road_quality

            return max(final_heuristic, 0.01)  # Mínimo valor

        except Exception:
            return 0.1

    def _handle_no_valid_neighbors(self):
        """Manejo MEJORADO de sin vecinos válidos"""
        if len(self.path) > 1:
            # Backtrack inteligente: ir al nodo con más opciones
            candidates = self.path[-5:] if len(self.path) >= 5 else self.path[:-1]
            best_candidate = None
            max_options = -1

            for candidate in candidates:
                neighbors = self.city_map.get_neighbors(candidate)
                valid_options = len([n for n in neighbors if n not in self.visited])
                if valid_options > max_options:
                    max_options = valid_options
                    best_candidate = candidate

            backtrack_node = best_candidate if best_candidate else self.path[-2]
            self.visited.discard(self.current)
            return backtrack_node
        else:
            self.visited.clear()
            self.visited.add(self.start)
            return self.start

    def _is_in_local_loop(self, next_node: int) -> bool:
        """Detección MEJORADA de bucles"""
        if len(self.path) < 3:
            return False

        # Verificar patrones cíclicos
        recent_nodes = self.path[-6:]  # Ventana más grande
        if len(set(recent_nodes)) < len(recent_nodes) * 0.7:  # 30% de repetición
            return True

        # Verificar si el siguiente nodo crea un ciclo pequeño
        if next_node in self.path[-8:]:
            cycle_length = len(self.path) - self.path.index(next_node)
            if cycle_length <= 4:  # Ciclos muy pequeños
                return True

        return False

    def _get_road_travel_time(self, start: int, end: int, current_time: float) -> float:
        """Wrapper para obtener tiempo de viaje"""
        if hasattr(self.city_map, 'get_road_travel_time'):
            return self.city_map.get_road_travel_time(start, end, current_time)
        else:
            # Fallback si el método no existe
            road_key = (start, end)
            if road_key in self.city_map.roads:
                base_time = self.city_map.roads[road_key].get('base_travel_time', 1.0)
                traffic = self.city_map.roads[road_key].get('traffic', 1.0)
                return base_time * traffic
            return 1.0  # Valor por defecto

    def move_to(self, intersection: int):
        """Movimiento MEJORADO con más lógica"""
        if self.current == intersection:
            return

        travel_time = self._get_road_travel_time(self.current, intersection, self.travel_time)
        self.travel_time += travel_time
        self.current = intersection
        self.path.append(intersection)
        self.visited.add(intersection)

        # Detección de bucles durante el movimiento
        if self._is_in_local_loop(intersection):
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 0.5)  # Reducción gradual

        if self.current == self.end:
            self.arrived = True
            self.state = VehicleState.ARRIVED
            self._reinforce_path()

    def _reinforce_path(self):
        """Refuerzo MEJORADO de feromonas"""
        # Calcular calidad considerando longitud y eficiencia
        path_length = len(self.path)
        optimal_length = self._estimate_optimal_path_length()
        length_efficiency = optimal_length / max(path_length, 1)

        path_quality = (100.0 / (self.travel_time + 1)) * length_efficiency

        # Reforzar todo el camino, pero con decaimiento
        for i in range(len(self.path) - 1):
            start, end = self.path[i], self.path[i + 1]
            road_key = (start, end)
            if road_key in self.city_map.roads:
                # Decaimiento basado en posición en el camino
                decay_factor = 0.9 ** i  # Primeros segmentos más importantes
                reinforcement = path_quality * decay_factor
                self.city_map.roads[road_key]['pheromone'] += reinforcement

    def _estimate_optimal_path_length(self) -> float:
        """Estimar longitud óptima del camino"""
        try:
            if (self.start in self.city_map.reverse_mapping and
                    self.end in self.city_map.reverse_mapping):
                start_osm = self.city_map.reverse_mapping[self.start]
                end_osm = self.city_map.reverse_mapping[self.end]

                return nx.shortest_path_length(
                    self.city_map.graph,
                    start_osm,
                    end_osm,
                    weight='length'
                )
        except:
            pass

        return abs(self.end - self.start) * 10  # Fallback

    def has_arrived(self) -> bool:
        return self.arrived

    def get_path_travel_time(self) -> float:
        return self.travel_time

    def get_path_efficiency(self) -> float:
        """Calcular eficiencia del camino encontrado"""
        if not self.arrived or len(self.path) < 2:
            return 0.0

        optimal_time = self._estimate_optimal_path_length() / 50.0  # Estimación
        return optimal_time / max(self.travel_time, 0.1)
    
    def select_next_intersection(self, alpha: float, beta: float) -> int:
        """Seleccionar siguiente intersección basada en el estado del vehículo"""
        # Actualizar estado primero
        self._update_vehicle_state()
        
        if self.state == VehicleState.ARRIVED:
            return self.current  # No moverse si ya llegó
        
        elif self.state == VehicleState.STUCK:
            return self._escape_strategy()
        
        elif self.state == VehicleState.BACKTRACKING:
            return self._backtrack_strategy()
        
        elif self.state == VehicleState.FOLLOWING_PHEROMONE:
            return self._pheromone_follow_strategy(alpha, beta)
        
        else:  # EXPLORING or default
            return self._exploration_strategy(alpha, beta)


class ACOTrafficOptimizer:
    """Optimizador MEJORADO con gestión avanzada"""

    def __init__(self, city_map):
        self.city_map = city_map
        self.vehicles = []
        self.iteration = 0
        self.best_paths = {}
        self.convergence_count = 0
        self.global_best_time = float('inf')
        self.adaptive_params = ACO_PARAMS.copy()

        # Estadísticas
        self.iteration_stats = {
            'arrived_vehicles': [],
            'average_travel_time': [],
            'convergence_rate': []
        }

    def add_vehicle(self, start: int, end: int):
        vehicle_id = len(self.vehicles)
        vehicle = ACOVehicle(vehicle_id, start, end, self.city_map)
        self.vehicles.append(vehicle)

        # Inicialización más agresiva de feromonas
        initial_path = self._compute_initial_path(start, end)
        if initial_path:
            self._initialize_pheromones(initial_path, strength=100.0)  # Más fuerte

        return vehicle

    def run_iteration(self) -> Dict:
        """Ejecutar iteración MEJORADA que no para hasta que todos lleguen"""
        self.iteration += 1

        # Mover todos los vehículos
        arrived_count = 0
        total_travel_time = 0.0

        for vehicle in self.vehicles:
            if not vehicle.has_arrived():
                next_intersection = vehicle.select_next_intersection(
                    self.adaptive_params['alpha'],
                    self.adaptive_params['beta']
                )
                vehicle.move_to(next_intersection)
            else:
                arrived_count += 1
                total_travel_time += vehicle.get_path_travel_time()

        # Actualizar feromonas
        self.update_pheromones()
        self.evaporate_pheromones()

        # Actualizar parámetros adaptativos
        self._update_adaptive_parameters(arrived_count)

        # Reforzar vehículos atascados periódicamente
        if self.iteration % 5 == 0:
            self._assist_stuck_vehicles()

        # Guardar estadísticas
        avg_time = total_travel_time / max(arrived_count, 1)
        self.iteration_stats['arrived_vehicles'].append(arrived_count)
        self.iteration_stats['average_travel_time'].append(avg_time)

        convergence_rate = arrived_count / len(self.vehicles)
        self.iteration_stats['convergence_rate'].append(convergence_rate)

        return {
            'arrived_count': arrived_count,
            'total_vehicles': len(self.vehicles),
            'convergence_rate': convergence_rate,
            'average_time': avg_time,
            'iteration': self.iteration
        }

    def run_until_all_arrive(self, max_iterations: int = 200) -> Dict:
        """Ejecutar hasta que TODOS los vehículos lleguen o máximo de iteraciones - VERSIÓN CORREGIDA"""
        import streamlit as st

        start_time = time.time()

        # Elementos de UI de Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()

        for i in range(max_iterations):
            result = self.run_iteration()

            # Actualizar UI
            progress = (i + 1) / max_iterations
            progress_bar.progress(progress)

            # CALCULAR arrived_count CORRECTAMENTE
            arrived_count = sum(1 for v in self.vehicles if v.has_arrived())
            total_vehicles = len(self.vehicles)
            convergence_rate = arrived_count / total_vehicles if total_vehicles > 0 else 0

            status_text.text(
                f"🔄 Iteración {self.iteration} | "
                f"Llegaron: {arrived_count}/{total_vehicles} | "
                f"Tasa: {convergence_rate:.1%}"
            )

            # Mostrar estadísticas en tiempo real
            with stats_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iteración", self.iteration)
                with col2:
                    st.metric("Vehículos Llegados", arrived_count)
                with col3:
                    st.metric("Tasa de Convergencia", f"{convergence_rate:.1%}")
                with col4:
                    # Calcular tiempo promedio solo de vehículos que llegaron
                    arrived_times = [v.get_path_travel_time() for v in self.vehicles if v.has_arrived()]
                    avg_time = np.mean(arrived_times) if arrived_times else 0
                    st.metric("Tiempo Promedio", f"{avg_time:.2f}")

            # Verificar si todos llegaron
            if arrived_count == total_vehicles:
                elapsed_time = time.time() - start_time
                status_text.text(f"✅ ¡TODOS los vehículos llegaron! Tiempo: {elapsed_time:.2f}s")
                progress_bar.progress(1.0)
                return {
                    'success': True,
                    'iterations': self.iteration,
                    'total_time': elapsed_time,
                    'final_stats': {
                        'arrived_count': arrived_count,
                        'total_vehicles': total_vehicles,
                        'success_rate': convergence_rate * 100
                    }
                }

            time.sleep(0.05)  # Pequeña pausa para visualización

        # Si llegamos al máximo de iteraciones
        elapsed_time = time.time() - start_time
        final_arrived = sum(1 for v in self.vehicles if v.has_arrived())
        final_total = len(self.vehicles)

        status_text.text(f"⏰ Máximo de iteraciones alcanzado. Tiempo: {elapsed_time:.2f}s")
        return {
            'success': False,
            'iterations': self.iteration,
            'total_time': elapsed_time,
            'final_stats': {
                'arrived_count': final_arrived,
                'total_vehicles': final_total,
                'success_rate': (final_arrived / final_total * 100) if final_total > 0 else 0
            }
        }

    def _update_adaptive_parameters(self, arrived_count: int):
        """Actualizar parámetros adaptativamente basado en desempeño"""
        convergence_rate = arrived_count / len(self.vehicles)

        # Ajustar alpha y beta según convergencia
        if convergence_rate < 0.3:  # Baja convergencia -> más exploración
            self.adaptive_params['beta'] = min(3.0, self.adaptive_params['beta'] + 0.1)
        elif convergence_rate > 0.8:  # Alta convergencia -> más explotación
            self.adaptive_params['alpha'] = min(2.0, self.adaptive_params['alpha'] + 0.05)

        # Ajustar evaporación dinámicamente
        if self.iteration > 20:
            diversity = self._calculate_population_diversity()
            self.adaptive_params['evaporation_rate'] = 0.4 + (diversity * 0.3)

    def _calculate_population_diversity(self) -> float:
        """Calcular diversidad de la población de vehículos"""
        if not self.vehicles:
            return 1.0

        # Calcular cuántos caminos únicos hay
        unique_paths = set()
        for vehicle in self.vehicles:
            if vehicle.path:
                path_signature = tuple(vehicle.path[:min(10, len(vehicle.path))])
                unique_paths.add(path_signature)

        return len(unique_paths) / len(self.vehicles)

    def get_best_path(self, start: int, end: int) -> Tuple[Optional[List[int]], float]:
        """Obtener el mejor camino conocido entre start y end - MÉTODO NUEVO"""
        try:
            if (start not in self.city_map.reverse_mapping or
                    end not in self.city_map.reverse_mapping):
                return None, float('inf')

            start_osm = self.city_map.reverse_mapping[start]
            end_osm = self.city_map.reverse_mapping[end]

            # Usar el camino más corto en el grafo como mejor estimación
            shortest_path = nx.shortest_path(
                self.city_map.graph,
                start_osm,
                end_osm,
                weight='length'
            )

            # Convertir de OSM a nuestros IDs
            our_path = []
            for osm_id in shortest_path:
                if osm_id in self.city_map.node_mapping:
                    our_path.append(self.city_map.node_mapping[osm_id])

            # Calcular distancia
            path_length = nx.shortest_path_length(
                self.city_map.graph,
                start_osm,
                end_osm,
                weight='length'
            )

            return our_path, path_length

        except (nx.NetworkXNoPath, KeyError, Exception):
            return None, float('inf')

    def _assist_stuck_vehicles(self):
        """Asistir a vehículos atascados con información global"""
        for vehicle in self.vehicles:
            if (not vehicle.has_arrived() and
                    len(vehicle.path) > self.adaptive_params['max_steps'] * 0.6):

                # Usar mejor camino conocido para ayudar
                best_path, _ = self.get_best_path(vehicle.current, vehicle.end)
                if best_path and len(best_path) > 1:
                    # No forzar directamente, pero influenciar
                    next_suggestion = best_path[1]
                    if next_suggestion in self.city_map.get_neighbors(vehicle.current):
                        # Reforzar esta opción localmente
                        road_key = (vehicle.current, next_suggestion)
                        if road_key in self.city_map.roads:
                            self.city_map.roads[road_key]['pheromone'] += 10.0

    def _compute_initial_path(self, start: int, end: int) -> Optional[List[int]]:
        """Versión CORREGIDA con múltiples estrategias"""
        try:
            if (start not in self.city_map.reverse_mapping or
                    end not in self.city_map.reverse_mapping):
                return None

            start_osm = self.city_map.reverse_mapping[start]
            end_osm = self.city_map.reverse_mapping[end]  # CORREGIDO: usar 'end' no 'self.end'

            # Intentar con diferentes estrategias
            strategies = [
                lambda: nx.shortest_path(self.city_map.graph, start_osm, end_osm, weight='length'),
                lambda: nx.shortest_path(self.city_map.graph, start_osm, end_osm, weight='travel_time'),
                lambda: nx.astar_path(self.city_map.graph, start_osm, end_osm, weight='length')
            ]

            for strategy in strategies:
                try:
                    shortest_path = strategy()
                    our_path = []
                    for osm_id in shortest_path:
                        if osm_id in self.city_map.node_mapping:
                            our_path.append(self.city_map.node_mapping[osm_id])

                    if len(our_path) > 1:
                        return our_path
                except:
                    continue

            return None

        except Exception:
            return None

    def _initialize_pheromones(self, path: List[int], strength: float = 50.0):
        """Inicialización MEJORADA de feromonas"""
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            road_key = (start, end)
            if road_key in self.city_map.roads:
                self.city_map.roads[road_key]['pheromone'] += strength

                # También inicializar camino inverso si existe
                reverse_key = (end, start)
                if reverse_key in self.city_map.roads:
                    self.city_map.roads[reverse_key]['pheromone'] += strength * 0.8

    def update_pheromones(self):
        """Actualización MEJORADA de feromonas"""
        delta_pheromones = {}

        for road in self.city_map.roads:
            delta_pheromones[road] = 0.0

        for vehicle in self.vehicles:
            if vehicle.has_arrived():
                # Calcular calidad considerando múltiples factores
                path_quality = self._calculate_path_quality(vehicle)

                for i in range(len(vehicle.path) - 1):
                    start, end = vehicle.path[i], vehicle.path[i + 1]
                    road_key = (start, end)
                    if road_key in delta_pheromones:
                        # Decaimiento a lo largo del camino
                        position_factor = 0.95 ** i
                        delta_pheromones[road_key] += path_quality * position_factor

        for road, delta in delta_pheromones.items():
            self.city_map.roads[road]['pheromone'] += delta

    def _calculate_path_quality(self, vehicle: ACOVehicle) -> float:
        """Calcular calidad de camino considerando múltiples métricas"""
        base_quality = self.adaptive_params['Q'] / (vehicle.get_path_travel_time() + 0.1)

        # Bonus por eficiencia
        efficiency = vehicle.get_path_efficiency()
        efficiency_bonus = 1.0 + (efficiency * 0.5)

        # Penalización por longitud excesiva
        optimal_length = vehicle._estimate_optimal_path_length()
        actual_length = len(vehicle.path)
        length_penalty = 1.0 if actual_length <= optimal_length * 1.5 else 0.7

        return base_quality * efficiency_bonus * length_penalty

    def evaporate_pheromones(self):
        """Evaporación MEJORADA con límites adaptativos"""
        for road in self.city_map.roads:
            current_pheromone = self.city_map.roads[road]['pheromone']

            # Evaporación más inteligente
            new_pheromone = current_pheromone * self.adaptive_params['evaporation_rate']

            # Aplicar límites
            new_pheromone = max(
                self.adaptive_params['min_pheromone'],
                min(self.adaptive_params['max_pheromone'], new_pheromone)
            )

            self.city_map.roads[road]['pheromone'] = new_pheromone

    def get_performance_report(self) -> Dict:
        """Generar reporte de desempeño detallado"""
        arrived_vehicles = [v for v in self.vehicles if v.has_arrived()]
        stuck_vehicles = [v for v in self.vehicles if not v.has_arrived()]

        return {
            'total_iterations': self.iteration,
            'total_vehicles': len(self.vehicles),
            'arrived_vehicles': len(arrived_vehicles),
            'stuck_vehicles': len(stuck_vehicles),
            'success_rate': len(arrived_vehicles) / len(self.vehicles),
            'average_travel_time': np.mean(
                [v.get_path_travel_time() for v in arrived_vehicles]) if arrived_vehicles else 0,
            'convergence_history': self.iteration_stats['convergence_rate'],
            'best_paths_found': len(self.best_paths)
        }