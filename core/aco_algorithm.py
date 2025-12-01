import time
import random
import heapq
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
import networkx as nx
from enum import Enum

# PARÁMETROS GLOBALES DEL ALGORITMO ACO - OPTIMIZADOS PARA RUTAS LARGAS
ACO_PARAMS = {
    'max_steps': 2000,  # ✅ AUMENTADO: Más pasos para rutas largas
    'max_stuck_count': 30,  # ✅ AUMENTADO: Más tolerancia antes de declarar "stuck"
    'alpha': 1.0,  # Peso de las feromonas
    'beta': 3.0,  # ✅ AUMENTADO: Más peso a la heurística (dirección al destino)
    'evaporation_rate': 0.3,  # ✅ REDUCIDO: Menos evaporación = más memoria
    'Q': 100.0,
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
    """Versión MEJORADA del vehículo con mejor manejo de rutas largas"""

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
        self.memory = []
        self.exploration_factor = 1.0
        
        # ✅ NUEVO: Calcular ruta óptima de referencia
        self.optimal_path_length = self._estimate_optimal_path_length()
        self.best_distance_so_far = float('inf')
        self.steps_without_improvement = 0
        
        # Parámetros desde config
        self.max_steps = ACO_PARAMS['max_steps']
        self.max_stuck_count = ACO_PARAMS['max_stuck_count']
        self.exploration_decay = 0.98  # Más conservador

    def select_next_intersection(self, alpha: float, beta: float) -> int:
        """✅ CORREGIDO: Mejor selección para rutas largas y no lineales"""
        
        # ✅ NUEVO: Verificar si estamos cerca del destino
        current_distance = self._get_distance_to_destination(self.current)
        
        # Si estamos MUY cerca (a 2 saltos), usar búsqueda directa
        if current_distance < 0.01:  # Muy cerca en coordenadas
            neighbors = self.city_map.get_neighbors(self.current)
            if self.end in neighbors:
                return self.end
        
        # Actualizar mejor distancia
        if current_distance < self.best_distance_so_far:
            self.best_distance_so_far = current_distance
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1
        
        # ✅ MEJORADO: Detectar estancamiento temprano
        if self.steps_without_improvement > 50:
            self.stuck_count += 1
            self.steps_without_improvement = 0
        
        # Actualizar estado del vehículo
        self._update_vehicle_state()
        
        # Estrategias según estado
        if self.state == VehicleState.STUCK:
            return self._escape_strategy()
        elif self.state == VehicleState.BACKTRACKING:
            return self._backtrack_strategy()
        elif self.state == VehicleState.FOLLOWING_PHEROMONE:
            return self._pheromone_follow_strategy(alpha, beta)
        else:  # EXPLORING
            return self._exploration_strategy(alpha, beta)

    def _get_distance_to_destination(self, node_id: int) -> float:
        """✅ NUEVO: Calcular distancia real al destino"""
        try:
            if (node_id in self.city_map.reverse_mapping and 
                self.end in self.city_map.reverse_mapping):
                
                node_osm = self.city_map.reverse_mapping[node_id]
                end_osm = self.city_map.reverse_mapping[self.end]
                
                # Usar distancia Euclidiana de las coordenadas
                x1, y1 = self.city_map.graph.nodes[node_osm]['x'], self.city_map.graph.nodes[node_osm]['y']
                x2, y2 = self.city_map.graph.nodes[end_osm]['x'], self.city_map.graph.nodes[end_osm]['y']
                
                return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except:
            pass
        
        # Fallback: distancia de IDs
        return abs(self.end - node_id)

    def _exploration_strategy(self, alpha: float, beta: float) -> int:
        """✅ CORREGIDO: Mejor exploración para rutas largas"""
        neighbors = self.city_map.get_neighbors(self.current)
        
        # ✅ CRÍTICO: Permitir revisitar nodos si no hay alternativas
        unvisited_neighbors = [n for n in neighbors if n not in self.visited]
        
        if not unvisited_neighbors and len(neighbors) > 0:
            # Si no hay nodos no visitados, elegir el que nos acerque más al destino
            neighbors_with_distance = [
                (n, self._get_distance_to_destination(n)) 
                for n in neighbors
            ]
            neighbors_with_distance.sort(key=lambda x: x[1])
            
            # Permitir revisitar el mejor nodo (el más cercano al destino)
            best_neighbor = neighbors_with_distance[0][0]
            
            # ✅ IMPORTANTE: Limpiar visitados si volvemos a un nodo
            self.visited = set(self.path[-10:])  # Solo recordar últimos 10 nodos
            
            return best_neighbor
        
        if not unvisited_neighbors:
            return self._handle_no_valid_neighbors()
        
        # Calcular probabilidades con FUERTE énfasis en heurística
        probabilities = []
        for neighbor in unvisited_neighbors:
            # ✅ MEJORADO: Dar MÁS peso a la dirección correcta
            base_prob = self._calculate_move_probability(neighbor, alpha, beta * 1.5)
            
            # Bonus adicional si el vecino nos acerca al destino
            neighbor_distance = self._get_distance_to_destination(neighbor)
            current_distance = self._get_distance_to_destination(self.current)
            
            if neighbor_distance < current_distance:
                base_prob *= 2.0  # ✅ BONUS: Duplicar probabilidad si nos acercamos
            
            probabilities.append(base_prob)
        
        total = sum(probabilities)
        if total == 0:
            chosen = random.choice(unvisited_neighbors)
        else:
            probabilities = [p / total for p in probabilities]
            chosen = random.choices(unvisited_neighbors, weights=probabilities)[0]
        
        self.exploration_factor *= self.exploration_decay
        return chosen

    def _pheromone_follow_strategy(self, alpha: float, beta: float) -> int:
        """✅ MEJORADO: Seguir feromonas pero sin olvidar el destino"""
        neighbors = self.city_map.get_neighbors(self.current)
        
        # ✅ Permitir revisitar si es necesario
        valid_neighbors = [n for n in neighbors if n not in self.visited]
        
        if not valid_neighbors:
            if len(neighbors) > 0:
                # Elegir el vecino más cercano al destino
                valid_neighbors = sorted(
                    neighbors, 
                    key=lambda n: self._get_distance_to_destination(n)
                )[:3]  # Top 3 más cercanos
                self.visited = set(self.path[-15:])
            else:
                return self._handle_no_valid_neighbors()
        
        # Combinar feromonas con dirección al destino
        candidate_scores = []
        for neighbor in valid_neighbors:
            road_key = (self.current, neighbor)
            if road_key in self.city_map.roads:
                pheromone = self.city_map.roads[road_key]['pheromone']
                heuristic = self._calculate_detailed_heuristic(neighbor)
                
                # ✅ BALANCE: 60% feromonas, 40% heurística
                score = (pheromone ** alpha) * (heuristic ** (beta * 0.7))
                candidate_scores.append((neighbor, score))
        
        if not candidate_scores:
            return random.choice(valid_neighbors)
        
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # ✅ 70% mejor, 30% exploración
        if random.random() < 0.7:
            return candidate_scores[0][0]
        else:
            return random.choice([c[0] for c in candidate_scores[:3]])

    def _backtrack_strategy(self) -> int:
        """✅ MEJORADO: Backtracking más inteligente"""
        if len(self.path) <= 1:
            self.visited.clear()
            self.visited.add(self.start)
            return self.start
        
        # ✅ Retroceder hasta encontrar un nodo con opciones no exploradas
        backtrack_steps = min(5, len(self.path) - 1)
        
        for step in range(1, backtrack_steps + 1):
            candidate = self.path[-step]
            neighbors = self.city_map.get_neighbors(candidate)
            unexplored = [n for n in neighbors if n not in self.visited]
            
            if len(unexplored) > 0:
                # Limpiar visited de los nodos posteriores
                for node in self.path[-step:]:
                    self.visited.discard(node)
                
                self.stuck_count = 0
                self.state = VehicleState.EXPLORING
                self.exploration_factor = 1.0
                
                return candidate
        
        # Si no encontramos nada, limpiar todo y empezar desde más atrás
        self.visited = set(self.path[-20:])
        self.stuck_count = 0
        return self.path[-backtrack_steps]

    def _escape_strategy(self) -> int:
        """✅ CRÍTICO: Estrategia de escape mejorada"""
        neighbors = self.city_map.get_neighbors(self.current)
        if not neighbors:
            return self.current
        
        # ✅ RESET COMPLETO: Olvidar casi todo el historial
        self.visited = set(self.path[-5:])  # Solo recordar últimos 5 nodos
        
        # Buscar el vecino que MÁS nos acerque al destino
        best_neighbor = None
        best_distance = float('inf')
        
        for neighbor in neighbors:
            distance = self._get_distance_to_destination(neighbor)
            if distance < best_distance:
                best_distance = distance
                best_neighbor = neighbor
        
        if best_neighbor:
            self.stuck_count = 0
            self.state = VehicleState.EXPLORING
            self.exploration_factor = 1.0
            return best_neighbor
        
        return random.choice(neighbors)

    def _calculate_move_probability(self, neighbor: int, alpha: float, beta: float) -> float:
        """✅ MEJORADO: Cálculo de probabilidad optimizado"""
        road_key = (self.current, neighbor)
        
        if road_key not in self.city_map.roads:
            return 0.0
        
        pheromone = self.city_map.roads[road_key]['pheromone']
        travel_time = self._get_road_travel_time(self.current, neighbor, self.travel_time)
        heuristic = self._calculate_detailed_heuristic(neighbor)
        
        if travel_time <= 0:
            travel_time = 0.1
        
        # ✅ BALANCE: Más peso a la heurística
        probability = (pheromone ** alpha) * (heuristic ** beta) / (travel_time + 0.1)
        
        return max(probability, 1e-6)

    def _calculate_detailed_heuristic(self, neighbor: int) -> float:
        """✅ CORREGIDO: Heurística más precisa"""
        try:
            if neighbor == self.end:
                return 10000.0  # ✅ VALOR ALTO si es el destino
            
            # ✅ Usar distancia real en coordenadas
            neighbor_distance = self._get_distance_to_destination(neighbor)
            
            if neighbor_distance < 0.0001:
                return 5000.0
            
            # Invertir distancia: menor distancia = mayor heurística
            heuristic = 100.0 / (neighbor_distance + 0.001)
            
            # Factor de calidad de carretera
            road_quality = 1.0
            if (self.current, neighbor) in self.city_map.roads:
                road_data = self.city_map.roads[(self.current, neighbor)]
                if road_data.get('is_major_road', False):
                    road_quality = 1.3
                elif road_data.get('traffic', 1.0) > 2.0:
                    road_quality = 0.7
            
            return heuristic * road_quality
            
        except Exception:
            return 1.0

    def _handle_no_valid_neighbors(self):
        """✅ CORREGIDO: Mejor manejo cuando no hay vecinos válidos"""
        if len(self.path) > 5:
            # Buscar el mejor nodo para retroceder
            candidates = self.path[-10:] if len(self.path) >= 10 else self.path[:-1]
            
            best_candidate = None
            best_distance = float('inf')
            
            for candidate in candidates:
                distance = self._get_distance_to_destination(candidate)
                neighbors = self.city_map.get_neighbors(candidate)
                unexplored = [n for n in neighbors if n not in self.visited]
                
                if len(unexplored) > 0 and distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
            
            if best_candidate:
                # Limpiar visited
                self.visited = set(self.path[-10:])
                return best_candidate
        
        # Último recurso: reset completo
        self.visited = set([self.start])
        self.stuck_count = 0
        return self.start

    def _is_in_local_loop(self, next_node: int) -> bool:
        """✅ Detección mejorada de bucles"""
        if len(self.path) < 3:
            return False
        
        # Buscar patrones repetitivos
        recent_nodes = self.path[-8:]
        if len(set(recent_nodes)) < len(recent_nodes) * 0.5:
            return True
        
        # Detectar ciclos pequeños
        if next_node in self.path[-6:]:
            return True
        
        return False

    def _update_vehicle_state(self):
        """✅ MEJORADO: Actualización de estado más inteligente"""
        if len(self.path) > self.max_steps * 0.9:
            self.state = VehicleState.STUCK
        elif self.stuck_count > self.max_stuck_count // 2:
            self.state = VehicleState.BACKTRACKING
        elif self.steps_without_improvement > 30:
            self.state = VehicleState.BACKTRACKING
        elif len(self.path) > 20 and self._is_making_progress():
            self.state = VehicleState.FOLLOWING_PHEROMONE
        else:
            self.state = VehicleState.EXPLORING

    def _is_making_progress(self) -> bool:
        """✅ Verificar progreso real"""
        if len(self.path) < 5:
            return True
        
        current_distance = self._get_distance_to_destination(self.current)
        prev_distance = self._get_distance_to_destination(self.path[-5])
        
        return current_distance < prev_distance * 1.1  # 10% de tolerancia

    def _get_road_travel_time(self, start: int, end: int, current_time: float) -> float:
        """Wrapper para obtener tiempo de viaje"""
        if hasattr(self.city_map, 'get_road_travel_time'):
            return self.city_map.get_road_travel_time(start, end, current_time)
        else:
            road_key = (start, end)
            if road_key in self.city_map.roads:
                base_time = self.city_map.roads[road_key].get('base_travel_time', 1.0)
                traffic = self.city_map.roads[road_key].get('traffic', 1.0)
                return base_time * traffic
            return 1.0

    def move_to(self, intersection: int):
        """✅ CORREGIDO: Movimiento con mejor control"""
        if self.current == intersection:
            return
        
        travel_time = self._get_road_travel_time(self.current, intersection, self.travel_time)
        
        if travel_time == float('inf'):
            travel_time = 100.0
        
        self.travel_time += travel_time
        self.current = intersection
        self.path.append(intersection)
        self.visited.add(intersection)
        
        # ✅ Detección mejorada de bucles
        if self._is_in_local_loop(intersection):
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 0.5)
        
        if self.current == self.end:
            self.arrived = True
            self.state = VehicleState.ARRIVED
            self._reinforce_path()

    def _reinforce_path(self):
        """✅ Refuerzo de feromonas más fuerte para rutas exitosas"""
        path_length = len(self.path)
        optimal_length = max(self.optimal_path_length, 10)
        length_efficiency = optimal_length / max(path_length, 1)
        
        # ✅ Más recompensa para rutas exitosas
        path_quality = (150.0 / (self.travel_time + 1)) * length_efficiency
        
        for i in range(len(self.path) - 1):
            start, end = self.path[i], self.path[i + 1]
            road_key = (start, end)
            if road_key in self.city_map.roads:
                # ✅ Refuerzo más uniforme en toda la ruta
                decay_factor = 0.95 ** i
                reinforcement = path_quality * decay_factor
                self.city_map.roads[road_key]['pheromone'] += reinforcement

    def _estimate_optimal_path_length(self) -> float:
        """✅ MEJORADO: Estimación más precisa"""
        try:
            if (self.start in self.city_map.reverse_mapping and
                self.end in self.city_map.reverse_mapping):
                
                start_osm = self.city_map.reverse_mapping[self.start]
                end_osm = self.city_map.reverse_mapping[self.end]
                
                path_length = nx.shortest_path_length(
                    self.city_map.graph,
                    start_osm,
                    end_osm,
                    weight='length'
                )
                
                # Número estimado de nodos
                estimated_nodes = max(int(path_length / 100), 10)  # ~100m por nodo
                return estimated_nodes
        except:
            pass
        
        return abs(self.end - self.start)

    def has_arrived(self) -> bool:
        return self.arrived

    def get_path_travel_time(self) -> float:
        return self.travel_time

    def get_path_efficiency(self) -> float:
        """Calcular eficiencia del camino encontrado"""
        if not self.arrived or len(self.path) < 2:
            return 0.0
        
        optimal_length = max(self.optimal_path_length, 1)
        actual_length = len(self.path)
        
        return optimal_length / actual_length


class ACOTrafficOptimizer:
    """✅ Optimizador MEJORADO para rutas largas"""

    def __init__(self, city_map):
        self.city_map = city_map
        self.vehicles = []
        self.iteration = 0
        self.best_paths = {}
        self.convergence_count = 0
        self.global_best_time = float('inf')
        self.adaptive_params = ACO_PARAMS.copy()
        
        self.iteration_stats = {
            'arrived_vehicles': [],
            'average_travel_time': [],
            'convergence_rate': []
        }

    def add_vehicle(self, start: int, end: int):
        vehicle_id = len(self.vehicles)
        vehicle = ACOVehicle(vehicle_id, start, end, self.city_map)
        self.vehicles.append(vehicle)
        
        # ✅ Inicialización más fuerte de feromonas
        initial_path = self._compute_initial_path(start, end)
        if initial_path:
            self._initialize_pheromones(initial_path, strength=200.0)
        
        return vehicle

    def run_iteration(self) -> Dict:
        """✅ Iteración optimizada"""
        self.iteration += 1
        
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
        
        self.update_pheromones()
        self.evaporate_pheromones()
        
        # ✅ Asistencia cada 10 iteraciones
        if self.iteration % 10 == 0:
            self._assist_stuck_vehicles()
        
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

    def run_until_all_arrive(self, max_iterations: int = 500) -> Dict:
        """✅ CORREGIDO: Ejecutar hasta que todos lleguen"""
        import streamlit as st
        
        start_time = time.time()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()
        
        for i in range(max_iterations):
            result = self.run_iteration()
            
            progress = (i + 1) / max_iterations
            progress_bar.progress(progress)
            
            arrived_count = sum(1 for v in self.vehicles if v.has_arrived())
            total_vehicles = len(self.vehicles)
            convergence_rate = arrived_count / total_vehicles if total_vehicles > 0 else 0
            
            status_text.text(
                f"🔄 Iteración {self.iteration} | "
                f"Llegaron: {arrived_count}/{total_vehicles} | "
                f"Tasa: {convergence_rate:.1%}"
            )
            
            with stats_container.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iteración", self.iteration)
                with col2:
                    st.metric("Vehículos Llegados", arrived_count)
                with col3:
                    st.metric("Tasa de Convergencia", f"{convergence_rate:.1%}")
                with col4:
                    arrived_times = [v.get_path_travel_time() for v in self.vehicles if v.has_arrived()]
                    avg_time = np.mean(arrived_times) if arrived_times else 0
                    st.metric("Tiempo Promedio", f"{avg_time:.2f}")
            
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
            
            time.sleep(0.05)
        
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

    def _assist_stuck_vehicles(self):
        """✅ CRÍTICO: Asistencia a vehículos atascados"""
        for vehicle in self.vehicles:
            if (not vehicle.has_arrived() and 
                (vehicle.stuck_count > 10 or vehicle.steps_without_improvement > 40)):
                
                # Buscar mejor camino conocido
                best_path, _ = self.get_best_path(vehicle.current, vehicle.end)
                
                if best_path and len(best_path) > 1:
                    # Reforzar FUERTEMENTE el camino sugerido
                    for i in range(min(5, len(best_path) - 1)):
                        start, end = best_path[i], best_path[i + 1]
                        road_key = (start, end)
                        if road_key in self.city_map.roads:
                            self.city_map.roads[road_key]['pheromone'] += 50.0
                
                # Reset del vehículo
                vehicle.visited = set(vehicle.path[-10:])
                vehicle.stuck_count = max(0, vehicle.stuck_count - 5)
                vehicle.steps_without_improvement = 0

    def _compute_initial_path(self, start: int, end: int) -> Optional[List[int]]:
        """✅ MEJORADO: Cálculo de ruta inicial"""
        try:
            if (start not in self.city_map.reverse_mapping or
                end not in self.city_map.reverse_mapping):
                return None
            
            start_osm = self.city_map.reverse_mapping[start]
            end_osm = self.city_map.reverse_mapping[end]
            
            # Probar múltiples estrategias
            strategies = [
                lambda: nx.shortest_path(self.city_map.graph, start_osm, end_osm, weight='length'),
                lambda: nx.dijkstra_path(self.city_map.graph, start_osm, end_osm, weight='length'),
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

    def _initialize_pheromones(self, path: List[int], strength: float = 200.0):
        """✅ MEJORADO: Inicialización más fuerte"""
        for i in range(len(path) - 1):
            start, end = path[i], path[i + 1]
            road_key = (start, end)
            if road_key in self.city_map.roads:
                self.city_map.roads[road_key]['pheromone'] += strength
                
                reverse_key = (end, start)
                if reverse_key in self.city_map.roads:
                    self.city_map.roads[reverse_key]['pheromone'] += strength * 0.8

    def get_best_path(self, start: int, end: int) -> Tuple[Optional[List[int]], float]:
        """✅ Obtener mejor camino conocido"""
        try:
            if (start not in self.city_map.reverse_mapping or
                end not in self.city_map.reverse_mapping):
                return None, float('inf')
            
            start_osm = self.city_map.reverse_mapping[start]
            end_osm = self.city_map.reverse_mapping[end]
            
            shortest_path = nx.shortest_path(
                self.city_map.graph,
                start_osm,
                end_osm,
                weight='length'
            )
            
            our_path = []
            for osm_id in shortest_path:
                if osm_id in self.city_map.node_mapping:
                    our_path.append(self.city_map.node_mapping[osm_id])
            
            path_length = nx.shortest_path_length(
                self.city_map.graph,
                start_osm,
                end_osm,
                weight='length'
            )
            
            return our_path, path_length
            
        except (nx.NetworkXNoPath, KeyError, Exception):
            return None, float('inf')

    def update_pheromones(self):
        """✅ Actualización de feromonas optimizada"""
        delta_pheromones = {}
        
        for road in self.city_map.roads:
            delta_pheromones[road] = 0.0
        
        for vehicle in self.vehicles:
            if vehicle.has_arrived():
                path_quality = self._calculate_path_quality(vehicle)
                
                for i in range(len(vehicle.path) - 1):
                    start, end = vehicle.path[i], vehicle.path[i + 1]
                    road_key = (start, end)
                    if road_key in delta_pheromones:
                        position_factor = 0.95 ** i
                        delta_pheromones[road_key] += path_quality * position_factor
        
        for road, delta in delta_pheromones.items():
            self.city_map.roads[road]['pheromone'] += delta

    def _calculate_path_quality(self, vehicle: ACOVehicle) -> float:
        """✅ Calcular calidad del camino"""
        base_quality = self.adaptive_params['Q'] / (vehicle.get_path_travel_time() + 0.1)
        
        efficiency = vehicle.get_path_efficiency()
        efficiency_bonus = 1.0 + (efficiency * 0.5)
        
        optimal_length = max(vehicle.optimal_path_length, 1)
        actual_length = len(vehicle.path)
        length_penalty = 1.0 if actual_length <= optimal_length * 1.5 else 0.7
        
        return base_quality * efficiency_bonus * length_penalty

    def evaporate_pheromones(self):
        """✅ Evaporación mejorada"""
        for road in self.city_map.roads:
            current_pheromone = self.city_map.roads[road]['pheromone']
            
            new_pheromone = current_pheromone * self.adaptive_params['evaporation_rate']
            
            new_pheromone = max(
                self.adaptive_params['min_pheromone'],
                min(self.adaptive_params['max_pheromone'], new_pheromone)
            )
            
            self.city_map.roads[road]['pheromone'] = new_pheromone

    def get_performance_report(self) -> Dict:
        """✅ Reporte de desempeño detallado"""
        arrived_vehicles = [v for v in self.vehicles if v.has_arrived()]
        stuck_vehicles = [v for v in self.vehicles if not v.has_arrived()]
        
        return {
            'total_iterations': self.iteration,
            'total_vehicles': len(self.vehicles),
            'arrived_vehicles': len(arrived_vehicles),
            'stuck_vehicles': len(stuck_vehicles),
            'success_rate': len(arrived_vehicles) / len(self.vehicles) if self.vehicles else 0,
            'average_travel_time': np.mean(
                [v.get_path_travel_time() for v in arrived_vehicles]) if arrived_vehicles else 0,
            'convergence_history': self.iteration_stats['convergence_rate'],
            'best_paths_found': len(self.best_paths),
            'vehicles_arrived': len(arrived_vehicles),
            'vehicles_stuck': len(stuck_vehicles)
        }