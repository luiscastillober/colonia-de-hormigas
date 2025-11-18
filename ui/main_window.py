"""
Ventana principal de la aplicación
"""

import tkinter as tk
from tkinter import ttk
import threading

from core.city_map import RealCityMap
from core.aco_algorithm import ACOTrafficOptimizer
from ui.control_panel import ControlPanel
from ui.map_display import MapDisplay
from config import UI_CONFIG, MESSAGES


class MainWindow:
    """Ventana principal que coordina todos los componentes de la UI"""

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("🚦 Sistema de Optimización de Tráfico ACO")
        self.window.geometry(UI_CONFIG['window_size'])

        # Estado de la aplicación
        self.city_map = None
        self.optimizer = None
        self.vehicles = []
        self.simulation_thread = None
        self.is_running = False
        self.showing_ids = False

        # Componentes de la UI
        self.control_panel = None
        self.map_display = None

        self.setup_main_window()

    def setup_main_window(self):
        """Configurar la ventana principal con paneles divididos"""
        # Frame principal dividido
        main_frame = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel de controles (izquierda)
        control_container = ttk.Frame(main_frame)
        main_frame.add(control_container, weight=1)

        # Panel del mapa (derecha)
        map_container = ttk.Frame(main_frame)
        main_frame.add(map_container, weight=2)

        # Inicializar componentes
        self.control_panel = ControlPanel(control_container, self)
        self.map_display = MapDisplay(map_container, self)

    def log(self, message: str):
        """Añadir mensaje al log del panel de control"""
        if self.control_panel:
            self.control_panel.log(message)

    def update_map_info(self):
        """Actualizar información del mapa en el panel de control"""
        if self.control_panel:
            self.control_panel.update_map_info()

    def update_map_display(self):
        """Actualizar la visualización del mapa"""
        if self.map_display:
            self.map_display.update_display()

    def load_map_by_name(self, place_name: str):
        """Cargar mapa por nombre de lugar"""

        def load_thread():
            try:
                self.city_map = RealCityMap()
                self.city_map.load_city_from_osm(place_name)
                self.window.after(0, lambda: self.log(
                    MESSAGES['map_loaded'].format(
                        len(self.city_map.intersections),
                        len(self.city_map.roads)
                    )
                ))
                self.window.after(0, self.update_map_info)
                self.window.after(0, lambda: self.log("💡 Mapa listo para visualización"))
                self.window.after(100, self.update_map_display)
            except Exception as e:
                self.window.after(0, lambda: self.log(f"❌ Error: {str(e)}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def load_map_by_coords(self, lat: float, lon: float, radius: int):
        """Cargar mapa por coordenadas"""

        def load_thread():
            try:
                self.city_map = RealCityMap()
                self.city_map.load_city_from_point(lat, lon, radius)
                self.window.after(0, lambda: self.log(
                    MESSAGES['map_loaded'].format(
                        len(self.city_map.intersections),
                        len(self.city_map.roads)
                    )
                ))
                self.window.after(0, self.update_map_info)
                self.window.after(0, lambda: self.log("💡 Mapa listo para visualización"))
                self.window.after(100, self.update_map_display)
            except Exception as e:
                self.window.after(0, lambda: self.log(f"❌ Error: {str(e)}"))

        threading.Thread(target=load_thread, daemon=True).start()

    def add_vehicle(self, origin: int, dest: int):
        """Añadir vehículo a la simulación"""
        if not self.city_map:
            return False

        try:
            if (origin not in self.city_map.intersections or
                    dest not in self.city_map.intersections):
                return False

            self.vehicles.append((origin, dest))
            vehicle_num = len(self.vehicles)
            self.log(MESSAGES['vehicle_added'].format(vehicle_num, origin, dest))
            return True

        except ValueError:
            return False

    def block_road(self, start: int, end: int):
        """Bloquear una calle"""
        if self.city_map:
            self.city_map.block_road(start, end)
            self.log(MESSAGES['road_blocked'].format(start, end))

    def add_traffic(self, start: int, end: int):
        """Añadir tráfico a una calle"""
        if self.city_map:
            self.city_map.add_traffic_to_road(start, end, 2.5)
            self.log(MESSAGES['traffic_added'].format(start, end))

    def run_simulation(self, iterations: int):
        """Ejecutar simulación ACO"""
        if not self.city_map or not self.vehicles:
            return

        if self.is_running:
            return

        self.log(MESSAGES['simulation_start'].format(iterations))
        self.is_running = True

        def simulation_thread():
            try:
                self.optimizer = ACOTrafficOptimizer(self.city_map)

                # Añadir vehículos al optimizador
                aco_vehicles = []
                for origin, dest in self.vehicles:
                    vehicle = self.optimizer.add_vehicle(origin, dest)
                    aco_vehicles.append(vehicle)

                # Ejecutar iteraciones
                for i in range(iterations):
                    if not self.is_running:
                        break

                    arrived_count = self.optimizer.run_iteration()
                    self.window.after(0, lambda idx=i: self.control_panel.update_progress(idx + 1))

                    # Actualizar mapa cada 5 iteraciones
                    if (i + 1) % 5 == 0:
                        self.window.after(0, self.update_map_display)

                    if (i + 1) % 5 == 0 or i == 0:
                        progress_msg = f"   Iteración {i + 1}: {arrived_count}/{len(aco_vehicles)} llegaron"
                        if arrived_count == len(aco_vehicles):
                            progress_msg += " ✅"
                        self.window.after(0, lambda msg=progress_msg: self.log(msg))

                # Resultados finales
                self.window.after(0, self.show_simulation_results)
                self.window.after(0, self.update_map_display)

            except Exception as e:
                self.window.after(0, lambda: self.log(f"❌ Error en simulación: {str(e)}"))
            finally:
                self.window.after(0, lambda: self.control_panel.enable_run_button())
                self.is_running = False

        self.simulation_thread = threading.Thread(target=simulation_thread, daemon=True)
        self.simulation_thread.start()

    def show_simulation_results(self):
        """Mostrar resultados de la simulación"""
        if not self.optimizer:
            return

        self.log("\n📊 RESULTADOS FINALES:")
        self.log("=" * 40)

        for i, vehicle in enumerate(self.optimizer.vehicles):
            status = "✅ LLEGÓ" if vehicle.has_arrived() else "🚦 EN CAMINO"
            self.log(f"Vehículo {i + 1}: {vehicle.start} → {vehicle.end} - {status}")
            if vehicle.has_arrived():
                self.log(f"   Ruta: {' → '.join(map(str, vehicle.path))}")
                self.log(f"   Tiempo de viaje: {vehicle.get_path_travel_time():.2f} unidades")
            self.log("")

    def stop_simulation(self):
        """Detener simulación en curso"""
        self.is_running = False
        self.log("⏹️ Simulación detenida por el usuario")

    def reset_simulation(self):
        """Reiniciar simulación"""
        self.is_running = False
        self.vehicles = []
        self.optimizer = None
        if self.control_panel:
            self.control_panel.reset_controls()
        self.log("🔄 Sistema reiniciado")
        self.update_map_info()

    def toggle_show_ids(self):
        """Activar/desactivar visualización de IDs"""
        self.showing_ids = not self.showing_ids
        action = "activada" if self.showing_ids else "desactivada"
        self.log(f"👁️ Visualización de IDs {action}")
        self.update_map_display()

    def get_random_intersections(self, count: int = 2):
        """Obtener intersecciones aleatorias"""
        if self.city_map:
            return self.city_map.get_random_intersections(count)
        return []

    def save_map_image(self, filename: str):
        """Guardar el mapa como imagen"""
        if not self.city_map:
            return False

        try:
            vehicles_to_show = self.optimizer.vehicles if self.optimizer else []
            fig = self.city_map.visualize_on_map(
                vehicles=vehicles_to_show,
                title="Mapa de Tráfico - Optimización ACO",
                show_ids=self.showing_ids
            )

            if fig:
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                return True
            return False

        except Exception as e:
            self.log(f"❌ Error al guardar mapa: {str(e)}")
            return False

    def run(self):
        """Ejecutar la aplicación"""
        self.window.mainloop()