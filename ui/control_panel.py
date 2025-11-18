"""
Panel de control de la aplicación
"""

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog

from config import MESSAGES


class ControlPanel:
    """Panel de control con todos los widgets de configuración"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        self.setup_control_panel()

    def setup_control_panel(self):
        """Configurar el panel de control con scroll"""
        # Canvas con scrollbar
        canvas = tk.Canvas(self.parent)
        scrollbar = ttk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Contenido del panel
        self.create_control_content()

    def create_control_content(self):
        """Crear el contenido del panel de control"""
        # Sección 1: Cargar Mapa
        self.create_map_section()

        # Sección 2: Herramientas de ID
        self.create_id_tools_section()

        # Sección 3: Configurar Obstáculos
        self.create_obstacles_section()

        # Sección 4: Vehículos
        self.create_vehicles_section()

        # Sección 5: Simulación
        self.create_simulation_section()

        # Sección 6: Log
        self.create_log_section()

    def create_map_section(self):
        """Crear sección para cargar mapa"""
        ttk.Label(self.scrollable_frame, text="📍 CARGAR MAPA",
                  font=('Arial', 11, 'bold')).grid(row=0, column=0, columnspan=3, pady=8, sticky=tk.W)

        # Entrada de lugar
        ttk.Label(self.scrollable_frame, text="Lugar:").grid(row=1, column=0, sticky=tk.W)
        self.place_entry = ttk.Entry(self.scrollable_frame, width=20)
        self.place_entry.insert(0, "Trujillo, Peru")
        self.place_entry.grid(row=1, column=1, padx=5, sticky=tk.W)

        ttk.Button(self.scrollable_frame, text="Cargar",
                   command=self.load_map_by_name).grid(row=1, column=2, padx=2)

        ttk.Button(self.scrollable_frame, text="📋 Info IDs",
                   command=self.show_id_info).grid(row=1, column=3, padx=2)

        # Información del mapa
        self.map_info_label = ttk.Label(self.scrollable_frame, text="No hay mapa cargado",
                                        foreground="red", font=('Arial', 9))
        self.map_info_label.grid(row=2, column=0, columnspan=4, pady=3, sticky=tk.W)

        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=3, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

    def create_id_tools_section(self):
        """Crear sección de herramientas de identificación"""
        ttk.Label(self.scrollable_frame, text="🔍 HERRAMIENTAS DE ID",
                  font=('Arial', 11, 'bold')).grid(row=4, column=0, columnspan=4, pady=8, sticky=tk.W)

        id_tools_frame = ttk.Frame(self.scrollable_frame)
        id_tools_frame.grid(row=5, column=0, columnspan=4, pady=3, sticky=tk.W)

        ttk.Button(id_tools_frame, text="📊 Listar Nodos",
                   command=self.list_first_nodes).pack(side=tk.LEFT, padx=1)
        ttk.Button(id_tools_frame, text="🛣️ Listar Calles",
                   command=self.list_roads).pack(side=tk.LEFT, padx=1)
        ttk.Button(id_tools_frame, text="🎲 Aleatorio",
                   command=self.show_random_node).pack(side=tk.LEFT, padx=1)

        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=6, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

    def create_obstacles_section(self):
        """Crear sección para configurar obstáculos"""
        ttk.Label(self.scrollable_frame, text="🚧 CONFIGURAR OBSTÁCULOS",
                  font=('Arial', 11, 'bold')).grid(row=7, column=0, columnspan=4, pady=8, sticky=tk.W)

        config_frame = ttk.Frame(self.scrollable_frame)
        config_frame.grid(row=8, column=0, columnspan=4, pady=3, sticky=tk.W)

        # Bloquear calles
        ttk.Label(config_frame, text="Bloquear:").grid(row=0, column=0, sticky=tk.W)
        self.block_start = ttk.Entry(config_frame, width=5)
        self.block_start.grid(row=0, column=1, padx=1)
        ttk.Label(config_frame, text="→").grid(row=0, column=2)
        self.block_end = ttk.Entry(config_frame, width=5)
        self.block_end.grid(row=0, column=3, padx=1)
        ttk.Button(config_frame, text="Bloquear",
                   command=self.block_road).grid(row=0, column=4, padx=2)

        # Añadir tráfico
        ttk.Label(config_frame, text="Tráfico:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.traffic_start = ttk.Entry(config_frame, width=5)
        self.traffic_start.grid(row=1, column=1, padx=1)
        ttk.Label(config_frame, text="→").grid(row=1, column=2)
        self.traffic_end = ttk.Entry(config_frame, width=5)
        self.traffic_end.grid(row=1, column=3, padx=1)
        ttk.Button(config_frame, text="+ Tráfico",
                   command=self.add_traffic).grid(row=1, column=4, padx=2)

        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=9, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

    def create_vehicles_section(self):
        """Crear sección para configurar vehículos"""
        ttk.Label(self.scrollable_frame, text="🚗 CONFIGURAR VEHÍCULOS",
                  font=('Arial', 11, 'bold')).grid(row=10, column=0, columnspan=4, pady=8, sticky=tk.W)

        vehicle_frame = ttk.Frame(self.scrollable_frame)
        vehicle_frame.grid(row=11, column=0, columnspan=4, pady=3, sticky=tk.W)

        ttk.Label(vehicle_frame, text="Origen:").pack(side=tk.LEFT)
        self.origin_entry = ttk.Entry(vehicle_frame, width=5)
        self.origin_entry.pack(side=tk.LEFT, padx=1)
        ttk.Label(vehicle_frame, text="Destino:").pack(side=tk.LEFT, padx=(5, 0))
        self.dest_entry = ttk.Entry(vehicle_frame, width=5)
        self.dest_entry.pack(side=tk.LEFT, padx=1)

        ttk.Button(vehicle_frame, text="+ Vehículo",
                   command=self.add_vehicle).pack(side=tk.LEFT, padx=2)
        ttk.Button(vehicle_frame, text="🎲 Aleatorio",
                   command=self.add_random_vehicle).pack(side=tk.LEFT, padx=2)

        # Lista de vehículos
        self.vehicle_listbox = tk.Listbox(self.scrollable_frame, height=4, width=50)
        self.vehicle_listbox.grid(row=12, column=0, columnspan=4, pady=3, sticky=tk.W)

        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=13, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

    def create_simulation_section(self):
        """Crear sección para ejecutar simulación"""
        ttk.Label(self.scrollable_frame, text="▶️ EJECUTAR SIMULACIÓN",
                  font=('Arial', 11, 'bold')).grid(row=14, column=0, columnspan=4, pady=8, sticky=tk.W)

        sim_frame = ttk.Frame(self.scrollable_frame)
        sim_frame.grid(row=15, column=0, columnspan=4, pady=3, sticky=tk.W)

        ttk.Label(sim_frame, text="Iteraciones:").pack(side=tk.LEFT)
        self.iterations_entry = ttk.Entry(sim_frame, width=6)
        self.iterations_entry.insert(0, "20")
        self.iterations_entry.pack(side=tk.LEFT, padx=3)

        self.run_button = ttk.Button(sim_frame, text="▶️ Ejecutar",
                                     command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="⏹️ Detener",
                   command=self.stop_simulation).pack(side=tk.LEFT, padx=2)
        ttk.Button(sim_frame, text="🔄 Reiniciar",
                   command=self.reset_simulation).pack(side=tk.LEFT, padx=2)

        # Barra de progreso
        self.progress = ttk.Progressbar(self.scrollable_frame, orient='horizontal',
                                        length=300, mode='determinate')
        self.progress.grid(row=16, column=0, columnspan=4, pady=5, sticky=tk.W)

        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=17, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)

    def create_log_section(self):
        """Crear sección del log"""
        ttk.Label(self.scrollable_frame, text="LOG:").grid(row=18, column=0, sticky=tk.W, pady=(8, 0))
        self.status_text = tk.Text(self.scrollable_frame, height=12, width=55)
        self.status_text.grid(row=19, column=0, columnspan=4, pady=3, sticky=tk.W)

        scrollbar = ttk.Scrollbar(self.scrollable_frame, command=self.status_text.yview)
        scrollbar.grid(row=19, column=4, sticky=(tk.N, tk.S))
        self.status_text.config(yscrollcommand=scrollbar.set)

    # Métodos de control de la UI
    def log(self, message: str):
        """Añadir mensaje al log"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.parent.update_idletasks()

    def update_map_info(self):
        """Actualizar información del mapa cargado"""
        if self.main_app.city_map and self.main_app.city_map.intersections:
            info = f"✅ Mapa cargado: {len(self.main_app.city_map.intersections)} intersecciones, {len(self.main_app.city_map.roads)} segmentos"
            self.map_info_label.config(text=info, foreground="green")
        else:
            self.map_info_label.config(text="No hay mapa cargado", foreground="red")

    def update_progress(self, value: int):
        """Actualizar barra de progreso"""
        self.progress['value'] = value

    def enable_run_button(self):
        """Habilitar botón de ejecución"""
        self.run_button.config(state='normal')

    def reset_controls(self):
        """Reiniciar controles"""
        self.vehicle_listbox.delete(0, tk.END)
        self.progress['value'] = 0
        self.status_text.delete(1.0, tk.END)
        self.update_map_info()

    # Métodos de eventos
    def load_map_by_name(self):
        """Cargar mapa por nombre"""
        place = self.place_entry.get()
        self.main_app.load_map_by_name(place)

    def show_id_info(self):
        """Mostrar información de IDs"""
        if not self.main_app.city_map:
            messagebox.showwarning("Advertencia", "Primero carga un mapa")
            return

        total_nodes = len(self.main_app.city_map.intersections)
        info = f"El mapa tiene {total_nodes} nodos (IDs 0-{total_nodes - 1})"
        messagebox.showinfo("Información de IDs", info)

    def list_first_nodes(self):
        """Listar primeros nodos"""
        if not self.main_app.city_map:
            messagebox.showwarning("Advertencia", "Primero carga un mapa")
            return

        self.main_app.log("\n📋 PRIMEROS 10 NODOS:")
        self.main_app.log("=" * 50)

        for i, (node_id, coords) in enumerate(list(self.main_app.city_map.intersections.items())[:10]):
            x, y = coords
            self.main_app.log(f"Nodo {node_id}: ({x:.4f}, {y:.4f})")

    def list_roads(self):
        """Listar calles de ejemplo"""
        if not self.main_app.city_map:
            messagebox.showwarning("Advertencia", "Primero carga un mapa")
            return

        self.main_app.log("\n🛣️ EJEMPLOS DE CALLES:")
        self.main_app.log("=" * 50)

        for i, ((start, end), road_data) in enumerate(list(self.main_app.city_map.roads.items())[:10]):
            distance = road_data['distance']
            self.main_app.log(f"Calle {start} → {end}: {distance:.2f} km")

    def show_random_node(self):
        """Mostrar nodo aleatorio"""
        if not self.main_app.city_map:
            messagebox.showwarning("Advertencia", "Primero carga un mapa")
            return

        nodes = self.main_app.get_random_intersections(1)
        if nodes:
            self.origin_entry.delete(0, tk.END)
            self.origin_entry.insert(0, str(nodes[0]))
            self.main_app.log(f"🎲 Nodo aleatorio: {nodes[0]}")

    def block_road(self):
        """Bloquear calle"""
        try:
            start = int(self.block_start.get())
            end = int(self.block_end.get())
            self.main_app.block_road(start, end)
        except ValueError:
            messagebox.showerror("Error", "IDs inválidos")

    def add_traffic(self):
        """Añadir tráfico"""
        try:
            start = int(self.traffic_start.get())
            end = int(self.traffic_end.get())
            self.main_app.add_traffic(start, end)
        except ValueError:
            messagebox.showerror("Error", "IDs inválidos")

    def add_vehicle(self):
        """Añadir vehículo"""
        try:
            origin = int(self.origin_entry.get())
            dest = int(self.dest_entry.get())

            if self.main_app.add_vehicle(origin, dest):
                vehicle_num = len(self.main_app.vehicles)
                self.vehicle_listbox.insert(tk.END, f"Vehículo {vehicle_num}: {origin} → {dest}")
                self.origin_entry.delete(0, tk.END)
                self.dest_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", "Uno o ambos nodos no existen")
        except ValueError:
            messagebox.showerror("Error", "IDs inválidos")

    def add_random_vehicle(self):
        """Añadir vehículo con nodos aleatorios"""
        nodes = self.main_app.get_random_intersections(2)
        if len(nodes) >= 2:
            self.origin_entry.delete(0, tk.END)
            self.dest_entry.delete(0, tk.END)
            self.origin_entry.insert(0, str(nodes[0]))
            self.dest_entry.insert(0, str(nodes[1]))
            self.add_vehicle()

    def run_simulation(self):
        """Ejecutar simulación"""
        try:
            iterations = int(self.iterations_entry.get())
            self.run_button.config(state='disabled')
            self.main_app.run_simulation(iterations)
        except ValueError:
            messagebox.showerror("Error", "Número de iteraciones inválido")

    def stop_simulation(self):
        """Detener simulación"""
        self.main_app.stop_simulation()

    def reset_simulation(self):
        """Reiniciar simulación"""
        self.main_app.reset_simulation()