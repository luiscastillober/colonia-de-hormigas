"""
Visualización del mapa en la interfaz
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib

matplotlib.use('Agg')


class MapDisplay:
    """Componente para visualizar el mapa en la interfaz"""

    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        self.current_figure = None
        self.canvas = None
        self.toolbar = None

        self.setup_map_display()

    def setup_map_display(self):
        """Configurar el área de visualización del mapa"""
        # Título
        ttk.Label(self.parent, text="🗺️ VISUALIZACIÓN DEL MAPA",
                  font=('Arial', 12, 'bold')).pack(pady=10)

        # Controles del mapa
        self.setup_map_controls()

        # Contenedor del mapa
        self.setup_map_container()

    def setup_map_controls(self):
        """Configurar controles del mapa"""
        map_controls = ttk.Frame(self.parent)
        map_controls.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(map_controls, text="🔄 Actualizar Mapa",
                   command=self.update_display).pack(side=tk.LEFT, padx=2)

        ttk.Button(map_controls, text="👁️ Toggle IDs",
                   command=self.toggle_ids).pack(side=tk.LEFT, padx=2)

        ttk.Button(map_controls, text="💾 Guardar Mapa",
                   command=self.save_map).pack(side=tk.LEFT, padx=2)

        ttk.Button(map_controls, text="🗑️ Limpiar",
                   command=self.clear_display).pack(side=tk.LEFT, padx=2)

    def setup_map_container(self):
        """Configurar contenedor del mapa"""
        self.map_container = ttk.Frame(self.parent, relief='sunken', borderwidth=2)
        self.map_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Label inicial
        self.placeholder_label = ttk.Label(
            self.map_container,
            text="El mapa se mostrará aquí después de cargar datos\n\n"
                 "💡 Instrucciones:\n"
                 "1. Carga un mapa usando el panel izquierdo\n"
                 "2. Configura vehículos y obstáculos\n"
                 "3. Ejecuta la simulación para ver las rutas optimizadas",
            font=('Arial', 10),
            justify=tk.CENTER
        )
        self.placeholder_label.pack(expand=True)

    def update_display(self):
        """Actualizar la visualización del mapa"""
        if not self.main_app.city_map:
            self.show_placeholder("No hay mapa cargado")
            return

        self.main_app.log("🔄 Actualizando visualización del mapa...")

        # Limpiar el contenedor
        self.clear_container()

        # Generar figura en un hilo separado
        def generate_figure():
            try:
                vehicles_to_show = self.main_app.optimizer.vehicles if self.main_app.optimizer else []
                fig = self.main_app.city_map.visualize_on_map(
                    vehicles=vehicles_to_show,
                    title="Mapa de Tráfico - Optimización ACO",
                    show_ids=self.main_app.showing_ids
                )
                return fig
            except Exception as e:
                self.main_app.log(f"❌ Error al generar figura: {str(e)}")
                return None

        # Actualizar UI en el hilo principal
        def update_ui():
            try:
                fig = generate_figure()
                if fig:
                    self.display_figure(fig)
                    self.main_app.log("✅ Mapa actualizado correctamente")
                else:
                    self.show_placeholder("Error al generar el mapa")
            except Exception as e:
                self.main_app.log(f"❌ Error al actualizar UI: {str(e)}")
                self.show_placeholder(f"Error: {str(e)}")

        # Ejecutar en hilo separado
        threading.Thread(target=update_ui, daemon=True).start()

    def display_figure(self, fig):
        """Mostrar figura de matplotlib en el contenedor"""
        self.current_figure = fig

        # Crear canvas de matplotlib
        self.canvas = FigureCanvasTkAgg(fig, self.map_container)
        self.canvas.draw()

        # Empacar canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Añadir toolbar de navegación
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_container)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_container(self):
        """Limpiar el contenedor del mapa"""
        for widget in self.map_container.winfo_children():
            widget.destroy()

    def show_placeholder(self, message: str):
        """Mostrar mensaje placeholder"""
        self.clear_container()
        self.placeholder_label = ttk.Label(
            self.map_container,
            text=message,
            font=('Arial', 10),
            justify=tk.CENTER
        )
        self.placeholder_label.pack(expand=True)

    def toggle_ids(self):
        """Alternar visualización de IDs"""
        self.main_app.toggle_show_ids()

    def save_map(self):
        """Guardar el mapa como imagen"""
        if not self.main_app.city_map:
            self.main_app.log("❌ No hay mapa cargado para guardar")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*")
                ],
                title="Guardar mapa como..."
            )

            if filename:
                if self.main_app.save_map_image(filename):
                    self.main_app.log(f"💾 Mapa guardado como: {filename}")
                else:
                    self.main_app.log("❌ Error al guardar el mapa")

        except Exception as e:
            self.main_app.log(f"❌ Error al guardar: {str(e)}")

    def clear_display(self):
        """Limpiar la visualización"""
        self.clear_container()
        self.show_placeholder("Visualización limpiada\n\nUsa 'Actualizar Mapa' para regenerar")
        self.main_app.log("🗑️ Visualización limpiada")