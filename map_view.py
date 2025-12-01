import folium
import osmnx as ox

class MapView:
    def __init__(self, graph):
        self.graph = graph

    def create_map(self, output_file="map.html"):
        # Obtener punto central del grafo
        nodes = ox.graph_to_gdfs(self.graph, nodes=True, edges=False)
        center_lat = nodes.geometry.y.mean()
        center_lon = nodes.geometry.x.mean()

        # Crear mapa interactivo
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Añadir calles al mapa
        folium.GeoJson(ox.graph_to_gdfs(self.graph, nodes=False, edges=True)).add_to(m)

        # Guardar HTML
        m.save(output_file)

        print(f"Mapa generado: {output_file}")
    

    def add_route(self, m, graph, path, color="red"):
        coords = []

        for node in path:
            x = graph.nodes[node]["x"]
            y = graph.nodes[node]["y"]
            coords.append((y, x))

        folium.PolyLine(coords, weight=5).add_to(m)

