"""
Sistema de Optimización de Tráfico ACO - Versión DASH
Aplicación principal con mapas interactivos clickeables
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Importar módulos del sistema
from core import EnhancedCityMap, ACOTrafficOptimizer, ACO_PARAMS

# ============================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True,
    title="🚗 Optimizador de Tráfico ACO"
)

# Estado global de la aplicación
class AppState:
    def __init__(self):
        self.city_map = None
        self.optimizer = None
        self.vehicles = []
        self.blocked_roads = set()
        self.traffic_roads = set()
        self.simulation_running = False
        self.current_mode = "select_start"  # select_start, select_end, block_road, add_traffic

app_state = AppState()

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def create_map_figure(city_map, vehicles=None, highlight_nodes=None):
    """Crear figura de Plotly con el mapa"""
    if not city_map or not city_map.intersections:
        return go.Figure()
    
    fig = go.Figure()
    
    # 1. Dibujar todas las calles en gris
    for (start, end), road_data in city_map.roads.items():
        if start in city_map.intersections and end in city_map.intersections:
            x0, y0 = city_map.intersections[start]['coords']
            x1, y1 = city_map.intersections[end]['coords']
            
            # Color según estado
            color = 'lightgray'
            width = 1
            
            if (start, end) in city_map.blocked_roads:
                color = 'red'
                width = 3
            elif (start, end) in city_map.high_traffic_roads:
                color = 'orange'
                width = 2
            
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color=color, width=width),
                hoverinfo='skip',
                showlegend=False
            ))
    
    # 2. Dibujar nodos (clickeables)
    node_ids = list(city_map.intersections.keys())
    node_coords = [city_map.intersections[nid]['coords'] for nid in node_ids]
    
    if node_coords:
        xs, ys = zip(*node_coords)
        
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                opacity=0.6,
                line=dict(width=1, color='darkblue')
            ),
            text=[f"Nodo {nid}" for nid in node_ids],
            customdata=node_ids,
            hovertemplate='<b>%{text}</b><br>Click para seleccionar<extra></extra>',
            showlegend=False
        ))
    
    # 3. Dibujar nodos destacados
    if highlight_nodes:
        highlight_coords = [city_map.intersections[nid]['coords'] 
                          for nid in highlight_nodes 
                          if nid in city_map.intersections]
        if highlight_coords:
            xs, ys = zip(*highlight_coords)
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                marker=dict(size=15, color='yellow', 
                          line=dict(width=2, color='black')),
                text=[f"Nodo {nid}" for nid in highlight_nodes],
                hoverinfo='text',
                showlegend=False
            ))
    
    # 4. Dibujar rutas de vehículos
    if vehicles:
        colors = px.colors.qualitative.Set1
        for idx, vehicle in enumerate(vehicles):
            if len(vehicle.path) > 1:
                path_coords = []
                for node_id in vehicle.path:
                    if node_id in city_map.intersections:
                        path_coords.append(city_map.intersections[node_id]['coords'])
                
                if len(path_coords) > 1:
                    xs, ys = zip(*path_coords)
                    color = colors[idx % len(colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        line=dict(color=color, width=3),
                        marker=dict(size=8, color=color),
                        name=f'Vehículo {idx + 1}',
                        hovertemplate=f'<b>Vehículo {idx + 1}</b><extra></extra>'
                    ))
    
    # Configurar layout
    fig.update_layout(
        title="🗺️ Mapa Interactivo - Click en los nodos para seleccionar",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

# ============================================
# LAYOUT DE LA APLICACIÓN
# ============================================

app.layout = dbc.Container([
    # Store para datos
    dcc.Store(id='map-data', data=None),
    dcc.Store(id='vehicles-data', data=[]),
    dcc.Store(id='selection-mode', data='select_start'),
    dcc.Store(id='temp-selection', data={}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-car me-2"),
                "Optimizador de Tráfico ACO"
            ], className="text-primary mb-3"),
            html.P("Sistema interactivo de optimización de rutas con Algoritmo de Colonia de Hormigas",
                  className="lead text-muted")
        ])
    ], className="mt-4 mb-3"),
    
    html.Hr(),
    
    # Contenido principal
    dbc.Row([
        # Panel izquierdo - Controles
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4([
                    html.I(className="fas fa-cog me-2"),
                    "Panel de Control"
                ])),
                dbc.CardBody([
                    # Tabs para organizar controles
                    dbc.Tabs([
                        # Tab 1: Cargar Mapa
                        dbc.Tab(label="🗺️ Mapa", tab_id="tab-map", children=[
                            html.Div([
                                html.H5("Cargar Mapa", className="mt-3"),
                                dbc.RadioItems(
                                    id='load-method',
                                    options=[
                                        {'label': 'Por Nombre', 'value': 'name'},
                                        {'label': 'Por Coordenadas', 'value': 'coords'}
                                    ],
                                    value='name',
                                    inline=True,
                                    className="mb-3"
                                ),
                                html.Div(id='load-inputs'),
                                dbc.Button(
                                    "📍 Cargar Mapa",
                                    id='btn-load-map',
                                    color="primary",
                                    className="w-100 mt-2"
                                ),
                                html.Div(id='map-status', className="mt-3")
                            ])
                        ]),
                        
                        # Tab 2: Vehículos
                        dbc.Tab(label="🚗 Vehículos", tab_id="tab-vehicles", children=[
                            html.Div([
                                html.H5("Configurar Vehículos", className="mt-3"),
                                dbc.Alert([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Click en el mapa para seleccionar inicio y destino"
                                ], color="info", className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Nodo Inicio:"),
                                        dbc.Input(
                                            id='input-start',
                                            type='number',
                                            placeholder='ID nodo',
                                            min=0
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Nodo Destino:"),
                                        dbc.Input(
                                            id='input-end',
                                            type='number',
                                            placeholder='ID nodo',
                                            min=0
                                        )
                                    ], width=6)
                                ], className="mb-2"),
                                
                                dbc.Button(
                                    "➕ Añadir Vehículo",
                                    id='btn-add-vehicle',
                                    color="success",
                                    className="w-100 mt-2"
                                ),
                                
                                html.Div(id='vehicles-list', className="mt-3")
                            ])
                        ]),
                        
                        # Tab 3: Obstáculos
                        dbc.Tab(label="🚧 Obstáculos", tab_id="tab-obstacles", children=[
                            html.Div([
                                html.H5("Configurar Obstáculos", className="mt-3"),
                                
                                dbc.RadioItems(
                                    id='obstacle-mode',
                                    options=[
                                        {'label': '🚧 Bloquear Calle', 'value': 'block'},
                                        {'label': '🚦 Añadir Tráfico', 'value': 'traffic'}
                                    ],
                                    value='block',
                                    className="mb-3"
                                ),
                                
                                html.Div(id='obstacle-controls'),
                                
                                html.Hr(),
                                
                                dbc.Button(
                                    "🗑️ Limpiar Obstáculos",
                                    id='btn-clear-obstacles',
                                    color="danger",
                                    outline=True,
                                    className="w-100"
                                )
                            ])
                        ]),
                        
                        # Tab 4: Simulación
                        dbc.Tab(label="▶️ Simulación", tab_id="tab-simulation", children=[
                            html.Div([
                                html.H5("Ejecutar Simulación", className="mt-3"),
                                
                                dbc.Label("Iteraciones:"),
                                dcc.Slider(
                                    id='iterations-slider',
                                    min=10,
                                    max=200,
                                    step=10,
                                    value=50,
                                    marks={i: str(i) for i in range(10, 201, 30)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                
                                dbc.Accordion([
                                    dbc.AccordionItem([
                                        dbc.Row([
                                            dbc.Col([
                                                dbc.Label("Alpha (feromonas):"),
                                                dcc.Slider(
                                                    id='alpha-slider',
                                                    min=0.1, max=3.0, step=0.1,
                                                    value=ACO_PARAMS['alpha'],
                                                    marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}
                                                )
                                            ], width=12, className="mb-3"),
                                            dbc.Col([
                                                dbc.Label("Beta (heurística):"),
                                                dcc.Slider(
                                                    id='beta-slider',
                                                    min=0.1, max=3.0, step=0.1,
                                                    value=ACO_PARAMS['beta'],
                                                    marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}
                                                )
                                            ], width=12, className="mb-3"),
                                            dbc.Col([
                                                dbc.Label("Evaporación:"),
                                                dcc.Slider(
                                                    id='evaporation-slider',
                                                    min=0.1, max=0.9, step=0.05,
                                                    value=ACO_PARAMS['evaporation_rate'],
                                                    marks={i: f"{i:.1f}" for i in [0.1, 0.3, 0.5, 0.7, 0.9]}
                                                )
                                            ], width=12)
                                        ])
                                    ], title="⚙️ Parámetros Avanzados")
                                ], className="mt-3 mb-3", start_collapsed=True),
                                
                                dbc.Button(
                                    "▶️ Ejecutar Simulación",
                                    id='btn-run-simulation',
                                    color="primary",
                                    className="w-100 mt-3"
                                ),
                                
                                html.Div(id='simulation-progress', className="mt-3")
                            ])
                        ])
                    ], id="control-tabs", active_tab="tab-map")
                ])
            ], className="mb-3")
        ], width=4),
        
        # Panel derecho - Mapa y Resultados
        dbc.Col([
            # Mapa
            dbc.Card([
                dbc.CardHeader([
                    html.H4([
                        html.I(className="fas fa-map me-2"),
                        "Visualización del Mapa"
                    ]),
                    dbc.Badge(id='mode-badge', color="info", className="ms-2")
                ]),
                dbc.CardBody([
                    dcc.Graph(
                        id='map-graph',
                        figure=go.Figure(),
                        config={'displayModeBar': True, 'scrollZoom': True}
                    )
                ])
            ], className="mb-3"),
            
            # Resultados
            html.Div(id='results-section')
        ], width=8)
    ]),
    
    # Footer
    html.Hr(className="mt-5"),
    dbc.Row([
        dbc.Col([
            html.P([
                html.I(className="fas fa-bug me-2"),
                "Sistema de Optimización ACO | ",
                html.A("GitHub", href="#", className="text-decoration-none"),
                " | Versión 2.0 (Dash)"
            ], className="text-muted text-center")
        ])
    ], className="mb-4")
], fluid=True, style={'maxWidth': '1800px'})

# ============================================
# CALLBACKS
# ============================================

# Callback: Actualizar inputs de carga según método
@app.callback(
    Output('load-inputs', 'children'),
    Input('load-method', 'value')
)
def update_load_inputs(method):
    place_style = {'display': 'block'} if method == 'name' else {'display': 'none'}
    coords_style = {'display': 'block'} if method == 'coords' else {'display': 'none'}
    
    return html.Div([
        dbc.Input(
            id='place-input',
            type='text',
            placeholder='Ej: Trujillo, Peru',
            value='Trujillo, Peru',
            style=place_style
        ),
        dbc.Row([
            dbc.Col([
                dbc.Input(id='lat-input', type='number', placeholder='Latitud', value=-8.109)
            ], width=4),
            dbc.Col([
                dbc.Input(id='lon-input', type='number', placeholder='Longitud', value=-79.022)
            ], width=4),
            dbc.Col([
                dbc.Input(id='dist-input', type='number', placeholder='Radio (m)', value=1000)
            ], width=4)
        ], style=coords_style)
    ])

# Callback: Cargar mapa
@app.callback(
    [Output('map-data', 'data'),
     Output('map-status', 'children')],
    Input('btn-load-map', 'n_clicks'),
    [State('load-method', 'value'),
     State('place-input', 'value'),
     State('lat-input', 'value'),
     State('lon-input', 'value'),
     State('dist-input', 'value')],
    prevent_initial_call=True
)
def load_map(n_clicks, method, place, lat, lon, dist):
    if not n_clicks:
        return None, ""
    
    try:
        app_state.city_map = EnhancedCityMap()
        
        if method == 'name':
            app_state.city_map.load_city_from_osm(place)
            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Mapa cargado: {len(app_state.city_map.intersections)} intersecciones"
            ], color="success")
        else:
            app_state.city_map.load_city_from_point(lat, lon, dist)
            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"Mapa cargado: {len(app_state.city_map.intersections)} intersecciones"
            ], color="success")
        
        return {'loaded': True}, status
        
    except Exception as e:
        return None, dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            f"Error: {str(e)}"
        ], color="danger")

# Callback: Actualizar mapa cuando se carga
@app.callback(
    Output('map-graph', 'figure'),
    [Input('map-data', 'data'),
     Input('vehicles-data', 'data'),
     Input('temp-selection', 'data')],
    prevent_initial_call=False
)
def update_map(map_data, vehicles_data, temp_selection):
    if not app_state.city_map:
        return go.Figure()
    
    # Obtener vehículos si hay optimizador
    vehicles = app_state.optimizer.vehicles if app_state.optimizer else None
    
    # Nodos destacados
    highlight_nodes = []
    if temp_selection and 'start' in temp_selection:
        highlight_nodes.append(temp_selection['start'])
    if temp_selection and 'end' in temp_selection:
        highlight_nodes.append(temp_selection['end'])
    
    return create_map_figure(app_state.city_map, vehicles, highlight_nodes)

# Callback: Manejar clicks en el mapa
@app.callback(
    [Output('temp-selection', 'data'),
     Output('input-start', 'value'),
     Output('input-end', 'value')],
    Input('map-graph', 'clickData'),
    [State('temp-selection', 'data'),
     State('control-tabs', 'active_tab')],
    prevent_initial_call=True
)
def handle_map_click(click_data, temp_selection, active_tab):
    if not click_data or not app_state.city_map:
        return temp_selection or {}, None, None
    
    # Obtener el nodo clickeado
    point = click_data['points'][0]
    if 'customdata' in point:
        node_id = point['customdata']
    else:
        return temp_selection or {}, None, None
    
    temp_selection = temp_selection or {}
    
    # Si estamos en tab de vehículos
    if active_tab == 'tab-vehicles':
        if 'start' not in temp_selection:
            temp_selection['start'] = node_id
            return temp_selection, node_id, None
        else:
            temp_selection['end'] = node_id
            start = temp_selection['start']
            return temp_selection, start, node_id
    
    return temp_selection, None, None

# Callback: Añadir vehículo
@app.callback(
    [Output('vehicles-data', 'data'),
     Output('vehicles-list', 'children')],
    Input('btn-add-vehicle', 'n_clicks'),
    [State('input-start', 'value'),
     State('input-end', 'value'),
     State('vehicles-data', 'data')],
    prevent_initial_call=True
)
def add_vehicle(n_clicks, start, end, vehicles_data):
    if not n_clicks or start is None or end is None:
        return vehicles_data or [], html.Div()
    
    if not app_state.city_map:
        return vehicles_data or [], dbc.Alert("⚠️ Carga un mapa primero", color="warning")
    
    vehicles_data = vehicles_data or []
    vehicles_data.append({'start': start, 'end': end})
    app_state.vehicles = vehicles_data
    
    # Crear lista visual
    vehicle_list = []
    for i, v in enumerate(vehicles_data):
        vehicle_list.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(f"🚗 Vehículo {i + 1}"),
                    html.P(f"{v['start']} → {v['end']}", className="mb-0")
                ])
            ], className="mb-2", color="light")
        )
    
    return vehicles_data, html.Div(vehicle_list)

# Callback: Ejecutar simulación
@app.callback(
    [Output('simulation-progress', 'children'),
     Output('results-section', 'children')],
    Input('btn-run-simulation', 'n_clicks'),
    [State('iterations-slider', 'value'),
     State('alpha-slider', 'value'),
     State('beta-slider', 'value'),
     State('evaporation-slider', 'value'),
     State('vehicles-data', 'data')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, iterations, alpha, beta, evaporation, vehicles_data):
    if not n_clicks:
        return html.Div(), html.Div()
    
    if not app_state.city_map or not vehicles_data:
        return dbc.Alert("⚠️ Carga un mapa y añade vehículos primero", color="warning"), html.Div()
    
    # Actualizar parámetros ACO
    ACO_PARAMS.update({
        'alpha': alpha,
        'beta': beta,
        'evaporation_rate': evaporation
    })
    
    try:
        # Crear optimizador
        app_state.optimizer = ACOTrafficOptimizer(app_state.city_map)
        
        # Añadir vehículos
        for v in vehicles_data:
            app_state.optimizer.add_vehicle(v['start'], v['end'])
        
        # Ejecutar simulación (simplificado para Dash)
        for i in range(iterations):
            app_state.optimizer.run_iteration()
        
        # Resultados
        arrived = sum(1 for v in app_state.optimizer.vehicles if v.has_arrived())
        total = len(app_state.optimizer.vehicles)
        success_rate = (arrived / total * 100) if total > 0 else 0
        
        progress = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"Simulación completada: {arrived}/{total} vehículos llegaron"
        ], color="success")
        
        # Crear sección de resultados
        results = dbc.Card([
            dbc.CardHeader(html.H4([
                html.I(className="fas fa-chart-bar me-2"),
                "Resultados de la Simulación"
            ])),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H3(str(total), className="text-center"),
                        html.P("Vehículos Totales", className="text-center text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3(str(arrived), className="text-center text-success"),
                        html.P("Llegaron", className="text-center text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3(f"{success_rate:.1f}%", className="text-center text-info"),
                        html.P("Tasa de Éxito", className="text-center text-muted")
                    ], width=3),
                    dbc.Col([
                        html.H3(str(iterations), className="text-center"),
                        html.P("Iteraciones", className="text-center text-muted")
                    ], width=3)
                ])
            ])
        ])
        
        return progress, results
        
    except Exception as e:
        return dbc.Alert(f"❌ Error: {str(e)}", color="danger"), html.Div()

# ============================================
# EJECUTAR APLICACIÓN
# ============================================

if __name__ == '__main__':
    print("🚀 Iniciando aplicación Dash...")
    print("📍 Abre: http://127.0.0.1:8050")
    app.run_server(debug=True, port=8050)