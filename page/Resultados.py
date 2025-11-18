import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Resultados Detallados",
    page_icon="📊"
)

st.title("📊 Resultados Detallados de la Simulación")

if 'simulation_results' not in st.session_state or not st.session_state.simulation_results:
    st.warning("⚠️ No hay resultados de simulación disponibles. Ejecuta una simulación primero.")
    st.stop()

results = st.session_state.simulation_results
optimizer = results['optimizer']

# Estadísticas generales
st.header("Estadísticas Generales")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_vehicles = len(optimizer.vehicles)
    st.metric("Vehículos Totales", total_vehicles)

with col2:
    arrived = sum(1 for v in optimizer.vehicles if v.has_arrived())
    st.metric("Vehículos que Llegaron", arrived)

with col3:
    success_rate = (arrived / total_vehicles * 100) if total_vehicles > 0 else 0
    st.metric("Tasa de Éxito", f"{success_rate:.1f}%")

with col4:
    st.metric("Iteraciones", results['iterations'])

# Tabla detallada de vehículos
st.header("Detalles por Vehículo")

vehicles_data = []
for i, vehicle in enumerate(optimizer.vehicles):
    vehicles_data.append({
        'Vehículo': i + 1,
        'Origen': vehicle.start,
        'Destino': vehicle.end,
        'Estado': '✅ Llegó' if vehicle.has_arrived() else '🚦 En camino',
        'Tiempo de Viaje': f"{vehicle.get_path_travel_time():.2f}",
        'Pasos': len(vehicle.path),
        'Ruta': ' → '.join(map(str, vehicle.path)) if vehicle.has_arrived() else 'N/A'
    })

df = pd.DataFrame(vehicles_data)
st.dataframe(df, use_container_width=True)

# Gráfico de progreso
st.header("Progreso de la Simulación")
if 'progress_data' in results:
    progress_df = pd.DataFrame(results['progress_data'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(progress_df['iteration'], progress_df['arrived_count'],
            marker='o', linewidth=2, markersize=4)
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Vehículos que Llegaron')
    ax.set_title('Progreso de la Simulación')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)