"""
Página "Acerca de"
"""

import streamlit as st

st.set_page_config(
    page_title="Acerca de",
    page_icon="ℹ️"
)

st.title("ℹ️ Acerca del Optimizador de Tráfico ACO")

st.markdown("""
## 🚦 ¿Qué es este sistema?

Este es un sistema de optimización de rutas urbanas que utiliza el algoritmo de 
**Colonia de Hormigas (ACO)** para encontrar las rutas más eficientes en una ciudad.

## 🎯 Características Principales

- **Mapas reales**: Usa datos de OpenStreetMap para ciudades reales
- **Algoritmo ACO**: Optimización basada en el comportamiento de hormigas
- **Configuración flexible**: Añade obstáculos, tráfico y vehículos
- **Visualización interactiva**: Ve los resultados en tiempo real
- **Web-based**: Accesible desde cualquier navegador

## 🔧 Tecnologías Utilizadas

- **Streamlit**: Framework web para aplicaciones de datos
- **OSMnx**: Librería para trabajar con mapas de OpenStreetMap
- **NetworkX**: Análisis de grafos y redes
- **Matplotlib**: Visualización de datos
- **Python 3.9+**: Lenguaje de programación

## 📊 ¿Cómo funciona?

1. **Carga un mapa** de cualquier ciudad del mundo
2. **Configura obstáculos** como calles bloqueadas o con tráfico
3. **Añade vehículos** con origen y destino específicos
4. **Ejecuta la simulación** con el algoritmo ACO
5. **Analiza los resultados** y las rutas optimizadas

## 🐜 Algoritmo ACO

El algoritmo de Colonia de Hormigas se inspira en cómo las hormigas reales encuentran 
los caminos más cortos entre su colonia y las fuentes de comida mediante feromonas.

## 📞 Soporte

Para reportar problemas o sugerir mejoras, contacta al equipo de desarrollo.
""")