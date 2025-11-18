import sys
import os

# Añadir directorios al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


def check_dependencies():
    """Verificar dependencias necesarias"""
    try:
        import osmnx as ox
        import networkx as nx
        import matplotlib
        print("✅ Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependencias faltantes: {e}")
        print("\n📦 Para instalar las dependencias necesarias:")
        print("pip install osmnx networkx matplotlib numpy")
        print("\n💡 En Windows, puede que necesites también:")
        print("pip install geopandas pyproj cartopy folium mapclassify")
        return False


def main():
    """Función principal"""
    print("=" * 60)
    print("🚦 SISTEMA DE OPTIMIZACIÓN DE TRÁFICO ACO")
    print("=" * 60)

    # Verificar dependencias
    if not check_dependencies():
        print("\n❌ No se pueden cargar las dependencias necesarias")
        sys.exit(1)

    # Importar después de verificar dependencias
    from ui.main_window import MainWindow

    print("\n🎯 Características:")
    print("• Optimización de rutas con algoritmo ACO")
    print("• Mapas reales usando OpenStreetMap")
    print("• Visualización interactiva integrada")
    print("• Configuración de obstáculos y tráfico")
    print("• Herramientas de identificación de IDs")
    print("=" * 60)

    try:
        # Iniciar aplicación
        app = MainWindow()
        app.run()

    except Exception as e:
        print(f"❌ Error crítico: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()