#!/bin/bash

# Instalar dependencias del sistema para OSMnx
apt-get update
apt-get install -y graphviz graphviz-dev

# Instalar Python dependencies
pip install -r requirements.txt

# Crear directorio para cache de OSMnx
mkdir -p .cache/osmnx