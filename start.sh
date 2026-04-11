#!/bin/bash
# Create admin_settings.json from template if it doesn't exist
if [ ! -f /app/admin_settings.json ]; then
    echo "Creating admin_settings.json from template..."
    cp /app/admin_settings.template.json /app/admin_settings.json
    echo "admin_settings.json created"
fi

# Set up figures directory
# On Render, figures are stored on persistent disk at /data/figures
# Locally they are in the project directory
if [ -d "/data" ]; then
    echo "Setting up persistent disk paths..."
    mkdir -p /data/figures
    mkdir -p /data/figures_metadata
    
    # Copy figures_metadata from app to persistent disk if not already there
    if [ ! -f /data/figures_metadata/figures_index.json ] && [ -f /app/figures_metadata/figures_index.json ]; then
        echo "Copying figures_metadata to persistent disk..."
        cp -r /app/figures_metadata/* /data/figures_metadata/
    fi
    
    # Symlink figures directories to persistent disk
    rm -rf /app/figures /app/figures_metadata
    ln -s /data/figures /app/figures
    ln -s /data/figures_metadata /app/figures_metadata
    
    echo "Persistent disk paths configured"
fi

echo "Starting Gas Turbine Combustion Expert API..."
exec uvicorn api.backend:app --host 0.0.0.0 --port 8000
