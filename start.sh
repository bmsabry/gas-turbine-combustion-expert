#!/bin/bash
# Create admin_settings.json from template if it doesn't exist
if [ ! -f /app/admin_settings.json ]; then
    echo "Creating admin_settings.json from template..."
    cp /app/admin_settings.template.json /app/admin_settings.json
    echo "admin_settings.json created"
fi
# Start the application
exec uvicorn api.backend:app --host 0.0.0.0 --port 8000
