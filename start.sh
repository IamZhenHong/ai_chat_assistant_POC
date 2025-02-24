#!/bin/bash
uvicorn fastapi_app:app --host 0.0.0.0 --port $PORT

chmod +x start.sh
