#!/bin/bash
uvicorn fastapi_app.main:app --host 0.0.0.0 --port $PORT
