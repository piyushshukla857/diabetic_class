#!/bin/bash

# Start serve.py in the background
python serve.py &

# Start app.py in the foreground
# python app.py

gunicorn --bind 0.0.0.0:5001 --timeout 120 app:app