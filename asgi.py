from flask import Flask
from app import app  # Import the Flask app from your app.py file

# Create an ASGI application callable
asgi_app = app
