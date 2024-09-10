FROM python:3.10-slim

WORKDIR /app

# Install git and other necessary build tools
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

    # COPY . /app/

# Install Python dependencies
RUN pip install  -r requirements.txt

COPY . /app/

EXPOSE 5001

EXPOSE 8000

# Copy the start script
COPY start.sh /app/start.sh

# Make sure the script is executable
RUN chmod +x /app/start.sh

# Use the start script as the entry point
CMD ["/bin/bash", "/app/start.sh"]
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]