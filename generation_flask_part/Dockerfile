# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        git \
        cmake \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first, to cache them if requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python
RUN pip install llama-cpp-python

# Copy the current directory contents into the container at /usr/src/app
# This step is placed later because this content is more likely to change
COPY . .

# Define environment variable (for non-sensitive data)
ENV NAME World
ENV API_KEY "Shoshoshoshona"

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "./app.py"]
