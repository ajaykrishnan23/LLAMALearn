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

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install Flask and Transformers
# RUN pip install --no-cache-dir flask transformers

# Download the model using huggingface-cli
# RUN pip install huggingface-cli
# RUN huggingface-cli download TheBloke/NeuralHermes-2.5-Mistral-7B-GGUF neuralhermes-2.5-mistral-7b.Q2_K.gguf --local-dir . --local-dir-use-symlinks False
# COPY neuralhermes-2.5-mistral-7b.Q2_K.gguf ./neuralhermes-2.5-mistral-7b.Q2_K.gguf



# Install llama-cpp-python with CMAKE_ARGS
# RUN CMAKE_ARGS="-DLLAMA_METAL=on" 
# RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" 

RUN pip install -r requirements.txt
# RUN pip install llama-cpp-python
# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "./app.py"]



