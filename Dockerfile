# Step 1: Use an official lightweight Python image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install ONLY the absolutely required system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file into the container
COPY requirements.txt .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your local app files into the container
COPY . .

# Step 7: Expose the standard Streamlit web port
EXPOSE 8501

# Step 8: Configure Streamlit to run smoothly inside a container network
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Step 9: Define the command to start the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]