# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your mathematical Python scripts into the container
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to boot up your SaaS Gateway
CMD ["uvicorn", "aegis_public_api:app", "--host", "0.0.0.0", "--port", "8000"]
