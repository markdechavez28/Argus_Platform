FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent files
COPY Agent/ ./Agent/

# Expose ports
EXPOSE 8501 8000

# Run the Streamlit app
CMD ["streamlit", "run", "Agent/argus_agent.py", "--server.port=8501", "--server.address=0.0.0.0"]
