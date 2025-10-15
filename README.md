Argus Agent Monitoring

## Requirements:

1. Python 3.9
2. Prometheus
3. Windows PowerShell
4. Docker Desktop

## Setup Instructions:

### Option 1: Docker Compose (Recommended)

```bash
# In VS Code terminal
docker-compose up -d
python Database\prometheus_exporter.py
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Metrics: http://localhost:8000/metrics

### Option 2: Streamlit Agent Only

```bash
# In VS Code terminal
streamlit run Agent\argus_agent.py
```

- Agent UI: http://localhost:8501

1. Run Argus Agent (VS Terminal)
   python Agent/argus_agent.py
   Note: It should run at http://localhost:8000/metrics
2. Run Prometheus (PowerShell)  
   .\prometheus.exe --config.file=prometheus.yml
   Note: It should run at: http://localhost:9090
   Note: Prom's own yml should be configured

P.S. Updated on Oct 10
