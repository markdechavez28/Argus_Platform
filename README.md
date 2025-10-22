Argus Agent Monitoring

## Requirements:

1. Python 3.9
2. Prometheus
3. Windows PowerShell
4. Docker Desktop

## Setup Instructions:

### Option 1: Run start.ps1 (Recommended)

```bash
# In VS Code terminal
./start.ps1
```
This will run all of it
- Argus Agent UI: http://localhost:8501
- Metrics Endpoint: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login: admin/admin)

To Stop
```bash
# In VS Code terminal
docker-compose down
```
**Note: Wait for at least 30 seconds or more after starting to load**
### Option 2: Manual Start (Docker Desktop)

```bash
# In VS Code terminal
 docker-compose up -d --build
```
To Stop
```bash
# In VS Code terminal
docker-compose down
```
**Note: Wait for at least 30 seconds or more after starting to load**

### Option 3: Streamlit Agent Only

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

### To set up the dashboard in Grafana
1. Navigate to Grafana Dashboard
3. Click "New"
2. import "Argus Environmental Telemetry Dashboard.json" from the github repository (filepath: \Argusplatform\Dashboards\Argus Environmental Telemetry Dashboar.json)
4. Add UID if applicable
5. Click Save and Open

### To update the dashboard in Grafana
1.  Copy the contents of the dashboard .json file
2.  Go to grafana and open the 'Argus Environmental Telemetry Dashboard'
3. Click the "Edit" button
4. Click the "Settings" button
5. Go to "JSON model" and paste the copied .json code
6. Click "Save Changes"
7. Click "go back to dashboard" to see updated file



P.S. Updated on Oct 20
