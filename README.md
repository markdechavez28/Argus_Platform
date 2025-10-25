# Argus Platform

A data-driven observability solution for monitoring environmental standards compliance in cloud-native operations.

---

## 🧩 Requirements

Before starting, make sure you have the following installed:

1. **Python 3.9**
2. **Prometheus**
3. **Windows PowerShell**
4. **Docker Desktop**

---

## ⚙️ Quick Start (Recommended)

### 1. Open Docker Desktop

Ensure Docker Desktop is **running** before proceeding.

### 2. Run the Start Script

```bash
# In VS Code terminal
./start.ps1
```

This script automatically builds and starts all Argus Platform services.

**Access the following components:**

- **Argus Agent UI:** [http://localhost:8501](http://localhost:8501)
- **Metrics Endpoint:** [http://localhost:8000/metrics](http://localhost:8000/metrics)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) (login: `admin / admin`)

**To Stop:**

```bash
docker-compose down
```

> ⏳ **Note:** Please wait at least 30 seconds after startup for all services to initialize properly.

---

## 🐳 Alternative Setup Options

### Option 2: Manual Docker Start

If you prefer to start containers manually:

```bash
docker-compose down
./start.ps1
docker-compose up -d --build Argus
```

**To Stop:**

```bash
docker-compose down
```

> ⏳ **Note:** Allow ~30 seconds for all services to fully load.

---

### Option 3: Run Streamlit Agent Only

If you only want to run the Streamlit Agent (without Prometheus/Grafana):

```bash
streamlit run Agent\argus_agent.py
```

**Access:**

- **Agent UI:** [http://localhost:8501](http://localhost:8501)
- **Metrics Endpoint:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

Or manually run:

```bash
# Run Agent
python Agent/argus_agent.py

# Run Prometheus (PowerShell)
.\prometheus.exe --config.file=prometheus.yml
```

> Prometheus config file (`prometheus.yml`) must be properly set up.

---

## 📊 Setting Up the Grafana Dashboard

1. Open **Grafana** → [http://localhost:3000](http://localhost:3000)
2. Click **+ (New)** → **Import**
3. Upload or paste the JSON file from:
   `Argusplatform/Dashboards/Argus Environmental Telemetry Dashboard.json`
4. (Optional) Add a custom **UID**
5. Click **Save & Open**

---

## 🔄 Updating the Grafana Dashboard

1. Copy contents of the updated dashboard `.json` file
2. In Grafana, open **Argus Environmental Telemetry Dashboard**
3. Click **Edit → Settings → JSON Model**
4. Paste the new JSON code
5. Click **Save Changes** → **Back to Dashboard**

---

## 🗓️ Notes

- **Last Updated:** October 23
- Ensure Docker Desktop is **running** before executing any script.
- For troubleshooting, verify that all services appear as **running** in Docker Desktop.
- P.S. This configuration should work. If assistance is needed, please contact markdechavez128@gmail.com
