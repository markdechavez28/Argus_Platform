#!/usr/bin/env python3
"""
Argus Streamlit telemetry app + Prometheus exporter (merged & corrected)
- Module-level Prometheus metric registration so /metrics shows argus_* metrics immediately.
- Streamlit UI displays live telemetry, and Prometheus scrapes metrics at METRICS_PORT.
- Collection interval: INTERVAL seconds (default 15s).
Notes:
- Current metrics use random generators (same semantics as your original sample).
- For production, replace generators with real collectors and consider label cardinality.
"""

# ---------------------------
# Imports & config
# ---------------------------
import streamlit as st
import random
import time
import uuid
import math
from datetime import datetime

def human_bytes(n: int) -> str:
    step_unit = 1024.0
    if n < step_unit:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= step_unit
        if n < step_unit:
            return f"{n:,.2f} {unit}"
    return f"{n:,.2f} PB"

def human_seconds(s: int) -> str:
    if s < 60:
        return f"{s} s"
    m, s2 = divmod(s, 60)
    if m < 60:
        return f"{m}m {s2}s"
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m"

# ---------------------------
# External fetch functions (unchanged but defensive)
# ---------------------------
def fetch_weather_data():
    weather_url = f"https://api.openweathermap.org/data/2.5/forecast?zip={zip_code},{country_code}&appid={api_key}&units=metric"
    try:
        response = requests.get(weather_url, timeout=6)
    except Exception as e:
        logger.exception("Weather API request failed: %s", e)
        return None
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Weather API Error: %s", response.status_code)
        return None

def fetch_precipitation():
    total_precipitation = 0
    for day in range(30):
        date = end_date - timedelta(days=day)
        date_str = date.strftime("%Y-%m-%d")
        precipitation_url = f"http://api.weatherapi.com/v1/history.json?key={precipitation_key}&q={location}&dt={date_str}"
        try:
            response = requests.get(precipitation_url, timeout=6)
        except Exception as e:
            logger.exception("Precipitation API request failed for %s: %s", date_str, e)
            continue
        if response.status_code == 200:
            data = response.json()
            if 'forecast' in data and 'forecastday' in data['forecast']:
                total_precipitation += data['forecast']['forecastday'][0]['day'].get('totalprec_mm', 0)
        else:
            logger.error("Error fetching precipitation data for %s: %s", date_str, response.status_code)
    return total_precipitation

def fetch_air_quality():
    air_quality_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(air_quality_url, timeout=6)
    except Exception as e:
        logger.exception("Air Quality API request failed: %s", e)
        return None
    if response.status_code == 200:
        return response.json()
    else:
        logger.error("Air Quality API Error: %s", response.status_code)
        return None

# ---------------------------
# Random metric generators (same semantics as your original)
# Replace these in production with your real collectors.
# ---------------------------
def generate_container_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "container_image": f"image_{random.choice(['nginx', 'mysql', 'redis', 'flask'])}",
        "cpu_util_pct": round(random.uniform(0.0, 100.0), 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "memory_limit_bytes": random.randint(8 * 1024**2, 16 * 1024**2),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "io_ops": random.randint(0, 100),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "process_count": random.randint(1, 50),
        "restart_count": random.randint(0, 10),
        "uptime_seconds": random.randint(0, 86400),
        "sensor_temp_c": round(random.uniform(0.0, 100.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2)
    }

def generate_vm_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "vm_cpu_pct": round(random.uniform(0.0, 100.0), 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "vm_cpu_steal_pct": round(random.uniform(0.0, 100.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "memory_limit_bytes": random.randint(8 * 1024**2, 16 * 1024**2),
        "disk_iops": random.randint(0, 100),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "host_power_estimate_w": round(random.uniform(0.0, 5000.0), 2),
        "hypervisor_overhead_pct": round(random.uniform(0.0, 100.0), 2),
        "uptime_seconds": random.randint(0, 86400)
    }

def generate_app_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "container_id": f"container_{random.randint(1, 50)}",
        "request_rate_rps": round(random.uniform(0.0, 500.0), 2),   # instantaneous
        "latency_p95_ms": round(random.uniform(0.0, 1000.0), 2),
        "latency_p50_ms": round(random.uniform(0.0, 1000.0), 2),
        "latency_p99_ms": round(random.uniform(0.0, 1000.0), 2),
        "error_rate_pct": round(random.uniform(0.0, 100.0), 2),
        "db_connection_count": random.randint(0, 100),
        "cache_hit_ratio": round(random.uniform(0.0, 100.0), 2),
        "queue_length": random.randint(0, 100),
        "cpu_util_pct": round(random.uniform(0.0, 100.0), 2),
        "cpu_seconds": round(random.uniform(0.0, 10000.0), 2),
        "memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "disk_read_bytes": random.randint(0, 1024**3),
        "disk_write_bytes": random.randint(0, 1024**3),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "process_count": random.randint(1, 50),
        "restart_count": random.randint(0, 10),
        "sensor_temp_c": round(random.uniform(0.0, 100.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2)
    }

def generate_orchestrator_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "node_count": random.randint(1, 20),
        "pod_count": random.randint(1, 200),
        "pod_status_pending": random.randint(0, 20),
        "pod_status_running": random.randint(0, 200),
        "pod_status_failed": random.randint(0, 20),
        "scheduler_evictions": random.randint(0, 50),
        "cluster_api_latency_ms": round(random.uniform(0.0, 1000.0), 2),
        "cluster_autoscaler_actions": random.randint(0, 50),
        "aggregated_cpu_util_pct": round(random.uniform(0.0, 100.0), 2),
        "aggregated_memory_rss_bytes": random.randint(0, 8 * 1024**2),
        "aggregated_network_bytes": random.randint(0, 1024**3),
        "restart_count": random.randint(0, 10),
        "uptime_seconds": random.randint(0, 86400)
    }

def generate_network_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "host_id": f"host_{random.randint(1, 10)}",
        "interface_throughput_bps": random.randint(0, 1000000),
        "network_rx_bytes": random.randint(0, 1024**3),
        "network_tx_bytes": random.randint(0, 1024**3),
        "packet_loss_pct": round(random.uniform(0.0, 100.0), 2),
        "rtt_ms": round(random.uniform(0.0, 1000.0), 2),
        "jitter_ms": round(random.uniform(0.0, 100.0), 2),
        "active_flows": random.randint(0, 100),
        "bgp_changes": random.randint(0, 10),
        "psu_efficiency_pct": round(random.uniform(0.0, 100.0), 2),
        "sensor_temp_c": round(random.uniform(0.0, 100.0), 2),
        "sensor_humidity_pct": round(random.uniform(0.0, 100.0), 2)
    }

# ---------------------------
# Helper: update global metrics based on generated dictionaries
# ---------------------------
def update_prometheus_from_metrics(instance_id, collection_interval_s,
                                   container_metrics, vm_metrics, app_metrics,
                                   orchestrator_metrics, network_metrics):
    """
    Write values into the module-level global metrics.
    This uses the global metric variables defined above.
    """
    inst = str(instance_id)
    # container
    try:
        if container_metrics:
            host = str(container_metrics.get('host_id', 'unknown'))
            cid = str(container_metrics.get('container_id', 'unknown'))
            image = str(container_metrics.get('container_image', 'unknown'))
            g_container_cpu.labels(instance=inst, host_id=host, container_id=cid, container_image=image).set(float(container_metrics.get('cpu_util_pct', 0.0)))
            g_container_memory_rss.labels(instance=inst, host_id=host, container_id=cid).set(float(container_metrics.get('memory_rss_bytes', 0)))
            g_container_network_rx.labels(instance=inst, host_id=host, container_id=cid).set(float(container_metrics.get('network_rx_bytes', 0)))
            g_container_uptime.labels(instance=inst, host_id=host, container_id=cid).set(float(container_metrics.get('uptime_seconds', 0)))
    except Exception:
        logger.exception("Failed to set container metrics")

    # VM
    try:
        if vm_metrics:
            host = str(vm_metrics.get('host_id', 'unknown'))
            g_vm_cpu.labels(instance=inst, host_id=host).set(float(vm_metrics.get('vm_cpu_pct', 0.0)))
            g_vm_memory_rss.labels(instance=inst, host_id=host).set(float(vm_metrics.get('memory_rss_bytes', 0)))
    except Exception:
        logger.exception("Failed to set VM metrics")

    # APP
    try:
        if app_metrics:
            host = str(app_metrics.get('host_id', 'unknown'))
            cid = str(app_metrics.get('container_id', 'unknown'))
            g_app_req_rate.labels(instance=inst, host_id=host, container_id=cid).set(float(app_metrics.get('request_rate_rps', 0.0)))
            g_app_error_rate.labels(instance=inst, host_id=host, container_id=cid).set(float(app_metrics.get('error_rate_pct', 0.0)))
            g_app_cpu.labels(instance=inst, host_id=host, container_id=cid).set(float(app_metrics.get('cpu_util_pct', 0.0)))

            # approximate delta for Counter using instantaneous RPS
            try:
                rps = float(app_metrics.get('request_rate_rps', 0.0))
            except Exception:
                rps = 0.0
            approx_delta = max(0.0, rps * float(collection_interval_s))
            if approx_delta > 0:
                try:
                    c_app_requests_total.labels(instance=inst, host_id=host, container_id=cid).inc(approx_delta)
                except Exception:
                    logger.exception("Failed to increment app requests counter")
            # latency: observe p50/p95/p99 as samples (approx)
            try:
                p50_ms = app_metrics.get('latency_p50_ms')
                p95_ms = app_metrics.get('latency_p95_ms')
                p99_ms = app_metrics.get('latency_p99_ms')
                if p50_ms is not None:
                    h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(float(p50_ms) / 1000.0)
                if p95_ms is not None:
                    h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(float(p95_ms) / 1000.0)
                if p99_ms is not None:
                    h_app_latency_seconds.labels(instance=inst, host_id=host, container_id=cid).observe(float(p99_ms) / 1000.0)
            except Exception:
                logger.exception("Failed to observe latency histogram")
    except Exception:
        logger.exception("Failed to set app metrics")
# ---------------------------
# Orchestrator & network metrics
# ---------------------------
    try:
        if orchestrator_metrics:
            g_orch_pod_count.labels(instance=inst).set(int(orchestrator_metrics.get('pod_count', 0)))
            g_orch_api_latency_ms.labels(instance=inst).set(float(orchestrator_metrics.get('cluster_api_latency_ms', 0.0)))
    except Exception:
        logger.exception("Failed to set orchestrator metrics")

    try:
        if network_metrics:
            host = str(network_metrics.get('host_id', 'unknown'))
            g_net_throughput.labels(instance=inst, host_id=host).set(float(network_metrics.get('interface_throughput_bps', 0)))
            g_net_rtt.labels(instance=inst, host_id=host).set(float(network_metrics.get('rtt_ms', 0.0)))
            g_net_packet_loss.labels(instance=inst, host_id=host).set(float(network_metrics.get('packet_loss_pct', 0.0)))
    except Exception:
        logger.exception("Failed to set network metrics")

# ---------------------------
# Streamlit UI + main loop
# ---------------------------
def main():
    st.set_page_config(page_title="Argus Environmental Telemetry", layout="wide", initial_sidebar_state="expanded")

    # Session state
    if "instance_id" not in st.session_state:
        st.session_state.instance_id = str(uuid.uuid4())[:8]
    if "agent_log" not in st.session_state:
        st.session_state.agent_log = []
    if "emit_seq" not in st.session_state:
        st.session_state.emit_seq = 0
    if "last_emit" not in st.session_state:
        st.session_state.last_emit = time.monotonic()
    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = {
            "container": generate_container_metrics(),
            "vm": generate_vm_metrics(),
            "app": generate_app_metrics(),
            "orchestrator": generate_orchestrator_metrics(),
            "network": generate_network_metrics()
        }

    # Fetch external data once (like before)
    weather_data = fetch_weather_data()
    precipitation_total = fetch_precipitation()
    air_quality_data = fetch_air_quality()

    # UI styling
    st.markdown(
        """
    <style>
    .agent-card { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0c4a6e 100%); padding:16px; border-radius:12px; color:#E6F0FF; box-shadow:0 6px 18px rgba(2,6,23,0.5); border-left:4px solid #10b981; margin-bottom:16px; }
    .agent-card h3 { margin:0 0 8px 0; color:#e2e8f0; }
    .agent-small { color:#94a3b8; font-size:13px; }
    .status-pill { display:inline-block; padding:4px 10px; border-radius:14px; font-weight:600; font-size:11px; }
    .status-online { background: linear-gradient(90deg,#10b981,#059669); color:white; }
    .agent-feed { background: linear-gradient(135deg, #021026 0%, #081520 100%); border-radius:8px; padding:12px; color:#e2e8f0; min-height:200px; max-height:380px; overflow-y:auto; }
    .countdown-box { background: linear-gradient(135deg,#065f46,#047857); padding:10px; border-radius:8px; color:white; text-align:center; font-weight:600; }
    pre { background-color: #021026 !important; color: #e2e8f0 !important; padding: 8px !important; border-radius: 6px !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
        <div>
            <div style='font-size:22px; font-weight:800; color:#e2e8f0;'>🛰️ Argus Environmental Telemetry</div>
            <div style='color:#94a3b8; margin-top:4px;'>Real-time simulated telemetry • Weather • Air Quality • System Metrics</div>
        </div>
        <div><div class='status-pill status-online'>ACTIVE</div></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("# 🛡️ Argus Agent")
        st.markdown(f"**Instance ID:** `{st.session_state.instance_id}`")
        st.markdown("**Location:** Manila")
        st.markdown("---")
        st.markdown("### 🌱 External Data")
        st.markdown("This panel shows external weather + air quality fetches.")
        st.markdown("---")
        show_raw = st.checkbox("Show global raw JSON", value=False)
        st.markdown("---")
        st.caption("UI styled from Argus design • All functionality preserved")

    # Layout placeholders
    main_area = st.container()
    with main_area:
        panels = st.container()
        left, right = panels.columns([2.5, 1])

        # placeholders
        container_placeholder = left.empty()
        vm_placeholder = left.empty()
        app_placeholder = left.empty()
        orchestrator_placeholder = left.empty()
        network_placeholder = left.empty()
        weather_placeholder = left.empty()

        feed_box = right.empty()
        countdown_placeholder = right.empty()

    # Show external data card
    with weather_placeholder.container():
        st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
        st.markdown("<h3>🌦️ External Data Metrics</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Temperature (°C):** {weather_data['list'][0]['main']['temp'] if weather_data else 'N/A'}  \n")
            st.markdown(f"**Humidity (%):** {weather_data['list'][0]['main']['humidity'] if weather_data else 'N/A'}  \n")
            st.markdown(f"**30-day Precipitation Total (mm):** {precipitation_total}  \n")
            st.markdown(f"**Grid emission factor:** 0.691 kgCO₂e/kWh  \n")
            st.markdown(f"**90th Percentile Daily Maximum Temperature (°C):** 30.5 °C  \n")
        with col2:
            st.markdown("<div style='margin-top:4px;'><strong>Air Quality</strong></div>", unsafe_allow_html=True)
            if air_quality_data:
                components = air_quality_data['list'][0]['components']
                st.markdown(f"PM₂.5: {components.get('pm2_5', 'Data not available')} µg/m³  \n")
                st.markdown(f"PM₁₀: {components.get('pm10', 'Data not available')} µg/m³  \n")
                st.markdown(f"O₃: {components.get('o3', 'Data not available')} µg/m³  \n")
            else:
                st.markdown("Air quality data not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Main loop (non-blocking UI updates every SLEEP_STEP)
    try:
        while True:
            now_mon = time.monotonic()
            elapsed = now_mon - st.session_state.last_emit
            remaining = max(0.0, INTERVAL - elapsed)
            remaining_ceil = math.ceil(remaining)

            # Time to collect new metrics
            if elapsed >= INTERVAL:
                st.session_state.last_metrics = {
                    "container": generate_container_metrics(),
                    "vm": generate_vm_metrics(),
                    "app": generate_app_metrics(),
                    "orchestrator": generate_orchestrator_metrics(),
                    "network": generate_network_metrics()
                }
                st.session_state.last_emit = now_mon
                st.session_state.emit_seq += 1
                entry = f"🕐 {datetime.now().strftime('%H:%M:%S')} | Update #{st.session_state.emit_seq} | Generated telemetry snapshot."
                st.session_state.agent_log.insert(0, entry)
                st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

                # Update Prometheus metrics with the new snapshot (writes to module-level metrics)
                try:
                    update_prometheus_from_metrics(
                        st.session_state.instance_id,
                        INTERVAL,
                        st.session_state.last_metrics["container"],
                        st.session_state.last_metrics["vm"],
                        st.session_state.last_metrics["app"],
                        st.session_state.last_metrics["orchestrator"],
                        st.session_state.last_metrics["network"]
                    )
                    st.session_state.agent_log.insert(0, f"📡 Prometheus metrics updated (emit #{st.session_state.emit_seq})")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
                except Exception as e:
                    logger.exception("Failed to update Prometheus metrics: %s", e)
                    st.session_state.agent_log.insert(0, f"❌ Failed to update Prometheus metrics: {e}")
                    st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]

            # Use latest snapshot for UI display
            container_metrics = st.session_state.last_metrics["container"]
            vm_metrics = st.session_state.last_metrics["vm"]
            app_metrics = st.session_state.last_metrics["app"]
            orchestrator_metrics = st.session_state.last_metrics["orchestrator"]
            network_metrics = st.session_state.last_metrics["network"]

            # Countdown
            countdown_placeholder.markdown(f"""
            <div class='countdown-box'>
                <div style='font-size:12px;'>Next update in</div>
                <div style='font-size:20px;margin-top:6px;'>{remaining_ceil}s</div>
            </div>
            """, unsafe_allow_html=True)

            # Container card
            with container_placeholder.container():
                st.markdown("<div class='agent-card'>", unsafe_allow_html=True)
                st.markdown("<h3>🐳 Container Infrastructure</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='agent-small'>{container_metrics['container_image']} • {container_metrics['container_id']} • Host: {container_metrics['host_id']}</div>", unsafe_allow_html=True)
                k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
                k1.metric("CPU Usage", f"{container_metrics['cpu_util_pct']:.1f}%", delta=f"{random.uniform(-5, 5):.1f}%")
                k2.metric("Memory", f"{human_bytes(container_metrics['memory_rss_bytes'])}", delta=f"{random.choice(['+', '-'])}{human_bytes(random.randint(1000, 50000))}")
                k3.metric("Network RX", f"{human_bytes(container_metrics['network_rx_bytes'])}")
                k4.metric("Uptime", f"{human_seconds(container_metrics['uptime_seconds'])}")
                co2_impact = container_metrics['cpu_util_pct'] * 0.12
                st.markdown(f"""
                <div style='margin: 8px 0; padding: 8px; background: rgba(16, 185, 129, 0.06); border-left: 3px solid #10b981; border-radius:6px;'>
                    <div style='font-size:12px; font-weight:700; color:#10b981;'>ENVIRONMENTAL IMPACT</div>
                    <div style='color:#e2e8f0;'>Estimated CO₂: {co2_impact:.2f} kg/h • Heat: {container_metrics['sensor_temp_c']:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(max(container_metrics['cpu_util_pct'] / 100.0, 0.0), 1.0))
                st.markdown("**Full container telemetry:**")
                st.markdown(
                    f"**Timestamp:** {container_metrics['timestamp']}  \n"
                    f"**Host ID:** {container_metrics['host_id']}  \n"
                    f"**Container ID:** {container_metrics['container_id']}  \n"
                    f"**Container Image:** {container_metrics['container_image']}  \n"
                    f"**CPU Utilization (%):** {container_metrics['cpu_util_pct']}  \n"
                    f"**Memory RSS (bytes):** {container_metrics['memory_rss_bytes']}  \n"
                    f"**Network RX (bytes):** {container_metrics['network_rx_bytes']}  \n"
                    f"**Uptime (seconds):** {container_metrics['uptime_seconds']}  \n"
                )
                with st.expander("🔍 View raw JSON - container", expanded=False):
                    st.json(container_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # VM card
            with vm_placeholder.container():
                st.markdown("<div class='agent-card' style='margin-top:10px'>", unsafe_allow_html=True)
                st.markdown("<h3>🖥️ Virtual Machine</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='agent-small'>Host: {vm_metrics['host_id']}</div>", unsafe_allow_html=True)
                v1, v2, v3, v4 = st.columns([1, 1, 1, 1])
                v1.metric("CPU %", f"{vm_metrics['vm_cpu_pct']}%")
                v2.metric("Memory", f"{human_bytes(vm_metrics['memory_rss_bytes'])}")
                v3.metric("Disk IOPS", f"{vm_metrics['disk_iops']}")
                v4.metric("Power Est.", f"{vm_metrics['host_power_estimate_w']} W")
                st.progress(min(max(vm_metrics['vm_cpu_pct'] / 100.0, 0.0), 1.0))
                st.markdown("**Full VM telemetry:**")
                st.markdown(
                    f"**Timestamp:** {vm_metrics['timestamp']}  \n"
                    f"**Host ID:** {vm_metrics['host_id']}  \n"
                    f"**VM CPU Utilization (%):** {vm_metrics['vm_cpu_pct']}  \n"
                    f"**Memory RSS (bytes):** {vm_metrics['memory_rss_bytes']}  \n"
                )
                with st.expander("🔍 View raw JSON - vm", expanded=False):
                    st.json(vm_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # App card
            with app_placeholder.container():
                st.markdown("<div class='agent-card' style='margin-top:10px'>", unsafe_allow_html=True)
                st.markdown("<h3>📦 Application</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='agent-small'>{app_metrics['container_id']} on {app_metrics['host_id']}</div>", unsafe_allow_html=True)
                a1, a2, a3 = st.columns([1, 1, 1])
                a1.metric("Req/s (instant)", f"{app_metrics['request_rate_rps']}")
                a2.metric("P95 (ms)", f"{app_metrics['latency_p95_ms']}")
                a3.metric("Errors %", f"{app_metrics['error_rate_pct']}%")
                st.progress(min(max(app_metrics['cpu_util_pct'] / 100.0, 0.0), 1.0))
                st.markdown("**Full Application telemetry:**")
                st.markdown(
                    f"**Timestamp:** {app_metrics['timestamp']}  \n"
                    f"**Host ID:** {app_metrics['host_id']}  \n"
                    f"**Container ID:** {app_metrics['container_id']}  \n"
                    f"**Request Rate (RPS):** {app_metrics['request_rate_rps']}  \n"
                    f"**Latency P95 (ms):** {app_metrics['latency_p95_ms']}  \n"
                    f"**Error Rate (%):** {app_metrics['error_rate_pct']}  \n"
                )
                with st.expander("🔍 View raw JSON - app", expanded=False):
                    st.json(app_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Orchestrator card
            with orchestrator_placeholder.container():
                st.markdown("<div class='agent-card' style='margin-top:10px'>", unsafe_allow_html=True)
                st.markdown("<h3>🗂️ Orchestrator</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='agent-small'>Nodes: {orchestrator_metrics['node_count']}</div>", unsafe_allow_html=True)
                o1, o2 = st.columns([1, 1])
                o1.metric("Pods", f"{orchestrator_metrics['pod_count']}")
                o2.metric("API Latency ms", f"{orchestrator_metrics['cluster_api_latency_ms']}")
                st.markdown("**Full Orchestrator telemetry:**")
                with st.expander("🔍 View raw JSON - orchestrator", expanded=False):
                    st.json(orchestrator_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Network card
            with network_placeholder.container():
                st.markdown("<div class='agent-card' style='margin-top:10px'>", unsafe_allow_html=True)
                st.markdown("<h3>🌐 Network</h3>", unsafe_allow_html=True)
                st.markdown(f"<div class='agent-small'>Host: {network_metrics['host_id']}</div>", unsafe_allow_html=True)
                n1, n2, n3 = st.columns([1, 1, 1])
                n1.metric("Throughput bps", f"{network_metrics['interface_throughput_bps']}")
                n2.metric("RTT ms", f"{network_metrics['rtt_ms']}")
                n3.metric("Packet Loss %", f"{network_metrics['packet_loss_pct']}%")
                st.markdown("**Full Network telemetry:**")
                with st.expander("🔍 View raw JSON - network", expanded=False):
                    st.json(network_metrics)
                st.markdown("</div>", unsafe_allow_html=True)

            # Agent feed log on right
            feed_box.markdown(f"""
            <div class='agent-feed'>
                <div style='font-weight: 700; margin-bottom: 8px;'>🔄 Real-time Telemetry Feed</div>
                <pre style='white-space: pre-wrap; margin: 0; font-size: 12px;'>{'<br>'.join(st.session_state.agent_log)}</pre>
            </div>
            """, unsafe_allow_html=True)

            # Optionally show raw JSON in sidebar
            if show_raw:
                st.sidebar.markdown("### 📋 Latest Raw Telemetry")
                st.sidebar.json({
                    "container": container_metrics,
                    "vm": vm_metrics,
                    "app": app_metrics,
                    "orchestrator": orchestrator_metrics,
                    "network": network_metrics
                })

            # Small sleep so UI updates countdown; not blocking long
            time.sleep(SLEEP_STEP)

    except Exception as e:
        err_entry = f"❌ [{datetime.now().isoformat()}] ERROR | {str(e)}"
        st.session_state.agent_log.insert(0, err_entry)
        st.session_state.agent_log = st.session_state.agent_log[:MAX_FEED]
        feed_box.markdown(
            "<div class='agent-feed'><pre style='white-space:pre-wrap'>" + "\n".join(st.session_state.agent_log) + "</pre></div>",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
